"""Opt-in scheduler observation; CPU submission durations are not DMA timings."""
from collections import Counter
import json
from pathlib import Path
import time


def scheduler_class(base_scheduler, output, rank, settings):
    import torch

    class ResourceObservedScheduler(base_scheduler):
        def __init__(self, *args, **kwargs):
            #output实验输出目录； rank当前tp进程编号；settings采样间隔，nsys窗口等配置
            super().__init__(*args, **kwargs)
            directory = Path(output)
            self._resource_log = (directory / f"resources.rank{rank}.jsonl").open("w", buffering=1024 * 1024)
            self._transfer_log = (directory / f"transfers.rank{rank}.jsonl").open("w", buffering=1024 * 1024)
            self._observer_settings = settings
            self._pending_count = 0
            self._completed_count = 0
            self._committed_tokens = 0
            self._forward_calls = 0
            self._transfer_counts = {"D2H": 0, "H2D": 0}
            self._transfer_bytes = {"D2H": 0, "H2D": 0}
            physical = self.kv_pool._physical_mem_gpu
            self._page_bytes = (physical.numel() // self.kv_pool.capacity) * physical.element_size()
            # 包装函数工厂与包装器思路相同
            self._original_transfers = {"offload": self.kv_pool.offload, "restore": self.kv_pool.restore}
            self.kv_pool.offload = self._transfer_wrapper("offload", "D2H")
            self.kv_pool.restore = self._transfer_wrapper("restore", "H2D")
            kind = settings.get("capture_kind", "none")
            self._capture_remaining = ["low", "pressure"] if kind == "both" else ([] if kind == "none" else [kind])
            self._capture_phase = None
            self._capture_range_id = None
            self._capture_windows = []
            self._observer_started = None
            self._capture_group = None
            if kind != "none":
                import torch.distributed as dist
                self._capture_group = dist.new_group(backend="gloo")

        def _write(self, stream, record):
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")

        def _transfer_wrapper(self, method, direction):
            original = self._original_transfers[method]

            def transfer(first, second):
                pages = len(first)
                self._transfer_counts[direction] += 1
                sequence = self._transfer_counts[direction]
                byte_count = pages * self._page_bytes
                self._transfer_bytes[direction] += byte_count
                label = f"resource.kv_{direction} seq={sequence} pages={pages} bytes={byte_count}"
                started = time.perf_counter()
                torch.cuda.nvtx.range_push(label)
                # 会先得到返回值，再执行finally，最后返回给调用者
                try:
                    return original(first, second)
                finally:
                    torch.cuda.nvtx.range_pop()
                    self._write(self._transfer_log, dict(
                        direction=direction, sequence=sequence, step=self.step_id,
                        rank=rank, pages=pages, bytes=byte_count,
                        submit_perf_counter=started, submit_wall_time=time.time(),
                        cpu_submit_seconds=time.perf_counter() - started))
            return transfer

        def admit_requests(self, pending_requests):
            result = super().admit_requests(pending_requests)
            #在run（）刚进入，尚未准入请求时，pending_requests就包含这组负载的全部请求
            self._pending_count = len(pending_requests)
            return result
        #计数补齐让_committed_tokens包含首token，最终能与保存的输出总数对上，completed是本轮完成的请求列表
        def update_requests(self, *args, **kwargs):
            normal_before = [(request, request.len_token_ids) for request in self.active_requests
                             if request.status.name == "NORMAL"]
            completed = super().update_requests(*args, **kwargs)
            for request, previous_length in normal_before:
                if request.len_token_ids > request.len_input:
                    self._committed_tokens += max(0, request.len_token_ids - previous_length)
            self._completed_count += len(completed)
            return completed
        #_committed_token表示所有请求最终保留的输出token数之和，如果只是正确性检测可以删除
        def verify_outputs(self, request, output_token_ids, **kwargs):
            result = super().verify_outputs(request, output_token_ids, **kwargs)
            self._committed_tokens += len(output_token_ids)
            return result

        def is_request_complete(self, request, check_idx=-1):
            previous_length = request.len_token_ids
            complete = super().is_request_complete(request, check_idx)
            self._committed_tokens -= previous_length - request.len_token_ids
            return complete
        # 统计活跃队列各状态（normal，draft，verify,stall）各有多少请求；统计活跃请求kv大小
        def _state(self, metadata):
            counts = Counter(request.status.name for request in self.active_requests)
            active_kv = sum(request.len_kv_cache for request in self.active_requests)
            # kv池总容量
            capacity = self.kv_pool.capacity * self.kv_pool.page_size
            #空闲页*每页能存多少个token
            free = self.kv_pool.num_free_pages * self.kv_pool.page_size
            # 卸载队列中的请求还占着多少GPU KV槽位
            gpu_tokens_offloaded_requests = sum(request.len_kv_cache for request in self.offload_requests)
            # 采样事件，请求数量，kv状态，CPU kv统计
            return dict(step=self.step_id, rank=rank, perf_counter=time.perf_counter(),
                        wall_time=time.time(), elapsed=time.perf_counter() - self._observer_started,
                        active_requests=len(self.active_requests), pending_requests=self._pending_count,
                        completed_requests=self._completed_count, offloaded_requests=len(self.offload_requests),
                        states={name: counts[name] for name in ["NORMAL", "DRAFT", "VERIFY", "STALL"]},
                        kv_capacity_tokens=capacity, kv_free_tokens=free,
                        allocated_kv_ratio=1 - free / capacity, active_kv_tokens=active_kv,
                        active_kv_ratio=active_kv / capacity,
                        gpu_tokens_offloaded_requests=gpu_tokens_offloaded_requests,
                        cpu_pages_used=self.kv_pool._MAX_CPU_CAPACITY - len(self.kv_pool._free_cpu),
                        #0 or 1
                        normal_prefill_requests=sum(request.status.name == "NORMAL" and
                                                    request.len_kv_cache <= request.len_input
                                                    for request in self.active_requests),
                        # forward中参与计算的token数
                        query_tokens=len(metadata["token_ids"]) if metadata is not None else 0,
                        # 非空计算输出
                        forward_calls=self._forward_calls,
                        committed_output_tokens=self._committed_tokens,
                        cuda_allocated_bytes=torch.cuda.memory_allocated(self.model.device),
                        cuda_reserved_bytes=torch.cuda.memory_reserved(self.model.device),
                        kv_transfer_calls=dict(self._transfer_counts), kv_transfer_bytes=dict(self._transfer_bytes))

        def _stop_capture(self, reason):
            if self._capture_phase is None:
                return
            torch.cuda.nvtx.range_end(self._capture_range_id)
            window = self._capture_windows[-1]
            # End the measured window before tool flushing and boundary waits.
            stopped = time.perf_counter()
            window.update(stop_perf_counter=stopped, stop_step=self.step_id, reason=reason,
                          cpu_window_seconds=stopped - window["start_perf_counter"],
                          committed_output_tokens=self._committed_tokens - window["start_committed_tokens"])
            if reason == "run_failed":
                window["aborted_without_collectives"] = True
                self._capture_phase = None
                self._capture_range_id = None
                return
            self._capture_boundary()
            api_started = time.perf_counter()
            if rank == 0:
                torch.cuda.cudart().cudaProfilerStop()
            window["profiler_stop_api_seconds"] = time.perf_counter() - api_started
            self._capture_barrier()
            window["stop_boundary_total_seconds"] = time.perf_counter() - stopped
            self._capture_phase = None
            self._capture_range_id = None

        def _capture_barrier(self):
            import torch.distributed as dist
            dist.barrier(group=self._capture_group)

        def _capture_boundary(self):
            from serve.scheduler.scheduler import stream_forward
            self._capture_barrier()
            # Only at capture boundaries, not each step or normal resource sample.
            stream_forward.synchronize()
            torch.cuda.current_stream().synchronize()
            self._capture_barrier()
            # 默认目标20秒，但这个检查只在采样时执行，因此实际窗口可能超过20秒。
# _capture_phase：现在正在采哪个窗口；None表示没有采集。_capture_windows：已经开始过的窗口及其记录。_capture_remaining：还有哪些窗口没采。
        def _capture(self, state):
            if self._capture_phase is None and not self._capture_remaining:
                return
            command = None
            if rank == 0:
                if self._capture_phase is not None:
                    window = self._capture_windows[-1]
                    if time.perf_counter() - window["start_perf_counter"] >= settings.get("window_seconds", 20):
                        command = "stop"
                else:
                    phase = self._capture_remaining[0]
                    if phase == "low":
                        ready = (self.step_id >= settings.get("low_min_step", 100)
                                 and state["active_kv_tokens"] >= settings.get("low_min_kv_tokens", 4096)
                                 and state["allocated_kv_ratio"] < .7 and state["pending_requests"] > 0)
                    else:
                        ready = (state["active_kv_ratio"] >= .95
                                 and state["pending_requests"] + state["offloaded_requests"] > 0
                                 and (state["offloaded_requests"] > 0 or self._transfer_counts["D2H"] > 0))
                    if ready:
                        command = "start"
            import torch.distributed as dist
            commands = [command]
            dist.broadcast_object_list(commands, src=0, group=self._capture_group)
            command = commands[0]
            if command == "stop":
                self._stop_capture("duration_reached")
            elif command == "start":
                phase = self._capture_remaining[0]
                self._capture_remaining.pop(0)
                self._capture_phase = phase
                boundary_started = time.perf_counter()
                self._capture_boundary()
                api_started = time.perf_counter()
                if rank == 0:
                    # 开始记录
                    torch.cuda.cudart().cudaProfilerStart()
                api_seconds = time.perf_counter() - api_started
                self._capture_barrier()
                # A window spans scheduler iterations and must not use the nested range stack.
                self._capture_range_id = torch.cuda.nvtx.range_start(f"resource.capture.{phase}")
                self._capture_windows.append(dict(phase=phase, start_step=self.step_id,
                                                 start_perf_counter=time.perf_counter(), start_state=state,
                                                 start_committed_tokens=self._committed_tokens,
                                                 profiler_start_api_seconds=api_seconds,
                                                 start_boundary_total_seconds=time.perf_counter() - boundary_started))

        def prepare_forward_inputs(self):
            result = super().prepare_forward_inputs()
            if result[0] is not None:
                self._forward_calls += 1
            if self.step_id % settings.get("interval", 100) == 0:
                state = self._state(result[0])
                self._write(self._resource_log, state)
                torch.cuda.nvtx.mark("resource.state " + json.dumps(state, separators=(",", ":")))
                self._capture(state)
                self._resource_log.flush()
                self._transfer_log.flush()
            return result

        def run(self, pending_requests):
            self._observer_started = time.perf_counter()
            successful = False
            try:
                outputs = super().run(pending_requests)
                successful = True
                return outputs
            finally:
                self._stop_capture("run_completed" if successful else "run_failed")
                for method, original in self._original_transfers.items():
                    setattr(self.kv_pool, method, original)
                self._resource_log.close()
                self._transfer_log.close()
                report = dict(rank=rank, successful=successful, settings=settings,
                              windows=self._capture_windows, missing_windows=self._capture_remaining,
                              transfer_calls=self._transfer_counts, transfer_bytes=self._transfer_bytes,
                              committed_output_tokens=self._committed_tokens,
                              local_kv_page_bytes=self._page_bytes, added_per_step_cuda_synchronization=False,
                              added_capture_boundary_stream_synchronization=settings.get("capture_kind", "none") != "none",
                              profiler_api_control_rank=0, forward_calls=self._forward_calls)
                (Path(output) / f"observer.rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")

    return ResourceObservedScheduler
