"""Public inference entry for the opt-in compact KV serving implementation."""

import json
import logging
import os
import time

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from serve.attention.kv_buffer_pool import KVBufferPool
from serve.distribute.tp import setup_tp_worker
from serve.model.model import Transformer
from serve.scheduler.packed_scheduler import PackedSpecScheduler, coordinated_call
from serve.utils import get_local_rank, setup_seed, str_to_dtype, wandb_logger


def run_packed_inference(reqs, model_name, *, max_batch_size, tp_ranks, max_mem_per_rank,
                         mem_fraction, dtype, reduce_method, chunk_size, clip_length,
                         spec_stride, budget_ratio, num_min_budget, temperature, seed,
                         result_dir, acceptance_log_path, gpu_pool_mib=None, cpu_pool_mib=None):
    device = torch.device(f"cuda:{get_local_rank()}")
    torch.cuda.set_device(device)
    dtype = str_to_dtype(dtype)
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)
    setup_seed(seed)
    model = Transformer.from_pretrained(repo_id=model_name, device=device, dtype=dtype)
    model = setup_tp_worker(model, tp_ranks, str(device), dtype, reduce_method)
    control = dist.new_group(backend="gloo") if tp_ranks > 1 else None
    gpu_pool = cpu_pool = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        limit = model.config.max_position_embeddings
        clip = min(clip_length, limit) if clip_length is not None else limit
        for request in reqs:
            remaining = clip - request.len_input
            request.len_output = min(remaining, request.len_output) if request.len_output is not None else remaining
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(device)
        budget = min(total * mem_fraction, total if max_mem_per_rank is None
                     else max_mem_per_rank * 1024**3)
        # Keep transient logits, model activations, and FlashInfer workspaces
        # outside the arena. Explicit budgets never silently shrink on failure.
        available = min(free, budget - torch.cuda.memory_allocated(device)) - 2 * 1024**3
        gpu_bytes = (min(int(available), 4 * 1024**3) if gpu_pool_mib is None
                     else int(gpu_pool_mib * 1024**2))
        cpu_bytes = (2 * gpu_bytes if cpu_pool_mib is None else int(cpu_pool_mib * 1024**2))
        def check_budget():
            if gpu_bytes < 1 or gpu_bytes > available or cpu_bytes < 1:
                raise ValueError("Invalid packed arena budgets or insufficient memory after model loading")
            return gpu_bytes, cpu_bytes
        coordinated_call(control, "arena budget check", check_budget)
        if control is not None and gpu_pool_mib is None:
            budgets = [None] * tp_ranks
            dist.all_gather_object(budgets, gpu_bytes, group=control)
            gpu_bytes = min(budgets)
            if cpu_pool_mib is None:
                cpu_bytes = 2 * gpu_bytes
        gpu_pool = coordinated_call(control, "GPU arena reservation",
                                     lambda: KVBufferPool(gpu_bytes, device=device),
                                     lambda pool: pool.close())
        cpu_pool = coordinated_call(control, "CPU arena reservation",
                                     lambda: KVBufferPool(cpu_bytes), lambda pool: pool.close())
        worker = PackedSpecScheduler(
            model, gpu_pool, cpu_pool, max_batch_size=max_batch_size, chunk_size=chunk_size,
            spec_stride=spec_stride, budget_ratio=budget_ratio, num_min_budget=num_min_budget,
            temperature=temperature, eos_token_id=(model.config.eos_token_id, tokenizer.eos_token_id),
            tp_ranks=tp_ranks, control_group=control, result_dir=result_dir,
            acceptance_log_path=acceptance_log_path,
        )
        started = time.perf_counter()
        with torch.inference_mode():
            completed = worker.run(reqs)
        elapsed = time.perf_counter() - started
        outputs = [dict(id=request.request_id, len_input=request.len_input,
                        len_output=len(request.token_ids) - request.len_input,
                        output_token_ids=list(request.token_ids[request.len_input:]),
                        tokens=tokenizer.decode(request.token_ids),
                        generated_tokens=tokenizer.decode(request.token_ids[request.len_input:]))
                   for request in completed]
        stats = dict(implementation="packed", elapsed_seconds=elapsed, requests=len(completed),
                     output_throughput=sum(row["len_output"] for row in outputs) / elapsed,
                     model_name=model_name, tp_ranks=tp_ranks, dtype=str(dtype),
                     temperature=temperature, seed=seed, chunk_size=chunk_size, clip_length=clip,
                     k=spec_stride, r=budget_ratio, max_batch_size=max_batch_size,
                     gpu_pool_bytes=gpu_bytes, cpu_pool_bytes=cpu_bytes, **worker.metrics)
        wandb_logger.log(dict(accepted_speculative_tokens=worker.accepted,
                              total_speculative_tokens=worker.proposed), categories="spec")
        rank = dist.get_rank() if control is not None else 0
        logging.info("Compact KV results: %s", stats)
        if result_dir is not None and rank == 0:
            for name, value in (("requests.json", outputs), ("summary.json", stats)):
                with open(os.path.join(result_dir, name), "w") as file:
                    json.dump(value, file, indent=2, ensure_ascii=False)
                    file.write("\n")
        return outputs
    finally:
        if cpu_pool is not None:
            cpu_pool.close(wait=True)
        if gpu_pool is not None:
            gpu_pool.close(wait=True)
        if control is not None:
            dist.destroy_process_group(control)
