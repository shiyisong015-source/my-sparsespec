"""Isolate native persistent attention versus independent cold-KV transfer.

Uses the serving BatchTopKAttention core, without a model, TP communication,
scheduler, allocation in measured intervals, or shared data dependencies.
Carveout changes apply only to the restore CUfunction in this process.
"""
import argparse
import ctypes
import json
from pathlib import Path
import random
import statistics
import time

import torch
import triton
import triton.language as tl

from serve.attention.backend import BatchTopKAttention
from serve.attention.kv_pack import _restore_kv_pages_bounded


@triton.jit
def low_shared_prelude(result, nanoseconds):
    start = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", constraints="=l",
                                    args=[], dtype=tl.int64, is_pure=False, pack=1)
    now = start
    while now - start < nanoseconds:
        now = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", constraints="=l",
                                      args=[], dtype=tl.int64, is_pure=False, pack=1)
    tl.store(result + tl.program_id(0), now - start)


def main(args):
    torch.cuda.set_device(0)
    torch.manual_seed(330)
    outdir = args.output
    outdir.mkdir(parents=True, exist_ok=False)
    # TP2 rank-local Qwen3-8B geometry; attention and restore are disjoint.
    layers, heads, dim = 36, 4, 128
    n = args.pages
    cold = torch.empty((layers, n, 2, 1, heads, dim), dtype=torch.float16, pin_memory=True)
    cold.fill_(.125)
    cold[:, 0].fill_(.5)
    destination = torch.empty_like(cold, device="cuda")
    pages = torch.arange(n, dtype=torch.int32, device="cuda")
    strides = cold.stride()

    def mapped():
        return _restore_kv_pages_bounded[(8,)](
            cold, destination, pages, n, layers, heads, dim,
            strides[0], strides[1], strides[2], strides[4],
            strides[0], strides[1], strides[2], strides[4],
            TOKENS=8, WIDTH=256, num_warps=8)

    compiled = mapped()
    torch.cuda.synchronize()
    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuCtxSetCacheConfig.argtypes = [ctypes.c_int]
    if args.context_max_shared:
        status = driver.cuCtxSetCacheConfig(1)
        if status:
            raise RuntimeError(f"cuCtxSetCacheConfig failed: {status}")
    driver.cuFuncSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    driver.cuFuncGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_void_p]
    driver.cuFuncSetCacheConfig.argtypes = [ctypes.c_void_p, ctypes.c_int]
    handle = ctypes.c_void_p(compiled.function)
    original = ctypes.c_int()
    status = driver.cuFuncGetAttribute(ctypes.byref(original), 9, handle)
    if status:
        raise RuntimeError(f"cuFuncGetAttribute failed: {status}")

    def carveout(value):
        result = driver.cuFuncSetAttribute(handle, 9, value)
        if result:
            raise RuntimeError(f"cuFuncSetAttribute failed: {result}")

    def cache_config(value):
        result = driver.cuFuncSetCacheConfig(handle, value)
        if result:
            raise RuntimeError(f"cuFuncSetCacheConfig failed: {result}")

    # Thirty-six sparse Draft requests plus one Verify request. Use the same
    # custom core and resource geometry as serving; do not rematerialize scores.
    kv_lens = [512] * 36 + [12000]
    q_lens = [1] * 36 + [33]
    indptr = lambda lens: torch.tensor([0, *list(__import__("itertools").accumulate(lens))],
                                       dtype=torch.int32, device="cuda")
    qi, ki = indptr(q_lens), indptr(kv_lens)
    kv_lengths = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    indices = torch.arange(sum(kv_lens), dtype=torch.int32, device="cuda")
    flat = torch.zeros((heads, 131072), dtype=torch.int32, device="cuda")
    # Each head uses the same synthetic sparse positions; storage is disjoint
    # from the independently restored cold buffer.
    flat[:, :sum(kv_lens)] = indices
    types = torch.tensor([1] * 36 + [2], dtype=torch.int32, device="cuda")
    kv = torch.randn((sum(kv_lens), 2, 1, heads, dim), dtype=torch.float16, device="cuda")
    q = torch.randn((sum(q_lens), heads * 4, dim), dtype=torch.float16, device="cuda")
    wrapper = BatchTopKAttention(num_kv_heads=heads, num_qo_heads=heads*4, head_dim=dim,
        q_data_type=torch.float16, kv_data_type=torch.float16,
        MAX_TOTAL_DRAFT_KV_LEN=131072, MAX_TOTAL_VERIFY_KV_LEN=262144, MAX_SPEC_STRIDE=33)
    wrapper.plan(qi, ki, kv_lengths, heads*4, heads, dim, dim, 1,
                 causal=True, q_data_type=torch.float16, kv_data_type=torch.float16)
    print("ATTENTION_PLAN_READY", flush=True)

    def attention():
        return wrapper.run_attention(q, kv, indices, flat, types)

    for _ in range(3):
        attention()
    torch.cuda.synchronize()
    reference, reference_lse = [tensor.clone() for tensor in attention()]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out, graph_lse = attention()
    torch.cuda.synchronize()
    compute, transfer = torch.cuda.Stream(), torch.cuda.Stream()
    default = torch.cuda.current_stream()
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    prelude_result = torch.empty(sms, dtype=torch.int64, device="cuda")
    if args.prelude_ms:
        low_shared_prelude[(sms,)](prelude_result, int(args.prelude_ms * 1e6), num_warps=4)
        torch.cuda.synchronize()
    conditions = [(mode, method) for mode in ("eager", "graph")
                  for method in ("none", "mapped_default", "mapped_max_shared", "mapped_hard_shared", "dma")]
    records = []
    print("PROBE_MEASUREMENT_START", flush=True)
    if args.profiler:
        torch.cuda.cudart().cudaProfilerStart()
    try:
        for rep in range(args.repeats):
            order = list(conditions)
            random.Random(330 + rep).shuffle(order)
            for mode, method in order:
                torch.cuda.synchronize()
                cache_config(1 if method == "mapped_hard_shared" else 0)
                carveout(100 if method in ("mapped_max_shared", "mapped_hard_shared") else original.value)
                origin, ts, te, end = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
                markers = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                           for _ in range(args.attentions)]
                origin.record(default)
                transfer.wait_event(origin); compute.wait_event(origin)
                torch.cuda.nvtx.range_push(f"probe mode={mode} transfer={method} rep={rep}")
                if args.prelude_ms:
                    with torch.cuda.stream(compute):
                        low_shared_prelude[(sms,)](prelude_result, int(args.prelude_ms * 1e6), num_warps=4)
                    time.sleep(.0005)
                with torch.cuda.stream(transfer):
                    ts.record()
                    if method.startswith("mapped"):
                        mapped()
                    elif method == "dma":
                        destination.copy_(cold, non_blocking=True)
                    te.record()
                if method != "none":
                    time.sleep(.002)
                with torch.cuda.stream(compute):
                    for a, b in markers:
                        a.record()
                        if mode == "graph":
                            graph.replay()
                            actual, actual_lse = graph_out, graph_lse
                        else:
                            actual, actual_lse = attention()
                        b.record()
                    end.record()
                torch.cuda.nvtx.range_pop()
                torch.cuda.synchronize()
                starts = [origin.elapsed_time(a) for a, b in markers]
                ends = [origin.elapsed_time(b) for a, b in markers]
                t0, t1 = origin.elapsed_time(ts), origin.elapsed_time(te)
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                torch.testing.assert_close(actual_lse, reference_lse, rtol=0, atol=0)
                row = dict(rep=rep, mode=mode, transfer=method,
                    transfer_start_ms=t0, transfer_end_ms=t1, transfer_ms=t1-t0,
                    attention_start_ms=starts, attention_end_ms=ends,
                    first_attention_ms=ends[0]-starts[0],
                    attention_batch_end_ms=origin.elapsed_time(end),
                    attentions_finished_before_transfer_end=sum(t0 < e < t1 for e in ends),
                    output_exact=True, lse_exact=True)
                records.append(row)
                print("PROBE_RESULT", json.dumps({k: v for k, v in row.items()
                      if k not in ("attention_start_ms", "attention_end_ms")}), flush=True)
    finally:
        torch.cuda.synchronize()
        if args.profiler:
            torch.cuda.cudart().cudaProfilerStop()
        carveout(original.value)
        cache_config(0)
    torch.testing.assert_close(destination[:, [0, n//2, n-1]].cpu(),
                               cold[:, [0, n//2, n-1]], rtol=0, atol=0)
    groups = []
    for mode, method in conditions:
        rows = [r for r in records if r["mode"] == mode and r["transfer"] == method]
        groups.append(dict(mode=mode, transfer=method, repeats=len(rows),
            median_transfer_ms=statistics.median(r["transfer_ms"] for r in rows),
            median_first_attention_ms=statistics.median(r["first_attention_ms"] for r in rows),
            median_attention_batch_end_ms=statistics.median(r["attention_batch_end_ms"] for r in rows),
            median_attentions_finished_before_transfer_end=statistics.median(
                r["attentions_finished_before_transfer_end"] for r in rows)))
    result = dict(gpu=torch.cuda.get_device_name(), properties=str(torch.cuda.get_device_properties(0)),
        restore_bytes=cold.numel()*cold.element_size(), restore_ctas=8,
        original_restore_carveout=original.value, attention_layers=args.attentions, prelude_ms=args.prelude_ms,
        context_max_shared=args.context_max_shared,
        attention_kv_lens=kv_lens, attention_q_lens=q_lens, groups=groups, records=records,
        notes=["Synthetic rank-local native attention; no TP, scheduler, shared transfer inputs, or cold-restore dependencies.",
               "Carveout is a per-function hint; verify effective sharedMemoryExecuted with Nsight.",
               "Attention event intervals include launch gaps; Nsight gives kernel execution intervals.",
               "DMA copies contiguous disjoint storage and is a mechanism control, not a production page-scatter implementation."])
    (outdir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("PROBE_COMPLETE", json.dumps(groups, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pages", type=int, default=16000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--attentions", type=int, default=36)
    parser.add_argument("--profiler", action="store_true")
    parser.add_argument("--prelude-ms", type=float, default=0,
                        help="Place a low-shared-memory kernel on all SMs before restoration (diagnostic only)")
    parser.add_argument("--context-max-shared", action="store_true",
                        help="Process-local context preference control; not a serving change")
    main(parser.parse_args())
