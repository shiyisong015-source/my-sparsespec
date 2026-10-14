"""Measure rank-local KV packing with a fixed program budget, outside serving."""

import argparse
import hashlib
import importlib
import itertools
import json
from pathlib import Path
import random
import statistics
import time

import torch

from serve.tests.test_kv_pack import reference_pack


def measure(function, samples, repeats):
    for _ in range(10):
        function()
    torch.cuda.synchronize()
    times = []
    for _ in range(samples):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            function()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / repeats)
    return dict(mean_ms=statistics.mean(times), median_ms=statistics.median(times),
                min_ms=min(times), max_ms=max(times), sample_means_ms=times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--module", default="serve.attention.kv_pack")
    parser.add_argument("--counts", type=int, nargs="+", default=[512, 2048, 4096])
    parser.add_argument("--programs", type=int, default=2)
    parser.add_argument("--warps", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--tiles", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    if min(args.counts + [args.programs, args.samples, args.repeats]) < 1 or max(args.counts) > 32768:
        parser.error("Invalid count/program/sample/repeat values")
    module = importlib.import_module(args.module)
    pack = module.pack_selected_kv
    torch.manual_seed(330)
    shape = (36, 32768, 2, 1, 4, 128)  # Qwen3-8B rank-local geometry under TP=2.
    source = torch.randn(shape, dtype=torch.float16, device="cuda")
    selections = torch.stack([
        torch.randperm(shape[1], device="cuda")[:max(args.counts)]
        for _ in range(shape[0] * shape[4])
    ]).reshape(shape[0], shape[4], -1).int()
    torch.cuda.synchronize()
    report = dict(module=args.module, source_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                  gpu=torch.cuda.get_device_name(), device_cc=torch.cuda.get_device_capability(),
                  torch=torch.__version__, seed=330, source_shape=shape,
                  samples=args.samples, repeats=args.repeats, programs=args.programs,
                  timing="CUDA event group means; warmup excluded; includes GPU idle launch gaps",
                  bandwidth="Nominal KV read+write bytes / time; not measured DRAM traffic",
                  model_loaded=False, cases=[], started_wall_time=time.time())
    for count in args.counts:
        destination = torch.empty((shape[0], count, *shape[2:]), dtype=source.dtype, device=source.device)
        if args.profile_only:
            for _ in range(11):
                pack(source, selections, destination, count, num_programs=args.programs,
                     num_warps=args.warps[0], tokens_per_tile=args.tiles[0])
            torch.cuda.synchronize()
            print("Profile launches complete", flush=True)
            return
        expected = reference_pack(source, selections, count)
        payload_bytes = destination.numel() * destination.element_size()
        configs = list(itertools.product(args.warps, args.tiles))
        random.Random(330 + count).shuffle(configs)
        if (8, 8) in configs:
            configs.remove((8, 8))
            configs.insert(0, (8, 8))
        case = dict(count=count, payload_bytes=payload_bytes, configurations=[])
        for warps, tile in configs:
            def run():
                pack(source, selections, destination, count, num_programs=args.programs,
                     num_warps=warps, tokens_per_tile=tile)
            run()
            torch.testing.assert_close(destination, expected, rtol=0, atol=0)
            result = dict(warps=warps, threads=32 * warps, tile=tile, correctness=True,
                          **measure(run, args.samples, args.repeats))
            result["nominal_read_write_GBps"] = 2 * payload_bytes / (result["median_ms"] * 1e6)
            case["configurations"].append(result)
            print(f"count={count} warps={warps} tile={tile} median={result['median_ms']:.4f} ms", flush=True)
        reference = measure(lambda: destination.copy_(expected), args.samples, args.repeats)
        case["contiguous_copy_reference"] = reference
        case["reference_note"] = "Contiguous copy excludes gather and is only a throughput reference."
        baseline = next((x for x in case["configurations"] if (x["warps"], x["tile"]) == (8, 8)), None)
        if baseline:
            for result in case["configurations"]:
                result["speedup_vs_default"] = baseline["median_ms"] / result["median_ms"]
        report["cases"].append(case)
        del expected, destination
    report["finished_wall_time"] = time.time()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
