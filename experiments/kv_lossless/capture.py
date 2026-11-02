"""Capture real compact cold KV after native selection; serving sources unchanged."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def prepare(output):
    from transformers import AutoTokenizer
    source = REPO / "log/v2/samples-longbench-seed330/confirm.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines() if line]
    tokenizer = AutoTokenizer.from_pretrained("/data/models/Qwen3-8B", local_files_only=True)
    requests = []
    for target in [2048, 8192, 16384]:
        row = min(rows, key=lambda item: abs(item["input_tokens"] - target))
        ids = tokenizer.apply_chat_template([dict(role="user", content=row["prompt"])],
            tokenize=True, add_generation_prompt=True, enable_thinking=True)
        requests.append(dict(sample_id=row["sample_id"], input_ids=ids, source=row["source"],
                             target_tokens=target, input_tokens=len(ids)))
    output.mkdir(parents=True, exist_ok=False)
    defaults = json.loads((REPO / "repro/serving-defaults.json").read_text())
    kwargs = dict(defaults, model_name="/data/models/Qwen3-8B", tp_ranks=2,
        max_batch_size=1, mem_fraction=.65, spec_stride=32, budget_ratio=.05,
        compact_selected_capacity=6144, compact_pool_pages=22000, compact_cpu_pages=22000,
        compact_dynamic_admission=True, max_num_batched_tokens=512)
    payload = dict(output=str(output), requests=requests, output_caps=[40] * len(requests),
                   kwargs=kwargs, capture_layers=[0, 11, 23, 35],
                   capture_source="first native selection after first verify following complete prefill",
                   input_file=str(source), input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   serving_sha256={str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted((REPO / "serve").rglob("*.py"))})
    save(output / "worker-payload.json", payload)
    print(json.dumps(dict(output=str(output), requests=[{k:v for k,v in r.items() if k != "input_ids"}
                                                       for r in requests]), ensure_ascii=False), flush=True)


def worker(path):
    import numpy as np
    import torch
    import torch.distributed as dist
    from serve.attention.kv_pack import pack_selected_kv
    from serve.attention.kv_partition import partition_pages
    from serve.request.kv_cache_ptr.compact import CompactCachePtr
    from experiments.reproduction.run_compact_acceptance_scan import worker as run_worker
    payload = json.loads(path.read_text())
    for name, expected in payload["serving_sha256"].items():
        if hashlib.sha256((REPO / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Serving source changed: {name}")
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    output = Path(payload["output"])
    original = CompactCachePtr.keep_full_resident
    captured = []

    def capture(cache):
        if not getattr(cache, "_lossless_captured", False):
            sample_index = len(captured)
            if sample_index >= len(payload["requests"]):
                raise RuntimeError("Unexpected cache adoption order")
            sample = payload["requests"][sample_index]
            # Native selection follows the first verify, whose accepted tail
            # can grow by the complete speculative window plus a bonus token.
            upper = sample["input_tokens"] + payload["kwargs"]["spec_stride"] + 2
            if not sample["input_tokens"] <= cache.len_kv_cache <= upper:
                raise RuntimeError(f"Unexpected first-selection context: {cache.len_kv_cache}; "
                                   f"expected {sample['input_tokens']}..{upper}")
            physical = cache.kv_pool._physical_mem_gpu
            full = torch.tensor(tuple(cache.indices), dtype=torch.int32, device=physical.device)
            plan = partition_pages(full, cache.selected_indices_buffer_gpu,
                cache.len_selected_indices_gpu, pool_capacity=cache.kv_pool.capacity,
                recent_count=cache.recent_count, validate=True)
            shape = (len(payload["capture_layers"]), plan.cold_count, *physical.shape[2:])
            cold = torch.empty(shape, dtype=physical.dtype, device="cpu", pin_memory=True)
            for index, layer in enumerate(payload["capture_layers"]):
                pack_selected_kv(physical[layer:layer + 1], plan.cold_pages[layer:layer + 1],
                                 cold[index:index + 1], plan.cold_count)
            torch.cuda.synchronize()
            bits = cold.numpy().view(np.uint16)
            if not np.isfinite(cold.numpy()).all():
                raise RuntimeError("Captured used KV contains nonfinite values")
            filename = f"sample{sample_index}.rank{rank}.cold.npy"
            np.save(output / filename, bits, allow_pickle=False)
            # Validate against direct source reads, independently of packing.
            for index, layer in enumerate(payload["capture_layers"]):
                for head in range(physical.shape[-2]):
                    ids = plan.cold_pages[layer, head, :plan.cold_count].long()
                    reference = physical[layer, ids, :, 0, head, :].cpu().numpy().view(np.uint16)
                    if not np.array_equal(reference, bits[index, :, :, 0, head, :]):
                        raise RuntimeError("Packed cold bytes do not match native source KV")
            record = dict(sample_index=sample_index, sample_id=sample["sample_id"],
                input_tokens=sample["input_tokens"], context_tokens=cache.len_kv_cache,
                cold_tokens_per_layer_head=plan.cold_count, hot_tokens_per_layer_head=plan.hot_count,
                selected_prefix=cache.len_selected_indices_gpu, recent=cache.recent_count,
                layers=payload["capture_layers"], rank=rank, dtype=str(physical.dtype),
                layout="layers,cold_slots,K_or_V,page_size_1,local_KV_heads,head_dim",
                shape=list(bits.shape), file=filename, bytes=bits.nbytes,
                sha256=hashlib.sha256(bits.tobytes()).hexdigest(),
                bitwise_matches_source=True, finite=True,
                zero_fraction=float(np.mean(cold.numpy() == 0)))
            captured.append(record)
            save(output / f"capture.rank{rank}.json", captured)
            cache._lossless_captured = True
            print("COLD_CAPTURE", json.dumps(record), flush=True)
            del bits, cold, plan
        return original(cache)

    CompactCachePtr.keep_full_resident = capture
    run_worker(path)
    if len(captured) != len(payload["requests"]):
        raise RuntimeError(f"Only captured {len(captured)} samples")
    save(output / f"capture-complete.rank{rank}.json", dict(successful=True, samples=len(captured)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--worker-payload", type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare.resolve())
    elif args.worker_payload:
        worker(args.worker_payload.resolve())
    else:
        parser.error("Choose --prepare or --worker-payload")
