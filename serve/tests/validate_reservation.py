"""Short real-model reservation validation via the public run_inference API.

Run under torchrun with a JSON payload containing kwargs, requests/input_ids,
output_caps and output. This is a correctness/liveness smoke, not a benchmark.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main(payload_path):
    import torch
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    import torch.distributed as dist
    from serve.request.request import Request
    import serve.run as engine
    import serve.scheduler.reservation_scheduler as scheduling

    payload = json.loads(payload_path.read_text())
    source_root = Path(__file__).resolve().parents[2]
    sources = ["serve/run.py", "serve/request/kv_cache_ptr/compact.py",
               "serve/scheduler/reservation_table.py", "serve/scheduler/reservation_scheduler.py"]
    source_sha256 = {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
                     for name in sources}
    output = Path(payload["output"])
    output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", 0))
    original = scheduling.ReservationCompactSpecScheduler
    observed = {}

    class ObservedScheduler(original):
        def transfer_timing_snapshot(self, elapsed_seconds=None):
            report = super().transfer_timing_snapshot(elapsed_seconds)
            observed.update(report=report, metrics=dict(self.metrics))
            return report

    scheduling.ReservationCompactSpecScheduler = ObservedScheduler
    rows, caps = payload["requests"], payload["output_caps"]
    requests = [Request(row["input_ids"], cap) for row, cap in zip(rows, caps, strict=True)]
    kwargs = dict(payload["kwargs"], result_dir=str(output / "engine"),
                  acceptance_log_path=str(output / f"rounds.rank{rank}.jsonl"))
    results = engine.run_inference(requests, **kwargs)
    metrics = observed["metrics"]
    if metrics["gpu_final_free_pages"] != metrics["gpu_capacity_pages"]:
        raise AssertionError("Reservation execution leaked KV pages")
    if observed["report"]["reservations"]["reservations"]:
        raise AssertionError("Completed requests retained calendar reservations")
    if observed["report"]["reservations"].get("releases"):
        raise AssertionError("Completed requests retained memory-release dependencies")
    if not metrics["restores"] or not metrics["migrations"]:
        raise AssertionError("Smoke did not exercise a complete offload/restore cycle")
    if payload.get("require_future_capacity") and not metrics.get("reservation_release_commits"):
        raise AssertionError("Smoke did not exercise confirmed future memory releases")
    if rank == 0:
        if len(results) != len(rows):
            raise AssertionError("Not every request completed")
        completion = []
        for index, (result, row) in enumerate(zip(results, rows, strict=True)):
            if result["id"] != index or result["len_input"] != len(row["input_ids"]):
                raise AssertionError("Output identity or prompt length changed")
            completion.append(dict(request_id=index, sample_id=row["sample_id"],
                                   input_tokens=result["len_input"], output_tokens=result["len_output"]))
        (output / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    if any(hashlib.sha256((source_root / name).read_bytes()).hexdigest() != digest
           for name, digest in source_sha256.items()):
        raise AssertionError("Scheduler source changed during validation; rerun with a fixed source")
    observed.update(rank=rank, complete=True, source_sha256=source_sha256)
    (output / f"validation.rank{rank}.json").write_text(json.dumps(observed, indent=2) + "\n")
    print("RESERVATION_VALIDATION", rank, json.dumps(metrics), flush=True)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-payload", type=Path, required=True)
    main(parser.parse_args().worker_payload.resolve())
