"""Run one fixed SparseSpec draft-length / KV-budget acceptance case.

Usage after `source scripts/activate_sparsespec.sh`:
  python scripts/run_acceptance_case.py --spec-stride 4 --budget-ratio 0.05
"""

import argparse
import json
import hashlib
import time
from pathlib import Path

from transformers import AutoTokenizer

from serve.request.request import Request
from serve.run import run_inference
from serve.utils import wandb_logger


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "external/SparseSpec/eval/benchmarks/data/livecodebench/test.jsonl"
INDICES = (3,)


def load_requests(tokenizer, desired_output_length, request_file=None):
    selected = set(INDICES) if request_file is None else None
    requests = []
    metadata = []
    with (request_file or DATA).open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if selected is not None and index not in selected:
                continue
            row = json.loads(line)
            prompt = row.get("prompt") or (
                "Solve this programming problem in Python 3.\n\n"
                f"{row['question_title']}\n\n{row['question_content']}"
            )
            if row.get("starter_code"):
                prompt += f"\n\nStarter code:\n{row['starter_code']}"
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if "input_tokens" in row:
                assert len(ids) == row["input_tokens"], "Tokenizer differs from validated sample"
            requests.append(Request(ids, desired_output_length))
            metadata.append(
                {"index": index, "question_id": row["question_id"], "input_tokens": len(ids)}
            )
            if selected is not None and len(requests) == len(INDICES):
                break
    assert requests
    return requests, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--model", default="/data/models/Qwen3-8B")
    parser.add_argument("--request-file", type=Path)
    parser.add_argument("--spec-stride", type=int, required=True)
    parser.add_argument("--budget-ratio", type=float, required=True)
    parser.add_argument("--output-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.65)
    parser.add_argument("--output-root", type=Path, default=ROOT / "external/SparseSpec/log/acceptance-sweep")
    args = parser.parse_args()
    if args.request_file is not None:
        args.request_file = args.request_file.absolute()
    assert 1 <= args.spec_stride <= 14
    assert 0 < args.budget_ratio <= 1
    assert args.max_model_len > 0
    assert 1 <= args.max_batch_size <= 100
    case_dir = args.output_root / Path(args.model).name / f"k{args.spec_stride}_r{args.budget_ratio:g}"
    case_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    requests, metadata = load_requests(tokenizer, args.output_tokens, args.request_file)
    for row in metadata:
        if row["input_tokens"] + args.output_tokens > args.max_model_len:
            raise ValueError(
                f"Request {row['question_id']} needs "
                f"{row['input_tokens'] + args.output_tokens} tokens, "
                f"exceeding max_model_len={args.max_model_len}"
            )
    started = time.perf_counter()
    outputs = run_inference(
        reqs=requests,
        model_name=args.model,
        max_batch_size=args.max_batch_size,
        tp_ranks=1,
        max_mem_per_rank=26,
        cuda_graph=False,
        async_cpu=False,
        dtype="bfloat16",
        kv_cache="pillar",
        admit_policy="offloading",
        batch_policy="greedy",
        chunk_size=256,
        clip_length=args.max_model_len,
        spec_stride=args.spec_stride,
        budget_ratio=args.budget_ratio,
        num_min_budget=16,
        temperature=args.temperature,
        dataset_name="livecodebench-fixed-requests",
        result_dir=str(case_dir),
        acceptance_log_path=str(case_dir / "rounds.jsonl"),
    )
    inference_call_seconds = time.perf_counter() - started
    assert len(outputs) == len(metadata), "Not all requests completed"
    accepted = int(wandb_logger.log_data.get("spec/accepted_speculative_tokens", 0))
    drafted = int(wandb_logger.log_data.get("spec/total_speculative_tokens", 0))
    rounds = [
        json.loads(line)
        for line in (case_dir / "rounds.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rounds, "No verification rounds were logged"
    assert rounds[0]["cold_start"] is True
    assert rounds[0]["draft_kv_len"] == rounds[0]["kv_len"]
    assert sum(row["actual_draft_tokens"] for row in rounds) == drafted
    assert sum(row["accepted_draft_tokens"] for row in rounds) == accepted
    report = {
        "model": args.model,
        "max_batch_size": args.max_batch_size,
        "inference_call_seconds": inference_call_seconds,
        "dataset": str(args.request_file or DATA),
        "sample_sha256": hashlib.sha256(args.request_file.read_bytes()).hexdigest() if args.request_file else None,
        "spec_stride": args.spec_stride,
        "budget_ratio": args.budget_ratio,
        "num_min_budget": 16,
        "temperature": args.temperature,
        "enable_thinking": False,
        "desired_output_tokens": args.output_tokens,
        "max_model_len": args.max_model_len,
        "cuda_graph": False,
        "async_cpu": False,
        "inputs": metadata,
        "outputs": [{"request_id": row["id"], "output_tokens": row["len_output"]} for row in outputs],
        "verification_rounds": len(rounds),
        "post_cold_rounds": sum(not row["cold_start"] for row in rounds),
        "accepted_speculative_tokens": accepted,
        "total_speculative_tokens": drafted,
        "acceptance_rate": accepted / drafted if drafted else None,
        "post_cold_acceptance_rate": (
            sum(row["accepted_draft_tokens"] for row in rounds if not row["cold_start"])
            / sum(row["actual_draft_tokens"] for row in rounds if not row["cold_start"])
            if any(not row["cold_start"] for row in rounds)
            else None
        ),
    }
    temporary = case_dir / "case.json.tmp"
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(case_dir / "case.json")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
