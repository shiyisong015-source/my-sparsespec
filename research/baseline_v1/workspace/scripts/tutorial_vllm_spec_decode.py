#!/usr/bin/env python3
"""中文入门实验：对比普通 greedy 与 vLLM ngram speculative decode。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# vLLM 0.28.0 的 ngram proposer 尚未接入 V2 model runner。
# 必须在首次 import vllm 前设置。
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "ngram"), required=True)
    parser.add_argument("--model", default="/data/models/Qwen3-4B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    return parser.parse_args()


def read_metric(metrics: list[object], name: str) -> float:
    return sum(
        float(getattr(metric, "value", 0))
        for metric in metrics
        if getattr(metric, "name", None) == name
    )


def main() -> None:
    args = parse_args()
    from vllm import LLM, SamplingParams

    unit = "春天 夏天 秋天 冬天 "
    prompts = [unit * 80 + "\n请严格继续上面的四季循环，不要解释："]
    speculative_config = None
    if args.mode == "ngram":
        speculative_config = {
            "method": "ngram",
            "num_speculative_tokens": 4,
            "prompt_lookup_min": 2,
            "prompt_lookup_max": 4,
        }

    llm = LLM(
        model=args.model,
        max_model_len=2048,
        max_num_seqs=1,
        gpu_memory_utilization=0.65,
        enforce_eager=True,
        enable_prefix_caching=False,
        disable_log_stats=False,
        seed=0,
        speculative_config=speculative_config,
    )
    try:
        params = SamplingParams(temperature=0, max_tokens=args.max_tokens)
        outputs = llm.generate(prompts, params)
        metrics = llm.get_metrics()
        records = [
            {
                "prompt": output.prompt,
                "token_ids": list(output.outputs[0].token_ids),
                "text": output.outputs[0].text,
            }
            for output in outputs
        ]
    finally:
        del llm

    num_drafts = read_metric(metrics, "vllm:spec_decode_num_drafts")
    draft_tokens = read_metric(metrics, "vllm:spec_decode_num_draft_tokens")
    accepted_tokens = read_metric(metrics, "vllm:spec_decode_num_accepted_tokens")
    acceptance_length = 1.0
    if num_drafts:
        acceptance_length += accepted_tokens / num_drafts

    payload = {
        "mode": args.mode,
        "model": args.model,
        "records": records,
        "metrics": {
            "num_drafts": num_drafts,
            "num_draft_tokens": draft_tokens,
            "num_accepted_tokens": accepted_tokens,
            "mean_acceptance_length": acceptance_length,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"mode={args.mode}")
    for key, value in payload["metrics"].items():
        print(f"{key}={value}")
    for index, record in enumerate(records, start=1):
        print(f"request={index} token_ids={record['token_ids']}")
        print(f"request={index} text={record['text']!r}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
