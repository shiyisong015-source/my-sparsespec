#!/usr/bin/env python3
"""Generate deterministic token IDs for wheel/source baseline comparison."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data/models/Qwen3-4B")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import vllm
    from vllm import LLM, SamplingParams

    prompts = [
        "用一句中文解释 KV cache 在大语言模型推理中的作用。",
        "Write one sentence explaining speculative decoding.",
    ]
    llm = LLM(
        model=args.model,
        max_model_len=2048,
        max_num_seqs=1,
        gpu_memory_utilization=0.65,
        enforce_eager=True,
        enable_prefix_caching=False,
        seed=0,
    )
    try:
        params = SamplingParams(temperature=0, max_tokens=16)
        outputs = llm.generate(prompts, params)
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

    payload = {
        "model": args.model,
        "vllm_runtime_version": vllm.__version__,
        "vllm_distribution_version": importlib.metadata.version("vllm"),
        "vllm_source": str(Path(vllm.__file__).resolve()),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
