#!/usr/bin/env python3
"""Exercise official LMCache on official vLLM, without self-speculation."""

from __future__ import annotations

import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data/models/Qwen3-4B")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument("--lmcache-host", default="127.0.0.1")
    parser.add_argument("--lmcache-port", type=int, default=15555)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("LMCACHE_CHUNK_SIZE", "256")
    os.environ.setdefault("LMCACHE_LOCAL_CPU", "True")
    os.environ.setdefault("LMCACHE_MAX_LOCAL_CPU_SIZE", "2.0")

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        enable_prefix_caching=False,
        kv_transfer_config=KVTransferConfig(
            kv_connector="LMCacheMPConnector",
            kv_role="kv_both",
            kv_connector_module_path=(
                "lmcache.integration.vllm.lmcache_mp_connector"
            ),
            kv_connector_extra_config={
                "lmcache.mp.host": args.lmcache_host,
                "lmcache.mp.port": args.lmcache_port,
            },
        ),
    )

    shared_prefix = (
        "LMCache stores reusable key and value states for repeated prefixes. "
        "This request belongs to the official vLLM control experiment. " * 80
    )
    prompts = [
        shared_prefix + "\nSummarize the purpose in one sentence.",
        shared_prefix + "\nState why this is a useful control experiment.",
    ]

    try:
        params = SamplingParams(temperature=0, max_tokens=8)
        for index, prompt in enumerate(prompts, start=1):
            output = llm.generate([prompt], params)[0]
            print(f"request={index} output={output.outputs[0].text!r}")
    finally:
        del llm

    print("Official vLLM + LMCache smoke test completed.")


if __name__ == "__main__":
    main()
