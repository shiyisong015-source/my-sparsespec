#!/usr/bin/env python3
"""Validate the unmodified official vLLM + LMCache control environment."""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path


EXPECTED = {
    "torch": "2.13.0",
    "vllm": "0.28.0",
    "lmcache": "0.5.5",
    "cupy": "14.2.0",
}


def report(name: str, ok: bool, detail: str) -> bool:
    print(f"[{'OK' if ok else 'FAIL':4}] {name:<12} {detail}")
    return ok


def main() -> int:
    passed: list[bool] = []
    passed.append(report(
        "python", sys.version_info[:2] == (3, 12),
        f"{sys.version.split()[0]} ({sys.executable})"))

    import torch
    import vllm

    versions: dict[str, str] = {
        "torch": importlib.metadata.version("torch"),
        "vllm": importlib.metadata.version("vllm"),
        "lmcache": importlib.metadata.version("lmcache"),
        "cupy": importlib.metadata.version("cupy-cuda13x"),
    }
    for name, version in versions.items():
        normalized = version.split("+", maxsplit=1)[0]
        passed.append(report(
            name, normalized == EXPECTED[name], version))

    source = Path(vllm.__file__).resolve().parent
    direct_url = importlib.metadata.distribution("vllm").read_text(
        "direct_url.json")
    passed.append(report(
        "official", "site-packages" in source.parts and direct_url is None,
        f"wheel at {source}"))

    from vllm.config.speculative import SpeculativeConfig
    has_sparse_attn = "sparse_attn" in str(SpeculativeConfig.__annotations__)
    passed.append(report(
        "no self-spec", not has_sparse_attn,
        "sparse_attn absent" if not has_sparse_attn else "sparse_attn present"))

    from lmcache.integration.vllm.lmcache_mp_connector import (
        LMCacheMPConnector,
    )
    passed.append(report(
        "connector", LMCacheMPConnector is not None,
        "LMCache external MP connector importable"))

    from lmcache import cuda_ops
    passed.append(report(
        "cuda ops", cuda_ops is not None,
        str(Path(cuda_ops.__file__).resolve())))
    passed.append(report(
        "cuda", torch.cuda.is_available(),
        f"available={torch.cuda.is_available()} torch_cuda={torch.version.cuda}"))

    print()
    if all(passed):
        print("Official vLLM + LMCache control environment is ready.")
        return 0
    print(f"Environment check failed: {passed.count(False)} item(s).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
