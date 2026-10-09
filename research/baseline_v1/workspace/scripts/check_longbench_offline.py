#!/usr/bin/env python3
"""Verify that the copied LongBench fixtures and dataset caches are usable."""

from __future__ import annotations

import json
import os
from pathlib import Path


def count_jsonl(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError(f"Expected an object in {path}")
            count += 1
    return count


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError("Activate the control environment in offline mode first")

    fixtures = (
        root / "data/workloads/lb_16k_18k.jsonl",
        root / "data/workloads/lbv2_30k_40k.jsonl",
    )
    for fixture in fixtures:
        print(f"[OK] {fixture.name}: {count_jsonl(fixture)} records")

    required = (
        root / ".cache/huggingface/hub/datasets--THUDM--LongBench",
        root / ".cache/huggingface/hub/datasets--THUDM--LongBench-v2",
        root / ".cache/huggingface/datasets/THUDM___long_bench",
        root / ".cache/huggingface/datasets/THUDM___long_bench-v2",
    )
    for path in required:
        if not path.is_dir():
            raise FileNotFoundError(path)
        print(f"[OK] cache: {path.relative_to(root)}")

    print("LongBench offline data check passed.")


if __name__ == "__main__":
    main()
