#!/usr/bin/env python3
"""Compare token IDs emitted by wheel and unmodified editable source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("control", type=Path)
    parser.add_argument("dev", type=Path)
    args = parser.parse_args()
    control = json.loads(args.control.read_text(encoding="utf-8"))
    dev = json.loads(args.dev.read_text(encoding="utf-8"))

    control_ids = [record["token_ids"] for record in control["records"]]
    dev_ids = [record["token_ids"] for record in dev["records"]]
    if control_ids != dev_ids:
        raise SystemExit("FAIL: wheel and editable-source token IDs differ")
    print(f"PASS: {len(control_ids)} prompts match token for token")


if __name__ == "__main__":
    main()
