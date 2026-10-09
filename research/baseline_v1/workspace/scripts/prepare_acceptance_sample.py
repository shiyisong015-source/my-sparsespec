"""Validate and freeze a reproducible random LiveCodeBench sample."""
import hashlib
import json
import math
from pathlib import Path
import random
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "external/SparseSpec/log/acceptance-sweep/lcb-random100-seed330"


def main():
    tokenizer = AutoTokenizer.from_pretrained("/data/models/Qwen3-8B", local_files_only=True)
    vocab_size = len(tokenizer)
    order = list(range(1055))
    random.Random(330).shuffle(order)
    candidates = set(order[:150])
    valid, rejected = {}, []
    with (ROOT / "external/SparseSpec/eval/benchmarks/data/livecodebench/test.jsonl").open() as stream:
        for index, line in enumerate(stream):
            if index not in candidates:
                continue
            row = json.loads(line)
            try:
                assert isinstance(row["question_content"], str) and row["question_content"].strip()
                prompt = "Solve this programming problem in Python 3.\n\n" + row["question_title"] + "\n\n" + row["question_content"]
                if row.get("starter_code"):
                    prompt += "\n\nStarter code:\n" + row["starter_code"]
                ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                    tokenize=True, add_generation_prompt=True, enable_thinking=False)
                assert ids and all(0 <= token < vocab_size for token in ids)
                # Reserve the entire decode budget plus a final speculative round.
                maximum = len(ids) + 1024 + 15
                assert maximum <= 32768
                assert max(16, math.ceil(maximum * 0.15)) < 6144
                valid[index] = {"source_index": index, "question_id": row["question_id"],
                                "prompt": prompt, "input_tokens": len(ids)}
            except (AssertionError, KeyError, TypeError) as error:
                rejected.append({"source_index": index, "reason": type(error).__name__})
    sample = [valid[index] for index in order if index in valid][:100]
    assert len(sample) == 100
    assert len({row["question_id"] for row in sample}) == 100
    OUT.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sample).encode()
    (OUT / "requests.jsonl").write_bytes(data)
    manifest = {"seed": 330, "population": 1055, "sample_size": 100,
        "model": "/data/models/Qwen3-8B", "enable_thinking": False,
        "output_tokens": 1024, "max_model_len": 32768,
        "sha256": hashlib.sha256(data).hexdigest(), "rejected": rejected,
        "indices": [row["source_index"] for row in sample],
        "input_length_min": min(row["input_tokens"] for row in sample),
        "input_length_max": max(row["input_tokens"] for row in sample)}
    (OUT / "sample_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
