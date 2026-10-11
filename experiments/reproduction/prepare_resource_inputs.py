"""Freeze distinct official samples and audit their original prompt construction."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
BENCHMARKS = REPO / "eval/benchmarks"
COUNTS = {"aime24": 30, "olympiadbench": 256, "livecodebench": 256}
REPEATS = {"aime24": 60, "olympiadbench": 3, "livecodebench": 2}


def digest(value, **kwargs):
    return hashlib.sha256(json.dumps(value, **kwargs).encode()).hexdigest()


def prepare(dataset, tokenizer, model, seed, reference_root):
    # Reuse the official helpers without importing an inference engine.
    from data_loader import load_data
    from parser import parse_ground_truth, parse_question
    from utils import construct_prompt

    data_path = BENCHMARKS / "data" / dataset / "test.jsonl"
    if not data_path.is_file():
        raise ValueError(f"Missing local dataset: {data_path}")
    examples = load_data(dataset, "test", str(BENCHMARKS / "data"))
    args = SimpleNamespace(prompt_type="qwen3-math-thinking", num_shots=0, adapt_few_shot=False)
    rows, prompts, tokens, rejected = [], [], [], []
    seen = set()
    for example in examples:
        sample_id = example["idx"]
        if str(sample_id) in seen:
            raise ValueError(f"Duplicate original sample ID in {dataset}: {sample_id}")
        seen.add(str(sample_id))
        example["question"] = parse_question(example, dataset)
        example["gt_cot"], example["gt_ans"] = parse_ground_truth(example, dataset)
        raw_prompt = construct_prompt(example, dataset, args)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": raw_prompt.strip()}], tokenize=False,
            add_generation_prompt=True, enable_thinking=True)
        ids = tokenizer(prompt)["input_ids"]
        prompts.append(prompt)
        tokens.append(ids)
        if not ids or len(ids) >= 32768:
            rejected.append(dict(sample_id=sample_id, input_tokens=len(ids)))
            continue
        rows.append(dict(sample_id=sample_id, input_ids=ids, prompt=prompt))
    # Reconstruct the previous full workload only for hash comparison, not execution.
    repeated_prompts = prompts * REPEATS[dataset]
    repeated_tokens = tokens * REPEATS[dataset]
    audit = dict(num_requests=len(repeated_tokens), input_tokens=sum(map(len, repeated_tokens)),
                 prompt_sha256=digest(repeated_prompts, ensure_ascii=False),
                 token_ids_sha256=digest(repeated_tokens))
    reference = reference_root / f"sparsespec-{dataset}-full/engine/workload.json"
    if not reference.is_file():
        raise ValueError(f"Missing previous workload audit: {reference}")
    previous = json.loads(reference.read_text())
    if any(previous.get(key) != value for key, value in audit.items()):
        raise ValueError(f"Official initial inputs do not match the previous audit: {dataset}")
    count = COUNTS[dataset]
    if len(rows) < count:
        raise ValueError(f"Only {len(rows)} eligible distinct samples; need {count} for {dataset}")
    selected = rows if dataset == "aime24" else random.Random(seed).sample(rows, count)
    if len(selected) != count:
        raise ValueError("AIME must contain exactly 30 distinct original samples")
    return dict(dataset=dataset, model_path=str(model), enable_thinking=True, seed=seed,
                prompt_type=args.prompt_type, num_shots=0, requests=selected,
                token_ids_sha256=digest([row["input_ids"] for row in selected]),
                eligible_samples=len(rows), rejected_samples=rejected,
                dataset_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                previous_workload_path=str(reference), previous_workload_match=True,
                reconstructed_previous_audit=audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=Path("/data/models/Qwen3-8B"))
    parser.add_argument("--seed", type=int, default=330)
    parser.add_argument("--reference-root", type=Path,
                        default=REPO / "log/reproduction/aligned-vllm011-m080-recovery2")
    args = parser.parse_args()
    output, model = args.output.resolve(), args.model.resolve()
    if not output.is_relative_to(REPO / "log") or output.exists():
        parser.error("Use a new directory under SparseSpec/log")
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false")
    sys.path.insert(0, str(BENCHMARKS))
    from transformers import AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    prepared = [prepare(dataset, tokenizer, model, args.seed, args.reference_root.resolve())
                for dataset in COUNTS]
    if torch.cuda.is_initialized():
        raise RuntimeError("Input preparation unexpectedly initialized CUDA")
    output.mkdir(parents=True)
    report = dict(seed=args.seed, requests={}, inputs={}, input_length_ranges={},
                  cuda_initialized=False, original_samples_repeated=False,
                  tokenizer_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in model.glob("*.json")},
                  prompt_source_sha256={name: hashlib.sha256((BENCHMARKS / name).read_bytes()).hexdigest()
                                        for name in ["math_eval.py", "data_loader.py", "parser.py", "utils.py"]})
    for data in prepared:
        path = output / f"{data['dataset']}.json"
        path.write_text(json.dumps(data, ensure_ascii=False) + "\n")
        lengths = [len(row["input_ids"]) for row in data["requests"]]
        report["requests"][data["dataset"]] = len(lengths)
        report["inputs"][data["dataset"]] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        report["input_length_ranges"][data["dataset"]] = dict(min=min(lengths), max=max(lengths),
                                                              mean=sum(lengths) / len(lengths))
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
