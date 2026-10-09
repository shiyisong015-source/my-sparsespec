"""One-request SparseSpec smoke test for the local Llama 3.1 8B model.

Run after: source scripts/activate_sparsespec.sh
"""

from transformers import AutoTokenizer

from serve.request.request import Request
from serve.run import run_inference


MODEL = "/data/models/Llama-3.1-8B"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tokens = tokenizer("Explain why the sky is blue in one sentence.")["input_ids"]
    request = Request(tokens, desired_output_length=4)
    results = run_inference(
        reqs=[request],
        model_name=MODEL,
        max_batch_size=1,
        tp_ranks=1,
        max_mem_per_rank=26,
        cuda_graph=False,
        async_cpu=False,
        dtype="bfloat16",
        kv_cache="pillar",
        admit_policy="offloading",
        batch_policy="greedy",
        chunk_size=256,
        clip_length=1024,
        spec_stride=4,
        budget_ratio=0.05,
        num_min_budget=16,
        temperature=0.0,
        dataset_name="local-smoke",
    )
    print(f"requests={len(results)}")
    print(f"output_tokens={results[0]['len_output']}")
    print(f"output={results[0]['generated_tokens']}")


if __name__ == "__main__":
    main()
