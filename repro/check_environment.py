"""Check dependencies and native library loading; GPU allocation is opt-in."""
import argparse
import ctypes
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("Use Python 3.12")
    repo = Path(__file__).resolve().parents[1]
    reference = json.loads((repo / "repro/environment/reference-environment.json").read_text())
    required = ["torch", "triton", "transformers", "vllm", "datasets", "numpy", "scipy", "matplotlib"]
    versions = {name: importlib.metadata.version(name) for name in required}
    for name in required:
        if versions[name].split("+")[0] != reference["python_packages"][name]:
            raise SystemExit(f"Version mismatch: {name}={versions[name]}")
    import torch
    if torch.version.cuda != "12.8":
        raise SystemExit(f"Expected CUDA 12.8 PyTorch, got {torch.__version__}")
    import flashinfer
    if not Path(flashinfer.__file__).resolve().is_relative_to(repo / "3rdparty/flashinfer"):
        raise SystemExit("Custom FlashInfer must be imported from the pinned submodule")
    prefix = Path(os.environ["INSTALL_PREFIX"])
    for name in ["include/raft/matrix/detail/select_k.cuh", "include/rmm/version_config.hpp",
                 "include/raft/version_config.hpp", "lib/librmm.so", "lib/librapids_logger.so"]:
        if not (prefix / name).is_file():
            raise SystemExit(f"Missing native dependency: {prefix / name}")
    ctypes.CDLL(str(prefix / "lib/librapids_logger.so"), mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(str(prefix / "lib/librmm.so"), mode=ctypes.RTLD_GLOBAL)
    from vllm import _custom_ops
    from serve.attention.backend import BatchTopKWrapper
    toolkit = subprocess.check_output([str(Path(os.environ["CUDA_HOME"]) / "bin/nvcc"),
                                       "--version"], text=True)
    if "release 12.8" not in toolkit:
        raise SystemExit(f"Use CUDA 12.8 nvcc for the pinned environment:\n{toolkit}")
    result = dict(python=sys.version, packages=versions, torch_build=torch.__version__,
                  flashinfer=str(flashinfer.__file__), native_prefix=str(prefix), nvcc=toolkit,
                  custom_allreduce_imported=True, gpu_checks_requested=args.gpu)
    if args.gpu:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is unavailable")
        result["gpus"] = [{"index": i, "name": torch.cuda.get_device_name(i),
                           "total_bytes": torch.cuda.get_device_properties(i).total_memory}
                          for i in range(torch.cuda.device_count())]
        # Compile and execute the native TopK on one explicitly selected GPU.
        wrapper = BatchTopKWrapper(MAX_TOTAL_VERIFY_KV_LEN=128, MAX_DRAFT_KV_LEN=16)
        values = torch.arange(128, dtype=torch.float16, device="cuda")[None, :]
        ids = torch.arange(128, dtype=torch.int32, device="cuda")
        zero = torch.zeros(1, dtype=torch.int32, device="cuda")
        lengths = torch.tensor([128], dtype=torch.int32, device="cuda")
        ks = torch.tensor([8], dtype=torch.int32, device="cuda")
        _, selected = wrapper.run(in_val=values, in_idx=ids, in_val_start_pos=zero,
            in_idx_start_pos=zero, in_lens=lengths, ks=ks, num_requests=1)
        if sorted(selected[0, 0, :8].cpu().tolist()) != list(range(120, 128)):
            raise SystemExit("Native TopK smoke test failed")
        result["native_topk_smoke"] = "passed"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
