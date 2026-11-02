"""TP worker control: set only the mapped-restore CUfunction carveout hint.

Reuse the existing natural-load Nsight observer and its original payload.
No serving source, scheduler policy, attention, or transfer stream is changed.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path


def worker(payload_path):
    import torch
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    from serve.attention import kv_pack
    from experiments.diagnostics.transfer_compute import worker as observed_worker
    payload = json.loads(payload_path.read_text())
    output = Path(payload["output"])
    rank = int(os.environ["RANK"])
    original = kv_pack._restore_kv_pages_bounded
    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuFuncSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    driver.cuFuncGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_void_p]
    changed = {}
    launches = 0

    class CarveoutProbe:
        def __getitem__(self, grid):
            original_launch = original[grid]

            def launch(*args, **kwargs):
                nonlocal launches
                compiled = original.warmup(*args, grid=grid, **kwargs)
                compiled._init_handles()
                key = str(compiled.hash)
                if key not in changed:
                    handle = ctypes.c_void_p(compiled.function)
                    before = ctypes.c_int()
                    status = driver.cuFuncGetAttribute(ctypes.byref(before), 9, handle)
                    if status:
                        raise RuntimeError(f"cuFuncGetAttribute failed: {status}")
                    status = driver.cuFuncSetAttribute(handle, 9, 100)
                    if status:
                        raise RuntimeError(f"cuFuncSetAttribute failed: {status}")
                    changed[key] = dict(original=before.value, requested=100,
                                        registers=compiled.n_regs, shared_bytes=compiled.metadata.shared)
                    print("RESTORE_CARVEOUT_PROBE", rank, changed[key], flush=True)
                launches += 1
                return original_launch(*args, **kwargs)

            return launch

    kv_pack._restore_kv_pages_bounded = CarveoutProbe()
    successful = False
    try:
        observed_worker(payload_path)
        successful = True
    finally:
        kv_pack._restore_kv_pages_bounded = original
        (output / f"carveout-control.rank{rank}.json").write_text(json.dumps(dict(
            rank=rank, successful=successful, restore_launches=launches, changed=changed,
            scope="Process-local restore CUfunction hint; original serving source unchanged"), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-payload", type=Path, required=True)
    worker(parser.parse_args().worker_payload.resolve())
