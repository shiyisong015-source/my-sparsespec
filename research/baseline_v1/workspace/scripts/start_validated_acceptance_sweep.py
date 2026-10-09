"""Wait for the active calibration and preflight before starting the grid."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "external/SparseSpec/log/acceptance-sweep/lcb-random100-seed330"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pids", nargs="+", type=int, required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    status = OUT / "job_status.json"
    status.write_text(json.dumps({"pid": os.getpid(), "state": "waiting_for_validation"}))
    try:
        for pid in args.wait_pids:
            while True:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(5)
        assert (OUT / "sample_manifest.json").exists(), "Sample preflight did not finish successfully"
        calibration = ROOT / "external/SparseSpec/log/acceptance-sweep/lcb-random8-seed330/Qwen3-8B/k14_r0.05/case.json"
        report = json.loads(calibration.read_text())
        assert len(report["inputs"]) == len(report["outputs"]) == 8
        boundary = OUT / "boundary-preflight/Qwen3-8B/k8_r0.05/case.json"
        assert boundary.exists(), "k8 boundary preflight did not complete successfully"
        boundary_report = json.loads(boundary.read_text())
        assert boundary_report["spec_stride"] == 8
        status.write_text(json.dumps({"pid": os.getpid(), "state": "running"}))
        subprocess.run([sys.executable, str(ROOT / "scripts/run_acceptance_sweep.py")], cwd=ROOT, check=True)
        status.write_text(json.dumps({"pid": os.getpid(), "state": "complete"}))
    except Exception as error:
        status.write_text(json.dumps({"pid": os.getpid(), "state": "failed", "error": str(error)}))
        raise


if __name__ == "__main__":
    main()
