"""Run batch sweeps with per-batch locks across GPU workers."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'external/SparseSpec/log/acceptance-sweep/lcb-random100-seed330'
BATCHES = (4, 16, 32, 64, 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--start-now', action='store_true')
    args = parser.parse_args()
    assert args.gpu in (0, 1, 2)
    status_path = OUT / ('batch_queue_status.json' if args.gpu == 0 else f'batch_queue_gpu{args.gpu}_status.json')
    status = {'pid': os.getpid(), 'state': 'waiting_for_batch1',
              'batches': list(BATCHES), 'completed_batches': [], 'gpu': args.gpu}

    def save():
        temporary = status_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(status, indent=2))
        temporary.replace(status_path)

    save()
    try:
        while not args.start_now:
            predecessor = json.loads((OUT / 'job_status.json').read_text())
            if predecessor['state'] == 'complete':
                break
            if predecessor['state'] == 'failed':
                raise RuntimeError('batch=1 failed; subsequent batches were not started')
            try:
                os.kill(predecessor['pid'], 0)
            except ProcessLookupError:
                raise RuntimeError('batch=1 controller exited without recording success')
            time.sleep(15)
        if not args.start_now:
            grid = json.loads((OUT / 'Qwen3-8B/grid.json').read_text())
            assert len(grid) == 15 and all(len(row['outputs']) == 100 for row in grid)
        for batch in BATCHES:
            # Hold the lock throughout the sweep so GPUs never write the same case.
            with (OUT / f'batch{batch}.lock').open('a') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    print(f'Batch {batch} assigned to another GPU; skipping', flush=True)
                    continue
                grid_path = OUT / f'batch{batch}/Qwen3-8B/grid.json'
                if grid_path.exists():
                    grid = json.loads(grid_path.read_text())
                    if len(grid) == 15:
                        print(f'Batch {batch} already complete; skipping', flush=True)
                        continue
                status.update(state='running', current_batch=batch)
                save()
                print(f'Starting max_batch_size={batch} on GPU {args.gpu}', flush=True)
                environment = os.environ.copy()
                environment['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
                with (OUT / f'batch{batch}-sweep.log').open('a') as log:
                    subprocess.run(
                        [sys.executable, str(ROOT / 'scripts/run_acceptance_sweep.py'),
                         '--max-batch-size', str(batch)], cwd=ROOT,
                        env=environment, stdout=log, stderr=subprocess.STDOUT, check=True,
                    )
                status['completed_batches'].append(batch)
                save()
        status.update(state='complete', current_batch=None)
        save()
    except Exception as error:
        status.update(state='failed', error=str(error))
        save()
        raise


if __name__ == '__main__':
    main()
