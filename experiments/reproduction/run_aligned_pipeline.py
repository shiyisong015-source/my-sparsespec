"""Gate identical-batch SparseSpec/vLLM full workloads on boundary preflights."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from analyze_results import summarize

REPO = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).with_name('run_official.py')
PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]
DATASETS = ['olympiadbench', 'livecodebench', 'aime24']
ENGINES = ['sparsespec', 'vllm']


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z'
    except FileNotFoundError:
        return False


def free_pairs():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                  '--format=csv,noheader,nounits'], text=True)
    states = {int(line.split(',')[0]): [int(x) for x in line.split(',')[1:]] for line in raw.splitlines()}
    return [pair for pair in PAIRS if all(states[g][0] < 128 and states[g][1] == 0 for g in pair)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mem-fraction', default=.85, type=float)
    parser.add_argument('--smoke-output', type=Path)
    parser.add_argument('--smoke-pid', type=int)
    parser.add_argument('--recovery-root', type=Path,
                        help='Preserve the old run and reuse validated preflights/completed cases')
    parser.add_argument('--idle-grace-seconds', type=float, default=60,
                        help='Require a GPU pair to remain idle before starting a new case')
    args = parser.parse_args()
    base = args.output.resolve()
    if not base.is_relative_to(REPO / 'log') or not 0 < args.mem_fraction < 1:
        parser.error('Use a log directory and an explicit valid memory fraction')
    base.mkdir(parents=True, exist_ok=True)
    status_path = base / 'pipeline-status.json'
    active, done, logs = {}, {'preflight': [], 'full': []}, []
    free_since = {}
    preflight_summary_written = False

    def save(state, error=None):
        record = dict(state=state, pid=os.getpid(), mem_fraction=args.mem_fraction,
                      idle_grace_seconds=args.idle_grace_seconds,
                      max_batch_size=128, completed=done,
                      active={key: {k: (str(v) if isinstance(v, Path) else v)
                                    for k, v in job.items() if k != 'process'} for key, job in active.items()})
        if error:
            record['error'] = str(error)
        temporary = base / 'pipeline-status.tmp'
        temporary.write_text(json.dumps(record, indent=2))
        temporary.replace(status_path)

    def verify(output, check_memory=True):
        report = json.loads((output / 'launch.json').read_text())
        if report.get('state') != 'complete':
            raise RuntimeError(f'Failed case: {output}; inspect runtime.log')
        if report.get('max_batch_size') != 128 or report.get('profile') != 'matched':
            raise RuntimeError(f'Unexpected batch/profile: {output}')
        if check_memory and report.get('mem_fraction') != args.mem_fraction:
            raise RuntimeError(f'Unexpected memory fraction: {output}')

    def match_workloads(phase):
        for data in DATASETS:
            s = json.loads((base / f'sparsespec-{data}-{phase}/engine/workload.json').read_text())
            v = json.loads((base / f'vllm-{data}-{phase}/engine/workload.json').read_text())
            for field in ['num_requests', 'prompt_sha256', 'token_ids_sha256', 'input_tokens']:
                if s[field] != v[field]:
                    raise RuntimeError(f'{data} {phase} workload mismatch: {field}')

    try:
        if status_path.exists():
            previous = json.loads(status_path.read_text())
            if previous['state'] == 'failed':
                raise RuntimeError('Preserve failed cases; use a new explicit run directory after resolving the failure')
            if previous['mem_fraction'] != args.mem_fraction:
                raise RuntimeError('Cannot change memory fraction on resume')
            done = previous['completed']
            active = {key: dict(job, output=Path(job['output']), gpus=tuple(job['gpus']), process=None)
                      for key, job in previous.get('active', {}).items()}
        elif args.recovery_root:
            recovery = args.recovery_root.resolve()
            if not recovery.is_relative_to(REPO / 'log') or recovery == base:
                raise ValueError('Recovery source must be a different log directory')
            for data in DATASETS:
                for engine in ENGINES:
                    key = f'{engine}-{data}'
                    source = recovery / f'{key}-preflight'
                    verify(source)
                    (base / f'{key}-preflight').symlink_to(source, target_is_directory=True)
                    done['preflight'].append(key)
                    source = recovery / f'{key}-full'
                    if source.exists() and json.loads((source / 'launch.json').read_text()).get('state') == 'complete':
                        verify(source)
                        (base / f'{key}-full').symlink_to(source, target_is_directory=True)
                        done['full'].append(key)
            save('running')
        else:
            if args.smoke_output is None or args.smoke_pid is None:
                parser.error('First launch must adopt an explicit vLLM smoke test')
            smoke = args.smoke_output.resolve()
            save('waiting_vllm_smoke')
            while alive(args.smoke_pid):
                time.sleep(10)
            # Smoke establishes environment compatibility. Each larger
            # preflight below must independently pass at the target fraction.
            verify(smoke, check_memory=False)

        while len(done['full']) < len(DATASETS) * len(ENGINES):
            for key, job in list(active.items()):
                code = job['process'].poll() if job['process'] else (None if alive(job['pid']) else 0)
                if code is None:
                    continue
                if code:
                    raise RuntimeError(f'{key} exited {code}; inspect {job["output"]}')
                verify(job['output'])
                done[job['phase']].append(f'{job["engine"]}-{job["dataset"]}')
                del active[key]
            phase = 'preflight' if len(done['preflight']) < 6 else 'full'
            if phase == 'full':
                match_workloads('preflight')
                if not preflight_summary_written:
                    summarize(base, 'preflight')
                    preflight_summary_written = True
            pending = [(engine, data) for data in DATASETS for engine in ENGINES
                       if f'{engine}-{data}' not in done[phase] and f'{engine}-{data}-{phase}' not in active]
            available = free_pairs()
            for pair in list(free_since):
                if pair not in available:
                    del free_since[pair]
            for pair in available:
                free_since.setdefault(pair, time.monotonic())
                if time.monotonic() - free_since[pair] < args.idle_grace_seconds:
                    continue
                if any(job['gpus'] == pair for job in active.values()) or not pending:
                    continue
                engine, data = pending.pop(0)
                key = f'{engine}-{data}-{phase}'
                output = base / key
                if output.exists():
                    verify(output)
                    done[phase].append(f'{engine}-{data}')
                    continue
                log = (base / f'{key}-controller.log').open('w')
                logs.append(log)
                command = [sys.executable, str(RUNNER), '--engine', engine, '--profile', 'matched',
                           '--dataset', data, '--phase', phase, '--mem-fraction', str(args.mem_fraction),
                           '--gpus', ','.join(map(str, pair)), '--output', str(output)]
                if phase == 'preflight':
                    command += ['--num-test-sample', '128', '--preflight-max-tokens', '1024']
                if args.recovery_root and phase == 'full' and engine == 'sparsespec':
                    source = args.recovery_root.resolve() / key
                    if (source / 'engine/summary.log').exists() and (source / 'engine/requests.json').exists():
                        command += ['--resume-from', str(source)]
                process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
                active[key] = dict(engine=engine, dataset=data, phase=phase, output=output,
                                   gpus=pair, pid=process.pid, process=process)
            save('running' if active else 'waiting_free_gpu_pairs')
            time.sleep(15)
        match_workloads('full')
        summarize(base)
        save('complete')
    except Exception as error:
        save('failed', error)
        raise


if __name__ == '__main__':
    main()
