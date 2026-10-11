"""Run official Qwen3-8B workloads, with explicit recorded alignment overrides."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
REPEATS = {'aime24': 60, 'livecodebench': 2, 'olympiadbench': 3}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=list(REPEATS), required=True)
    p.add_argument('--gpus', required=True, help='Two physical GPU indices, e.g. 0,1')
    p.add_argument('--phase', choices=['preflight', 'full'], required=True)
    p.add_argument('--engine', choices=['sparsespec', 'vllm'], default='sparsespec')
    p.add_argument('--profile', choices=['script', 'matched'], default='matched')
    p.add_argument('--mem-fraction', type=float, default=.85)
    p.add_argument('--num-test-sample', type=int, default=None,
                   help='Explicit boundary preflight size; full always uses the entire official workload')
    p.add_argument('--preflight-max-tokens', type=int, default=256)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume-from', type=Path,
                   help='Replay a saved SparseSpec first epoch with unchanged engine settings')
    args = p.parse_args()
    if not 0 < args.mem_fraction < 1:
        p.error('mem-fraction must be between zero and one')
    if args.num_test_sample is not None and (args.phase != 'preflight' or args.num_test_sample < 1):
        p.error('num-test-sample is only for explicit preflight cases')
    if args.preflight_max_tokens < 1:
        p.error('preflight-max-tokens must be positive')
    gpus = [int(x) for x in args.gpus.split(',')]
    if len(set(gpus)) != 2 or any(gpu < 0 for gpu in gpus):
        p.error('Qwen3-8B official configuration requires two distinct GPUs')
    output = args.output.resolve()
    if not output.is_relative_to(REPO / 'log'):
        p.error('Keep reproduction artifacts under SparseSpec/log')
    if output.exists() and any(output.iterdir()):
        p.error('Use a new output directory to preserve existing results')
    previous = None
    if args.resume_from:
        source = args.resume_from.resolve()
        if args.engine != 'sparsespec' or not source.is_relative_to(REPO / 'log'):
            p.error('Only SparseSpec cases under log can replay saved generation')
        previous = json.loads((source / 'launch.json').read_text())
        expected = dict(engine='sparsespec', dataset=args.dataset, phase=args.phase, profile=args.profile,
                        max_batch_size=128, mem_fraction=args.mem_fraction, dtype='float16',
                        temperature=.6, enable_thinking=True, spec_stride=8, budget_ratio=.05,
                        num_min_budget=192, chunk_size=256)
        if any(previous.get(key) != value for key, value in expected.items()):
            p.error('Saved generation engine/workload settings do not match')
        if args.phase == 'preflight':
            prior_command = previous['command']
            if int(prior_command[prior_command.index('--max_tokens_per_call') + 1]) != args.preflight_max_tokens:
                p.error('Saved preflight generation length does not match')
        for name, digest in previous['source_sha256'].items():
            if name.startswith('serve/') and hashlib.sha256((REPO / name).read_bytes()).hexdigest() != digest:
                p.error(f'Engine source changed since cached generation: {name}')
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                         '--format=csv,noheader,nounits'], text=True)
    available = {int(line.split(',')[0]): [int(x) for x in line.split(',')[1:]]
                 for line in inventory.splitlines()}
    for gpu in gpus:
        if gpu not in available or available[gpu][0] >= 128 or available[gpu][1] != 0:
            p.error(f'GPU {gpu} is unavailable')
    output.mkdir(parents=True)
    # The evaluation entry prefixes "outputs/" when this directory is absent.
    (output / 'evaluation').mkdir()
    if previous:
        (output / 'engine').mkdir()
        for filename in ['summary.log', 'requests.json', 'workload.json']:
            shutil.copy2(source / 'engine' / filename, output / 'engine' / filename)
        (output / 'cached-generation-provenance.json').write_text(json.dumps(dict(
            source_case=str(source), source_launch=previous,
            sha256={name: hashlib.sha256((source / 'engine' / name).read_bytes()).hexdigest()
                    for name in ['summary.log', 'requests.json', 'workload.json']}), indent=2))
    cmd = ([sys.executable, 'math_eval.py', '--use_specgen', '--specgen_isolated_epochs'] if args.engine == 'sparsespec'
           else [sys.executable, 'math_eval.py', '--use_vllm'])
    cmd += ['--model_name_or_path', '/data/models/Qwen3-8B',
           '--prompt_type', 'qwen3-math-thinking', '--apply_chat_template', '--enable_thinking',
           '--split', 'test', '--data_names', args.dataset, '--temperature', '.6',
           '--num_test_sample', str(args.num_test_sample or 4) if args.phase == 'preflight' else '-1',
           '--repeat_dataset', '1' if args.phase == 'preflight' else str(REPEATS[args.dataset]),
           '--save_outputs', '--output_dir', str(output / 'evaluation'), '--overwrite']
    if args.engine == 'sparsespec':
        cmd += [
           '--tp_ranks', '2', '--kv_cache', 'pillar', '--max_batch_size', '128',
           '--admit_policy', 'offloading', '--cuda_graph', '--async_cpu', '--mem_fraction', str(args.mem_fraction),
           '--spec_stride', '8', '--budget_ratio', '.05', '--num_min_budget', '192',
           '--batch_policy', 'greedy', '--chunk_size', '256', '--specgen_result_dir', str(output / 'engine')]
        if previous:
            cmd += ['--specgen_resume_dir', str(source / 'engine')]
    else:
        cmd += ['--n_sampling', '1', '--top_p', '1',
                '--vllm_gpu_memory_utilization', str(args.mem_fraction),
                '--vllm_result_dir', str(output / 'engine'),
                '--vllm_max_num_seqs', '128' if args.profile == 'matched' else '256']
        if args.profile == 'matched':
            cmd += ['--vllm_dtype', 'float16', '--vllm_seed', '330', '--vllm_max_model_len', '40960']
    if args.phase == 'preflight':
        cmd += ['--max_tokens_per_call', str(args.preflight_max_tokens)]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpus, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               ENABLE_INTRA_NODE_COMM='1', TOKENIZERS_PARALLELISM='false')
    if args.engine == 'vllm':
        env['VLLM_USE_V1'] = '1'
        env['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    env.pop('PYTHONOPTIMIZE', None)
    # Keep assertions for preflight; the official full launcher uses python -O.
    if args.phase == 'full':
        env['PYTHONOPTIMIZE'] = '1'
    report = dict(dataset=args.dataset, phase=args.phase, engine=args.engine, profile=args.profile,
                  gpus=gpus, command=cmd,
                  pid=os.getpid(), state='running', temperature=.6, max_batch_size=128,
                  spec_stride=8, budget_ratio=.05, num_min_budget=192, chunk_size=256,
                  enable_thinking=True, dtype='float16' if args.engine == 'sparsespec' or args.profile == 'matched' else 'auto',
                  cuda_graph=True, async_cpu=True if args.engine == 'sparsespec' else 'vLLM V1 default',
                  mem_fraction=args.mem_fraction,
                  alignment_overrides=dict(memory=args.mem_fraction != .85, profile=args.profile),
                  optimized_python=args.phase == 'full',
                  isolated_generation_epochs=args.engine == 'sparsespec',
                  resumed_generation_source=str(source) if previous else None,
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                  git_diff_sha256=hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=REPO)).hexdigest(),
                  source_sha256={str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for tree in ['serve', 'eval/benchmarks', 'experiments/reproduction']
                                 for path in (REPO / tree).rglob('*.py')},
                  gpu_inventory=subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total',
                                                        '--format=csv'], text=True),
                  throughput_source='engine/summary.log' if args.engine == 'sparsespec' else 'engine/summary.json',
                  packages={package: importlib.metadata.version(package)
                            for package in ['torch', 'vllm', 'transformers']})
    report['engine_environment'] = {key: env[key] for key in ['VLLM_USE_V1', 'VLLM_WORKER_MULTIPROC_METHOD'] if key in env}
    if args.engine == 'vllm':
        report['max_batch_size'] = 128 if args.profile == 'matched' else 256
        for key in ['spec_stride', 'budget_ratio', 'num_min_budget', 'chunk_size']:
            report.pop(key)
    status = output / 'launch.json'
    status.write_text(json.dumps(report, indent=2))
    started = time.perf_counter()
    try:
        with (output / 'runtime.log').open('w') as log:
            result = subprocess.run(cmd, cwd=REPO / 'eval/benchmarks', env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        report.update(exit_code=result.returncode, launcher_elapsed_seconds=time.perf_counter() - started)
        if result.returncode:
            raise RuntimeError(f'Official entry failed with exit code {result.returncode}')
        summary = 'summary.log' if args.engine == 'sparsespec' else 'summary.json'
        if not (output / 'engine' / summary).exists():
            raise RuntimeError('Entry exited without an engine summary')
        report['state'] = 'complete'
    except Exception as error:
        report.update(state='failed', error=str(error))
        raise
    finally:
        status.write_text(json.dumps(report, indent=2))
    print(json.dumps({'state': report['state'], 'output': str(output)}))


if __name__ == '__main__':
    main()
