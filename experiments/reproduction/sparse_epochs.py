"""Keep CUDA/TP lifecycle inside a fresh worker for each official generation call."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def read_metrics(directory):
    raw = (Path(directory) / 'summary.log').read_text()
    return {label.strip(): float(number) for label, number in
            re.findall(r'\|\s*([^|\n]+?)\s*\|\s*([\d.eE+-]+)\s*\|', raw)}


def run_epoch(token_ids, max_tokens, kwargs, allow_cached=False):
    directory = Path(kwargs['result_dir'])
    directory.mkdir(parents=True, exist_ok=True)
    outputs_path = directory / 'requests.json'
    if allow_cached:
        if not outputs_path.exists() or not (directory / 'summary.log').exists():
            raise ValueError('Cached epoch needs both outputs and its original engine timing')
        print(f'Replaying saved epoch: {directory}', flush=True)
    else:
        if outputs_path.exists():
            raise ValueError(f'Refusing to overwrite completed generation: {directory}')
        payload = directory / 'generation-input.json'
        payload.write_text(json.dumps(dict(token_ids=token_ids, max_tokens=max_tokens, kwargs=kwargs)))
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                   f"--nproc_per_node={kwargs['tp_ranks']}", '--', str(Path(__file__).resolve()),
                   '--payload', str(payload)]
        # The parent remains CPU-only. Each child owns its model, NCCL group,
        # graph buffers, and default CUDA streams until this call completes.
        subprocess.run(command, check=True, env=os.environ.copy())
    outputs = json.loads(outputs_path.read_text())
    if len(outputs) != len(token_ids):
        raise ValueError('Saved generation request count mismatch')
    if [item['len_input'] for item in outputs] != list(map(len, token_ids)):
        raise ValueError('Saved generation input lengths mismatch')
    return outputs, read_metrics(directory)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--payload', required=True, type=Path)
    args = parser.parse_args()
    data = json.loads(args.payload.read_text())
    from serve.request.request import Request
    from serve.run import run_inference
    from serve.utils import is_first_rank
    kwargs = data['kwargs']
    if not is_first_rank():
        kwargs['result_dir'] = None
    requests = [Request(ids, data['max_tokens']) for ids in data['token_ids']]
    run_inference(reqs=requests, **kwargs)
    import torch.distributed as dist
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
