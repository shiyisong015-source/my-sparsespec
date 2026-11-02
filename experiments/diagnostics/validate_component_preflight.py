"""Validate multi-window functionality before a natural-EOS profile scan."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def validate(root, gpu, repo):
    manifest=json.loads((root/'manifest.rank0.json').read_text())
    if (manifest['tp']!=1 or manifest['pool_pages']!=98304 or manifest['gpu']!=str(gpu)
            or manifest.get('buffer_batch_capacity')!=128):
        raise ValueError('Preflight TP/GPU/pool/buffer capacity mismatch')
    if hashlib.sha256(Path(manifest['arguments']['input']).read_bytes()).hexdigest()!=manifest['input_sha256']:
        raise ValueError('Preflight input changed')
    for name,digest in manifest['source_sha256'].items():
        if hashlib.sha256((repo/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Preflight source changed; use a fresh preflight directory')
    requested=manifest['arguments']['record_windows']
    coverage=json.loads((root/'b128/sampling-coverage.rank0.json').read_text())
    if requested!=3 or coverage['complete_windows']!=requested or coverage['missing_windows']:
        raise ValueError('Preflight must complete three recording windows')
    summary=json.loads((root/'component-summary.json').read_text())
    counts={direction:sum(rank['transfers'][direction]['count'] for case in summary['cases']
                         for rank in case['ranks']) for direction in ('repack','restore')}
    if not all(counts.values()):
        raise ValueError('Preflight did not sample both KV transfer directions')
    layers=json.loads((Path(manifest['arguments']['model'])/'config.json').read_text())['num_hidden_layers']
    required=('batch_attention_core','attention_selection_scores','mlp_input_gemm','mlp_activation','mlp_output_gemm')
    cycles,forwards,windows=[],[],[]
    for window in sorted(root.glob('b128_w*')):
        observed=json.loads((window/'observer.rank0.json').read_text())
        metrics=observed['metrics']
        if (observed['window_status']!='complete' or not metrics['migrations'] or not metrics['restores']
                or metrics['gpu_final_free_pages']!=metrics['gpu_capacity_pages']):
            raise ValueError(f'Incomplete preflight lifecycle: {window}')
        calls=defaultdict(Counter)
        for line in (window/'components.rank0.jsonl').open():
            row=json.loads(line);calls[row['producer_step']][row['component']]+=1
            if row['component']=='forward' and 'gpu_ms' in row:forwards.append(row['gpu_ms'])
        boundaries={row['step'] for row in observed['capture_boundaries']}
        sampled=[row for line in (window/'batches.rank0.jsonl').open()
                 if (row:=json.loads(line))['detail_sample'] and row['step'] not in boundaries]
        if len({row['step']%2 for row in sampled})<2:
            raise ValueError(f'Both attention buffers not covered: {window}')
        for state in sampled:
            if any(calls[state['step']][name]!=layers for name in required):
                raise ValueError(f"Incomplete layer instrumentation: {window}, step={state['step']}")
        timeline=json.loads((window/'verify-wait-summary.rank0.json').read_text())
        if timeline['status']!='complete':raise ValueError(f'Unfinished Verify timeline: {window}')
        cycles.extend(json.loads(line) for line in (window/'verify-wait-cycles.rank0.jsonl').open())
        windows.append(dict(window_id=observed['window_id'],sampled_forwards=len(sampled),layers=layers))
    if len(windows)!=requested or not forwards:
        raise ValueError('Missing windows or forward measurements')
    stages=('restore_submit_begin','restore_pages_allocated','restore_submitted',
            'restore_ready_observed','restore_page_table_committed')
    restored=[row for row in cycles if row['need_time_known'] and not row['right_censored']
              and all(row['milestones'][name] is not None for name in stages)
              and row['milestones']['restore_device_ready'] is not None]
    if not restored:raise ValueError('No complete allocation/restore/commit/Verify lifecycle')
    for row in restored:
        times=[row['milestones'][name]['cpu_ns'] for name in stages]+[row['verify_submit']['cpu_ns']]
        if times!=sorted(times):raise ValueError('Restore host milestones out of order')
    estimate=statistics.median(forwards)
    (root/'iteration-estimate.json').write_text(json.dumps(dict(median_ms=estimate,samples=len(forwards),
        source='Single-GPU B128 three-window short preflight CUDA Event median; not a long-decode timing guarantee'),indent=2)+'\n')
    (root/'iteration-ms.txt').write_text(str(estimate)+'\n')
    report=dict(passed=True,gpu=gpu,windows=windows,sampled_transfers=counts,
                complete_restored_verify_cycles=len(restored),reservation_iteration_ms=estimate)
    (root/'preflight-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--repo',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(validate(args.root,args.gpu,args.repo),indent=2))
