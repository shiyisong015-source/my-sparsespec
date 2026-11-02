"""Audit three transfer/forward claims from an existing Nsight capture."""
import argparse
from bisect import bisect_right
from collections import defaultdict
import json
from pathlib import Path
import re
import sqlite3

import numpy as np

from analyze_transfer_compute import process_id, union, duration, overlap


def analyze(case):
    db = sqlite3.connect(f"file:{case / 'nsys.sqlite'}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    names = dict(db.execute('SELECT id,value FROM StringIds'))
    ranges = defaultdict(list)
    for row in db.execute('SELECT * FROM NVTX_EVENTS WHERE end IS NOT NULL'):
        label = row['text'] or names.get(row['textId'], '')
        if label.startswith('diag.forward '):
            ranges[row['globalTid']].append(dict(start=row['start'], end=row['end'],
                step=int(re.search(r'step=(\d+)', label)[1]), label=label,
                pid=process_id(row['globalTid'])))
    for rows in ranges.values():
        rows.sort(key=lambda r:r['start'])
    starts = {tid:[r['start'] for r in rows] for tid,rows in ranges.items()}
    launches = {}
    for row in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME'):
        tid = row['globalTid']
        if tid not in starts:
            continue
        idx = bisect_right(starts[tid], row['start']) - 1
        if idx >= 0 and row['end'] <= ranges[tid][idx]['end']:
            launches[(process_id(tid),row['correlationId'])] = ranges[tid][idx]
    grouped = defaultdict(list)
    for row in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL'):
        key = (process_id(row['globalPid']),row['correlationId'])
        if key in launches:
            grouped[(key[0],launches[key]['step'])].append((row['start'],row['end']))
    forward_rows = []
    for (pid,step), intervals in sorted(grouped.items()):
        bounds = min(a for a,b in intervals),max(b for a,b in intervals)
        active = union(intervals,*bounds)
        forward_rows.append(dict(pid=pid,step=step,start_ns=bounds[0],end_ns=bounds[1],
            elapsed_ms=(bounds[1]-bounds[0])/1e6,kernel_active_ms=duration(active)/1e6,
            kernel_count=len(intervals)))
    (case/'forward-gpu-intervals.json').write_text(json.dumps(forward_rows,indent=2)+'\n')
    audit = json.loads((case/'restore-submission-audit.json').read_text())
    # Restore kernels are matched to the audited CPU transfer launch labels.
    restore_ranges = defaultdict(list)
    for row in db.execute('SELECT * FROM NVTX_EVENTS WHERE end IS NOT NULL'):
        label = row['text'] or names.get(row['textId'], '')
        if label.startswith('diag.kv.restore '):
            restore_ranges[row['globalTid']].append((row['start'],row['end'],label))
    restore_calls = {}
    for row in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME'):
        for a,b,label in restore_ranges.get(row['globalTid'],[]):
            if a <= row['start'] and row['end'] <= b:
                restore_calls[(process_id(row['globalTid']),row['correlationId'])] = label
    restores = []
    for row in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL'):
        pid = process_id(row['globalPid'])
        label = restore_calls.get((pid,row['correlationId']))
        if label and any(n in names.get(row['shortName'],'') for n in ('_scatter_kv_pages','_restore_kv_pages_bounded')):
            restores.append(dict(pid=pid,label=label,start_ns=row['start'],end_ns=row['end']))
    device_results = []
    for pid in sorted({r['pid'] for r in forward_rows}):
        fs = [r for r in forward_rows if r['pid']==pid]
        rs = [r for r in restores if r['pid']==pid]
        normal = [f['elapsed_ms'] for f in fs if not any(f['start_ns']<r['end_ns'] and r['start_ns']<f['end_ns'] for r in rs)]
        med = float(np.median(normal))
        for r in rs:
            r['duration_ms'] = (r['end_ns']-r['start_ns'])/1e6
            r['equivalent_median_forward_count'] = r['duration_ms']/med
            r['forward_starts_inside'] = sum(r['start_ns']<=f['start_ns']<r['end_ns'] for f in fs)
            r['forward_completions_inside'] = sum(r['start_ns']<f['end_ns']<=r['end_ns'] for f in fs)
            r['whole_forwards_inside'] = sum(r['start_ns']<=f['start_ns'] and f['end_ns']<=r['end_ns'] for f in fs)
            r['overlapping_forward_steps'] = [f['step'] for f in fs if f['start_ns']<r['end_ns'] and r['start_ns']<f['end_ns']]
        device_results.append(dict(pid=pid,forward_count=len(fs),normal_forward_count=len(normal),
            forward_elapsed_ms_p25_p50_p90=np.percentile(normal,[25,50,90]).tolist(),restores=rs))
    states = [json.loads(s) for s in (case/'states.rank0.jsonl').open()]
    states = [s for s in states if s['event']=='state']
    state_rows=[]
    for s in states:
        reqs=s['requests']
        selected=[r for r in reqs if r['computing']]
        waiting=[r for r in reqs if r['status']=='VERIFY' and not r['computing']]
        state_rows.append(dict(step=s['step'],batch=len(selected),queries=s['queries'],live=len(reqs),pending=s['pending'],
            waiting_verify=len(waiting),transferring=sum(r['transferring'] for r in reqs),
            free_pages=s['free_pages'],draft_full_cold_pages=sum(max(0,r['physical']-r['hot']) for r in reqs if r['status']=='DRAFT' and r['phase']=='resident'),
            waiting_verify_cold_pages=sum(max(0,r['logical']-r['hot']) for r in waiting),
            phases=s['phases'],metrics=s['metrics']))
    before_after=[]
    for r in device_results[0]['restores']:
        step=int(re.search(r'step=(\d+)',r['label'])[1])
        nearby=[s for s in state_rows if step-3<=s['step']<=step+4]
        before_after.append(dict(label=r['label'],states=nearby))
    batches=[s['batch'] for s in state_rows]
    claims=dict(devices=device_results,state_count=len(states),
        batch_min_p25_p50_p90_max=[min(batches),*np.percentile(batches,[25,50,90]).tolist(),max(batches)],
        batch_first=state_rows[0],batch_last=state_rows[-1],
        downward_batch_steps=sum(b<a for a,b in zip(batches,batches[1:])),
        upward_batch_steps=sum(b>a for a,b in zip(batches,batches[1:])),restore_batch_neighborhoods=before_after,
        caveats=['Forward intervals are GPU min-start/max-end of kernels correlated with a model-forward CPU NVTX range; gaps and TP waits are included.',
            'Equivalent forward count divides transfer duration by the median of other profiled forwards; it is not actual concurrent progress or an unprofiled performance prediction.',
            'A mixed-batch forward has multiple request phases. Request count alone does not describe its FLOPs or latency.',
            'The capture is observational; changes due to EOS/admission and recovery must not be labeled a causal transfer effect without a control.'])
    (case/'transfer-claims.json').write_text(json.dumps(claims,ensure_ascii=False,indent=2)+'\n')
    (case/'batch-states.json').write_text(json.dumps(state_rows,ensure_ascii=False,indent=2)+'\n')
    plot(case,device_results,state_rows,forward_rows)
    print(json.dumps(dict(devices=device_results,state_count=len(states),batch=claims['batch_min_p25_p50_p90_max'],
        downward=claims['downward_batch_steps'],upward=claims['upward_batch_steps']),ensure_ascii=False,indent=2))


def plot(case,devices,states,forwards):
    import os
    os.environ.setdefault('MPLCONFIGDIR',str(case/'matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pid=devices[0]['pid']
    fs=sorted([f for f in forwards if f['pid']==pid],key=lambda f:f['step'])
    origin=json.loads((case/'analysis.json').read_text())['window_ns'][0]
    bystep={s['step']:s for s in states}
    matched=[f for f in fs if f['step'] in bystep]
    x=[(f['start_ns']-origin)/1e9 for f in matched]
    fig,axs=plt.subplots(3,1,figsize=(13,7),sharex=True)
    axs[0].plot(x,[bystep[f['step']]['batch'] for f in matched],'.',label='Requests per forward (samples)')
    axs[0].step(x,[bystep[f['step']]['live'] for f in matched],where='post',label='Live requests',alpha=.7)
    axs[1].step(x,[bystep[f['step']]['waiting_verify'] for f in matched],where='post',label='Unscheduled VERIFY requests')
    axs[2].plot(x,[f['elapsed_ms'] for f in matched],'.-',linewidth=.7,label='GPU forward elapsed')
    for ax in axs:
        for r in devices[0]['restores']:
            ax.axvspan((r['start_ns']-origin)/1e9,(r['end_ns']-origin)/1e9,color='orange',alpha=.25)
        ax.legend(loc='upper right',fontsize=8)
        ax.grid(alpha=.2)
    axs[0].set_ylabel('Request count')
    axs[1].set_ylabel('Waiting VERIFY')
    axs[2].set_ylabel('Forward (ms)')
    axs[2].set_xlabel('Capture time (s); orange = mapped cold KV restore')
    fig.tight_layout()
    for suffix in ['png','pdf']:
        fig.savefig(case/f'batch-forward-restore.{suffix}',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,required=True)
    analyze(p.parse_args().case.resolve())
