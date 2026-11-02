"""Summarize completed fixed-cohort cases without probability/noise files."""
import argparse
import json
import os
from pathlib import Path
import statistics


def analyze(case):
    observers=[json.loads((case/f'observer.rank{r}.json').read_text()) for r in [0,1]]
    if not all(o['successful'] for o in observers):raise RuntimeError('Incomplete TP cohort')
    rows=[json.loads(x) for x in (case/'states.rank0.jsonl').open()]
    start=next(r for r in rows if r['event']=='measurement_start')
    states=[r for r in rows if r['event']=='state']
    results=[]
    for rank in [0,1]:
        events=json.loads((case/f'gpu-events.rank{rank}.json').read_text())
        fs=[r for r in events if r['kind']=='forward']
        rs=[r for r in events if r['kind']=='H2D']
        first=min((r['start_ms'] for r in rs),default=float('inf'))
        calibration=[f for f in fs if f['end_ms']<first and f['phases']=={'DRAFT':start['cohort']}]
        # Remove initial release/launch transients; use identical-request-count
        # pure-Draft forwards before the first restore, when available.
        calibration=calibration[5:]
        lat=[f['end_ms']-f['start_ms'] for f in calibration]
        transfers=[]
        for r in rs:
            concurrent=[f for f in fs if f['start_ms']<r['end_ms'] and r['start_ms']<f['end_ms']]
            inside=[f for f in concurrent if r['start_ms']<=f['start_ms'] and f['end_ms']<=r['end_ms']]
            transfers.append(dict(request=r['transfer_request'],step=r['step'],duration_ms=r['end_ms']-r['start_ms'],
                overlapping_forward_steps=[f['step'] for f in concurrent],whole_forwards_inside=len(inside),
                forward_starts_inside=sum(r['start_ms']<=f['start_ms']<r['end_ms'] for f in concurrent)))
        results.append(dict(rank=rank,forward_count=len(fs),restores=transfers,
            calibration_forwards=len(lat),pure_draft_median_ms=statistics.median(lat) if lat else None,
            draft_queries_per_second=start['cohort']*1000/statistics.median(lat) if lat else None))
    summary=dict(case=str(case),phase=start['phase'],measured_cohort=start['cohort'],
        devices=results,completion_reasons={reason:sum(r.get('reason')==reason for r in rows if r['event']=='complete')
            for reason in ['eos','length_limit','diagnostic_round_limit']},
        notes=['Controlled all-hot initial cohort, not a natural traffic throughput benchmark.',
               'Forward event intervals include TP/attention waits; overlap alone does not prove useful SM activity.',
               'Aligned first-wave collapse is a constructed boundary condition; consult native_offsets and natural-precheck separately.',
               'Native-offset first windows may be shorter than k=32; subsequent windows use the native rules.'])
    (case/'drain-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    os.environ.setdefault('MPLCONFIGDIR',str(case/'matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fs=json.loads((case/'gpu-events.rank0.json').read_text())
    times={f['step']:f['start_ms']/1000 for f in fs if f['kind']=='forward'}
    states=[s for s in states if s['step'] in times]
    x=[times[s['step']] for s in states]
    fig,axes=plt.subplots(3,1,figsize=(12,7),sharex=True)
    for label,predicate in [('Scheduled Draft',lambda r:r['status']=='DRAFT' and r['computing']),
                            ('Scheduled Verify',lambda r:r['status']=='VERIFY' and r['computing']),
                            ('Unscheduled Verify',lambda r:r['status']=='VERIFY' and not r['computing'])]:
        axes[0].plot(x,[sum(predicate(r) for r in s['requests']) for s in states],'.-',markersize=2,linewidth=.6,label=label)
    axes[1].plot(x,[s['queries'] for s in states],'.',label='Query rows per forward')
    by={f['step']:f for f in fs if f['kind']=='forward'}
    axes[2].plot(x,[by[s['step']]['end_ms']-by[s['step']]['start_ms'] for s in states],'.',label='GPU forward elapsed (includes waits)')
    for ax in axes:
        for r in fs:
            if r['kind']=='H2D':ax.axvspan(r['start_ms']/1000,r['end_ms']/1000,color='orange',alpha=.2)
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    axes[0].set_ylabel('Requests')
    axes[1].set_ylabel('Query tokens')
    axes[2].set_ylabel('Elapsed (ms)');axes[2].set_xlabel('Measurement time (s); orange = cold restore launch bracket')
    fig.suptitle(f"{start['cohort']} requests; r=.05, k<=32; {start['phase']}")
    fig.tight_layout();fig.savefig(case/'draft-drain.png',dpi=160);plt.close(fig)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    root=p.parse_args().root.resolve()
    summaries=[]
    for case in sorted(root.iterdir()):
        if case.is_dir() and all((case/f'observer.rank{r}.json').exists() for r in [0,1]):
            summaries.append(analyze(case))
    (root/'completed-analysis.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2)+'\n')
    print('Analyzed',len(summaries),'completed cases')
