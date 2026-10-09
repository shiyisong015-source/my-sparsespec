"""Describe scheduling opportunities from fixed-policy logs; no counterfactual speedups."""
import csv
import hashlib
import json
import re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/'external/SparseSpec/log/acceptance-sweep/lcb-random100-seed330'
OUT = ROOT/'external/SparseSpec/log/dynamic-scheduling-analysis'
BS = [1,4,16,32,64,100]
KS = [4,8,10,12,14]
RS = [.05,.1,.15]


def mean_ci(values):
    values = np.asarray(values)
    rng = np.random.default_rng(330)
    samples = values[rng.integers(0,len(values),size=(3000,len(values)))].mean(1)
    return {'mean':float(values.mean()),'ci95':np.quantile(samples,[.025,.975]).tolist()}


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    cases=[];series={};request_stats=[];manifest=[];phase=[]
    expected_sha=json.loads((SRC/'sample_manifest.json').read_text())['sha256']
    for b in BS:
        base=SRC/'Qwen3-8B' if b==1 else SRC/f'batch{b}/Qwen3-8B'
        for k in KS:
            for r in RS:
                d=base/f'k{k}_r{r:g}';report=json.loads((d/'case.json').read_text());raw=(d/'rounds.jsonl').read_bytes()
                assert report['sample_sha256']==expected_sha
                assert len(report['inputs'])==len(report['outputs'])==100
                assert report.get('max_batch_size',1)==b
                text=(d/'summary.log').read_text()
                def metric(name):return float(re.search(re.escape(name)+r'\s*\|\s*([\d.]+)',text)[1])
                rows=[json.loads(line) for line in raw.splitlines()];groups={}
                for row in rows:
                    if not row['cold_start']:
                        assert row['actual_draft_tokens']==k
                        groups.setdefault(row['request_id'],[]).append(row)
                assert set(groups)==set(range(100))
                all_a=np.array([x['accepted_draft_tokens'] for g in groups.values() for x in g])
                assert abs(all_a.mean()/k-report['post_cold_acceptance_rate'])<1e-12
                per=[]
                for rid,g in groups.items():
                    a=np.array([v['accepted_draft_tokens'] for v in g]);series[b,k,r,rid]=g
                    per.append(a.mean()/k)
                    request_stats.append({'batch_limit':b,'k':k,'r':r,'rid':rid,'rounds':len(g),
                                          'mean_accepted':float(a.mean()),'acceptance':float(a.mean()/k),
                                          'full_fraction':float((a==k).mean()),'input_tokens':report['inputs'][rid]['input_tokens']})
                    if k==14:
                        for t in range(16,len(g)-15,16):
                            # Only complete contiguous, non-cold windows.
                            if any(g[j+1]['round']!=g[j]['round']+1 for j in range(t-16,t+15)):continue
                            h=a[t-16:t]/k;f=a[t:t+16]/k
                            phase.append({'batch_limit':b,'r':r,'rid':rid,'history':float(h.mean()),
                                          'future':float(f.mean()),'future_full':float((f==1).mean()),
                                          'history_full':float((h==1).mean()),'kv_len':g[t]['kv_len'],
                                          'progress':g[t]['kv_len']-report['inputs'][rid]['input_tokens']})
                # Decompose pooled-round variance into request means and within-request variance.
                avg=float(all_a.mean());between=sum(len(g)*(np.mean([v['accepted_draft_tokens'] for v in g])-avg)**2 for g in groups.values())/len(all_a)
                total_var=float(all_a.var())
                logfiles=sorted(base.glob(f'k{k}_r{r:g}_attempt*.log'))
                heartbeat=[];offloads=0
                if logfiles:
                    logs=logfiles[-1].read_text(errors='replace')
                    heartbeat=[int(x) for x in re.findall(r'\[Heartbeat\].*?batch size (\d+)',logs)]
                    offloads=len(re.findall(r'Offloaded request',logs))
                row={'batch_limit':b,'k':k,'r':r,'tps':metric('Output Throughput (tok/s)'),
                     'seconds':metric('time (s)'),'total_output':metric('Total Output Tokens'),
                     'rounds':len(all_a),'mean_accepted':avg,'full_fraction':float((all_a==k).mean()),
                     'request_acceptance_p10_p50_p90':np.quantile(per,[.1,.5,.9]).tolist(),
                     'between_request_variance_fraction':float(between/total_var),
                     'heartbeat_max_batch':max(heartbeat) if heartbeat else None,
                     'heartbeat_mean_batch':float(np.mean(heartbeat)) if heartbeat else None,'offload_log_events':offloads}
                cases.append(row);manifest.append({'path':str(d),'rounds_sha256':hashlib.sha256(raw).hexdigest(),
                    'case_sha256':hashlib.sha256((d/'case.json').read_bytes()).hexdigest(),
                    'summary_sha256':hashlib.sha256((d/'summary.log').read_bytes()).hexdigest()})
    assert len(cases)==90
    regime=[]
    for b in BS:
        grid=[v for v in cases if v['batch_limit']==b];best=max(grid,key=lambda v:v['tps']);base=next(v for v in grid if v['k']==4 and v['r']==.15)
        regime.append({'batch_limit':b,'best':{x:best[x] for x in ['k','r','tps']},
                       'near_best_95pct':[[v['k'],v['r']] for v in grid if v['tps']>=.95*best['tps']],
                       'best_vs_fixed_k4_r015_pct':100*(best['tps']/base['tps']-1)})
    gains=[]
    for b in BS:
        for k in KS:
            vectors={r:np.array([np.mean([x['accepted_draft_tokens'] for x in series[b,k,r,i]])/k for i in range(100)]) for r in RS}
            for lo,hi in [(.05,.1),(.1,.15)]:
                delta=vectors[hi]-vectors[lo]
                gains.append({'batch_limit':b,'k':k,'r_low':lo,'r_high':hi,'delta_acceptance':mean_ci(delta),
                              'delta_p10_p50_p90':np.quantile(delta,[.1,.5,.9]).tolist(),
                              'fraction_requests_improved':float((delta>0).mean()),
                              'note':'Different generation trajectories, paired by prompt only.'})
    # Predictability of a high-acceptance window: held-out requests, thresholds fit on train only.
    shuffle=np.random.default_rng(330).permutation(100);folds=np.empty(100,dtype=int)
    for j,ids in enumerate(np.array_split(shuffle,5)):folds[ids]=j
    history_tests=[]
    for r in RS:
        subset=[x for x in phase if x['r']==r];bins=[]
        for fold in range(5):
            train=[x for x in subset if folds[x['rid']]!=fold];test=[x for x in subset if folds[x['rid']]==fold]
            # Each regime gets separate thresholds to avoid comparing different loads.
            for b in BS:
                tr=[x for x in train if x['batch_limit']==b];ts=[x for x in test if x['batch_limit']==b]
                q=np.quantile([x['history'] for x in tr],[.25,.75])
                for x in ts:
                    label='low' if x['history']<=q[0] else ('high' if x['history']>=q[1] else 'middle')
                    bins.append({**x,'label':label})
        vals={}
        for label in ['low','middle','high']:
            group=[x for x in bins if x['label']==label]
            vals[label]={'windows':len(group),'future_mean_accepted':14*float(np.mean([x['future'] for x in group])),
                         'future_full_fraction':float(np.mean([x['future_full'] for x in group]))}
        deltas=[]
        for rid in range(100):
            matched=[]
            for b in BS:
                low=[x['future'] for x in bins if x['rid']==rid and x['batch_limit']==b and x['label']=='low']
                high=[x['future'] for x in bins if x['rid']==rid and x['batch_limit']==b and x['label']=='high']
                if low and high:matched.append(14*(np.mean(high)-np.mean(low)))
            if matched:deltas.append(np.mean(matched))
        history_tests.append({'r':r,'bins':vals,'paired_within_request_high_minus_low_tokens':mean_ci(deltas),'paired_requests':len(deltas),'pairing':'same request and batch, averaged over batches before request bootstrap',
                              'interpretation':'Association under fixed k=14; not a causal benefit from changing k/r.'})
    # Reproducibility of ratio sensitivity across batches, not throughput predictability.
    sensitivity=[]
    for k in [4,14]:
        ds=[]
        for b in BS:
            d=np.array([np.mean([x['accepted_draft_tokens'] for x in series[b,k,.15,i]])-
                        np.mean([x['accepted_draft_tokens'] for x in series[b,k,.05,i]]) for i in range(100)])/k
            ds.append(d)
        corr=np.corrcoef(ds)
        sensitivity.append({'k':k,'mean_pairwise_correlation_of_ratio_gain':float(corr[np.triu_indices(6,1)].mean())})
    # Cross-request prediction checks whether history adds information after known state.
    from analyze_acceptance_predictability import fit_predict
    x=np.array([[np.log1p(v['kv_len']),v['progress']/1000,v['history'],v['history_full']] for v in phase])
    y=np.array([v['future'] for v in phase]);rid=np.array([v['rid'] for v in phase])
    case=np.array([BS.index(v['batch_limit'])*3+RS.index(v['r']) for v in phase])
    predictions={name:np.zeros(len(y)) for name in ['case_mean','state','state_history']}
    for fold in range(5):
        train=folds[rid]!=fold;test=~train
        for name,cols in [('case_mean',[]),('state',[0,1]),('state_history',[0,1,2,3])]:
            predictions[name][test],_=fit_predict(x,y,case,rid,train,test,18,cols)
    request_errors={}
    for name,pred in predictions.items():
        request_errors[name]=np.array([np.mean([np.abs(pred[(rid==i)&(case==c)]-y[(rid==i)&(case==c)]).mean()*14
                                               for c in np.unique(case[rid==i])]) for i in np.unique(rid)])
    history_prediction={'windows':len(y),'requests':int(len(np.unique(rid))),'features':['log_kv_len','generated_tokens_1k','past16_acceptance','past16_full_fraction'],
                        'five_fold_request_held_out_mae_tokens':{name:float(v.mean()) for name,v in request_errors.items()},
                        'mae_improvement_vs_state':mean_ci(request_errors['state']-request_errors['state_history']),
                        'mae_improvement_vs_case_mean':mean_ci(request_errors['case_mean']-request_errors['state_history'])}
    assert all(np.isfinite(v).all() for v in request_errors.values())
    (OUT/'k14_history_predictions.json').write_text(json.dumps({name:p.tolist() for name,p in predictions.items()}))
    output={'history_prediction':history_prediction,'cases':cases,'regime_choices':regime,'ratio_gains':gains,'k14_history_tests':history_tests,'ratio_sensitivity_stability':sensitivity,
            'sources':manifest,'sample_sha256':expected_sha,'fold_by_request':folds.tolist(),
            'limitations':['No per-forward or per-request elapsed times.', 'One timing run per configuration.',
                           'Batch labels are configured maxima, not constant active batch.',
                           'Different parameter runs follow different sampled trajectories.',
                           'History tests require 16 past and 16 future rounds and exclude short/tail segments.',
                           'Same 100 prompts across batches; they are not independent repetitions.',
                           'Context length and generation stage can confound historical associations.']}
    (OUT/'k14_history_windows.json').write_text(json.dumps(phase,indent=2))
    (OUT/'analysis.json').write_text(json.dumps(output,indent=2))
    with (OUT/'case_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(cases[0]));writer.writeheader();writer.writerows(cases)
    (OUT/'per_request_metrics.json').write_text(json.dumps(request_stats,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True)
    for ax,b in zip(axs.flat,BS):
        grid=[v for v in cases if v['batch_limit']==b];best=max(v['tps'] for v in grid)
        for r in RS:
            vs=sorted([v for v in grid if v['r']==r],key=lambda v:v['k'])
            ax.plot([v['k'] for v in vs],[100*v['tps']/best for v in vs],'-o',label=f'r={r:g}')
        ax.axhline(95,color='gray',ls='--',alpha=.5);ax.set_title(f'Max batch = {b}');ax.set_xticks(KS);ax.grid(alpha=.2);ax.legend()
    for ax in axs[-1]:ax.set_xlabel('Draft length k')
    for ax in axs[:,0]:ax.set_ylabel('Throughput / best in regime (%)')
    fig.tight_layout();fig.savefig(OUT/'throughput_opportunity.png',dpi=180);fig.savefig(OUT/'throughput_opportunity.pdf');plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(12,3.6),sharey=True)
    for ax,row in zip(axs,history_tests):
        ax.bar(['Low history','Middle','High history'],[row['bins'][lab]['future_mean_accepted'] for lab in ['low','middle','high']]);ax.set_ylim(0,14);ax.set_title(f'k=14, r={row["r"]}');ax.tick_params(axis='x',rotation=20)
    axs[0].set_ylabel('Next 16 rounds: mean accepted length');fig.tight_layout();fig.savefig(OUT/'history_signal.png',dpi=180);plt.close(fig)
    for key in ['regime_choices','k14_history_tests','ratio_sensitivity_stability']:print(key,json.dumps(output[key]),flush=True)
    print('between-request variance fraction k14:',[(v['batch_limit'],v['r'],round(v['between_request_variance_fraction'],3)) for v in cases if v['k']==14])
    print('heartbeat actual batch ranges:',[(b,max(v['heartbeat_max_batch'] or 0 for v in cases if v['batch_limit']==b),sum(v['offload_log_events'] for v in cases if v['batch_limit']==b)) for b in BS])


if __name__=='__main__':main()
