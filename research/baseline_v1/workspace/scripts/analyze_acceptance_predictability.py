"""CPU-only, request-held-out ridge models of SparseSpec accepted draft length."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STATE = ['log_kv_len', 'log_draft_kv_len', 'effective_retention',
         'generated_tokens_1k', 'log_round', 'log_input_len']
HISTORY = ['last_accept_ratio', 'mean8', 'mean16', 'mean32', 'std16',
           'full_accept_fraction16', 'zero_accept_fraction16', 'recent8_minus_previous8']


def features(segment, t, input_len, k):
    row = segment[t]
    past = np.array([r['accepted_draft_tokens'] / k for r in segment[max(0,t-32):t]])
    last16 = past[-16:]
    state = [np.log1p(row['kv_len']), np.log1p(row['draft_kv_len']),
             row['draft_kv_len']/row['kv_len'], (row['kv_len']-input_len)/1000,
             np.log1p(row['round']), np.log1p(input_len)]
    history = [past[-1], past[-8:].mean(), last16.mean(), past.mean(), last16.std(),
               (last16==1).mean(), (last16==0).mean(), past[-8:].mean()-past[-16:-8].mean()]
    return state + history


def build(root):
    records = {1: [], 16: []}
    sources = []
    cases = sorted(root.glob('*/case.json'))
    sample_sha = None
    for ci, path in enumerate(cases):
        report = json.loads(path.read_text())
        assert len(report['inputs']) == len(report['outputs']) == 100
        assert report.get('max_batch_size', 1) == 1
        if sample_sha is None: sample_sha = report['sample_sha256']
        assert sample_sha == report['sample_sha256']
        k = report['spec_stride']
        rpath = path.parent/'rounds.jsonl'
        raw = rpath.read_bytes()
        sources.append({'case':path.parent.name, 'k':k, 'r':report['budget_ratio'],
                        'rounds_sha256':hashlib.sha256(raw).hexdigest(), 'case_path':str(path)})
        groups = {}
        for line in raw.splitlines():
            row = json.loads(line)
            groups.setdefault(row['request_id'], []).append(row)
        for rid, rows in groups.items():
            segments = [[]]
            for row in rows:
                if row['cold_start'] or row['actual_draft_tokens'] != k:
                    if segments[-1]: segments.append([])
                    continue
                assert 0 <= row['accepted_draft_tokens'] <= k
                if segments[-1]: assert row['round'] == segments[-1][-1]['round']+1
                segments[-1].append(row)
            for segment in segments:
                for h in records:
                    for t in range(16, len(segment)-h+1, 16):
                        y = np.mean([x['accepted_draft_tokens']/k for x in segment[t:t+h]])
                        records[h].append({'rid':rid, 'case':ci, 'k':k, 'round':segment[t]['round'],
                                           'y':float(y), 'x':features(segment,t,report['inputs'][rid]['input_tokens'],k)})
    return records, sources, sample_sha


def weights(case, rid):
    pairs = np.c_[case,rid]
    _, inv, counts = np.unique(pairs, axis=0, return_inverse=True, return_counts=True)
    w = 1/counts[inv]
    return w/w.mean()


def fit_predict(x, y, cases, rid, train, test, ncase, cols, alpha=100.):
    # Case intercepts unpenalized; shared standardized slopes have fixed L2 penalty.
    w = weights(cases[train],rid[train])
    selected = x[:,cols]
    mean = np.average(selected[train],axis=0,weights=w)
    scale = np.sqrt(np.average((selected[train]-mean)**2,axis=0,weights=w))
    scale[scale<1e-10] = 1
    design = np.c_[np.eye(ncase)[cases], (selected-mean)/scale]
    penalty = np.r_[np.full(ncase,1e-9), np.full(len(cols),alpha)]
    coef = np.linalg.solve(design[train].T@(w[:,None]*design[train])+np.diag(penalty),
                           design[train].T@(w*y[train]))
    return np.clip(design[test]@coef,0,1), {'columns':cols,'mean':mean.tolist(),'scale':scale.tolist(),
                                         'coefficients':coef.tolist(),'alpha':alpha}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=ROOT/'external/SparseSpec/log/acceptance-sweep/lcb-random100-seed330/Qwen3-8B')
    parser.add_argument('--output',type=Path,default=ROOT/'external/SparseSpec/log/acceptance-model/first-fit')
    args = parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    data,sources,sample_sha = build(args.input)
    rng=np.random.default_rng(330); shuffled=rng.permutation(100); folds=np.empty(100,dtype=int)
    for fold, ids in enumerate(np.array_split(shuffled,5)): folds[ids]=fold
    summary={'sample_sha256':sample_sha,'sources':sources,'fold_by_request':folds.tolist(),
             'feature_names':STATE+HISTORY,'protocol':{'folds':5,'past_only':True,'stride':16,
             'minimum_history':16,'ridge_alpha':100,'weighting':'equal case-request pairs',
             'scope':'unseen requests within observed k/r, batch=1','bootstrap_unit':'request',
             'limitations':['Only anchors with a complete future horizon are evaluated.',
                            'No throughput or parameter-switching benefit is established.',
                            'KV length and budget are correlated; coefficients are not causal.']},'targets':{}}
    fitted={}; charts={}
    for horizon, rows in data.items():
        x=np.array([v['x'] for v in rows]); y=np.array([v['y'] for v in rows]);
        rid=np.array([v['rid'] for v in rows]); case=np.array([v['case'] for v in rows]);k=np.array([v['k'] for v in rows])
        pred={name:np.zeros(len(y)) for name in ['case_mean','last_round','last16','state','history','state_history']}
        models={'case_mean':[], 'state':list(range(6)), 'history':list(range(6,14)), 'state_history':list(range(14))}
        for fold in range(5):
            train=folds[rid]!=fold;test=~train
            for name,cols in models.items():
                pred[name][test],_=fit_predict(x,y,case,rid,train,test,len(sources),cols)
            pred['last_round'][test]=x[test,6];pred['last16'][test]=x[test,8]
        pair_metrics={}; metrics={}
        for name,p in pred.items():
            by_request=[]
            for request_id in np.unique(rid):
                per_case=[]
                for ci in np.unique(case[rid==request_id]):
                    mask=(rid==request_id)&(case==ci);err=p[mask]-y[mask]
                    per_case.append([np.abs(err).mean(),(err**2).mean(),np.abs(err*k[mask]).mean()])
                by_request.append(np.mean(per_case,axis=0))
            pair_metrics[name]=np.array(by_request)
            agg=pair_metrics[name].mean(0)
            metrics[name]={'mae_ratio':float(agg[0]),'rmse_ratio':float(np.sqrt(agg[1])),'mae_tokens':float(agg[2])}
        differences={};brng=np.random.default_rng(331)
        for reference in ['case_mean','state']:
            delta=pair_metrics[reference][:,0]-pair_metrics['state_history'][:,0]
            bs=delta[brng.integers(0,len(delta),size=(4000,len(delta)))].mean(1)
            differences[reference]={'mae_reduction':float(delta.mean()),'ci95':np.quantile(bs,[.025,.975]).tolist()}
        per_case=[]
        for ci,source in enumerate(sources):
            ids=np.unique(rid[case==ci]); m={}
            for name,p in pred.items():
                m[name]=float(np.mean([np.abs(p[(case==ci)&(rid==i)]-y[(case==ci)&(rid==i)]).mean() for i in ids]))
            per_case.append({'case':source['case'],'request_count':len(ids),'mae_ratio':m})
        all_rows=np.ones(len(y),dtype=bool)
        _,fitted[str(horizon)]=fit_predict(x,y,case,rid,all_rows,all_rows,len(sources),models['state_history'])
        summary['targets'][str(horizon)]={'anchors':len(y),'requests':len(np.unique(rid)), 'metrics':metrics,
                                         'paired_mae_reduction_ci95':differences,'per_case':per_case}
        with (args.output/f'predictions_h{horizon}.jsonl').open('w') as stream:
            for i,row in enumerate(rows):
                result={key:row[key] for key in ['rid','case','k','round']}
                result.update(fold=int(folds[rid[i]]),actual_accepted_length=float(y[i]*k[i]),
                              predictions={name:float(p[i]*k[i]) for name,p in pred.items()})
                stream.write(json.dumps(result)+'\n')
        charts[horizon]=(y,pred,rid,case,k,np.array([v['round'] for v in rows]))
        print(json.dumps({'horizon':horizon,**summary['targets'][str(horizon)]},ensure_ascii=False),flush=True)
    (args.output/'metrics.json').write_text(json.dumps(summary,indent=2))
    (args.output/'fitted_models.json').write_text(json.dumps({'sources':sources,'feature_names':STATE+HISTORY,'models':fitted},indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['case_mean','last_round','last16','state','history','state_history']
    labels=['Case mean','Last round','Last 16','State','History','State + history']
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,h in zip(axes,[1,16]):
        ax.bar(np.arange(len(names)),[100*summary['targets'][str(h)]['metrics'][n]['mae_ratio'] for n in names])
        ax.set_xticks(np.arange(len(names)),labels,rotation=35,ha='right');ax.set_ylabel('MAE (% of draft length)')
        ax.set_title('Next round' if h==1 else 'Mean of next 16 rounds')
    fig.suptitle('Five-fold held-out-request predictions; lower is better');fig.tight_layout()
    fig.savefig(args.output/'prediction_error.png',dpi=180);fig.savefig(args.output/'prediction_error.pdf');plt.close(fig)
    y,pred,rid,case,k,rounds=charts[16];ci=next(i for i,v in enumerate(sources) if v['case']=='k8_r0.1')
    counts=[((rid==i)&(case==ci)).sum() for i in range(100)];chosen=int(np.argsort(counts)[len(counts)//2])
    mask=(rid==chosen)&(case==ci)
    fig,ax=plt.subplots(figsize=(8,4));ax.plot(rounds[mask],y[mask]*k[mask],'-o',label='Observed next-16 mean')
    for name in ['case_mean','state_history']:ax.plot(rounds[mask],pred[name][mask]*k[mask],'-o',label=name)
    ax.set(xlabel='Verification round at prediction',ylabel='Accepted draft tokens / round',ylim=(0,8),title=f'k=8, r=0.1, request {chosen} (held out)')
    ax.legend();fig.tight_layout();fig.savefig(args.output/'example_request.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
