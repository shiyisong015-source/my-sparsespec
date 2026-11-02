"""Run prepared fixed-cohort cases sequentially on one idle TP2 pair."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from experiments.acceptance.run import require_idle,save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--gpus',required=True)
    p.add_argument('--cases',nargs='+',required=True)
    a=p.parse_args()
    root=a.root.resolve();pair=list(map(int,a.gpus.split(',')))
    if len(pair)!=2 or len(set(pair))!=2:raise ValueError('Need distinct TP2 GPUs')
    status=dict(gpus=pair,cases={},state='starting')
    path=root/f'queue-{pair[0]}-{pair[1]}.json'
    save(path,status)
    for name in a.cases:
        case=root/name
        data=json.loads((case/'worker-payload.json').read_text())
        if data['kwargs']['budget_ratio']!=.05 or data['kwargs']['spec_stride']!=32:raise ValueError('Unexpected r/k')
        if (case/'runtime.log').exists():raise RuntimeError('Case already started; refusing overwrite: '+name)
        inventory=require_idle(pair)
        command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                 'experiments/diagnostics/draft_drain.py','--worker-payload',str(case/'worker-payload.json')]
        status.update(state='running',current=name)
        status['cases'][name]=dict(state='running',started=time.time(),command=command,inventory=inventory)
        save(path,status)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,pair)),ENABLE_INTRA_NODE_COMM='1',OMP_NUM_THREADS='1')
        with (case/'runtime.log').open('x') as log:
            result=subprocess.run(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
        valid=result.returncode==0 and all(json.loads((case/f'observer.rank{r}.json').read_text())['successful'] for r in [0,1])
        status['cases'][name].update(state='complete' if valid else 'failed',finished=time.time(),exit_code=result.returncode)
        save(path,status)
        if not valid:
            status['state']='failed';save(path,status)
            raise RuntimeError('Fixed-cohort case failed; queue stopped: '+name)
    status['state']='complete';save(path,status)


if __name__=='__main__':main()
