"""Independent GPU workers claim v2 cases with flock, with provenance-aware resume."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

REPO=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--request-file',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True,choices=[0,1,2])
    p.add_argument('--phase',choices=['screen','confirm'],default='screen')
    p.add_argument('--seed',type=int,default=330)
    args=p.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    request_file=args.request_file.resolve();sha=hashlib.sha256(request_file.read_bytes()).hexdigest()
    ks=[4,8,14] if args.phase=='screen' else [4,6,8,12,14]
    jobs=[(b,k,r,'sparse') for b in [4,16,32] for k in ks for r in [.05,.1,.15,.25]]
    jobs += [(b,4,.15,'ar') for b in [4,16,32]]
    random.Random(args.seed+args.gpu).shuffle(jobs)
    status={'gpu':args.gpu,'pid':os.getpid(),'state':'running','completed':[]}
    statusfile=args.output/f'gpu{args.gpu}_status.json'
    def save():
        temp=statusfile.with_suffix('.json.tmp');temp.write_text(json.dumps(status,indent=2));temp.replace(statusfile)
    save()
    try:
        for batch,k,r,mode in jobs:
            name=f'{mode}_b{batch}_k{k}_r{r:g}_seed{args.seed}'
            case=args.output/name
            with (args.output/f'{name}.lock').open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                if (case/'case.json').exists():
                    report=json.loads((case/'case.json').read_text())
                    assert report['sample_sha256']==sha and report['seed']==args.seed
                    assert report['batch']==batch and report['k']==k and report['r']==r and report['mode']==mode
                    assert report['output_tokens']==2048 and not report['disable_timing']
                    continue
                status['current_case']=name;save()
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(args.gpu),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
                with (args.output/f'{name}.log').open('a') as log:
                    subprocess.run([sys.executable,str(REPO/'experiments/v2/run_case.py'),
                        '--request-file',str(request_file),'--batch',str(batch),'--k',str(k),'--r',str(r),
                        '--mode',mode,'--seed',str(args.seed),'--output',str(case)],cwd=REPO,env=env,
                        stdout=log,stderr=subprocess.STDOUT,check=True)
                status['completed'].append(name);save()
        status.update(state='complete',current_case=None);save()
    except Exception as error:
        status.update(state='failed',error=str(error));save();raise


if __name__=='__main__':main()
