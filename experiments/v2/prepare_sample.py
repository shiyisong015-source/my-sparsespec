"""Validate local LongBench JSONL exports with a pinned official prompt-template JSON."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-dir',type=Path,required=True,help='Contains gov_report.jsonl, qmsum.jsonl, multi_news.jsonl')
    p.add_argument('--templates',type=Path,required=True,help='LongBench dataset2prompt.json; version/hash is saved')
    p.add_argument('--model',default='/data/models/Qwen3-8B')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=330)
    p.add_argument('--screen-size',type=int,default=32)
    p.add_argument('--confirm-per-source',type=int,default=32)
    args=p.parse_args()
    if (args.output/"manifest.json").exists():
        p.error("A sample manifest already exists; choose a new output directory")
    if args.screen_size < 4 or args.confirm_per_source < 1:
        p.error("screen-size must be >=4 and confirm-per-source must be positive")
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    templates=json.loads(args.templates.read_text());valid=[];rejected=[];sources=[]
    for name in ['gov_report','qmsum','multi_news']:
        source=args.source_dir/f'{name}.jsonl';raw=source.read_bytes()
        sources.append({'name':name,'path':str(source.resolve()),'sha256':hashlib.sha256(raw).hexdigest()})
        for index,line in enumerate(raw.splitlines()):
            row=json.loads(line);prompt=templates[name].format(**row)
            ids=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
            n=len(ids);bound=n+2048+15
            if not ids or n>20480 or bound>32768 or math.ceil(bound*.25)>=6144:
                rejected.append({'source':name,'index':index,'input_tokens':n});continue
            bucket=0 if n<2048 else 1 if n<8192 else 2 if n<16384 else 3
            valid.append({'sample_id':f'{name}:{row.get("_id",index)}','source':name,'source_index':index,
                          'input_tokens':n,'length_bucket':bucket,'prompt':prompt})
    rng=random.Random(args.seed);rng.shuffle(valid)
    screen=[];selected=set()
    for bucket in range(4):
        candidates=[v for v in valid if v['length_bucket']==bucket]
        for row in candidates[:args.screen_size//4]:screen.append(row);selected.add(row['sample_id'])
    for row in valid:
        if len(screen)>=args.screen_size:break
        if row['sample_id'] not in selected:screen.append(row);selected.add(row['sample_id'])
    assert len(screen)==args.screen_size, 'Insufficient valid screening data'
    confirm=[]
    for name in ['gov_report','qmsum','multi_news']:
        candidates=[v for v in valid if v['source']==name and v['sample_id'] not in selected]
        assert len(candidates)>=args.confirm_per_source, f'Insufficient held-out data: {name}'
        confirm.extend(candidates[:args.confirm_per_source])
    rng.shuffle(screen);rng.shuffle(confirm)
    assert len(set(v['sample_id'] for v in screen+confirm))==len(screen+confirm)
    args.output.mkdir(parents=True,exist_ok=True)
    manifest={'seed':args.seed,'model':args.model,'enable_thinking':False,'sources':sources,
              'templates_sha256':hashlib.sha256(args.templates.read_bytes()).hexdigest(),'rejected':rejected,
              'limits':{'input':20480,'output':2048,'reserved_tokens':15,'max_r':.25},'samples':{}}
    for name,rows in [('screen',screen),('confirm',confirm)]:
        data=''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in rows);(args.output/f'{name}.jsonl').write_text(data)
        manifest['samples'][name]={'count':len(rows),'sha256':hashlib.sha256(data.encode()).hexdigest(),
                                  'length_bins':[sum(v['length_bucket']==i for v in rows) for i in range(4)],
                                  'sources':{s:sum(v['source']==s for v in rows) for s in ['gov_report','qmsum','multi_news']}}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest['samples'],indent=2))


if __name__=='__main__':main()
