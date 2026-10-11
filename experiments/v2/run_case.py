"""Run one timed Qwen3 case from validated prompt-only JSONL."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--request-file', type=Path, required=True)
    p.add_argument('--model', default='/data/models/Qwen3-8B')
    p.add_argument('--mode', choices=['sparse','ar'], default='sparse')
    p.add_argument('--k', type=int, default=4)
    p.add_argument('--r', type=float, default=.15)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--output-tokens', type=int, default=2048)
    p.add_argument('--max-model-len', type=int, default=32768)
    p.add_argument('--chunk-size', type=int, default=256)
    p.add_argument('--seed', type=int, default=330)
    p.add_argument('--temperature', type=float, default=.65)
    p.add_argument('--max-mem-gib', type=float, default=26)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--disable-timing', action='store_true')
    p.add_argument('--async-cpu', action='store_true',
                   help='Enable CPU/GPU overlap; requires --disable-timing')
    args = p.parse_args()
    if args.async_cpu and not args.disable_timing:
        p.error('--async-cpu requires --disable-timing: step timing only supports synchronous execution')
    if not (1<=args.k<=14 and 0<args.r<=1 and args.batch>0 and args.output_tokens>0):
        p.error('Expected k in [1,14], r in (0,1], positive batch and output length')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'case.json').exists():p.error('Output already has a completed case; choose another directory')
    from transformers import AutoTokenizer
    from serve.request.request import Request
    from serve.run import run_inference
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    config = json.loads((Path(args.model)/'config.json').read_text())
    assert args.max_model_len<=config['max_position_embeddings']
    if args.mode=='sparse':
        assert (args.k+1)*config['num_attention_heads']//config['num_key_value_heads']<=64
    records=[json.loads(l) for l in args.request_file.read_text().splitlines() if l.strip()]
    assert records
    reqs=[];meta=[]
    for i,row in enumerate(records):
        messages=row.get('messages') or [{'role':'user','content':row['prompt']}]
        ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
        assert row.get('input_tokens',len(ids))==len(ids), 'Sample tokenizer mismatch'
        bound=len(ids)+args.output_tokens+(args.k+1 if args.mode=='sparse' else 1)
        assert bound<=args.max_model_len, f'Sample {i}: total length {bound} exceeds limit'
        if args.mode=='sparse':
            import math
            assert max(16,math.ceil(bound*args.r))<6144, f'Sample {i}: sparse KV selection capacity exceeded'
        reqs.append(Request(ids,args.output_tokens))
        meta.append({'request_id':i,'sample_id':row.get('sample_id',row.get('question_id',str(i))),
                     'source':row.get('source','provided'),'input_tokens':len(ids)})
    cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    cfg.update(sample_sha256=hashlib.sha256(args.request_file.read_bytes()).hexdigest(),
               enable_thinking=False,cuda_graph=False,async_cpu=args.async_cpu,inputs=meta,
               cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
               git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
               git_diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff','HEAD'],cwd=REPO)).hexdigest(),
               gpu_inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv'],text=True))
    cfg['source_sha256'] = {str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest()
                            for tree in ['serve','experiments/v2'] for p in (REPO/tree).rglob('*.py')}
    (args.output/'config.json').write_text(json.dumps(cfg,indent=2))
    start=time.perf_counter()
    outputs=run_inference(reqs=reqs,model_name=args.model,max_batch_size=args.batch,tp_ranks=1,
        max_mem_per_rank=args.max_mem_gib,cuda_graph=False,async_cpu=args.async_cpu,dtype='bfloat16',
        kv_cache='pillar' if args.mode=='sparse' else 'full',admit_policy='offloading',batch_policy='greedy',
        chunk_size=args.chunk_size,clip_length=args.max_model_len,spec_stride=args.k,budget_ratio=args.r,
        num_min_budget=16,temperature=args.temperature,seed=args.seed,dataset_name='qwen3-v2-fixed-sample',
        result_dir=str(args.output),acceptance_log_path=str(args.output/'rounds.jsonl') if args.mode=='sparse' else None,
        step_log_path=None if args.disable_timing else str(args.output/'steps.jsonl'))
    elapsed=time.perf_counter()-start
    assert len(outputs)==len(meta)
    total=sum(x['len_output'] for x in outputs)
    if not args.disable_timing:
        timing=json.loads((args.output/'steps.summary.json').read_text())
        assert timing['successful'] and timing['committed_tokens']==total, (timing['committed_tokens'],total)
    cfg.update(inference_call_seconds=elapsed,total_output_tokens=total,
               outputs=[{'request_id':x['id'],'output_tokens':x['len_output']} for x in outputs])
    tmp=args.output/'case.json.tmp';tmp.write_text(json.dumps(cfg,indent=2));tmp.replace(args.output/'case.json')
    print(json.dumps({'complete':True,'requests':len(outputs),'total_output_tokens':total,'output':str(args.output)}))


if __name__=='__main__':main()
