"""Fixed-r real-model draft depletion diagnostics; no probability/noise traces.

Warm-up parks each request only after its first completed verify and cold
partition. This deliberately creates an all-hot initial cohort. Aligned and
native-admission-offset starts are separate mechanism controls, not production
traffic claims. The serving admission algorithm is not changed.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def prepare(root):
    original=json.loads((REPO/'log/transfer-compute-current-20261021/20261021-201204/worker-payload.json').read_text())
    rows=sorted(original['requests'],key=lambda r:len(r['input_ids']))
    bins=[rows[i:i+8] for i in range(0,64,8)]
    rng=random.Random(330)
    for b in bins:rng.shuffle(b)
    ordered=[b[i] for i in range(8) for b in bins]
    source={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (REPO/'serve').rglob('*.py')}
    root.mkdir(parents=True,exist_ok=False)
    for n,phase in [(8,'aligned'),(16,'aligned'),(32,'aligned'),(64,'aligned'),(64,'native_offsets')]:
        out=root/f'b{n}-{phase}'
        out.mkdir()
        kwargs=dict(original['kwargs'],max_batch_size=n,budget_ratio=.05,spec_stride=32,
                    max_num_batched_tokens=512,mem_fraction=.75)
        selected=ordered[:n]
        payload=dict(output=str(out.resolve()),kwargs=kwargs,requests=selected,output_caps=[4096]*n,
            source_sha256=source,cohort=True,phase=phase,measurement_verify_rounds=8,
            description='Controlled cold-backed initial cohort; no new requests after release; natural EOS retained',
            length_summary=dict(min=min(len(r['input_ids']) for r in selected),max=max(len(r['input_ids']) for r in selected),
                                mean=sum(len(r['input_ids']) for r in selected)/n),record_signals=False)
        save(out/'worker-payload.json',payload)
    print(root,flush=True)


def worker(path):
    import torch
    import torch.distributed as dist
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    import serve.scheduler.dynamic_compact_scheduler as module
    import serve.attention.kv_pack as packing
    from serve.request.kv_cache_ptr.compact import CompactCachePtr
    from experiments.reproduction import run_compact_acceptance_scan as runner
    payload=json.loads(path.read_text())
    out=Path(payload['output'])
    rank=int(os.environ['RANK'])
    for name,h in payload['source_sha256'].items():
        if hashlib.sha256((REPO/name).read_bytes()).hexdigest()!=h:raise RuntimeError('Serving source changed: '+name)
    # Our scheduler below classifies both natural and diagnostic completion.
    runner.completion_audit_class=lambda base:base

    class Diagnostic(module.DynamicCompactSpecScheduler):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.started=False
            self.origin=None
            self.initial_offsets={}
            self.round_base={}
            self.forward_meta={}
            self.gpu_records=[]
            self.journal=(out/f'states.rank{rank}.jsonl').open('w',buffering=1)
            self.saved_kernels=[]
            owner=self
            class KernelProxy:
                def __init__(self,original,direction):self.original,self.direction=original,direction
                def __getitem__(self,grid):
                    launch=self.original[grid]
                    def call(*a,**kw):
                        if not owner.started:return launch(*a,**kw)
                        first=torch.cuda.Event(enable_timing=True);last=torch.cuda.Event(enable_timing=True)
                        first.record()
                        result=launch(*a,**kw)
                        last.record()
                        owner.gpu_records.append((dict(kind=self.direction,step=owner.step_id,
                            transfer_request=getattr(owner,'launch_request',None)),first,last))
                        return result
                    return call
            for name,direction in [('_restore_kv_pages_bounded','H2D'),('_pack_selected_kv','D2H')]:
                original=getattr(packing,name)
                self.saved_kernels.append((name,original))
                setattr(packing,name,KernelProxy(original,direction))
            original_forward=self.model.forward
            def forward(*a,**kw):
                if not self.started:return original_forward(*a,**kw)
                first=torch.cuda.Event(enable_timing=True);last=torch.cuda.Event(enable_timing=True)
                first.record()
                torch.cuda.nvtx.range_push(f'drain.forward step={self.step_id}')
                try:result=original_forward(*a,**kw)
                finally:torch.cuda.nvtx.range_pop()
                last.record()
                self.gpu_records.append((dict(kind='forward',step=self.step_id,**self.forward_meta),first,last))
                return result
            self.model.forward=forward

        def write(self,event,**values):
            self.journal.write(json.dumps(dict(event=event,step=self.step_id,rank=rank,
                perf=time.perf_counter(),measurement=self.started,**values),separators=(',',':'))+'\n')

        def add_request(self,request,is_restore=False):
            result=super().add_request(request,is_restore)
            self.initial_offsets.setdefault(request.request_id,int(request.initial_idx))
            self.write('admit',request=request.request_id)
            return result

        def _eligible(self,request):
            if (not self.started and request.status.name=='DRAFT'
                    and isinstance(request.kv_cache_ptr,CompactCachePtr)
                    and request.kv_cache_ptr.phase in ('resident','hot')):
                return False
            return super()._eligible(request)

        def _submit_transfer(self,request,direction):
            ptr=request.kv_cache_ptr
            self.write('transfer_submit',request=request.request_id,direction=direction,
                cold_pages=ptr.cold_count,physical_pages=len(ptr.indices),logical=ptr.len_kv_cache)
            self.launch_request=request.request_id
            torch.cuda.nvtx.range_push(f'drain.{direction} request={request.request_id} step={self.step_id}')
            try:return super()._submit_transfer(request,direction)
            finally:
                torch.cuda.nvtx.range_pop()
                self.launch_request=None

        def _poll_transfers(self):
            before={rid:job.direction for rid,job in self._transfers.items()}
            result=super()._poll_transfers()
            for rid in before.keys()-self._transfers.keys():self.write('transfer_commit',request=rid,direction=before[rid])
            return result

        def start_measurement(self):
            self.compute_stream.synchronize();self.transfer_stream.synchronize()
            torch.cuda.current_stream().synchronize()
            dist.barrier(group=self.group)
            for r in self._requests.values():
                if r.status.name!='DRAFT' or r.len_draft_probs!=0 or r.kv_cache_ptr.phase!='hot':
                    raise RuntimeError('Cohort barrier contains unsettled/non-hot request')
                offset=int(self.initial_offsets[r.request_id]) if payload['phase']=='native_offsets' else 0
                r.cur_spec_idx=r.initial_idx=offset
                self.round_base[r.request_id]=self.acceptance_rounds[r.request_id]
            self.origin=torch.cuda.Event(enable_timing=True)
            self.origin.record()
            self.compute_stream.wait_event(self.origin);self.transfer_stream.wait_event(self.origin)
            self.started=True
            self.write('measurement_start',cohort=len(self._requests),phase=payload['phase'],
                offsets={r.request_id:r.initial_idx for r in self._requests.values()},
                free_pages=self.kv_pool.num_free_pages,capacity=self.kv_pool.capacity)
            if rank==0:print('DRAIN_MEASUREMENT_START',len(self._requests),payload['phase'],flush=True)

        def admit_requests(self,pending):
            if not self.started:
                self._poll_retired();self._poll_transfers()
                current=list(self._requests.values())
                if (not pending and current and not self._transfers
                        and all(r.status.name=='DRAFT' and isinstance(r.kv_cache_ptr,CompactCachePtr)
                                and r.kv_cache_ptr.phase=='hot' for r in current)):
                    self.start_measurement()
                elif not self._transfers:
                    victims=[r for r in current if r.status.name=='DRAFT'
                        and isinstance(r.kv_cache_ptr,CompactCachePtr) and r.kv_cache_ptr.phase=='resident'
                        and r.len_draft_probs==0]
                    if victims:self._submit_transfer(min(victims,key=lambda r:r.request_id),'repack')
            if self.started and pending:raise RuntimeError('New admission during fixed-cohort measurement')
            result=super().admit_requests(pending)
            if self.step_id%100==0 and rank==0:
                print('DRAIN_PROGRESS',self.step_id,self.started,len(self._requests),len(pending),flush=True)
            return result

        def prepare_forward_inputs(self):
            host,tensors=super().prepare_forward_inputs()
            if self.started:
                reqs=list(self._requests.values())
                chosen={r.request_id for r in self.active_requests if r.status.name!='STALL'}
                rows=[dict(id=r.request_id,status=r.status.name,phase=r.kv_cache_ptr.phase,
                    computing=r.request_id in chosen,transferring=r.request_id in self._transfers,
                    draft_tokens=r.len_draft_probs,context=r.len_kv_cache,
                    hot=r.kv_cache_ptr.len_loading_indices,physical=len(r.kv_cache_ptr.indices)) for r in reqs]
                phases=dict(Counter(r.status.name for r in self.active_requests if r.status.name!='STALL'))
                self.forward_meta=dict(batch=len(chosen),queries=0 if host is None else len(host['token_ids']),phases=phases)
                self.write('state',requests=rows,free_pages=self.kv_pool.num_free_pages,
                           transfers=len(self._transfers),**self.forward_meta)
            return host,tensors

        def is_request_complete(self,request,check_idx=-1):
            complete=super().is_request_complete(request,check_idx)
            reason=None
            if complete:
                limit=request.len_oracle_ctx
                if request.status.name=='NORMAL':at_limit=limit is not None and request.len_kv_cache+check_idx>=limit-1
                else:at_limit=limit is not None and request.len_kv_cache>=limit
                reason='length_limit' if at_limit else 'eos'
            elif (self.started and request.status.name=='STALL'
                    and self.acceptance_rounds.get(request.request_id,0)-self.round_base.get(request.request_id,0)
                        >=payload['measurement_verify_rounds']):
                complete=True;reason='diagnostic_round_limit'
            if complete:
                request.resource_finish_reason=reason
                self.write('complete',request=request.request_id,reason=reason)
            return complete

        def wait_for_ready_requests(self):
            before=time.perf_counter()
            result=super().wait_for_ready_requests()
            if self.started:self.write('no_ready_wait',seconds=time.perf_counter()-before)
            return result

        def run(self,requests):
            successful=False
            try:
                result=super().run(requests)
                self.compute_stream.synchronize();self.transfer_stream.synchronize()
                torch.cuda.current_stream().synchronize()
                records=[]
                if self.origin is None:raise RuntimeError('Cohort never reached measurement')
                for meta,first,last in self.gpu_records:
                    records.append(dict(meta,start_ms=self.origin.elapsed_time(first),end_ms=self.origin.elapsed_time(last)))
                save(out/f'gpu-events.rank{rank}.json',records)
                save(out/f'observer.rank{rank}.json',dict(successful=True,cohort_started=True,
                    initial_requests=len(payload['requests']),measured_requests=len(self.round_base),
                    phase=payload['phase'],metrics=dict(self.metrics),probability_noise_recording=False))
                successful=True
                return result
            finally:
                self.write('end',successful=successful)
                self.journal.close()
                for name,original in self.saved_kernels:setattr(packing,name,original)
    module.DynamicCompactSpecScheduler=Diagnostic
    runner.worker(path)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare',type=Path)
    p.add_argument('--worker-payload',type=Path)
    args=p.parse_args()
    if args.prepare:prepare(args.prepare)
    else:worker(args.worker_payload.resolve())
