"""Isolate mapped-host restore scheduling; no serving changes or profiler.

Compare the shipped scatter grid with a diagnostic persistent grid. All
indices/buffers are preallocated; compute touches disjoint storage. The
persistent variant is a mechanism control, not a production implementation.
"""
import argparse
import json
from pathlib import Path
import time

import torch
import triton
import triton.language as tl

from serve.attention.kv_pack import _scatter_kv_pages


@triton.jit
def probe_copy(src, dst, N:tl.constexpr, BLOCK:tl.constexpr):
    x=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    tl.store(dst+x,tl.load(src+x,mask=x<N,other=0),mask=x<N)


@triton.jit
def persistent_restore(src,dst,pages,count:tl.constexpr,LAYERS:tl.constexpr,HEADS:tl.constexpr,DIM:tl.constexpr,TOKENS:tl.constexpr):
    tiles=tl.cdiv(count,TOKENS)
    lane=tl.arange(0,2*DIM)
    for task in range(tl.program_id(0),LAYERS*HEADS*tiles,tl.num_programs(0)):
        lh=task//tiles
        layer=lh//HEADS
        head=lh%HEADS
        j=(task%tiles)*TOKENS+tl.arange(0,TOKENS)
        p=tl.load(pages+j,mask=j<count,other=0).to(tl.int64)
        kv=lane//DIM
        d=lane%DIM
        s=((((layer.to(tl.int64)*count+j[:,None])*2+kv[None,:])*HEADS+head)*DIM+d[None,:])
        t=((((layer.to(tl.int64)*count+p[:,None])*2+kv[None,:])*HEADS+head)*DIM+d[None,:])
        v=tl.load(src+s,mask=j[:,None]<count,other=0)
        tl.store(dst+t,v,mask=j[:,None]<count)


def main(args):
    torch.cuda.set_device(0)
    device=torch.device('cuda:0')
    L,H,D=36,4,128
    n=args.pages
    host=torch.empty((L,n,2,1,H,D),dtype=torch.float16,pin_memory=True)
    host.fill_(.125)
    host[:,0].fill_(.5)
    destination=torch.empty_like(host,device=device)
    pages=torch.arange(n,device=device,dtype=torch.int32)
    source=torch.arange(65536,device=device,dtype=torch.float32)
    target=torch.empty_like(source)
    a=torch.randn((256,4096),device=device,dtype=torch.float16)
    b=torch.randn((4096,4096),device=device,dtype=torch.float16)
    c=torch.empty((256,4096),device=device,dtype=torch.float16)
    transfer=torch.cuda.Stream()
    compute=torch.cuda.Stream()
    default=torch.cuda.default_stream()
    sms=torch.cuda.get_device_properties(0).multi_processor_count
    strides=host.stride()
    def restore(kind):
        if kind=='original':
            _scatter_kv_pages[(triton.cdiv(n,8),L,H)](host,destination,destination,pages,n,H,D,
                strides[0],strides[1],strides[2],strides[4],0,0,
                strides[0],strides[1],strides[2],strides[4],
                RESTORE=True,TOKENS=8,WIDTH=2*D,num_warps=8)
        elif kind.startswith('bounded'):
            programs=sms if kind=='bounded' else int(kind.split('_')[1])
            persistent_restore[(programs,)](host,destination,pages,n,L,H,D,8,num_warps=8)
    # Complete compilation and library initialization before measured intervals.
    restore('original')
    restore('bounded')
    probe_copy[(128,)](source,target,65536,512,num_warps=4)
    torch.mm(a,b,out=c)
    torch.cuda.synchronize()
    # Validate all mapped bytes for both implementations outside timing.
    checks={}
    for kind in ['original','bounded','bounded_8','bounded_32']:
        destination.zero_()
        restore(kind)
        torch.cuda.synchronize()
        checks[kind]=bool(torch.equal(destination.cpu(),host))
        if not checks[kind]: raise RuntimeError('Bad restored data: '+kind)
    expected=c.clone()
    records=[]
    # Alternate conditions so the host source/cache and clock state are shared.
    for rep in range(args.repeats):
        for kind,which in [('none','default'),('original','none'),('bounded','none'),
                           ('original','default'),('original','nonblocking'),('bounded','default'),('bounded','nonblocking'),
                           ('bounded_8','none'),('bounded_8','nonblocking'),('bounded_32','none'),('bounded_32','nonblocking')]:
            torch.cuda.synchronize()
            origin,rs,re,ps,pe,ge=[torch.cuda.Event(enable_timing=True) for _ in range(6)]
            origin.record(default)
            transfer.wait_event(origin)
            cs=default if which=='default' else compute
            cs.wait_event(origin)
            begin=time.perf_counter()
            with torch.cuda.stream(transfer):
                rs.record()
                restore(kind)
                re.record()
            submit_ms=(time.perf_counter()-begin)*1e3
            if kind!='none': time.sleep(.002)
            if which!='none':
                with torch.cuda.stream(cs):
                    ps.record()
                    probe_copy[(128,)](source,target,65536,512,num_warps=4)
                    pe.record()
                    torch.mm(a,b,out=c)
                    ge.record()
            torch.cuda.synchronize()
            start,end=origin.elapsed_time(rs),origin.elapsed_time(re)
            row=dict(rep=rep,restore=kind,compute_stream=which,restore_start_ms=start,restore_end_ms=end,
                restore_ms=end-start,cpu_restore_submit_ms=submit_ms)
            if which!='none':
                t0,t1,t2=[origin.elapsed_time(e) for e in [ps,pe,ge]]
                row.update(probe_start_marker_ms=t0,probe_end_ms=t1,gemm_end_ms=t2,
                    probe_queue_and_run_ms=t1-t0,gemm_after_probe_ms=t2-t1,
                    probe_completed_during_restore=start<t1<end,
                    gemm_completed_during_restore=start<t2<end,
                    probe_lead_before_restore_end_ms=end-t1,gemm_lead_before_restore_end_ms=end-t2,
                    probe_correct=bool(torch.equal(source,target)),gemm_correct=bool(torch.equal(c,expected)))
                if not row['probe_correct'] or not row['gemm_correct']: raise RuntimeError('Compute mismatch')
            records.append(row)
    import statistics
    groups=[]
    for key in sorted({(r['restore'],r['compute_stream']) for r in records}):
        rows=[r for r in records if (r['restore'],r['compute_stream'])==key]
        group=dict(restore=key[0],compute_stream=key[1],repeats=len(rows),
            median_restore_ms=statistics.median(r['restore_ms'] for r in rows),
            median_cpu_submit_ms=statistics.median(r['cpu_restore_submit_ms'] for r in rows))
        if key[1]!='none':
            group.update(median_probe_queue_and_run_ms=statistics.median(r['probe_queue_and_run_ms'] for r in rows),
                median_gemm_after_probe_ms=statistics.median(r['gemm_after_probe_ms'] for r in rows),
                probe_completed_during_restore=sum(r['probe_completed_during_restore'] for r in rows),
                gemm_completed_during_restore=sum(r['gemm_completed_during_restore'] for r in rows),
                median_probe_lead_ms=statistics.median(r['probe_lead_before_restore_end_ms'] for r in rows),
                median_gemm_lead_ms=statistics.median(r['gemm_lead_before_restore_end_ms'] for r in rows))
        groups.append(group)
    output=dict(device=torch.cuda.get_device_name(),sms=sms,geometry=list(host.shape),bytes=host.numel()*host.element_size(),
        validation=checks,groups=groups,records=records,
        notes=['Original uses the shipped restore kernel unchanged, with preallocated indices to isolate GPU scheduling.',
               'Bounded uses a diagnostic persistent loop; bounded_8/32 limit the whole grid to 8/32 CTAs; bounded uses one CTA per SM. It is not a serving implementation change.',
               'CUDA event start markers may execute before a queued compute kernel can obtain resources; probe queue_and_run includes waiting.',
               'Independent GEMM is a compute feasibility probe, not a full model forward or a throughput benchmark.',
               'No CUDA synchronization is inserted between transfer submission and compute submission. Final synchronization only collects completed events.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:output[k] for k in ['device','sms','geometry','bytes','validation','groups']},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pages',type=int,default=16000)
    p.add_argument('--repeats',type=int,default=5)
    main(p.parse_args())
