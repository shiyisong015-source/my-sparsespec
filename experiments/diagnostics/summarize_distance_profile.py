"""Compare completed TP1 B32 windows with explicit memory and PCIe placement."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def load(path):
    return json.loads(path.read_text())


def distance_audit(path):
    counts=dict(plans=0,plans_with_future_sources=0,max_claimed_pages=0,
                nearer_victim_violations=0,farthest_first_order_violations=0,
                preemptions=0,strictly_nearer_preemptions=0,release_backed_chunks=0)
    with path.open() as stream:
        for line in stream:
            row=json.loads(line)
            if row['event']=='plan':
                victims=row['victims'];counts['plans']+=1
                counts['plans_with_future_sources']+=bool(victims)
                counts['max_claimed_pages']=max(counts['max_claimed_pages'],sum(v['pages'] for v in victims))
                counts['nearer_victim_violations']+=sum(v['distance']<row['distance'] for v in victims)
                counts['farthest_first_order_violations']+=sum(b['distance']>a['distance'] for a,b in zip(victims,victims[1:]))
            elif row['event']=='plan_preempt':
                counts['preemptions']+=1
                counts['strictly_nearer_preemptions']+=row['new_distance']<row['old_distance']
            elif row['event']=='chunk_submit' and row['direction']=='release_backed':
                counts['release_backed_chunks']+=1
    counts['scope']='Logged plan creation and preemption; not a reconstruction of all live candidates at every instant.'
    return counts


def main(root):
    plan_path=root/'plan.json'
    plan=load(plan_path if plan_path.exists() else root.parent/'plan.json')
    rows=[]
    details={}
    manifests=[]
    placements=[]
    for policy in ('reservation','distance_swap'):
        directory=root/policy/'profile'
        manifest=load(directory/'manifest.rank0.json')
        summary=load(directory/'component-summary.json')
        placement=load(root/policy/'placement.json')
        completion=load(directory/'b32/completion.rank0.json')
        job=load(root/policy/'job.json')
        if job['status']!='complete':
            raise RuntimeError(f'Profile did not finish: {policy}')
        coverage=summary['sampling_coverage']['b32']
        if coverage['status']!='complete':
            raise RuntimeError(f'Incomplete windows: {policy}: {coverage}')
        manifests.append(manifest)
        placements.append(placement)
        details[policy]=dict(placement=placement,manifest=manifest,
            completion_scope=completion.get('termination_scope'),unfinished=len(completion['unfinished_request_ids']),
            runtime_peak_reserved_bytes=completion['metrics']['runtime_peak_reserved_bytes'],
            non_torch_peak_observed_bytes=completion['metrics'].get('non_torch_peak_observed_bytes'),
            source_hashes=manifest['source_sha256'])
        peak_upper=(completion['metrics']['runtime_peak_reserved_bytes']+
                    completion['metrics'].get('non_torch_peak_observed_bytes',0))
        details[policy]['observed_peak_upper_bytes']=peak_upper
        details[policy]['total_gpu_budget_bytes']=manifest['gpu_total_bytes']*manifest['mem_fraction']
        details[policy]['memory_budget_status']=(
            'within_budget' if peak_upper<=details[policy]['total_gpu_budget_bytes'] else 'exceeded')
        if policy=='distance_swap':
            details[policy]['distance_audit']=distance_audit(directory/'b32/distance-swap.rank0.jsonl')
        waits=summary.get('verify_timelines',{})
        for case in summary['cases']:
            rank=case['ranks'][0]
            if rank['window_status']!='complete':
                raise RuntimeError(f"Short capture: {case['case']}")
            window=directory/case['case']
            batches=[json.loads(line) for line in (window/'batches.rank0.jsonl').read_text().splitlines() if line.strip()]
            load_stats=rank['window_load']
            wait=waits.get(f"{case['case']}/verify-timeline.rank0",{})
            traffic=rank['transfers']
            down=traffic.get('down' if policy=='distance_swap' else 'repack',{})
            up=traffic.get('up' if policy=='distance_swap' else 'restore',{})
            key=lambda name:load_stats.get(name,{}).get('mean')
            cpu=rank['cpu']
            row=dict(policy=policy,gpu=placement['gpu'],window=rank['window_id'],
                seconds=rank['window_elapsed_seconds'],forwards=rank['window_native_counter_deltas']['forwards'],
                mean_batch=key('actual_batch'),mean_D=key('D'),mean_V=key('V'),mean_P=key('P'),
                mean_queries=key('query_tokens'),mean_context=key('context_median'),
                mean_live=key('live_requests'),minimum_free_pages=min(b['free_kv_pages'] for b in batches),
                verified_tokens=rank.get('window_verified_output_tokens'),
                verified_output_tps=rank.get('window_verified_output_tps'),
                verify_wait_mean_ms=wait.get('host_need_to_submit_ms',{}).get('mean'),
                verify_wait_p90_ms=wait.get('host_need_to_submit_ms',{}).get('p90'),
                d2h_GB=down.get('bytes',0)/1e9,h2d_GB=up.get('bytes',0)/1e9,
                d2h_calls=down.get('count',0),h2d_calls=up.get('count',0),
                forward_gpu_event_mean_ms=statistics.mean(rank['all_decode_forward_ms']) if rank['all_decode_forward_ms'] else None,
                forward_cpu_wall_mean_ms=statistics.mean(rank['all_decode_forward_cpu_ms']) if rank.get('all_decode_forward_cpu_ms') else None,
                cpu_admission_mean_ms=cpu.get('cpu_admission',{}).get('mean_ms'),
                cpu_metadata_mean_ms=cpu.get('metadata_prepare',{}).get('mean_ms'))
            fine=[cell for cell in summary.get('kernel_components_by_DVP',[]) if cell.get('case')==case['case']]
            row['fine_component_groups']=len(fine)
            rows.append(row)
    a,b=manifests
    for name in ('input_sha256','request_ids','input_lengths','pool_pages','buffer_batch_capacity','gpu_total_bytes',
                 'runtime_reserve_bytes','mem_fraction','query_budget','source_sha256'):
        if a[name]!=b[name]:
            raise RuntimeError(f'Comparison configuration mismatch: {name}')
    if placements[0]['pcie_root']==placements[1]['pcie_root']:
        raise RuntimeError('Comparison used the same PCIe root')
    if set(placements[0]['cpu_affinity']) & set(placements[1]['cpu_affinity']):
        raise RuntimeError('Comparison CPU sets overlap')
    fields=list(rows[0])
    with (root/'window-comparison.csv').open('w') as output:
        writer=csv.DictWriter(output,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)
    report=dict(plan=plan,policies=details,windows=rows,
        notes=['Three windows from one trajectory are not independent repetitions.',
               'Context distributions differ when the policies progress at different speeds.',
               'Verified output counts include carry-in rounds and subtract native EOS trimming; unfinished requests are not EOS.',
               'D2H/H2D spans are GPU operation envelopes, not pure PCIe DMA or additive elapsed time.',
               'New policy uses chronological history backing including a duplicate of selected KV; transfer bytes must be reported.',
               'CUDA/NVTX collected; privileged SM/HBM hardware counters unavailable.',
               'Different GPUs and NUMA sockets; no claim of perfect hardware equivalence.'])
    (root/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    def value(number,precision=2):
        return '—' if number is None else f'{number:.{precision}f}'
    audit=details['distance_swap']['distance_audit']
    text=[f"本次比较使用 mem_fraction={a['mem_fraction']} 总显存预算，包含模型与实测运行缓冲。两侧 KV 池相同：{a['pool_pages']} 页（{a['pool_pages']*a['kv_page_bytes_per_rank']/1024**3:.3f} GiB）。",
          f"输入固定 LCB128、seed330、k=8、r=.05、B32、query2048、元数据容量128。GPU{placements[0]['gpu']} / {placements[0]['pcie_root']} / NUMA{placements[0]['numa_node']} 与 GPU{placements[1]['gpu']} / {placements[1]['pcie_root']} / NUMA{placements[1]['numa_node']} 独立，CPU 与内存均本地绑定。",
          '', '| 策略 | 窗口 | 实际准入 | 实际 batch | D / V / P | 有效输出 token/s | Verify 等待 ms | 上下文均值 | D2H / H2D GB |',
          '|---|---:|---:|---:|---|---:|---:|---:|---:|']
    for row in rows:
        text.append(f"| {row['policy']} | {row['window']} | {value(row['mean_live'])} | {value(row['mean_batch'])} | {value(row['mean_D'])} / {value(row['mean_V'])} / {value(row['mean_P'])} | {value(row['verified_output_tps'])} | {value(row['verify_wait_mean_ms'])} | {value(row['mean_context'],0)} | {value(row['d2h_GB'])} / {value(row['h2d_GB'])} |")
    text+=['', '显存峰值保守估计（PyTorch reserved 峰值 + 窗口边界实测 non-Torch 最大值；两者不保证同时出现）：'+
           '；'.join(f"{name} {detail['observed_peak_upper_bytes']/2**30:.3f} GiB，{detail['memory_budget_status']}"
                    for name,detail in details.items())+'。',
           f"日志计划排序检查：{audit['plans']} 个计划中，{audit['plans_with_future_sources']} 个认领未来释放来源；牺牲更近请求 {audit['nearer_victim_violations']} 次，未按距离从远到近排序 {audit['farthest_first_order_violations']} 次。仅核对计划创建快照，不重建所有时刻的候选集合。",
           '采窗后安全排空并退出，未完成请求保持未完成；不报告整批自然 EOS 吞吐。三窗来自同次运行的不同阶段，上下文不同，不能作为三次独立重复。',
           'CUDA/NVTX 和组件 Event 已采；SM/HBM 硬件计数器因权限缺失。CPU、forward 和传输区间有重叠，父子组件不能相加。新策略的历史副本表示与旧 compact cold-only 布局的字节量不同，表中给出真实搬运量。',
           '详细组件与次数见各策略 profile/component-latencies.csv；完整配置、源文件哈希、显存标定、失败尝试和 PCIe 占用检查保留在本结果目录。']
    (root/'comparison_zh.md').write_text('\n\n'.join(text[:2])+'\n'+'\n'.join(text[2:])+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for policy in ('reservation','distance_swap'):
        group=sorted((r for r in rows if r['policy']==policy),key=lambda r:r['window'])
        for axis,key,label in zip(axes.flat,('mean_batch','verified_output_tps','verify_wait_mean_ms','mean_context'),
                                  ('Actual D+V+P','Verified output tokens/s','Verify wait, mean ms','Context median, window mean')):
            axis.plot([r['window'] for r in group],[r[key] for r in group],marker='o',label=policy)
            axis.set_xlabel('Window (same run, not independent repeats)');axis.set_ylabel(label)
            axis.set_xticks([r['window'] for r in group])
            axis.grid(alpha=.25);axis.legend()
    fig.tight_layout();fig.savefig(root/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
