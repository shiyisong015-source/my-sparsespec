"""Component accounting and Nsight hardware catalogs, preserving units."""
import argparse
from bisect import bisect_right
from collections import defaultdict
import csv
import json
from pathlib import Path
import re
import sqlite3
import statistics

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from verify_timeline import analyze_verify_timeline


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def merged(intervals):
    result=[]
    for start,end in sorted(intervals):
        if result and start<=result[-1][1]:
            result[-1][1]=max(end,result[-1][1])
        else:
            result.append([start,end])
    return result


class EnclosingRangeIndex:
    """Latest-start enclosing NVTX range in O(log N), including unmatched APIs."""
    def __init__(self, values):
        self.values = sorted(values, key=lambda row: row[0])
        self.starts = [row[0] for row in self.values]
        self.size = 1 << max(0, (len(values)-1).bit_length())
        self.ends = [-1] * (2*self.size)
        for index, (_, end, _) in enumerate(self.values):
            self.ends[self.size+index] = end
        for index in range(self.size-1, 0, -1):
            self.ends[index] = max(self.ends[2*index], self.ends[2*index+1])

    def enclosing(self, start, end):
        limit = bisect_right(self.starts, start)

        def search(node, lower, upper):
            if lower >= limit or self.ends[node] < end:
                return None
            if upper-lower == 1:
                return self.values[lower][2]
            middle = (lower+upper)//2
            right = search(2*node+1, middle, upper)
            return right if right is not None else search(2*node, lower, middle)

        return search(1, 0, self.size)


def latency_statistics(values):
    values=np.asarray(values,dtype=float)
    return dict(samples=len(values),measured_sum_ms=float(values.sum()),mean_ms=float(values.mean()),
                p50_ms=float(np.percentile(values,50)),p90_ms=float(np.percentile(values,90)),
                maximum_ms=float(values.max()))


def distance_transfer_rows(root, case, rank):
    match=re.search(r"_w(\d+)$",case.name)
    if not match:
        return []
    parent=case.name[:match.start()]
    path=root/parent/f"distance-swap.rank{rank}.jsonl"
    if not path.exists():
        return []
    window=int(match[1])
    with path.open() as stream:
        return [row for line in stream if (row:=json.loads(line)).get("window_id")==window]


def host_transfer_intervals(root, case, rank):
    """Host submission/completion intervals, including distance-swap chunks."""
    path=case/f"transfers.rank{rank}.jsonl"
    intervals=[]
    if path.exists():
        with path.open() as stream:
            intervals=[r for line in stream if (r:=json.loads(line))["event"]=="commit"]
    pending={}
    for row in distance_transfer_rows(root,case,rank):
        if row["event"] not in ("chunk_submit","chunk_complete"):
            continue
        key=tuple(row[name] for name in ("request_id","generation","direction","offset"))
        if row["event"]=="chunk_submit":
            pending[key]=row["cpu_ns"]
        elif key in pending:
            intervals.append(dict(direction="repack" if row["direction"]=="down" else "restore",
                submit_cpu_ns=pending.pop(key),commit_cpu_ns=row["cpu_ns"],scope="history_chunk"))
    return intervals


def component_latency_tables(root, hardware):
    """Parent intervals, exclusive kernel work and request waits stay separate.

    Fine GPU timings are measured kernel work over annotated forwards only.
    Their measured sum is never extrapolated to the entire capture window.
    """
    rows=[]
    labels={"batch_attention_core":"Attention core", "attention_selection_scores":"Attention scores",
            "mlp_input_gemm":"MLP input GEMM", "mlp_activation":"MLP activation",
            "mlp_output_gemm":"MLP output GEMM", "mlp_compute":"MLP compute total",
            "other_forward_kernels":"Other forward kernels",
            "tp_forward":"TP collectives", "forward":"Forward total",
            "ordinary_sampling":"Token sampling", "rejection_sampling":"Rejection sampling",
            "selection_update":"KV selection / index update", "metadata_prepare":"Metadata preparation",
            "cpu_admission":"CPU admission / scheduling", "cpu_result_processing":"CPU result processing",
            "cpu_result_readback_and_patch":"CPU result readback / patch",
            "cpu_rejection_readback_and_parse":"CPU rejection readback / parse",
            "cpu_wait_for_ready":"CPU wait for ready work", "cpu_tp_control":"CPU TP control",
            "observer_window_control":"Observer window control", "repack":"KV repack / offload",
            "restore":"KV restore", "history_d2h":"History chunk D2H",
            "history_h2d":"History chunk H2D", "hot_gather":"Selected hot-view gather"}
    def append(case,rank,domain,component,cohort,unit,values,**extra):
        if values:
            order,lane={
                "observer_window_control":(5,"CPU observer"),
                "cpu_admission":(10,"CPU current iteration"),
                "repack":(20,"KV async branch"),"restore":(20,"KV async branch"),
                "history_d2h":(20,"KV async branch"),"history_h2d":(20,"KV async branch"),
                "hot_gather":(20,"KV async branch"),
                "cpu_wait_for_ready":(25,"CPU empty-ready-queue path"),
                "metadata_prepare":(30,"CPU/control current iteration"),
                "forward":(40,"GPU current forward: parent"),
                "batch_attention_core":(41,"GPU forward: repeated per layer"),
                "attention_selection_scores":(42,"GPU forward: repeated per layer"),
                "mlp_compute":(43,"GPU forward: MLP parent"),
                "mlp_input_gemm":(44,"GPU forward: repeated per layer"),
                "mlp_activation":(45,"GPU forward: repeated per layer"),
                "mlp_output_gemm":(46,"GPU forward: repeated per layer"),
                "tp_forward":(48,"GPU forward: collectives at multiple sites"),
                "other_forward_kernels":(49,"GPU forward: multiple unseparated sites"),
                "ordinary_sampling":(50,"GPU current forward output"),
                "cpu_result_processing":(60,"CPU previous forward result: parent"),
                "cpu_result_readback_and_patch":(61,"CPU previous forward result"),
                "rejection_sampling":(62,"GPU/control previous Verify result"),
                "cpu_rejection_readback_and_parse":(63,"CPU previous Verify result"),
                "selection_update":(64,"GPU/control previous result"),
                "cpu_tp_control":(65,"CPU control at multiple sites"),
            }.get(component,(90,"Unclassified"))
            window=re.search(r"_w(\d+)$",case)
            rows.append(dict(case=case,window_id=int(window[1]) if window else 1,rank=rank,domain=domain,component=component,
                phase_order=order,lane=lane,
                label=labels.get(component,component),cohort=cohort,unit=unit,
                **latency_statistics(values),**extra))
    cells=defaultdict(lambda:defaultdict(float))
    for report in hardware:
        for cell in report.get("kernel_components",[]):
            key=(cell["case"],cell["pid"],cell["physical_gpu"],cell["producer_step"])
            cells[key][cell["component"]]+=cell["kernel_union_ms"]
    status={}
    for case in sorted(root.glob("b*")):
        if not case.is_dir():continue
        for file in sorted(case.glob("components.rank*.jsonl")):
            rank=int(file.stem.split("rank")[-1])
            observer_path=case/f"observer.rank{rank}.json"
            if not observer_path.exists():continue
            observer=json.loads(observer_path.read_text())
            boundary={r["step"] for r in observer["capture_boundaries"]}
            manifest=json.loads((root/f"manifest.rank{rank}.json").read_text())
            device_ids=[int(v) for v in manifest["gpu"].split(",")]
            device=device_ids[rank]
            state_path=case/f"batches.rank{rank}.jsonl"
            states={r["step"]:r for line in state_path.open() if (r:=json.loads(line))}
            groups=defaultdict(list)
            coarse_gpu={"forward","ordinary_sampling","rejection_sampling","selection_update","metadata_prepare"}
            cpu_names={"cpu_admission","metadata_prepare","cpu_result_processing","cpu_result_readback_and_patch",
                       "cpu_rejection_readback_and_parse","cpu_wait_for_ready","cpu_tp_control","observer_window_control"}
            for line in file.open():
                row=json.loads(line);name=row["component"];step=row["producer_step"]
                if step in boundary or row["executing_step"] in boundary:continue
                cohorts=["all_window"]
                if step in states:
                    s=states[step];cohorts.append(f"D={s['D']},V={s['V']},P={s['P']}")
                for cohort in cohorts:
                    if name in coarse_gpu and "gpu_ms" in row:
                        groups[("gpu_stream_envelope",name,cohort)].append(row["gpu_ms"])
                    if name in cpu_names:
                        groups[("cpu_wall",name,cohort)].append(row["cpu_wall_ms"])
            for (domain,name,cohort),values in groups.items():
                append(case.name,rank,domain,name,cohort,"per recorded call",values)
            transfer_path=case/f"transfers.rank{rank}.jsonl"
            transfers=[r for line in transfer_path.open() if (r:=json.loads(line))["event"]=="commit"] if transfer_path.exists() else []
            for direction in ("repack","restore"):
                append(case.name,rank,"gpu_transfer_stream_envelope",direction,"all_window","per KV transaction",
                       [r["gpu_stream_span_ms"] for r in transfers if r["direction"]==direction])
            swap=distance_transfer_rows(root,case,rank)
            for direction,name in (("down","history_d2h"),("up","history_h2d")):
                append(case.name,rank,"gpu_transfer_stream_envelope",name,"all_window","per history chunk",
                       [r["gpu_ms"] for r in swap if r["event"]=="chunk_complete"
                        and r.get("direction")==direction and "gpu_ms" in r])
            append(case.name,rank,"gpu_transfer_stream_envelope","hot_gather","all_window","per selected-view construction",
                   [r["gpu_ms"] for r in swap if r["event"]=="hot_complete"])
            fine=defaultdict(list)
            forward_names=["batch_attention_core","attention_selection_scores","mlp_input_gemm","mlp_activation",
                           "mlp_output_gemm","other_forward_kernels","tp_forward"]
            for (name,pid,gpu,step),components in cells.items():
                if name!=case.name or gpu!=device or step in boundary or step not in states:continue
                s=states[step]
                # A coarse forward NVTX range also exists on unsampled steps;
                # never mistake its unclassified work for an MLP measurement.
                required={"mlp_input_gemm","mlp_output_gemm","mlp_activation","batch_attention_core"}
                if not s.get("detail_sample",True) or not required.issubset(components):continue
                values={k:components.get(k,0.) for k in forward_names}
                values["other_forward_kernels"]=sum(components.get(k,0.) for k in ("forward","attention_block","mlp_block"))
                values["tp_forward"]=sum(v for k,v in components.items() if k.startswith("tp_"))
                for cohort in ("sampled_forwards",f"D={s['D']},V={s['V']},P={s['P']}"):
                    for key,value in values.items():fine[(key,cohort)].append(value)
            # Shares use exclusive forward kernel work only. Parent CUDA event
            # intervals, transfers and CPU/request waits are not denominators.
            denominators={cohort:sum(sum(v) for (name,c),v in fine.items() if c==cohort)
                          for _,cohort in fine}
            for (name,cohort),values in fine.items():
                denominator=denominators[cohort]
                append(case.name,rank,"gpu_kernel_work",name,cohort,"sum across layers per sampled forward",values,
                       forward_kernel_work_share_pct=100*sum(values)/denominator if denominator else None)
            for cohort in denominators:
                values=[sum(parts) for parts in zip(*(fine[(name,cohort)] for name in
                         ("mlp_input_gemm","mlp_activation","mlp_output_gemm")))]
                append(case.name,rank,"gpu_kernel_work_parent","mlp_compute",cohort,
                       "sum across layers per sampled forward",values)
            status[f"{case.name}/rank{rank}"]=("collected" if fine else "not_collected: annotated Nsight kernel data required")
    rows.sort(key=lambda r:(int(re.match(r"b(\d+)",r["case"])[1]),r["window_id"],r["rank"],r["phase_order"],r["domain"],r["component"],r["cohort"]))
    fields=["case","window_id","rank","phase_order","lane","domain","component","label","cohort","unit","samples","measured_sum_ms",
            "mean_ms","p50_ms","p90_ms","maximum_ms","forward_kernel_work_share_pct"]
    with (root/"component-latencies.csv").open("w") as out:
        writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    lines=["# 采样窗口组件延迟", "", "单位：ms。P50/P90是实测分位数；细GPU组件每个样本为该forward跨所有层的kernel工作时间。", "",
           "- `gpu_stream_envelope`：CUDA Event区间，包含提交空隙/依赖；Forward total是父区间。",
           "- `gpu_kernel_work`：Nsight关联的细组件，仅细采样forward，不外推全窗口；占比只针对同一组采样forward的kernel工作时间。",
           "- `gpu_kernel_work_parent`：子项合计供阅读，不能与子项重复相加。MLP合计为输入GEMM、激活及输出GEMM，不含TP通信。",
           "- CPU wall、传输stream跨度与请求等待分别解释，不能相加得到端到端时间；请求等待另见verify-wait-summary。",
           "- `measured_sum_ms`只累计实际样本；缺失Nsight细组件表示未采集，不表示0。", "",
           "## 执行顺序", "",
           "按CPU提交的逻辑流程分组：调度/提交KV事务 → metadata → 本轮forward → 本轮普通采样 → 上轮结果处理/拒绝采样 → 回退无效KV/更新选择索引。",
           "恢复必须在对应Verify读取KV前完成；打包卸载依赖此前已完成的rejection和选择。KV事务异步运行，可与其他请求的forward重叠。",
           "forward内部逐层重复：QKV/归一化/RoPE → 写入本轮KV → attention与所需score计算 → 输出投影 → MLP；最后生成logits。",
           "当前KV写入、回退以及概率scatter/gather未逐项独立计时；分别包含在forward或结果处理父区间中。其他forward kernel与TP控制分散在多个位置，不代表只在表中该位置执行。", "",
           "各D/V/P组合见CSV；各窗口独立汇总，不把相隔窗口间的运行时间计为采样时间。", "",
           "| Batch/窗口 | rank | 执行支线 | 类型 | 组件 | 次数 | 均值 | P50 | P90 | 实测累计 | kernel工作占比 |",
           "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["cohort"] not in ("all_window","sampled_forwards"):continue
        share=row.get("forward_kernel_work_share_pct")
        lines.append(f"| {row['case']} | {row['rank']} | {row['lane']} | {row['domain']} | {row['label']} | {row['samples']} | "
                     f"{row['mean_ms']:.3f} | {row['p50_ms']:.3f} | {row['p90_ms']:.3f} | {row['measured_sum_ms']:.3f} | "
                     +(f"{share:.1f}%" if share is not None else "—")+" |")
    lines.extend(["", "## 细GPU组件采集状态", ""]+[f"- {key}: {value}" for key,value in status.items()])
    (root/"component-latencies_zh.md").write_text("\n".join(lines)+"\n")
    return dict(csv="component-latencies.csv",markdown="component-latencies_zh.md",fine_gpu_status=status)


def attribute_kernels(db,tables,names):
    if not {"NVTX_EVENTS","CUPTI_ACTIVITY_KIND_KERNEL"}<=tables:
        return [],{}
    ranges=defaultdict(list)
    pattern=re.compile(r"^cp\.(\S+) case=(\S+) producer=(-?\d+) executing=(-?\d+)(?: D=(\d+) V=(\d+) P=(\d+) Q=(\d+))?")
    for r in db.execute("SELECT start,end,globalTid,text,textId FROM NVTX_EVENTS WHERE end IS NOT NULL"):
        match=pattern.match(r["text"] or names.get(r["textId"],""))
        if match:
            label,case,producer,executing,d,v,p,q=match.groups()
            ranges[r["globalTid"]].append((r["start"],r["end"],dict(
                component=label,case=case,producer_step=int(producer),executing_step=int(executing),
                D=int(d) if d else None,V=int(v) if v else None,P=int(p) if p else None,Q=int(q) if q else None)))
    for values in ranges.values(): values.sort(key=lambda r:r[0])
    indices={tid:EnclosingRangeIndex(values) for tid,values in ranges.items()}
    api={}
    pid_of=lambda value:(value>>24)&0xffffff
    for table in ["CUPTI_ACTIVITY_KIND_RUNTIME","CUPTI_ACTIVITY_KIND_DRIVER"]:
        if table not in tables: continue
        for r in db.execute(f"SELECT start,end,globalTid,correlationId FROM {table}"):
            tid=r["globalTid"]
            if tid not in ranges: continue
            label=indices[tid].enclosing(r["start"],r["end"])
            if label is not None:
                api[(pid_of(tid),r["correlationId"])]=label
    mapping={}
    if "TARGET_INFO_CUDA_DEVICE" in tables:
        mapping={(r["pid"],r["cudaId"]):r["gpuId"] for r in db.execute("SELECT * FROM TARGET_INFO_CUDA_DEVICE")}
    cells={}
    intervals=defaultdict(list)
    for r in db.execute("SELECT start,end,globalPid,deviceId,correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        pid=pid_of(r["globalPid"])
        label=api.get((pid,r["correlationId"]))
        if label is None: continue
        gpu=mapping.get((pid,r["deviceId"]),r["deviceId"])
        key=(label["case"],pid,gpu,label["producer_step"],label["component"])
        if key not in cells:
            cells[key]=dict(**label,pid=pid,physical_gpu=gpu,kernels=0,kernel_duration_sum_ms=0.,intervals=[])
        cell=cells[key]
        cell["kernels"]+=1
        cell["kernel_duration_sum_ms"]+=(r["end"]-r["start"])/1e6
        cell["intervals"].append([r["start"],r["end"]])
        intervals[(gpu,label["component"])].append([r["start"],r["end"]])
    output=[]
    for cell in cells.values():
        spans=merged(cell.pop("intervals"))
        cell["kernel_union_ms"]=sum(b-a for a,b in spans)/1e6
        output.append(cell)
    return output,{key:merged(values) for key,values in intervals.items()}


def summarize_hardware(database):
    db = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    report = dict(database=str(database), metrics=[], status="missing_hardware_samples")
    names = dict(db.execute("SELECT id,value FROM StringIds")) if "StringIds" in tables else {}
    cells,component_intervals=attribute_kernels(db,tables,names)
    report["kernel_components"]=cells
    report["kernel_attribution_scope"]="Kernel -> CUDA correlation ID -> innermost NVTX range on launching CPU thread. Exclusive component labels, with no CPU launch gaps; collective kernels may still wait for peers."
    cases = set()
    if "NVTX_EVENTS" in tables:
        for row in db.execute("SELECT text,textId FROM NVTX_EVENTS"):
            name = row["text"] or names.get(row["textId"], "")
            match = re.search(r"\bcase=(b\d+(?:_w\d+)?)\b", name)
            if match:
                cases.add(match[1])
    report["case_labels"] = sorted(cases)
    if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
        blocking=defaultdict(list)
        for row in db.execute("SELECT start,end,nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
            api=names.get(row["nameId"],"")
            if re.search(r"cuda(Device|Stream|Event)Synchronize|cudaMemcpy(?!Async)",api):
                blocking[api].append((row["end"]-row["start"])/1e6)
        report["cpu_cuda_blocking_api"]={name:dict(calls=len(values),total_ms=sum(values),
            median_ms=statistics.median(values)) for name,values in blocking.items()}
        report["blocking_api_scope"]="Runtime API wall duration includes runtime overhead; capture-boundary synchronizations are also present. Not pure Python CPU work."
    if not {"GPU_METRICS", "TARGET_INFO_GPU_METRICS"} <= tables:
        db.close()
        return report
    catalog = [dict(row) for row in db.execute("SELECT * FROM TARGET_INFO_GPU_METRICS")]
    save(database.with_suffix(".metrics-catalog.json"), catalog)
    wanted = {(r["typeId"],r["metricId"]):r["metricName"] for r in catalog
              if re.search(r"SM.*Active|SM.*Issue|Tensor.*Active|DRAM|PCIe|Compute Warps|Unallocated Warps",r["metricName"],re.I)}
    series = defaultdict(list)
    for row in db.execute("SELECT timestamp,typeId,metricId,value FROM GPU_METRICS"):
        key = row["typeId"],row["metricId"]
        if key in wanted and np.isfinite(row["value"]):
            series[key].append((row["timestamp"],row["value"]))
    if not series:
        return report
    report["status"] = "sampled_hardware_counters"
    report["hardware_by_component"]=[]
    for key, data in series.items():
        values = np.asarray([v for _,v in data])
        report["metrics"].append(dict(gpu=key[0]&0xff, name=wanted[key], samples=len(values),
            mean=float(values.mean()), median=float(np.median(values)),p90=float(np.percentile(values,90)),
            maximum=float(values.max())))
        times=np.asarray([t for t,_ in data],dtype=np.int64)
        for (gpu,component),intervals in component_intervals.items():
            if gpu!=(key[0]&0xff): continue
            spans=np.asarray(intervals,dtype=np.int64)
            index=np.searchsorted(spans[:,0],times,side="right")-1
            mask=(index>=0)&(times<spans[np.maximum(index,0),1])
            observed=values[mask]
            if len(observed):
                report["hardware_by_component"].append(dict(gpu=gpu,component=component,name=wanted[key],
                    samples=len(observed),mean=float(observed.mean()),median=float(np.median(observed)),
                    p90=float(np.percentile(observed,90))))
    selected = [(k,v) for k,v in series.items() if re.search(r"SM.*Active|SM.*Issue|Tensor.*Active|DRAM|PCIe",wanted[k],re.I)]
    gpus = sorted({key[0]&0xff for key,_ in selected})
    fig,axes = plt.subplots(3,len(gpus),figsize=(7*len(gpus),9),squeeze=False)
    lower = min(ts for values in series.values() for ts,_ in values)
    for key,data in selected:
        name,gpu=wanted[key],key[0]&0xff
        i=0 if re.search(r"SM.*Active|SM.*Issue|Tensor.*Active",name,re.I) else 1 if "DRAM" in name else 2
        times,values=np.array(data).T
        bins=((times-lower)//1_000_000).astype(int)
        unique=np.unique(bins)
        sums=np.bincount(bins,weights=values)
        counts=np.bincount(bins)
        axes[i,gpus.index(gpu)].plot(unique/1000,sums[unique]/counts[unique],label=name,lw=1)
    for i in range(3):
        for j,gpu in enumerate(gpus):
            axes[i,j].set_title(f"GPU {gpu}: "+["SM / Tensor","HBM interface","PCIe interface"][i])
            axes[i,j].set_xlabel("Seconds from capture origin")
            axes[i,j].set_ylabel("Native counter value (see legend units)")
            axes[i,j].legend(fontsize=7)
            axes[i,j].grid(alpha=.2)
    fig.suptitle(" / ".join(sorted(cases)) or database.stem)
    fig.tight_layout(rect=[0,0,1,.96])
    fig.savefig(database.with_suffix(".hardware.png"),dpi=160)
    plt.close(fig)
    report["notes"]=["Hardware values retain original metric names and units. Percentages are not GB/s.",
        "Charts average samples into 1ms bins; raw samples remain in SQLite.",
        "Window aggregates include compute, TP waits and CPU launch gaps; SM Active is not free-SM count.",
        "This summary is per capture window; a finite sampling interval cannot be assigned precisely to a shorter kernel."]
    report["notes"].append("hardware_by_component uses sample timestamps inside correlated kernel intervals; the hardware sampling window may mix adjacent kernels, and low sample counts are not reliable per-kernel estimates.")
    db.close()
    return report


def analyze_case(case):
    batchfiles=sorted(case.glob("batches.rank*.jsonl")) or sorted(case.glob("batches.rank*.json"))
    if not batchfiles:
        return None
    batches=[({row["step"]:row for line in p.open() if line.strip() and (row:=json.loads(line))}
              if p.suffix==".jsonl" else {int(k):v for k,v in json.loads(p.read_text()).items()})
             for p in batchfiles]
    signature=lambda rows:[(k,v["D"],v["V"],v["P"],v["query_tokens"]) for k,v in sorted(rows.items())]
    if any(signature(b)!=signature(batches[0]) for b in batches[1:]):
        raise RuntimeError(f"TP batch counts disagree: {case}")
    summaries=[]
    for rank,states in enumerate(batches):
        observer=json.loads((case/f"observer.rank{rank}.json").read_text())
        boundary={b["step"] for b in observer["capture_boundaries"]}
        metrics=observer["metrics"]
        windowed=observer.get("recording_scope")=="window"
        starts=[row for row in observer["capture_boundaries"] if row["start"]]
        ends=[row for row in observer["capture_boundaries"] if not row["start"]]
        window_delta={}
        if windowed and starts and ends:
            window_delta={key:ends[-1]["metrics"].get(key,0)-starts[0]["metrics"].get(key,0)
                for key in ["forwards","request_verify_rounds","migrations","restores","d2h_bytes","h2d_bytes",
                            "profile_verified_output_tokens", "swap_d2h_pages", "swap_h2d_pages"]}
        policy=observer.get("policy","resident")
        ordered_states=sorted(states.values(),key=lambda row:row["cpu_time_ns"])
        load={}
        for key in ("actual_batch","D","V","P","context_median","pending_requests","live_requests","query_tokens"):
            values=[r[key] for r in ordered_states if key in r]
            if values:
                load[key]=dict(mean=float(np.mean(values)),median=float(np.median(values)),
                               minimum=min(values),maximum=max(values),first=values[0],last=values[-1])
        if policy=="resident":
            assert not any(metrics[k] for k in ["migrations","restores","d2h_bytes","h2d_bytes","preemptions","replay_starts"])
        assert metrics["gpu_final_free_pages"]==metrics["gpu_capacity_pages"]
        per=defaultdict(lambda:defaultdict(float))
        cpu=defaultdict(lambda:defaultdict(float))
        forward_spans=[]
        all_cpu_wait=[]
        for line in (case/f"components.rank{rank}.jsonl").open():
            row=json.loads(line)
            if row["component"]=="cpu_wait_for_ready":
                all_cpu_wait.append(row["cpu_wall_ms"])
            if row["component"]=="forward" and "gpu_ms" in row:
                forward_spans.append([row["gpu_start_ms"],row["gpu_end_ms"]])
            step=row["producer_step"]
            if row["executing_step"] in boundary:
                continue
            name=row["component"]
            if "gpu_ms" in row:
                per[step][name]+=row["gpu_ms"]
            cpu[step][name]+=row["cpu_wall_ms"]
        transfer_file=case/f"transfers.rank{rank}.jsonl"
        transfers=[]
        if transfer_file.exists():
            transfers=[r for line in transfer_file.open() if (r:=json.loads(line))["event"]=="commit"]
        byte_counts=defaultdict(lambda:defaultdict(int))
        for row in transfers:
            byte_counts[row["submit_step"]][row["direction"]]+=row["bytes"]
        for direction,key in [("repack","d2h_bytes"),("restore","h2d_bytes")]:
            expected=metrics[key]
            if windowed:
                # Boundary stream fences complete GPU transfers before native
                # host commit polling. Remove carry-in and add carry-out bytes
                # rather than compare sampled transfers with whole-run totals.
                expected=window_delta.get(key,0)
                if starts and ends:
                    expected-=sum(row["bytes"] for row in starts[0]["pending_transfers"] if row["direction"]==direction)
                    expected+=sum(row["bytes"] for row in ends[-1]["pending_transfers"] if row["direction"]==direction)
            if sum(row["bytes"] for row in transfers if row["direction"]==direction)!=expected:
                raise RuntimeError(f"Observed/native transfer byte mismatch: {case} rank{rank} {direction}")
        steps=[]
        for step,state in sorted(states.items()):
            if step in boundary or "forward" not in per[step]:
                continue
            gpu=dict(per[step])
            detailed="mlp_block" in gpu
            if detailed:
                gpu["mlp_compute"]=sum(gpu.get(k,0) for k in ["mlp_input_gemm","mlp_activation","mlp_output_gemm"])
                leaves=["batch_attention_core","attention_selection_scores","mlp_compute",
                        "tp_all_reduce_attention","tp_all_reduce_mlp","tp_all_reduce_embedding","tp_vocab_all_gather"]
                gpu["forward_other"]=gpu["forward"]-sum(gpu.get(k,0) for k in leaves)
            steps.append(dict(**state,rank=rank,detailed=detailed,fine_nvtx="mlp_block" in cpu[step],
                              gpu_ms=gpu,cpu_wall_ms=dict(cpu[step]),
                              kv_d2h_bytes=byte_counts[step]["repack"],kv_h2d_bytes=byte_counts[step]["restore"]))
        with (case/f"steps.rank{rank}.jsonl").open("w") as out:
            for row in steps:
                out.write(json.dumps(row)+"\n")
        groups=defaultdict(list)
        for row in steps:
            groups[(row["D"],row["V"],row["P"])].append(row)
        cells=[]
        for (d,v,p),group in sorted(groups.items()):
            names=sorted({key for row in group for key in row["gpu_ms"]})
            timing={name:dict(samples=len(values),mean_ms=statistics.mean(values),median_ms=statistics.median(values))
                    for name in names if (values:=[row["gpu_ms"][name] for row in group if name in row["gpu_ms"]])}
            cells.append(dict(D=d,V=v,P=p,actual_batch=d+v+p,steps=len(group),gpu=timing))
        save(case/f"by-DVP.rank{rank}.json",cells)
        detailed=[s for s in steps if s["detailed"] and s["P"]==0]
        component_names=["forward","batch_attention_core","attention_selection_scores","mlp_compute",
                         "tp_all_reduce_attention","tp_all_reduce_mlp","tp_all_reduce_embedding","tp_vocab_all_gather",
                         "forward_other","ordinary_sampling","rejection_sampling","selection_update"]
        group_summary={key:dict(samples=len(values),mean_ms=statistics.mean(values),median_ms=statistics.median(values))
                       for key in component_names if (values:=[s["gpu_ms"][key] for s in detailed if key in s["gpu_ms"]])}
        forward_union=merged(forward_spans)
        transfer_summary={}
        for direction in ["repack","restore"]:
            group=[row for row in transfers if row["direction"]==direction]
            spans=merged([[row["gpu_start_ms"],row["gpu_end_ms"]] for row in group])
            # Both intervals share one CUDA-event origin, even across streams.
            overlap=0.;i=j=0
            while i<len(spans) and j<len(forward_union):
                a,b=spans[i];c,d=forward_union[j]
                overlap+=max(0.,min(b,d)-max(a,c))
                if b<=d:i+=1
                else:j+=1
            transfer_summary[direction]=dict(count=len(group),bytes=sum(row["bytes"] for row in group),
                gpu_stream_union_ms=sum(b-a for a,b in spans),overlap_with_forward_envelopes_ms=overlap,
                mean_gpu_stream_span_ms=statistics.mean(row["gpu_stream_span_ms"] for row in group) if group else None,
                transfers_spanning_multiple_forward_submissions=sum(row["forwards_while_inflight"]>=2 for row in group),
                mean_forwards_while_inflight=statistics.mean(row["forwards_while_inflight"] for row in group) if group else None)
        if policy=="distance_swap":
            swap=distance_transfer_rows(case.parent,case,rank)
            manifest=json.loads((case.parent/f"manifest.rank{rank}.json").read_text())
            page_bytes=manifest["kv_page_bytes_per_rank"]
            for direction,key in (("down","swap_d2h_pages"),("up","swap_h2d_pages")):
                group=[r for r in swap if r["event"]=="chunk_complete" and r.get("direction")==direction]
                pages=sum(r["pages"] for r in group)
                if pages!=window_delta.get(key,0):
                    raise RuntimeError(f"Distance-swap chunk/native byte mismatch: {case} {direction}")
                values=[r["gpu_ms"] for r in group if "gpu_ms" in r]
                transfer_summary[direction]=dict(count=len(group),bytes=pages*page_bytes,
                    measured_chunks=len(values),gpu_chunk_span_sum_ms=sum(values),
                    mean_gpu_chunk_span_ms=statistics.mean(values) if values else None,
                    scope="per chunk after producer dependencies; GPU span is not physical PCIe DMA time")
        completion_path=case/f"completion.rank{rank}.json"
        completion=json.loads(completion_path.read_text()) if completion_path.exists() else {}
        outputs=completion.get("outputs",[])
        output_tokens=sum(row["output_tokens"] for row in outputs)
        elapsed=metrics.get("elapsed_seconds",0)
        summaries.append(dict(rank=rank,policy=policy,metrics=metrics,steps=len(steps),decode_steps=sum(s["P"]==0 for s in steps),
             window_id=observer.get("window_id",1),parent_run=observer.get("parent_run",case.name),
             full_run_metrics_shared_across_windows=observer.get("full_run_metrics_shared_across_windows",False),
             window_load=load,
             sampling_scope="window" if windowed else "whole_run",
             window_status=observer.get("window_status","legacy"),
             window_elapsed_seconds=(ends[-1]["cpu_ns"]-starts[0]["cpu_ns"])/1e9 if starts and ends else 0.,
             window_native_counter_deltas=window_delta,
             completed_requests=len(outputs),output_tokens=output_tokens,
             end_to_end_output_tps=output_tokens/elapsed if elapsed and completion.get("complete") else None,
             window_verified_output_tokens=(window_delta["profile_verified_output_tokens"]
                 if starts and "profile_verified_output_tokens" in starts[0]["metrics"] else None),
             window_verified_output_tps=(window_delta["profile_verified_output_tokens"]/
                 ((ends[-1]["cpu_ns"]-starts[0]["cpu_ns"])/1e9)
                 if starts and ends and "profile_verified_output_tokens" in starts[0]["metrics"] else None),
             termination_counts={name:sum(row.get("stop_reason")==name for row in outputs)
                                 for name in ["eos","serving_context_boundary","development_output_cap"]},
             transfers=transfer_summary,
             cpu_wait_for_ready=dict(calls=len(all_cpu_wait),total_ms=sum(all_cpu_wait)),
             sampled_decode_steps=len(detailed),sampled_nvtx_decode_steps=sum(s["P"]==0 and s["fine_nvtx"] for s in steps),
             sampled_decode_components=group_summary,
             all_decode_forward_ms=[s["gpu_ms"]["forward"] for s in steps if s["P"]==0],
             all_decode_forward_cpu_ms=[s["cpu_wall_ms"]["forward"] for s in steps if s["P"]==0 and "forward" in s["cpu_wall_ms"]],
             cpu={key:dict(samples=len(values),mean_ms=statistics.mean(values),median_ms=statistics.median(values))
                  for key in ["cpu_result_readback_and_patch","cpu_rejection_readback_and_parse","cpu_tp_control",
                              "metadata_prepare","cpu_admission","cpu_result_processing"]
                  if (values:=[s["cpu_wall_ms"][key] for s in steps if s["P"]==0 and key in s["cpu_wall_ms"]])}))
    return dict(case=case.name,ranks=summaries)


def pcie_case_views(root,samples):
    """Align interface telemetry with host submission/transfer state.

    NVML polls are not instantaneous byte counters. Phase labels describe
    in-flight host transactions; they do not isolate KV bytes from TP traffic.
    """
    if not samples:
        return []
    gpus=sorted({row["gpu"] for row in samples})
    manifest_path=root/"manifest.rank0.json"
    visible=(json.loads(manifest_path.read_text()).get("gpu","") if manifest_path.exists() else "")
    rank_of={int(value):rank for rank,value in enumerate(visible.split(",")) if value.strip()}
    telemetry={gpu:sorted((row for row in samples if row["gpu"]==gpu),key=lambda r:r["cpu_time_ns"])
               for gpu in gpus}
    reports=[]
    for case in sorted(root.glob("b*")):
        state_path=case/"batches.rank0.jsonl"
        legacy=case/"batches.rank0.json"
        if state_path.exists():
            states=[json.loads(line) for line in state_path.open() if line.strip()]
        elif legacy.exists():
            states=list(json.loads(legacy.read_text()).values())
        else:
            continue
        if not states:
            continue
        states.sort(key=lambda r:r["cpu_time_ns"])
        lower=states[0]["cpu_time_ns"]
        upper=states[-1]["cpu_time_ns"]+100_000_000
        for rank in rank_of.values():
            for row in host_transfer_intervals(root,case,rank):
                upper=max(upper,row["commit_cpu_ns"]+100_000_000)
        fig,axes=plt.subplots(len(gpus)+2,1,figsize=(13,3*len(gpus)+4),sharex=True)
        report=dict(case=case.name,gpus=[])
        rank_zero=[]
        for ax,gpu in zip(axes,gpus):
            rows=[r for r in telemetry[gpu] if lower<=r["cpu_time_ns"]<=upper]
            transfers=[]
            rank=rank_of.get(gpu)
            if rank is not None:
                transfers=host_transfer_intervals(root,case,rank)
            if rank==0:
                rank_zero=transfers
            times=np.asarray([r["cpu_time_ns"] for r in rows],dtype=np.int64)
            phase=np.full(len(rows),"no_kv_inflight",dtype=object)
            active={}
            for direction in ("repack","restore"):
                spans=merged([(r["submit_cpu_ns"],r["commit_cpu_ns"]) for r in transfers if r["direction"]==direction])
                valid=np.zeros(len(times),dtype=bool)
                if spans:
                    starts=np.asarray([s for s,e in spans],dtype=np.int64)
                    ends=np.asarray([e for s,e in spans],dtype=np.int64)
                    index=np.searchsorted(starts,times,side="right")-1
                    valid=(index>=0)&(times<ends[np.maximum(index,0)])
                active[direction]=valid
                phase[valid]=direction+"_inflight"
            phase[active["repack"] & active["restore"]]="both_directions_inflight"
            stats={}
            for key,label in [("rx_kib_per_s","RX"),("tx_kib_per_s","TX")]:
                values=np.asarray([r[key]*1024/1e9 for r in rows])
                plot_t=(times-lower)/1e9
                gaps=np.flatnonzero(np.diff(plot_t)>2.)+1
                ax.plot(np.insert(plot_t,gaps,np.nan),np.insert(values,gaps,np.nan),label=label,lw=.9)
                stats[label]={}
                for name in ["all","no_kv_inflight","repack_inflight","restore_inflight","both_directions_inflight"]:
                    chosen=values if name=="all" else values[phase==name]
                    stats[label][name]=dict(samples=len(chosen),
                        mean_gbps=float(chosen.mean()) if len(chosen) else None,
                        p90_gbps=float(np.percentile(chosen,90)) if len(chosen) else None,
                        maximum_observed_gbps=float(chosen.max()) if len(chosen) else None)
            ax.set_ylabel(f"GPU {gpu}\nPCIe GB/s");ax.grid(alpha=.2);ax.legend(loc="upper right")
            intervals=np.diff(times)/1e6
            report["gpus"].append(dict(gpu=gpu,rank=rank,interface=stats,
                polling_interval_ms=dict(samples=len(intervals),
                    median=float(np.median(intervals)) if len(intervals) else None,
                    p90=float(np.percentile(intervals,90)) if len(intervals) else None,
                    maximum=float(intervals.max()) if len(intervals) else None)))
        t=[(r["cpu_time_ns"]-lower)/1e9 for r in states]
        for name in ["D","V","P"]:
            axes[-2].step(t,[r[name] for r in states],where="post",label=name,lw=.9)
        axes[-2].set_ylabel("Submitted requests");axes[-2].legend(loc="upper right");axes[-2].grid(alpha=.2)
        for y,direction,color in [(0,"repack","tab:orange"),(1,"restore","tab:blue")]:
            spans=[((s-lower)/1e9,(e-s)/1e9) for s,e in merged([
                (r["submit_cpu_ns"],r["commit_cpu_ns"]) for r in rank_zero if r["direction"]==direction])]
            axes[-1].broken_barh(spans,(y-.3,.6),facecolors=color)
        axes[-1].set_yticks([0,1],["Repack / D2H","Restore / H2D"])
        axes[-1].set_ylabel("KV in flight\n(rank 0 host state)")
        axes[-1].set_xlabel("Seconds from first batch metadata submission")
        axes[-1].grid(alpha=.2)
        fig.suptitle(f"{case.name}: PCIe interface bandwidth, submitted D/V/P and KV in flight")
        fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(case/"pcie-bandwidth.png",dpi=160);plt.close(fig)
        report["scope"]="Decimal GB/s from NVML interface samples; includes KV, TP and metadata. In-flight phases use per-rank host submit/commit times (distance-swap: per chunk), not exact GPU execution intervals. Both directions may overlap. Sparse/boundary samples cannot establish a physical bandwidth ceiling or isolate KV-only traffic."
        save(case/"pcie-bandwidth.json",report)
        reports.append(report)
    return reports


def main(root):
    cases=[r for case in sorted(root.glob("b*")) if case.is_dir() and (r:=analyze_case(case))]
    verify_timelines={f"{path.parent.name}/{path.stem}":analyze_verify_timeline(
        path.parent,int(path.stem.split("rank")[-1]))
        for path in sorted(root.glob("b*/verify-timeline.rank*.jsonl"))}
    hardware=[summarize_hardware(p) for p in sorted(root.glob("nsys*.sqlite"))]
    latency_tables=component_latency_tables(root,hardware)
    correlated=defaultdict(lambda:defaultdict(float))
    descriptions={}
    for capture in hardware:
        for cell in capture.get("kernel_components",[]):
            key=(cell["case"],cell["physical_gpu"],cell["producer_step"])
            correlated[key][cell["component"]]+=cell["kernel_union_ms"]
            if cell["D"] is not None:
                descriptions[key]=(cell["D"],cell["V"],cell["P"])
    kernel_groups=defaultdict(list)
    for key,components in correlated.items():
        if "mlp_input_gemm" not in components or key not in descriptions:
            continue  # Unannotated/partial forwards are not fine component samples.
        d,v,p=descriptions[key]
        kernel_groups[(key[0],key[1],d,v,p)].append(dict(
            mlp_compute_ms=sum(components[k] for k in ["mlp_input_gemm","mlp_activation","mlp_output_gemm"]),
            attention_core_ms=components["batch_attention_core"],
            attention_scores_ms=components["attention_selection_scores"],
            tp_communication_and_wait_ms=sum(value for name,value in components.items()
                                             if name.startswith("tp_")),
            ordinary_sampling_ms=components["ordinary_sampling"],
            rejection_sampling_ms=components["rejection_sampling"],
            forward_unclassified_and_projections_ms=components["forward"]+components["attention_block"]))
    kernel_cells=[]
    for (case,gpu,d,v,p),samples in sorted(kernel_groups.items()):
        kernel_cells.append(dict(case=case,gpu=gpu,D=d,V=v,P=p,samples=len(samples),
            **{name:statistics.mean(row[name] for row in samples) for name in samples[0]}))
    if kernel_cells:
        with (root/"kernel-components-by-DVP.csv").open("w") as out:
            writer=csv.DictWriter(out,fieldnames=list(kernel_cells[0]))
            writer.writeheader();writer.writerows(kernel_cells)
    coverage={str(path.parent.name):json.loads(path.read_text()) for path in sorted(root.glob("b*/sampling-coverage.rank0.json"))}
    load_rows=[]
    for case in cases:
        for rank in case['ranks']:
            load=rank['window_load']
            row=dict(case=case['case'],parent_run=rank['parent_run'],window_id=rank['window_id'],rank=rank['rank'],
                     status=rank['window_status'],duration_seconds=rank['window_elapsed_seconds'])
            for name,stats in load.items():
                row.update({f'{name}_{field}':value for field,value in stats.items()})
            load_rows.append(row)
    if load_rows:
        with (root/'sampling-windows.csv').open('w') as out:
            fields=sorted({key for row in load_rows for key in row})
            writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader();writer.writerows(load_rows)
    report=dict(cases=cases,sampling_coverage=coverage,component_latency_tables=latency_tables,verify_timelines=verify_timelines,hardware=hardware,hardware_status="present" if any(
        h["status"]=="sampled_hardware_counters" for h in hardware) else "not_collected",
        kernel_components_by_DVP=kernel_cells,
        notes=["forward contains attention/MLP/TP: do not add parent and child intervals.",
               "GPU events include submission gaps and dependencies. CPU wall includes Python/launch/wait; it is not pure synchronization.",
               "CPU result readback and rejection readback are recorded separately, attributed to the producing forward.",
               "CPU result processing overlaps current forward; parent wall intervals cannot be summed to form elapsed time.",
               "Fine components are sampled; per-cell sample counts are retained. Prefill and capture-boundary steps are excluded from decode summaries."])
    report["notes"].append("Default fine GPU events are disabled: use hardware[].kernel_components for actual fine kernel execution times; components/steps GPU fields retain only coarse event intervals.")
    report["notes"].append("kernel_components_by_DVP averages correlated kernel work per annotated forward. A control result outside a capture may be missing: a zero rejection field in this table is not evidence that no rejection ran; use the full per-step coarse events for rejection timing.")
    report["notes"].append("KV transfer event spans include stream dependencies, GPU gather/scatter and offload/restore, not pure PCIe DMA. Their overlap with forward event envelopes is pipeline overlap, not a measurement of useful simultaneous SM execution. CPU wait-for-ready may also include retirement or calendar work; CUDA/NVTX traces establish the narrower cause.")
    report["notes"].append("For windowed runs, steps/components/transfers/PCIe describe only the recording window. Native metrics, completion counts and end_to_end_output_tps cover the full run. Transfer bytes are reconciled against boundary counter deltas with carry-in/out corrections, not against full-run traffic. Missing/short windows are explicitly labelled.")
    report["notes"].append("Multiple windows from the same parent_run share the same full-run totals; never add those duplicated totals or treat windows as independent workload runs. Each bN_wXX has independent events, clock origin and boundary accounting.")
    pcie=root/"pcie-nvml.jsonl"
    if pcie.exists():
        samples=[json.loads(line) for line in pcie.read_text().splitlines()]
        if samples:
            fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True)
            origin=min(r["cpu_time_ns"] for r in samples)
            for gpu in sorted({r["gpu"] for r in samples}):
                group=[r for r in samples if r["gpu"]==gpu]
                t=np.asarray([(r["cpu_time_ns"]-origin)/1e9 for r in group])
                gaps=np.flatnonzero(np.diff(t)>2.)+1
                for ax,key in zip(axes,["rx_kib_per_s","tx_kib_per_s"]):
                    values=np.asarray([r[key]*1024/1e9 for r in group])
                    ax.plot(np.insert(t,gaps,np.nan),np.insert(values,gaps,np.nan),label=f"GPU {gpu}")
                    ax.set_ylabel(key.split("_")[0].upper()+" GB/s")
                    ax.grid(alpha=.2); ax.legend()
            axes[-1].set_xlabel("Seconds from NVML sampler start")
            fig.suptitle("PCIe total interface traffic (KV + TP + metadata; inspect policy and transfer logs)")
            fig.tight_layout()
            fig.savefig(root/"pcie-nvml.png",dpi=160)
            plt.close(fig)
            report["pcie_by_case"]=pcie_case_views(root,samples)
    save(root/"component-summary.json",report)
    print(json.dumps(dict(cases=len(cases),hardware_status=report["hardware_status"],
                          hardware_reports=len(hardware)),indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    main(parser.parse_args().root)
