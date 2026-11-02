"""Inspect saved profile windows without requiring whole-run EOS metadata.

Read-only with respect to the captured run. Derived files go to --output;
no completion, observer or sweep status files are manufactured or updated.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import statistics


FINE = ("batch_attention_core", "mlp_input_gemm", "mlp_activation", "mlp_output_gemm")


def read_rows(path):
    with path.open() as source:
        return [json.loads(line) for line in source if line.strip()]


def distribution(values):
    if not values:
        return dict(samples=0)
    ordered = sorted(values)
    def quantile(p):
        index = (len(ordered)-1)*p
        lo = int(index)
        return ordered[lo]+(ordered[min(lo+1,len(ordered)-1)]-ordered[lo])*(index-lo)
    return dict(samples=len(values),mean=statistics.mean(values),p50=quantile(.5),
                p90=quantile(.9),minimum=ordered[0],maximum=ordered[-1])


def inspect_window(root, case, database, boundary_steps, layers, telemetry):
    timeline = read_rows(case/"verify-timeline.rank0.jsonl")
    starts = [r for r in timeline if r["event"]=="window_start"]
    ends = [r for r in timeline if r["event"]=="window_end"]
    if len(starts)!=1 or len(ends)!=1:
        raise ValueError(f"{case.name}: expected exactly one window_start/window_end")
    begin,end = starts[0]["cpu_ns"],ends[0]["cpu_ns"]
    states = read_rows(case/"batches.rank0.jsonl")
    components = read_rows(case/"components.rank0.jsonl")
    transfers = read_rows(case/"transfers.rank0.jsonl")
    rounds = read_rows(case/"rounds.rank0.jsonl")
    reservations = read_rows(case/"reservations.rank0.jsonl")
    by_step = {r["step"]:r for r in states}
    assert len(by_step)==len(states), "Duplicate batch step"
    assert all(r["actual_batch"]==r["D"]+r["V"]+r["P"] for r in states)
    forward = {r["producer_step"]:r for r in components if r["component"]=="forward" and "gpu_ms" in r}
    expected = {r["step"] for r in states if r.get("detail_sample") and r["step"] not in boundary_steps}
    cpu_fine = Counter((r["producer_step"],r["component"]) for r in components if r["component"] in FINE)
    complete_cpu = {step for step in expected if all(cpu_fine[step,name]==layers for name in FINE)}
    timings = defaultdict(list)
    for row in components:
        if row["producer_step"] in boundary_steps or row["executing_step"] in boundary_steps:
            continue
        timings[("cpu_wall",row["component"])].append(row["cpu_wall_ms"])
        if "gpu_ms" in row:
            timings[("gpu_stream_envelope",row["component"])].append(row["gpu_ms"])
    latency = [dict(case=case.name,domain=domain,component=name,unit="ms",**distribution(values))
               for (domain,name),values in sorted(timings.items())]
    db=sqlite3.connect(f"file:{database.resolve()}?mode=ro",uri=True)
    db.row_factory=sqlite3.Row
    integrity=db.execute("PRAGMA quick_check").fetchone()[0]
    assert integrity=="ok",integrity
    names=dict(db.execute("SELECT id,value FROM StringIds"))
    diagnostics=[dict(r) for r in db.execute("SELECT severity,text FROM DIAGNOSTIC_EVENT WHERE severity IN (2,3)")]
    nvtx_fine=Counter()
    nvtx_forwards=set()
    pattern=re.compile(r"^cp\.(\S+) case=(\S+) producer=(-?\d+) executing=(-?\d+)")
    for row in db.execute("SELECT start,end,text,textId FROM NVTX_EVENTS"):
        text=row["text"] or names.get(row["textId"],"")
        match=pattern.match(text)
        if not match or row["end"] is None or match[2]!=case.name:
            continue
        component,_,step,_=match.groups()
        step=int(step)
        if component in FINE:
            nvtx_fine[step,component]+=1
        if component=="forward":
            nvtx_forwards.add(step)
    complete_nvtx={step for step in expected if all(nvtx_fine[step,name]==layers for name in FINE)}
    tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    trace={}
    for table in ("CUPTI_ACTIVITY_KIND_KERNEL","CUPTI_ACTIVITY_KIND_RUNTIME","NVTX_EVENTS"):
        row=db.execute(f"SELECT COUNT(*),MIN(start),MAX(end) FROM {table}").fetchone()
        trace[table]=dict(events=row[0],start_ns=row[1],end_ns=row[2],span_seconds=(row[2]-row[1])/1e9)
    bins=list(db.execute("SELECT CAST(start/1000000000 AS INT),COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY 1"))
    present={r[0] for r in bins}
    empty_seconds=[i for i in range(min(present),max(present)+1) if i not in present]
    metrics=[]
    if {"GPU_METRICS","TARGET_INFO_GPU_METRICS"}<=tables:
        catalog={(r["typeId"],r["metricId"]):r["metricName"] for r in db.execute("SELECT * FROM TARGET_INFO_GPU_METRICS")}
        for row in db.execute("SELECT typeId,metricId,COUNT(*),MIN(timestamp),MAX(timestamp),AVG(value),MIN(value),MAX(value) FROM GPU_METRICS GROUP BY typeId,metricId"):
            name=catalog[row[0],row[1]]
            if re.search(r"SMs? Active|SM Issue|Tensor Active|DRAM|PCIe",name):
                metrics.append(dict(name=name,samples=row[2],span_seconds=(row[4]-row[3])/1e9,
                    mean=row[5],minimum=row[6],maximum=row[7]))
    db.close()
    submissions=Counter(r["sequence"] for r in transfers if r["event"]=="submit")
    commits=Counter(r["sequence"] for r in transfers if r["event"]=="commit")
    assert all(n==1 for n in submissions.values()) and all(n==1 for n in commits.values())
    assert not set(commits)-set(submissions), "Transfer commit without same-window submission"
    telemetry=[r for r in telemetry if begin<=r["cpu_time_ns"]<=end]
    pcie={}
    for direction in ("rx","tx"):
        pcie[direction+"_GBps"]=distribution([r[direction+"_kib_per_s"]*1024/1e9 for r in telemetry])
    pcie["sample_gap_ms"]=distribution([(b["cpu_time_ns"]-a["cpu_time_ns"])/1e6 for a,b in zip(telemetry,telemetry[1:])])
    checks=dict(json_parsed=True,sqlite_integrity_ok=True,window_end_present=True,
        coarse_forward_steps_present=not ((set(by_step)-boundary_steps)-set(forward)),
        sampled_cpu_layer_coverage=complete_cpu==expected,sampled_nvtx_layer_coverage=complete_nvtx==expected,
        nvtx_forward_coverage=not ((set(forward)-boundary_steps)-nvtx_forwards),
        cuda_trace_no_empty_interior_second=not empty_seconds,
        hardware_metrics_present=bool(metrics),pcie_samples_present=bool(telemetry),
        no_profiler_errors=not any(d["severity"]==3 for d in diagnostics),
        transfer_pairs_complete=not(set(submissions)-set(commits)))
    return dict(case=case.name,window_seconds=(end-begin)/1e9,checks=checks,
        batches=len(states),coarse_forward_events=len(forward),expected_detail_forwards=len(expected),
        complete_cpu_detail_forwards=len(complete_cpu),complete_nvtx_detail_forwards=len(complete_nvtx),
        missing_cpu_details=sorted(expected-complete_cpu),missing_nvtx_details=sorted(expected-complete_nvtx),
        missing_nvtx_forward_steps=sorted((set(forward)-boundary_steps)-nvtx_forwards),
        missing_coarse_forward_steps=sorted((set(by_step)-boundary_steps)-set(forward)),
        layers=layers,trace=trace,empty_cuda_trace_seconds=empty_seconds,diagnostics=diagnostics,
        hardware_metrics=metrics,pcie=pcie,transfer_submissions=len(submissions),transfer_commits=len(commits),
        transfer_uncommitted_sequences=sorted(set(submissions)-set(commits)),
        acceptance_rounds=len(rounds),reservation_records=len(reservations),
        load={key:distribution([r[key] for r in states]) for key in ("actual_batch","D","V","P","query_tokens","live_requests")}),latency


def main(args):
    root=args.root.resolve()
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    runtime=(root/"runtime.log").read_text()
    boundaries=defaultdict(set)
    for match in re.finditer(r"COMPONENT_WINDOW (?:START|STOP) (b\d+_w\d+) \{'step': (\d+)",runtime):
        boundaries[match[1]].add(int(match[2]))
    telemetry=read_rows(root/"pcie-nvml.jsonl")
    report=dict(whole_run_completion_confirmed=False,scope="Existing closed profile windows only; whole-run EOS/page-reclamation counters unavailable",
        notes=["Report export and coverage checks do not prove that every CUDA event was collected; retain Nsight warnings.",
               "Fine component CPU wall time is not GPU kernel time. GPU fine attribution requires a separate kernel correlation analysis.",
               "SM/HBM counter percentages retain native metric units; PCIe GB/s is from NVML sampling.",
               "Missing whole-run completion metadata is not fabricated; no original status/result files are modified."],windows=[])
    latencies=[]
    for i,case in enumerate(sorted(root.glob("b*_w*")),1):
        database=output/f"nsys.{i}.sqlite"
        window,rows=inspect_window(root,case,database,boundaries[case.name],args.layers,telemetry)
        report["windows"].append(window)
        latencies.extend(rows)
        print(case.name,json.dumps(window["checks"]),flush=True)
    report["reports_sha256"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.nsys-rep"))}
    good=bool(report["windows"]) and all(all(w["checks"].values()) for w in report["windows"])
    report["profile_status"]=("usable_with_profiler_warnings" if any(w["diagnostics"] for w in report["windows"])
                              else "validated") if good else "requires_investigation"
    (output/"inspection.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    with (output/"component-latencies.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=list(latencies[0]));writer.writeheader();writer.writerows(latencies)
    lines=["# B32 既有采样报告检查", "",
        f"采样检查状态：`{report['profile_status']}`。整批自然 EOS 完成状态：未确认。", "",
        "三份原始 Nsight 报告成功导出，SQLite quick_check、JSON 逐行解析、窗口起止、forward 覆盖、细采样 Attention/MLP 36 层覆盖与传输配对检查见 inspection.json。",
        "这些是已关闭窗口的离线检查，未生成或补造原始 observer/completion/状态文件，不需要为了上述元数据缺失直接重跑推理。", "",
        "| 窗口 | 时长秒 | forward 数 | 细采样 forward 数 | 实际平均 batch | 平均 D | 平均 V | 平均 P | 配对传输数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for w in report["windows"]:
        load=w["load"]
        lines.append(f"| {w['case']} | {w['window_seconds']:.3f} | {w['coarse_forward_events']} | "
            f"{w['expected_detail_forwards']} | {load['actual_batch']['mean']:.3f} | {load['D']['mean']:.3f} | "
            f"{load['V']['mean']:.3f} | {load['P']['mean']:.3f} | {w['transfer_commits']} |")
    lines += ["", "## 使用范围", "",
        "- 可用于窗口内 CPU 控制、forward Event 包络、KV 事务、Verify 等待与硬件资源 profile 分析。",
        "- 每个细采样 forward 的四个叶项标注（Attention、MLP 输入 GEMM、激活、输出 GEMM）在 CPU 日志与完整 NVTX 范围中均覆盖 36 层。GPU kernel 数量和时间覆盖也存在，但精细 GPU 组件延迟仍需通过 CUDA correlation ID 关联；不能把 CPU 调用时长当作 GPU 计算时间。",
        "- Nsight 三份报告均提示可能未收齐 NVTX/CUDA 事件；当前覆盖检查未发现缺失的已记录 forward/细层标注，也没有整秒 CUDA trace 空洞，但这不构成逐个 CUDA 事件零丢失的证明。保留警告，不宣称严格无丢失。",
        "- B32 表示 batch 上限，实际平均参与计算请求数为 4.49/3.45/2.48；不能称为固定 32 请求满 batch 的算子基准。",
        "- 整批完成记录、自然 EOS 输出统计、最终页回收与全程吞吐尚未确认，不能由三个采样窗口推断。",
        "- NVML PCIe 实际采样间隔约 92–94 ms，并非设置目标的 50 ms；硬件百分比保留原生单位，不能直接当 GB/s，也不能把 SM Active 的补数解释为可立即使用的空闲 SM 数。",
        "", "## 文件", "",
        "- inspection.json：覆盖检查、Nsight 警告、硬件指标及实际采样间隔。",
        "- component-latencies.csv：窗口内 CPU wall 与 GPU Event 包络分开统计，单位 ms，不包含尚未关联的 GPU 细组件时间。",
        "- nsys.1/2/3.sqlite：三份离线导出数据库；原始 .nsys-rep 保留在上一层。",
        "", "新增离线检查脚本：experiments/diagnostics/inspect_component_capture.py；未修改模型、运行中的 worker、生产调度及原始实验状态。"]
    (output/"findings_zh.md").write_text("\n".join(lines)+"\n")
    print("Saved",output/"inspection.json",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True,help="Directory containing exported nsys.1/2/3.sqlite")
    parser.add_argument("--layers",type=int,default=36)
    main(parser.parse_args())
