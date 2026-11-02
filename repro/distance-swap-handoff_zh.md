# 距下一次 Verify 的 KV 交换：新机器交接（2026-11-02）

分支：`research/sparsespec-reproduction`。本次交接包含实现、组件采集和离线分析；模型权重、环境缓存、原始 profile 不进入 Git。

## 当前实现

- 独立策略 `distance_swap`，仅支持 **TP1、eager、compact KV、CPU/传输异步**。
- 距离为 `0 if VERIFY else spec_stride - cur_spec_idx`。优先替换距下一次 Verify 更远的历史页，恢复从近到远；Draft 的 selected + recent/tail 热视图保留。
- 独立后台线程推进双向分块交换，每块默认 256 页。可认领未来释放页形成计划，GPU 目标页仍在真正空闲后分块分配；不等整个请求卸载完再恢复。
- **最新修改：移除了多请求的最坏情况聚合容量保护。** 不再预留“所有请求最终热预算＋一次最终完整 Verify”而把 B32 压成 27 个准入；按当前 prefill chunk、真实空闲页及 query/metadata/batch 限制准入。单请求可执行性检查仍保留；最终是否有 32 个 ready 请求需看实际记录。
- 原生 token sampling / rejection 未改。Verify 的完整视图保护延续到 CPU 结果消费和新 selection 发布。
- 历史 backing 按时间顺序，包含 selected 页的历史副本；GPU 另有 selected 热副本。因此字节量与旧 compact cold-only 表示不同，比较必须报告实际搬运量。

实现细节见 [实现说明](../research/request_distance_swap_implementation_2026-11-02_zh.md) 与 [设计稿](../research/request_distance_swap_design_2026-11-02_zh.md)。预约版另有可关闭的传输队列修复，见 [说明](../research/reservation_transfer_queue_2026-11-02_zh.md)。

## 拉取与环境

```bash
git fetch origin
git switch research/sparsespec-reproduction
git pull --ff-only origin research/sparsespec-reproduction
git submodule update --init --recursive 3rdparty/flashinfer 3rdparty/raft
# 首次安装才运行；已有对应环境可直接激活。
bash repro/bootstrap.sh
source repro/activate.sh
python repro/check_environment.py
```

环境固定版本、CUDA 12.8 工具链和 Qwen3-8B 权重准备见 [README_zh.md](README_zh.md)。本机还执行过 `python -m pip install --no-deps --no-build-isolation -e 3rdparty/flashinfer`，以建立 FlashInfer data 资源链接；若环境检查报告这些资源缺失，可在激活后执行同一命令。A100 可设置 `TORCH_CUDA_ARCH_LIST=8.0`；其他卡按实际架构设置。

准备固定 LCB128 输入，不需要旧服务器的 log：

```bash
python - <<'PY'
import json
from pathlib import Path
rows = [json.loads(s) for s in Path('experiments/acceptance/inputs/lcb128-seed330.jsonl').read_text().splitlines() if s.strip()]
assert len(rows) == 128
out = Path('.repro/profile-input.json')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({'requests': [{'sample_id': r['sample_id'], 'input_ids': r['input_ids']} for r in rows]}) + '\n')
PY
```

## 0.9 总预算口径

使用组件入口的 `--mem-fraction .9 --runtime-reserve-bytes N`，两种策略取同一个 N：

```text
KV 页数 = floor((CUDA 可用总显存 × 0.9 − 原生 FP16 模型预算 − N) / 每页字节数)
```

N 包括模型估算误差、运行张量、采样/元数据缓冲、allocator 与采集器开销，以及显式余量。它需要在新机器重新标定，不能只传 `.9`，也不能将 `serve/run.py` 原有“模型＋KV”公式当成这里的总预算。

本机 A100 40GB 的参考值：CUDA 总量 42,404,806,656 字节，原生模型预算 16,000,000,000 字节，每页 147,456 字节；长测标定 N=10,597,572,608 字节（含 1 GiB 余量），得到 78,442 页、10.772 GiB。**这只是本机起点，不是新机通用常量。**

标定方法：先用明确标为开发短测的较小固定池运行两种策略，覆盖 B32 prefill 和 CUDA/NVTX 采集，读取各自 `profile/b32/completion.rank0.json` 的 metrics 与 manifest：

```text
非 KV 峰值估计 = runtime_peak_reserved_bytes
              + non_torch_peak_observed_bytes − kv_pool_bytes_per_rank
N = 两种策略较大的非 KV 峰值估计 − model_budget_estimate_bytes + 余量
```

开发短测可在下面命令末尾追加 `--pool-pages 65536 --output-tokens 32 --warmup-seconds .5 --record-seconds 2 --window-gap-seconds 1`，并省略正式的 `--mem-fraction/--runtime-reserve-bytes`；65536 仅是本机 A100 的短测起点。短测仍可能漏掉长运行峰值，正式结束后须再次核对 `peak_reserved + observed_non_torch` 是否处于 0.9 预算内，超出则重标定。两者峰值未必同时出现，该和是保守估计。

预约版还应在新机短测的 P=0 连续批次记录中，用相邻 `cpu_time_ns` 间隔的中位数标定 `--reservation-iteration-ms`；这是近似逻辑时钟，不是 CUDA deadline 保证。本机参考值 39.211797 ms 也不应当作跨机器保证。

## 正式 profile 命令

安装 Nsight Systems CLI 并让 `nsys` 在 PATH 中。下面记录 CUDA/NVTX，不要求特权 SM/HBM 计数器；无权限时这些硬件指标保持缺失。

先设置本机参数，再运行。`profile_reserve_bytes` 和 `profile_iteration_ms` 的示例值来自上一节，迁移后按本机标定修改。

```bash
source repro/activate.sh
profile_model=/your/models/Qwen3-8B
profile_gpu=0
profile_reserve_bytes=10597572608
profile_iteration_ms=39.211797
profile_policy=distance_swap
profile_output=log/distance-swap-b32-run1
profile_nsys=$(command -v nsys)
test -n "$profile_nsys" || exit 1
mkdir -p "$profile_output"
profile_extra=()
if [[ $profile_policy == reservation ]]; then
  profile_extra=(--no-reservation-transfer-queue
    --reservation-iteration-ms "$profile_iteration_ms"
    --reservation-iteration-source "New-host adjacent P0 submission interval calibration")
fi
python experiments/diagnostics/pcie_isolated_exec.py \
  --gpu "$profile_gpu" --record "$profile_output/placement.json" -- \
  "$profile_nsys" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=repeat \
    --output="$profile_output/nsys" \
  "$VIRTUAL_ENV/bin/python" experiments/diagnostics/component_profile.py \
    --model "$profile_model" --input .repro/profile-input.json \
    --output "$profile_output/profile" --policy "$profile_policy" \
    --requests 128 --batches 32 --buffer-batch-capacity 128 --query-budget 2048 \
    --mem-fraction .9 --runtime-reserve-bytes "$profile_reserve_bytes" \
    --distance-swap-chunk-pages 256 --capture --windows-only \
    --warmup-seconds 120 --record-seconds 180 --record-windows 3 \
    --window-gap-seconds 180 --detail-every 8 --window-check-steps 1 \
    "${profile_extra[@]}" > "$profile_output/runtime.log" 2>&1
```

第二次运行改 `profile_policy=reservation`、`profile_output=log/reservation-b32-run1`，重新执行 `profile_extra` 到命令结束部分。**旧预约基线必须带 `--no-reservation-transfer-queue`**；该开关省略时测到的是修复后的预约执行器。

默认建议两策略在同一空闲卡顺序跑。需要并发时，先用 `nvidia-smi topo -m` 与 sysfs 核对独立 PCIe root；不能只看卡号不同。`pcie_isolated_exec.py` 会按 root 检查已有任务、加锁并绑定本地 CPU/NUMA 内存，但它只做启动时检查。原机额外监控过同 root 的外来任务，发现争用就终止自己的采集；新机也须在运行期间监测，受干扰窗口不能当作隔离对照。该包装器要求 Linux/sysfs、NVML、libnuma 和允许 `set_mempolicy` 的运行环境。

每次运行使用新的输出目录。固定条件是 Qwen3-8B FP16、原 thinking 模板、temperature=.6、seed330、k8、r=.05、最小热预算192、B32、query2048、metadata capacity128。正式运行不追加输出截断。`--windows-only` 在三窗完成后安全排空，未完成请求仍标为未完成；不能据此报告全批自然 EOS 吞吐。

本次组件采集入口是 `experiments/diagnostics/component_profile.py`。`experiments.acceptance.run` 校验旧 k63 serving 快照的 SHA，当前调度源码已改变，直接使用该接受长度实验入口会被其快照校验拒绝。

## 导出与分析

进程退出后每策略应生成三份报告。仍在使用上节对应的 `profile_output` 时运行：

```bash
for profile_index in 1 2 3; do
  "$profile_nsys" export --type=sqlite \
    --output="$profile_output/profile/nsys.$profile_index.sqlite" \
    "$profile_output/nsys.$profile_index.nsys-rep"
done
python experiments/diagnostics/analyze_component_profile.py \
  --root "$profile_output/profile"
```

主要看 `component-summary.json`、`sampling-windows.csv`、`component-latencies.csv`、`kernel-components-by-DVP.csv`、每窗 `verify-wait-summary.rank0.json` 和 PCIe 图。核对三窗完整、最终页数全回收、实际准入/实际 D+V+P、有效已验证输出 token/s、上下文和真实 D2H/H2D 字节；forward/Attention/MLP、CPU 调度/metadata/结果消费分别解释，不能将重叠区间相加。

`summarize_distance_profile.py` 是原机两任务控制器的配套汇总，要求 `plan.json`、`job.json` 等额外控制器元数据；仅运行上述迁移命令时直接使用各策略 `analyze_component_profile.py` 结果。

## 已完成与待做

移除聚合保护后，12 项调度回归通过，包括不因最终上下文拒绝当前可执行 prefill、真实页不足仍停止准入。推送前相关回归 88 项中 86 项通过、2 项 GPU 测试在 CPU 环境跳过，诊断脚本另有 14 项通过；两项 A100 GPU 存储数据测试此前已单独通过。可按以下命令复跑：

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest serve.tests.test_distance_swap.SchedulerTests -v
CUDA_VISIBLE_DEVICES=0 python -m unittest serve.tests.test_distance_swap.GPUStorageTests -v
```

此前带聚合保护的版本完成过一组 Qwen3-8B 三窗采集；原生 `.9` 公式曾预填充 OOM，短测缓冲预留也曾低估长测开销。修正预算后的另一轮因同 PCIe 域外来任务终止。**最新“无聚合保护”版本尚未完成正式性能采集**，不能将旧版三窗或受干扰窗口当作它的结果。下一步是在新机完成上述相同口径的两策略对照。
