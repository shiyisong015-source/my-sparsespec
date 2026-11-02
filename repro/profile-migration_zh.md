# 2026-11-02 当前研究代码迁移

远程仓库：`https://github.com/shiyisong015-source/my-sparsespec`，分支 `research/sparsespec-reproduction`。

本快照包含当前 compact KV、原生池内冷热布局、异步 CPU/传输、预约表与释放依赖、恢复 kernel 并发程序数限制、接受长度诊断、组件采样与离线检查。旧实验结果不因代码更新而自动成为当前快照的性能结论。

最新快照已包含预约传输队列修复、距下一次 Verify 的分块交换，以及取消最坏情况聚合准入预留。运行当前版本和 0.9 总预算 profile 请读[最新交接说明](distance-swap-handoff_zh.md)。下文保留较早预约版的固定池流程；原因分析见[原交接文档](../research/batch_readiness_handoff_2026-11-02_zh.md)，不代表最新版本尚未实施。

## 获取代码及环境

新目录：

```bash
git clone --branch research/sparsespec-reproduction \
  https://github.com/shiyisong015-source/my-sparsespec.git
cd my-sparsespec
git submodule update --init --recursive 3rdparty/flashinfer 3rdparty/raft
bash repro/bootstrap.sh
source repro/activate.sh
python repro/check_environment.py
```

已有干净克隆：

```bash
git fetch origin
git switch research/sparsespec-reproduction
git pull --ff-only origin research/sparsespec-reproduction
git submodule update --init --recursive 3rdparty/flashinfer 3rdparty/raft
```

这里 `origin` 指新机器克隆后的个人仓库；参考机将个人仓库命名为 `personal`，其 `origin` 是上游官方仓库。环境与原生库安装要求见 README_zh.md；环境不通过复制旧服务器 venv 迁移。Python 3.12、PyTorch 2.8.0/cu128、vLLM 0.11.0 wheel、固定 FlashInfer/RAFT 子模块与原生依赖归档均有对应版本或哈希记录。

模型权重需在新服务器准备；用 `--model` 指定位置。完整 log、Nsight 二进制与原始数据库不存入 Git。本次附带的 LCB128 固定输入已在 Git 中，因此下面的组件 profile 不依赖旧服务器的 log 数据。

## 准备相同 LCB128 输入

激活环境后，在仓库根目录运行；仅生成输入文件，不启动 GPU：

```bash
python - <<'PY'
import json
from pathlib import Path
source = Path('experiments/acceptance/inputs/lcb128-seed330.jsonl')
rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
assert len(rows) == 128
requests = [dict(sample_id=r['sample_id'], input_ids=r['input_ids']) for r in rows]
output = Path('.repro/profile-input.json')
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(dict(requests=requests)) + '\n')
print(output.resolve())
PY
```

这些 input IDs 与参考机使用的 128 条 LCB prompt 相同；输入文件序列化与其他 payload 字段不同，因此文件 SHA256 不应与旧 worker-payload 强行相等。检查样本 ID、顺序和 input IDs。

## 单卡组件 profile 的可迁移入口

以下示例用 GPU0，必须替换为新机器上允许使用的空闲卡。它直接调用 Python 入口，避开旧启动 shell 的绝对路径。下面没有启用 Nsight 硬件采集，只记录组件 Event/CPU、传输、批次和等待信息；细层 NVTX 的 GPU kernel 时间、SM/HBM 计数器需要下一节的 Nsight 包装。

```bash
source repro/activate.sh
CUDA_VISIBLE_DEVICES=0 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
python experiments/diagnostics/component_profile.py \
  --model /your/models/Qwen3-8B --input .repro/profile-input.json \
  --output log/component-profile-migrated-b32 \
  --policy reservation --batches 32 --pool-pages 98304 --buffer-batch-capacity 128 \
  --warmup-seconds 120 --record-seconds 180 --record-windows 3 \
  --window-gap-seconds 180 --detail-every 8
```

k=8、r=.05、温度 .6、seed330、FP16、query budget2048、chunk256；正式请求自然 EOS 或达到32768总长度边界，不额外限制输出。KV池为98304页，Qwen3-8B TP1下为13.5GiB；CPU冷池是按请求延迟分配的逻辑容量，不能理解为立即 pin 数百 GiB。新硬件容量不同可显式改 pool-pages，但比较性能时需要记录区别。批次上限不等于实际参与计算的 D/V/P 数量。

首次迁移可先显式追加 `--output-tokens 32 --warmup-seconds .5 --record-seconds 2 --window-gap-seconds 1` 做短测，输出到不同目录；它是开发预检，不作为自然 EOS 正式实验结果。不要并行在同一张卡上启动短测与正式实验。

### Nsight 硬件采集

新机器需要自行安装 Nsight Systems CLI；Git 中不打包参考机 `log/` 下的 Nsight 安装。GPU硬件计数器需已有驱动权限，或者由sudo启动collector、通过 `--run-as` 让worker仍属于普通用户。以下假设 `nsys` 已可在当前终端解析，示例GPU0：

```bash
source repro/activate.sh
profile_repo="$PWD"
profile_python="$VIRTUAL_ENV/bin/python"
profile_nsys="$(command -v nsys)"
mkdir -p log
sudo "$profile_nsys" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --gpu-metrics-devices=0 --gpu-metrics-frequency=1000 \
  --run-as="$(id -un)" --capture-range=cudaProfilerApi --capture-range-end=repeat \
  --output="$profile_repo/log/migrated-nsys" \
  env CUDA_VISIBLE_DEVICES=0 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
      SSPEC_VENV="$VIRTUAL_ENV" CUDA_HOME="$CUDA_HOME" \
  bash -c 'cd "$1"; source repro/activate.sh; exec "$2" experiments/diagnostics/component_profile.py \
    --model "$3" --input .repro/profile-input.json --output log/component-profile-migrated-b32 \
    --capture --policy reservation --batches 32 --pool-pages 98304 --buffer-batch-capacity 128 \
    --warmup-seconds 120 --record-seconds 180 --record-windows 3 --window-gap-seconds 180 --detail-every 8' \
  -- "$profile_repo" "$profile_python" /your/models/Qwen3-8B
```

这是上一条普通采集命令的替代方式，不能同时启动。新机应重新标定 `--reservation-iteration-ms`，不能直接把参考机31ms视为跨硬件保证；Python入口会测搬运有效带宽。计数器频率和NVML采样间隔以实际记录为准，设置值不是实测频率保证。

旧 `experiments/diagnostics/run_component_profile.sh`、其sweep控制器及若干历史诊断脚本仍包含参考机模型、venv、Nsight或日志路径。不要在新机器直接原样调用旧sudo扫描命令；上面的显式入口可独立运行。需要迁移自动硬件扫描时，应先配置新机Nsight位置、GPU/PCIe拓扑及CPU亲和性，再做功能预检。

## 结果完整性

- 采样窗口与全部128请求自然 EOS 是不同完成状态；当前 observer/全程完成元数据仍在整批结束后写入。
- 三个窗口已结束而整批未收尾时，可以使用 `inspect_component_capture.py` 检查已经导出的报告与JSONL；保留Nsight警告，不补造全程完成统计。
- `inspect_component_capture.py --root <单档目录> --output <独立检查目录>` 要求检查目录中已有 `nsys.1.sqlite`、`nsys.2.sqlite`、`nsys.3.sqlite`，按窗口顺序导出；详见其参数帮助。
- 数据库导出使用 `nsys export --type=sqlite --output=<检查目录>/nsys.1.sqlite <报告>.nsys-rep`；另外两个窗口同理。
- 旧server的B32已确认三个窗口可读，仍带Nsight事件可能缺失的警告，不能据此宣称整批推理正常完成或严格零丢失。

本次迁移检查以本机CPU测试、语法与输入检查为范围；尚未在另一台物理服务器从零安装并实跑。
