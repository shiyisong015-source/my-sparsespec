# 在另一台服务器运行当前 SparseSpec 研究版本

本目录对应 2026-10-20 的代码与实验环境。主实验是 Qwen3-8B、TP2、FP16、thinking 开启、temperature=0.6、seed=330，CUDA Graph、CPU 异步调度及 KV 异步传输开启。运行的是改动版 SparseSpec self-spec。

## 1. 获取代码

```bash
git clone --branch research/sparsespec-reproduction \
  git@github.com:shiyisong015-source/my-sparsespec.git
cd my-sparsespec
bash repro/bootstrap.sh
```

需要 Linux x86_64、Python 3.12、GCC/G++、可用的 NVIDIA 驱动及 CUDA 12.8 编译工具链。参考机器为 Ubuntu 22.04、A100 PCIe 40GB。`bootstrap.sh` 只在仓库的 `.repro/` 下创建环境和原生依赖；默认使用 `python3.12`。也可指定 `SSPEC_PYTHON=/path/to/python3.12` 和 `SSPEC_VENV=/path/to/new-venv`。

Python 完整版本锁位于 `environment/requirements-lock.txt`，共189个包。先装 PyTorch 2.8.0 的 cu128 wheel，再按观测环境的全量精确版本安装；使用 `--no-deps` 避免重新求解改变已有版本。FlashInfer 必须使用仓库固定子模块，不能替换成普通 PyPI FlashInfer。运行依赖 vLLM 0.11.0 wheel 的自定义 all-reduce 算子；不需要编译 `3rdparty/vllm` fork。

需要切换国内 Python 镜像时可在执行 bootstrap 前设置 `PIP_INDEX_URL`。PyTorch 三个 wheel 仍由脚本明确从官方 cu128 索引安装。

## 2. CUDA 编译环境

如果新服务器已有 `/usr/local/cuda-12.8`，激活脚本自动使用它。也可显式指定：

```bash
export CUDA_HOME=/your/cuda-12.8
source repro/activate.sh
python repro/check_environment.py
```

如果没有 CUDA 12.8 toolkit，但已有 conda，可只在当前仓库安装参考机的相同工具链，不改系统驱动：

```bash
conda create --yes --prefix "$PWD/.repro/cuda12" \
  --file repro/environment/cuda12-linux-64-explicit.txt
source repro/activate.sh
python repro/check_environment.py
```

`activate.sh` 设置 CUDA/RMM/RAFT 的头文件和动态库搜索路径、固定子模块的 `PYTHONPATH` 以及独立 FlashInfer JIT 缓存。首次运行会编译 kernel。可针对卡型设置 `TORCH_CUDA_ARCH_LIST=8.0`（A100）或 `9.0`（H100）；不设置时由 PyTorch 读取可见设备。

`environment/native-linux-x86_64.tar.gz` 包含当前实际使用的808个原生文件：RAFT/RMM/rapids_logger 头文件及 `librmm.so`、`librapids_logger.so`。bootstrap 解包并逐文件验证 SHA256。库按参考 Linux/CUDA 环境打包；其他架构需自行从源码构建，不能直接复用这个二进制包。来源、版本、许可证和哈希在 `native-manifest.json`，环境详情在 `reference-environment.json`。

## 3. 模型

模型权重不放入 Git。可以使用已有 Qwen3-8B，或在能联网的机器下载官方权重，例如：

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-8B', local_dir='/data/models/Qwen3-8B')
PY
```

国内也可以从 ModelScope 的 `Qwen/Qwen3-8B` 下载；修改下方命令中的 `--model` 即可使用任意本地位置。仓库另附原模型的小文件包（config、tokenizer、权重索引和许可证，不含权重）：

```bash
tar -xzf repro/environment/qwen3-8b-metadata.tar.gz -C /data/models/Qwen3-8B
```

这会把目标模型目录的小文件恢复为参考版本。启动器检查其 SHA256 及五个权重分片的文件大小，避免 tokenizer 或 checkpoint 不完整。权重本体没有做完整 SHA256，因此分片大小检查不能发现所有内容损坏。

## 4. 先检查环境和短测

先选择自己有权限使用的空闲卡。以下示例使用 0、1；卡号通过参数指定，不再绑定旧服务器的 2、3、4、5。

```bash
source repro/activate.sh
CUDA_VISIBLE_DEVICES=0 python repro/check_environment.py --gpu

python -u -m experiments.acceptance.run \
  --model /data/models/Qwen3-8B --gpus 0,1 --ratios .05 \
  --requests 4 --batch-size 4 --max-output-tokens 256 --cpu-pages 8192 \
  --output log/acceptance-smoke
```

环境 GPU 检查会在第一张可见卡编译/执行一次 TopK。短测是4条、输出上限256、较小 CPU 池的可运行性验证，不作为正式吞吐数据。输出目录必须尚不存在。启动器只在所选两张卡空闲时提交任务，不占用其他卡。

## 5. 正式 k63 / r 扫描

```bash
python -u -m experiments.acceptance.run \
  --model /data/models/Qwen3-8B --gpus 0,1 \
  --k 63 --ratios .05 .10 .15 .20 .25 \
  --output log/acceptance-k63-rscan
```

同一 TP2 GPU 对串行运行所有 r，每个请求每档跑一次。需要用4张卡并行时，可在两个终端使用不重叠的 GPU 对和输出目录：

```bash
# 终端一
python -u -m experiments.acceptance.run --model /data/models/Qwen3-8B \
  --gpus 0,1 --ratios .05 .15 .25 --output log/acceptance-k63-part1
# 终端二
python -u -m experiments.acceptance.run --model /data/models/Qwen3-8B \
  --gpus 2,3 --ratios .10 .20 --output log/acceptance-k63-part2
```

| 参数 | 默认值/说明 |
|---|---|
| 数据 | `experiments/acceptance/inputs/lcb128-seed330.jsonl`，同一128条请求、顺序、prompt 和 input IDs |
| 输出 | 自然 EOS；每请求上限 `32768 - input_tokens` |
| batch 上限 | 128；实际 batch 随资源和阶段变化 |
| k | 63；支持指定1到63，分析绘图脚本仅针对k63 |
| r | .05/.10/.15/.20/.25 |
| chunk / 最小 KV 预算 | 256 / 192 |
| GPU 内存比例 | 0.8，可用 `--mem-fraction` 显式调整并记录 |
| 单请求选集容量 | r≤.15 使用6144，否则9216 |
| CPU KV 页预算 | 每rank 1,016,087，可用 `--cpu-pages` 显式调整 |

Qwen3-8B FP16/TP2 每页每rank为73,728字节，默认 CPU KV 池预算约69.77GiB/rank，两rank约139.54GiB，此外还需模型、Python和其他主机内存。新机器内存较小时需要显式调低 `--cpu-pages`；这个改变会影响资源压力，比较结果时必须报告。启动器不会在 OOM 后静默减小 batch、r 或 k。实验不自动恢复半途失败的结果；保留原目录，修正后使用新目录。

`--prepare-only` 只检查输入/模型、生成 payload，不启动 GPU。`--max-output-tokens` 会改变工作负载，只用于明确标记的短测。`--requests N` 使用已固定随机样本的前N条，不再重新抽样。

## 6. 记录和分析

每档输出到 `log/.../r0.05/lcb128-natural/`：

- `rounds.jsonl` / `rounds.rank1.jsonl`：每次verify的实际接受数，排除bonus；cold_start单独标记。
- `selection-signals.rankN.jsonl/.f32`：按下一轮标记的每层/head选集分数统计。
- `probability-signals.rankN.jsonl/.f32`：当前草稿每位置的α、q、p及首token熵/置信度。
- `uniform-signals.rankN.jsonl/.f64`：实际用于拒绝判定的u，float64，按请求和轮次对齐；未使用位置填NaN。
- `uniform-validation.json`：通过u和p/q重建首次拒绝点，核对原生接受长度，并检查TP两rank记录。
- `request-audit.json`、`progress.rankN.json`、`engine/`、`worker-payload.json`、`runtime.log`：完成情况、进度、原结果和实际配置。

新增u采集只截获原生函数返回的同一张量，不新增随机抽样。u仍在verify时生成；本版本没有实现提前抽u或依u提前停止draft。旧的k63扫描未记录u，不能把旧结果当作u方案的证据。

```bash
python -m experiments.acceptance.analyze --root log/acceptance-k63-rscan --fit-signals
python -m experiments.acceptance.verify_uniforms \
  log/acceptance-k63-rscan/r0.05/lcb128-natural
```

分析生成接受长度分布图、统计表、请求隔离的5折上限预测和报告。`--fit-signals` 用于完整128请求数据，不用于4条短测。接受63仍是右删失，不代表真实上限恰好63。记录分数/u的额外归约和读回有观察开销，带诊断的耗时不能直接当作无诊断吞吐对照。

`experiments/acceptance/reference/` 保存迁移时已完成结果的摘要快照，未包含大体积逐轮日志。旧k32文档为历史记录，其中相对链接仍指向当时的本地完整分析；新实验使用上述正式入口。

## 7. 验证范围

`repro/validation_zh.md` 记录本次打包的实际检查。跨服务器的驱动、GPU型号、PCIe拓扑、NUMA和CPU性能会改变吞吐；先通过短测，再启动完整扫描。保留生成的manifest和硬件信息，与旧机器比较时报告差异。
