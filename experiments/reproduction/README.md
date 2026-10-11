# Qwen3-8B SparseSpec / vLLM 对齐复现

**来源标记：下述对齐结果使用研究改动版 SparseSpec，不是未经修改的官方源码复现。** 相对上游 `5c9616c` 有 attention tile 分界、请求/KV生命周期及调度计时等改动。官方干净版核验另存 `log/reproduction/official-clean-audit/`；两种版本的结果不可混称。

当前主目录：`log/reproduction/aligned-vllm011-m080-recovery2/`。

2026-10-10 上午检查：旧目录两个 vLLM 正式案例已完整完成；两个 SparseSpec 案例完成首轮全量生成后在评测包装层失败，AIME 正式案例未启动。修复后使用新目录 `log/reproduction/aligned-vllm011-m080-recovery/`，旧失败记录保留。

双方固定 TP2、FP16、温度0.6、thinking、seed330、最大 batch128、内存比例0.80、最大模型长度40960。SparseSpec 固定 k8、r0.05、最小KV预算192、chunk256、CUDA graph、异步CPU调度。vLLM 使用 v0.11.0 / V1，保留其原生图、chunked prefill、连续批处理和前缀缓存配置，实际值在 `engine/effective_config.json`。

论文的 vLLM b8b302c 正是 v0.11.0 标签。两种系统使用同一隔离环境 `.venv-vllm011`，PyTorch2.8.0+cu128、transformers4.55.2；原 SparseSpec 环境保持原样。

内存比例0.85的原环境正式运行和新环境大样本预检都保留 OOM 日志。0.80是A10040GB适配，未降低batch；本轮不能宣称完全匹配H100论文的温度0.65、batch256和2048随机请求。

## 进度与结果

从工作区根目录读取：

```bash
cat external/SparseSpec/log/reproduction/aligned-vllm011-m080-recovery2/pipeline-status.json
```

六组预检均通过、初始请求哈希一致后自动进入正式运行。AIME24重复60次（1800请求）、OlympiadBench重复3次（2025请求）、LiveCodeBench重复2次（2110请求）。数据集分别运行；只使用空闲GPU对。失败保留结果并停止后续启动，不自动调整参数。

正式运行不使用预检的1024输出上限。原评测流程可能追加工具反馈轮次，逐轮保存并汇总所有生成窗口。不同采样实现不保证输出完全相同，报告实际token数量与平均输出长度。官方LCB入口不执行代码测试，acc=0是占位值；比较报告记为空。

自动汇总到 `full-comparison_zh.md`、`full-comparison.json`、`full-cases.csv`。也可手动更新已完成案例的报告：

```bash
source scripts/activate_reproduction.sh
python external/SparseSpec/experiments/reproduction/analyze_results.py \
  --root external/SparseSpec/log/reproduction/aligned-vllm011-m080-recovery2
```

## 修改与计时

- `run_official.py`：固定负载、记录显式覆盖、检查GPU、版本和源码哈希；vLLM进程使用spawn。
- `run_aligned_pipeline.py`：较大样本预检、相同初始负载校验、GPU预留、可恢复调度、自动汇总。
- `math_eval.py`：可选vLLM dtype/seed/保存目录、实际配置及输入哈希；SparseSpec按epoch保存结果；延后引擎导入，避免加载无关引擎。
- `analyze_results.py`：核对完整性与可比条件；SparseSpec汇总各epoch的引擎summary时间，vLLM汇总generate时间；排除初始化和质量评测。vLLM窗口包含其内部tokenization，保留此计时差异。
- 工作区 `scripts/activate_reproduction.sh`：独立环境、FlashInfer编译缓存及cuRAND头文件路径。

## 包装层接续修复（2026-10-10）

- `evaluate.py`：CPU评分进程使用 spawn，避免 fork 继承 CUDA 自定义 all-reduce 对象。评分异常显式失败，不伪装为准确率0。
- `math_eval.py` / `sparse_epochs.py`：可选 `--specgen_isolated_epochs` 在 CPU 包装进程中逐轮启动独立 TP worker。每轮仍调用相同 `run_inference`、相同 seed/模型/算法参数；worker退出后才执行CPU工具/评分，避免第二轮重复初始化进程组和缓存类状态。
- `run_official.py --resume-from`：仅在参数、所有 `serve/` 源码哈希和初始输入哈希一致时重放已保存首轮输出。复制到新目录并保存来源与文件校验和；使用原首轮引擎时间，追加轮次单独记录。旧失败案例不改为成功。
- `run_aligned_pipeline.py --recovery-root`：复用通过的六组预检及已完成的两个vLLM案例；接续两个SparseSpec案例，再完成原计划的两组AIME。
- `analyze_results.py`：独立worker的各轮推测计数求和；保留旧同进程累计计数的兼容读取。

接续推理初始化仍排除在引擎时间外；额外子进程启动开销不作为SD算法加速收益。正式吞吐沿用各轮 `engine/summary.log` 的原生成窗口。

`recovery/`首次接续时，其他用户的作业在空闲检查后开始占用GPU3–6。只停止了我们的冲突任务，保存 `external-contention-interruption.json`，未操作外部进程。OlympiadBench评分已完成并复用到 `recovery2/`；剩余任务仅使用持续空闲至少60秒的GPU对，避免在其他作业重启的短暂空窗中抢占设备。

完成请求稀疏指针生命周期修复及两个CPU回归测试见 `serve/scheduler/spec_scheduler.py`、`serve/request/kv_cache_ptr/pillar.py`、`serve/tests/test_selection_completion.py`。完整经过与失败记录见 `log/reproduction/config_zh.md`。

## 单次生成的 r 扫描与资源诊断（2026-10-11）

本轮固定 Qwen3-8B、TP2、k8、batch 上限128、温度0.6、thinking、graph、异步 CPU 调度，扫描 r=.05/.10/.15。最大总上下文统一32768，以满足 r=.15 的单请求选择 buffer 上限。AIME24使用30条，OlympiadBench和LiveCodeBench分别固定随机抽256条，seed330；每个原始样本在每个配置中只生成一次。同一数据集三个 r 使用相同输入 token 和顺序。保留 EOS 或长度上限终止，无判分、工具反馈或追加生成。

| 文件 | 作用 |
| --- | --- |
| `prepare_resource_inputs.py` | 按官方提示词和 tokenizer 制作固定输入，核对已有完整负载哈希，拒绝重复样本 |
| `run_resource_case.py` | CPU 启动器写一个 payload，torchrun 启动两个新 worker；保存来源、原始结果和完成原因 |
| `resource_observer.py` | 采样请求状态、KV 占用、CUDA 分配量及 KV 传输事件；同步两张卡的 nsys 窗口 |
| `selection_probe.py` | 在独立诊断中异步复制实际选择索引，保存物理页到逻辑 token 的映射 |
| `footprint_stats.py` | 计算按 token、layer、head 保留时的选择并集和可释放字节 |
| `analyze_resource_cases.py` | 画接受长度分布，区分首次全上下文轮与后续稀疏轮，比较相同负载的吞吐 |
| `analyze_nsys_resources.py` | 关联 NVTX/CUDA API/DMA，统计计算和 KV 传输的时间并集及重叠 |
| `analyze_footprints.py` | 合并同一请求同一步的 TP2 快照，画不同粒度的潜在可释放比例 |
| `run_resource_pipeline.py` | 先完成一组完整 LCB，再扩展九组扫描，最后单独执行 nsys 和选择快照 |

在 SparseSpec 目录执行（必须先激活工作区的复现环境）：

```bash
source /data/users/songshiyi/vericache/scripts/activate_reproduction.sh
python experiments/reproduction/run_resource_pipeline.py \
  --inputs log/reproduction/resource-motivation/inputs-distinct-seed330 \
  --output log/reproduction/resource-motivation/r-scan-distinct-seed330 \
  --nsys log/reproduction/resource-motivation/tools/nsys-cli/opt/nvidia/nsight-systems-cli/2026.5.1/bin/nsys \
  --preflights log/reproduction/resource-motivation/preflight-r015-observer \
    log/reproduction/resource-motivation/preflight-r015-selection \
    log/reproduction/resource-motivation/preflight-r015-nsys-aligned \
    log/reproduction/resource-motivation/preflight-counter-eos \
    log/reproduction/resource-motivation/calibration-gpu23 \
    log/reproduction/resource-motivation/calibration-gpu45 \
    log/reproduction/resource-motivation/calibration-gpu67
```

输出目录必须尚不存在。只使用持续空闲60秒的 GPU 对，轮换配置；失败保留并停止后续启动，不隐式降低参数。进度在 `pipeline-status.json`，每组运行日志在 `runtime.log`；主分析在 `analysis/`，快照分析在 `footprint-analysis/`。逐文件改动记录在 `log/reproduction/resource-motivation/code-review-2026-10-11/`。

用户追加四张 GPU 后，当前扫描使用 `--skip-full-case-gate --max-parallel-jobs 3`：八组预检已通过即可扩展，最多三个 TP2 任务。已有任务接续还需 `--resume`，并先退出旧 CPU 控制进程；模型 worker 和当前单案例启动器保持运行。接续核对任务命令、PID、队列与预检来源，不重复生成样本。nsys 仍等待普通扫描全部结束后单独运行。并行扫描共享主机和 PCIe 资源，后续吞吐解释需保留执行并发信息；不能视为独占机器的测量。

### 测量边界

- 普通吞吐扫描没有新增逐步 CUDA 同步。nsys 的窗口边界会等待两张卡，并调用 profiler API；窗口统计排除这些启动/停止开销，整个诊断运行的吞吐不参与排名。
- 选择快照的 D2H 有开销，因此另跑。三种可释放比例是假设相应粒度存储后的空间上界；当前实现仍保留整页，不把不同时间的请求快照相加成批级收益。
- 传输事件的 CPU 提交时间不等于 DMA 耗时。DMA 字节统计包含与窗口交叠的完整复制事件，不按边界裁剪字节。
- kernel 时间包括模型计算、采样、TP 通信和 KV 准备，不等于 SM 利用率。当前权限不足以采集 SM/Tensor/HBM 硬件计数；缺失的高压窗口在 observer 中显式保存。
- 随机采样不保证逐 token 相同；短预检仅排查大幅测量开销与 GPU 差异，不提供正式置信区间。原调度器按 verify 轮完成长度检查，实际输出可能略超过配置上限，审计文件保存超出数量。
