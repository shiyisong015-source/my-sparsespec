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
