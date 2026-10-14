# 原生页池 compact KV

实现日期：2026-10-14；基线提交 `aaffc0fa4bb92836fd042ababa1dfcf809e7b208`。本版作为异步与准入策略改进前的本地提交检查点。`kv_cache="compact"` 是本文入口；`kv_cache="packed"` 保留 aaffc0f 的 Triton 原型。

## 使用

在仓库目录激活 `/data/users/songshiyi/vericache/scripts/activate_reproduction.sh`，每次运行前检查所选 GPU 空闲。当前实现要求 eager、同步 CPU、原生 attention。以下 Python API 已经公开入口 TP1 与 TP2 测试验证：

```python
from serve.request.request import Request
from serve.run import run_inference

outputs = run_inference(
    [Request(input_token_ids, desired_output_length=512)],
    "/data/models/Qwen3-8B",
    kv_cache="compact", spec_stride=4, budget_ratio=0.1,
    num_min_budget=128, max_batch_size=4, chunk_size=256,
    compact_pool_pages=4096, compact_cpu_pages=8192,
    temperature=0.0, tp_ranks=1,
    result_dir="log/my-compact-run",
)
```

`input_token_ids` 是调用者已分词的非空 prompt。Qwen thinking 由分词时的 chat template 控制；本轮 AIME 使用 `enable_thinking=True`。页数示例只适用于能满足该预算的请求；20K prompt 需要显著更大的原池和搬运峰值预算。不要将上述页数直接用于所有长度。

CLI 同样支持 `--kv_cache compact --compact_pool_pages ... --compact_cpu_pages ...`。可执行的本机验证脚本见 [commands.sh](../../log/kv-offload-native-review-20261014/commands.sh)，默认只显示使用方法，各 GPU 子命令先 fresh 检查。脚本不会运行整套实验或发布代码。

## 数据与执行流程

原 `KVPool` 保持 `[layer, capacity, 2, 1, local_KV_head, dim]` 布局、`alloc_page/free_page` 和 full prefill 地址/append。新增 `CompactCachePtr` 只负责独立 ownership、阶段页表和搬运事务：

1. 原 FullCachePtr prefill；转移 ownership 后 cold-start draft 仍读完整 KV；首轮 verify 仍为 full。
2. 原 rejection 只截无效 tail，原 scores/TopK 在冻结的 verify 页表顺序上选择源页。每 layer/head 有相同 hot 数量，源 token 可以不同。
3. 在同一 GPU 池分配共同 hot 目的页；各层/head gather 到相应 slot，cold 直接写 pinned CPU。所有层和 TP rank 完成后释放旧源页。
4. draft 原生 attention 读 hot + recent/tail。verify 前从 mapped pinned CPU 直接 scatter 回原池新页；共同页表为 `[selected_hot, restored_cold, chronological_recent, chronological_tail]`。
5. 后续 verify 继续调用原 persistent attention：统一 softmax、因果 query suffix、内部 split-KV reduction。没有新独立 merge。随后重复 rejection、selection、repack；终止请求不做下一轮 repack。

三个长度必须区分：`len_kv_cache` 是完整逻辑位置；`len(indices)` 是当前物理 append 页表长度；`len_loading_indices` 是 draft 可见长度。RoPE 使用逻辑位置，append 使用物理位置。`owned_pages` 是唯一 ownership 集合，不能用阶段页表代替。

QKV/MLP/RoPE、原 attention/score/TopK 数学和输出采样/rejection 继续复用。保留混合 batch 密集前向及一活跃 chunked-prefill。TP 的 token 由 rank0 原 sampler 决定，在 broadcast 前统一 int32；每 rank 的 KV 选择可不同，但计数与生命周期一致。

## 限制与预算

- `k>=1`、`0<r<=1`、`min_budget>=k+2`，最大热预算严格 `<6144`；`GQA*(k+1)<=64`。原 public `run_inference` 按需上调 `RejectionSampler.MAX_SPEC_LEN`；其默认 10 不是公共入口的固定 k 上限。直接构造 scheduler 的测试/调用者需先完成原 metadata 和概率缓冲初始化。
- batch draft 总索引 `<128*1024`，verify `<256*1024`，query NNZ `<32768`，batch `<4096`。冷启动 draft 包含整个 prompt，不能只按热比例估算。超预算请求等待；单请求无法满足保守峰值时明确失败。
- 初次迁移至少 old full + new hot；后续包括同批 append、全部恢复 cold、串行迁移中的最大 replacement hot。CPU live 预算包含 old/new cold 重叠。源码固定 k/r，不支持层间 overlap、async CPU 或 serving CUDA Graph。
- 只保留一个生产 GPU KV pool；关闭未使用的 legacy 1.5× CPU full buffer。`free_page` 增加池内可用页，**不降低已经预分配的 CUDA tensor/reserved 显存**。降低物理显存需要显式减小 `compact_pool_pages` 并验证峰值。
- `compact_cpu_pages` 限制活跃 cold Tensor slots；PyTorch pinned host cache 可能保留释放的块，不是 OS 锁页/reserved 硬上限。`freed_source_pages` 是累计归还旧源页次数，不是净释放页数。
- public 入口沿用原 `clip_length/max_position_embeddings` 输出预算裁剪；compact 要求非空输入和正的有限输出预算。原 speculative 终止检查可能使实际输出超过请求上限，报告使用实际 token 数；接近模型最大位置时需预留 `k+1` 的尾部空间，未改变原输出终止语义。
- TP setup/allocation errors 在模型 collective 前协调；CUDA fault 仍为致命错误，不能靠 Gloo 自动恢复。

## 验证与性能口径

完整证据见 [实施报告](../../log/kv-offload-native-review-20261014/report_zh.md)，设计与旧原型事实见 [交接文档](../../research/kv_offload_handoff_astra_2026-10-14.md)。`log/` 中大部分产物仅在本机，不能假定 fresh clone 包含全部日志。

Qwen TP1/TP2 在相同 batch、query、KV 内容和顺序下与原生对照 logits 逐位一致，覆盖搬运后 epoch；实际 KV 搬运也做逐位校验。完整前缀分块重算 teacher 在个别 mixed batch 超出原严格阈值，原生对照出现相同差异；前缀反序压力也有严格失败记录。因此不承诺任意重排/批形状的 bitwise 输出一致。

原 RAFT TopK 对固定真实 score 输入重复调用会产生不同排列和并列选择。自由运行的 temperature=0 长轨迹因此也不完全可重复。配对传输计时使用**仅测试用**固定原生有效选择记录；仍执行原 TopK，再复制相同记录到输出，两 compact arm 的完整输出、接受数、批次工作量一致后才计时。生产不使用 trace，也未改 TopK。

AIME 8 条、TP1/BF16、B4/k4/r.1/min128、thinking 开、temperature=0、eager 三轮：原生自由运行 33.294 秒；固定选择 compact resident 36.120 秒；固定选择 compact CPU 40.793 秒。两 compact arm 差 4.673 秒（12.94%）。原生与 compact 长轨迹不同，差值不能解释成纯 attention 或搬运开销。Trace 复制和 host 检查开销包含在测量中，报告另列代表性 microbench，不作精确相减。

同形状原生/compact 的 prefill、draft、verify forward logits 均相同；graph microbench GPU forward 约 17.33/13.23/14.76 ms。图只用于 micro 测量，服务端仍为 eager。搬运 host wall 包含分配与隐式等待，eager event 包含提交间隙；二者不能相加，也不能称为纯 kernel 时间。

## 完整 AIME 接受率扫描

`experiments/reproduction/run_compact_acceptance_scan.py` 复用原 r 扫描的 AIME24 全部30题已分词输入、顺序及输出预算，Qwen3-8B、TP2、FP16、k8、温度0.6、thinking 开、最大总长度32768，分别运行 r=.05/.10/.15 各一次。启动脚本限定 GPU 4/5 与 6/7 两组，第三档等待一组完成后启动；不重新运行原生基线。输入文件和旧扫描日志需要已存在于本地，fresh clone 不能直接假定具备这些产物。

扫描保存每轮接受长度、每100步进度、实际 batch 分布、完成原因与原始输出；未单独记录本次 model forward 时间。相较旧扫描关闭了 CUDA Graph 和 async CPU，因此吞吐差异包含执行模式、搬运与等待的差异。输出位于 `log/kv-offload-acceptance-aime-20261014/`；提交启动代码不表示整组实验已经完成。
