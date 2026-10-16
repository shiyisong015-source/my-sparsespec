# 原生页池 compact KV

实现日期：2026-10-14；基线提交 `aaffc0fa4bb92836fd042ababa1dfcf809e7b208`。本版作为异步与准入策略改进前的本地提交检查点。`kv_cache="compact"` 是本文入口；`kv_cache="packed"` 保留 aaffc0f 的 Triton 原型。

## 使用

在仓库目录激活 `/data/users/songshiyi/vericache/scripts/activate_reproduction.sh`，每次运行前检查所选 GPU 空闲。默认 compact 入口使用 eager、同步 CPU、原生 attention；CPU 流水入口见下文。以下同步 Python API 已经公开入口 TP1 与 TP2 测试验证：

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

## 按需异步 KV 搬运（显式启用）

Python API 设置 `compact_async_transfers=True`，或 CLI 添加
`--compact_async_transfers`，选择 `AsyncCompactSpecScheduler`。默认仍运行上述同步 compact 路径。
单独启用这个开关只实现 KV 事务与其他就绪请求的计算重叠，CPU 保持同步。
CPU/GPU 流水入口见下节；可显式开启 `cuda_graph=True`，没有层间搬运或动态 k/r。

1. verify 后完成新选择；只要没有页压力，全量 KV 保留在 GPU，draft 使用原 sparse 索引。
2. 准入、追加或恢复缺页时，从全量驻留请求中选择一个可卸载请求。
   原计算流记录 producer event；独立 transfer stream 等待它，再分割热/冷 KV。
3. 在途请求不参与 forward，源页保持锁定；其余就绪请求继续计算。
4. 各 TP rank 查询自身完成事件并交换状态，只有全部完成才提交页表、释放源页。
   中途卸载的草稿 tail 在搬运后仍可被 rejection 回退。
5. 只有到达 verify 且 cold 在 CPU 的请求才申请恢复。恢复完成后才允许该请求 verify。
6. 没有任何就绪请求时才显式等待在途事件。第一版最多一个 KV 事务在途，
   保守预留一次压紧需要的热目的页；无法满足预算时明确失败，不隐式降低 k/r。

`compact-metrics.rank*.json` 新增 `pressure_offloads`、`transfers_submitted`、
`resident_updates`、`transfer_idle_waits`、`forwards_with_transfer_inflight` 等计数。
最后一项只证明提交 forward 时已有搬运事务尚未提交，**不证明两个 GPU kernel 实际重叠**；
实际重叠需要时间线证据。`transfer-timing.rank*.json` 的事件等待是无就绪请求时的
CPU 阻塞；TP 轮询状态交换开销另计入对应方向的协调时间，不能与纯 PCIe 时间混用。

验证日志位于 `log/kv-offload-async-20261014/`。短测使用原 AIME 已分词输入的前4题，
Qwen3-8B/FP16/TP2、k8/r.05/min192、thinking 开、temperature=.6，输出上限512。
该短测验证生命周期和压力路径，不作为完整 AIME 吞吐或接受率结论。

完整异步 KV AIME 扫描使用 `experiments/reproduction/run_async_compact_scan.py`：

```bash
python -u -m experiments.reproduction.run_async_compact_scan \
  --reference-payload log/kv-offload-acceptance-aime-20261014/r0.05/worker-payload.json \
  --output log/kv-offload-async-aime-20261014 \
  --gpu-pairs '2,3;4,5;6,7'
```

先激活上述环境，使用新输出目录，并在持久终端会话中运行控制器。
GPU 编号必须按实际授权填写；本轮仅 GPU2–7。每档 r=.05/.10/.15 跑原30题一次，
保留 TP2、k8、FP16、thinking、temperature=.6 和完整自然 EOS 输出预算。
控制器每30秒检查指定双卡是否空闲，保持运行中的旧任务，记录每个任务退出状态，失败不自动重跑。
CPU 活跃冷 KV 页预算按全部请求和一次 replacement 的上界设置，按需分配 pinned Tensor，
不一次预分配整个 CPU 预算。`controller-status.json` 保存排队状态和已运行任务的进度。

## CPU/GPU 延迟一轮流水

同时设置 `compact_async_transfers=True, async_cpu=True`，选择
`AsyncCPUCompactSpecScheduler`。CLI 对应 `--compact_async_transfers --async_cpu`。
这与单独的 KV 异步开关不同；模型 forward 的 CUDA Graph 可独立开启，见下节。

复用原版 GPU dummy-token patch 和 `VERIFY -> STALL -> DRAFT` 状态机：
提交当前轮 forward/采样后，CPU 消费上一轮结果；STALL 请求只参与结果处理，query 数为0。
下一轮 GPU 输入直接从上一轮 GPU token tensor 修补，不等待当前轮回传。
没有把正常 batch 拆成两个互不相交的小 batch。

为适配 compact ownership，额外保护以下依赖：

- 搬运流等待源请求最后一次计算事件，以及控制流的选取结果事件。
- 全 STALL 轮次不 forward，但重新生成当前请求顺序的选择页表，避免 TopK 使用旧行位置。
- shared score snapshot 的下一轮覆盖在上一轮 TopK/选择消费事件之后排入计算流；
  不对尚未记录的事件发出等待。
- retire 延迟释放 KV 和 draft probability buffer，等待计算/控制消费者均完成且全部 TP rank 同意。
  如果页压力来自这些延迟释放页，只等待相关事件，不误报不可行。
- TP sampling/rejection 沿用原生各 rank 本地执行方式，不新增 token、概率表或
  rejection 结果广播，也不创建专用 rejection NCCL group。各 rank 使用相同 seed，
  并保持随机操作顺序和形状一致。KV 搬运及延迟回收的完成状态仍按一致顺序用 Gloo 协调。

`cpu_pipeline_submissions`、`cpu_stall_only_rounds`、`cpu_updates_with_forward_inflight`、
`deferred_retirements` 等指标保存到原 metrics 文件。`cpu_updates_with_forward_inflight`
是处理上一轮结果时当前计算事件尚未完成的计数，不代表严格的 GPU kernel overlap 测量。
CPU readback/result-update wall 时间可能重叠，不能相加当纯模型开销。

验证产物位于 `log/kv-offload-cpu-async-20261014/`；源码启用入口存在不等于全配置已验证，
实际通过项以该目录的测试日志和报告为准。

上述完整扫描命令添加 `--async-cpu` 即可启用 CPU 流水；`--ratios .05` 可只运行该比例一次。
CPU 与 GPU 内部仍然通过事件维护数据依赖，并非完全无同步。

### 正常运行与 TP 诊断模式（2026-10-15）

默认 `compact_debug_checks=False`：准入、metadata、上一轮结果处理直接在各 rank
本地执行，不逐阶段交换错误状态，也不逐轮广播 ready 请求列表作一致性核对。
原来的各 rank 独立 CPU 调度结构保持不变。常驻路径仍可能在首次准入和延迟回收时通信；
实际 KV 事务的提交/回滚协调、全部 rank 就绪后提交搬运/回收、准入容量决定继续保留。

诊断时 API 设置 `compact_debug_checks=True`，CLI 添加 `--compact_debug_checks`，
扫描控制器添加 `--compact-debug-checks`，恢复逐阶段错误交换和 ready 列表核对。
该开关须在所有 TP rank 上一致，独立于 Transformer debug 模式。

TP 调度运行阶段发生未处理异常时，打印 rank 与原 traceback 后以非零状态立即退出该
worker，由 torchrun 等外部进程管理器终止同组其他 worker。不在出错进程等待 CUDA
stream、Graph 析构或通信组销毁，避免其他 rank 卡在 collective 时清理过程也一直等待。
这不是异常恢复；调用方必须有整组进程管理能力，不能在失败的 TP 进程里继续推理。
TP1 保持普通异常传播及清理路径。成功结束仍完整等待消费者、回收页并释放 Graph。

### KV 追加长度 metadata

`SpecScheduler` 首次收集 `all_kv_len_arr` 时读取 cache 的 `append_kv_length`，
随后统一上传一次。原生 Full/Streaming/Pillar cache 返回已有的 `len_kv_cache`；
Compact cache 返回 `len(indices)`，即 page-size-one 下当前物理阶段页表的长度。
Compact metadata 分发不再重建或再次上传该数组。逻辑 RoPE 位置继续使用
`len_kv_cache`，attention 读取长度继续使用 `len_loading_kv_cache`。

### CPU 页表诊断快照

常驻 verify 不再通过 `tuple(indices)` 复制完整 CPU 页表，也不在 rejection 后或 pack
入口用完整前缀逐项比较。`freeze_verify_view()` 保留名称，只记录已有 verify epoch；
阶段、完整长度、tail 回退范围和页 ownership 检查继续执行。rejection 只删除末尾页，
pack/restore 通过现有阶段与事件依赖维护页序。绕过这些接口直接修改 indices 的同长度
重排不再由完整 CPU 快照捕获。

这项删除也适用于 TP 诊断模式；`compact_debug_checks` 只控制上述 TP 阶段/批次核对。
attention/TopK 使用的实际 GPU 页表、GPU score snapshot 和搬运时必需的源页表仍保留。

## CUDA Graph（2026-10-15 新增适配）

Python API 设置 `cuda_graph=True`；原有 CLI 使用 `--cuda_graph`。
异步扫描控制器 `experiments/reproduction/run_async_compact_scan.py` 使用
`--cuda-graph`，可与 `--async-cpu` 组合，且会记录实际开关。

捕获入口为 `PillarCachePtr` 及其子类建立稀疏 metadata，使用实际 cache 类的
`draft_kv_indices_tensor_buf`。compact 复用原生按 query 数分桶、双 phase 的
模型 forward Graph；不捕获 CPU 准入、动态分配、rejection/TopK、冷热分割或搬运。
这些操作通过现有事件依赖与 forward 对接。KV pool 与 attention metadata
buffer 的地址保持稳定，页号、物理长度、逻辑 RoPE 位置在 replay 前更新。

小 KV pool 可能无法容纳某些捕获档位的初始化工作量；原生捕获器会跳过这些档位。
运行时没有合适档位就回退 eager，不改变请求的 batch 上限、k/r 或 KV pool 预算。
不能仅凭开关为真认定所有 forward 都使用了 Graph；查看 summary 中的
`Graph Captured Ratio` 和 `Graph Padded Ratio`。捕获耗时不计入 engine 推理时间，
Graph 的额外显存及 padding 开销仍需考虑。

compact 推理结束后显式等待 GPU 工作完成并 reset 两个 phase 的 Graph，
再允许外层销毁 TP 通信组。含 NCCL 操作的 Graph 若因 scheduler 引用环仍存活，
可能使 NCCL 销毁一直等待；不能只依赖 Python 析构顺序。释放不清空吞吐统计，
且只在结束时执行，不在每次 forward 新增全设备同步。

适配验证日志：`log/kv-offload-cuda-graph-20261015/`。实际完成项以该目录的
`status.json`、测试日志和报告为准；不把正在运行的测试列为通过。

## 数据与执行流程

原 `KVPool` 保持 `[layer, capacity, 2, 1, local_KV_head, dim]` 布局、`alloc_page/free_page` 和 full prefill 地址/append。新增 `CompactCachePtr` 只负责独立 ownership、阶段页表和搬运事务：

1. 原 FullCachePtr prefill；转移 ownership 后 cold-start draft 仍读完整 KV；首轮 verify 仍为 full。
2. 原 rejection 只截无效 tail，原 scores/TopK 在冻结的 verify 页表顺序上选择源页。每 layer/head 有相同 hot 数量，源 token 可以不同。
3. 在同一 GPU 池分配共同 hot 目的页；各层/head gather 到相应 slot，cold 直接写 pinned CPU。所有层和 TP rank 完成后释放旧源页。
4. draft 原生 attention 读 hot + recent/tail。verify 前从 mapped pinned CPU 直接 scatter 回原池新页；共同页表为 `[selected_hot, restored_cold, chronological_recent, chronological_tail]`。
5. 后续 verify 继续调用原 persistent attention：统一 softmax、因果 query suffix、内部 split-KV reduction。没有新独立 merge。随后重复 rejection、selection、repack；终止请求不做下一轮 repack。

三个长度必须区分：`len_kv_cache` 是完整逻辑位置；`len(indices)` 是当前物理 append 页表长度；`len_loading_indices` 是 draft 可见长度。RoPE 使用逻辑位置，append 使用物理位置。`owned_pages` 是唯一 ownership 集合，不能用阶段页表代替。

QKV/MLP/RoPE、原 attention/score/TopK 数学和输出采样/rejection 继续复用。保留混合 batch 密集前向及一活跃 chunked-prefill。TP 各 rank 本地调用原 sampler 和 rejection sampler，采样 token 按原生流程转为 int32；每 rank 的 KV 选择可不同，但输出 token、计数与生命周期须一致。

## 限制与预算

- `k>=1`、`0<r<=1`、`min_budget>=k+2`，最大热预算严格 `<6144`；`GQA*(k+1)<=64`。原 public `run_inference` 按需上调 `RejectionSampler.MAX_SPEC_LEN`；其默认 10 不是公共入口的固定 k 上限。直接构造 scheduler 的测试/调用者需先完成原 metadata 和概率缓冲初始化。
- batch draft 总索引 `<128*1024`，verify `<256*1024`，query NNZ `<32768`，batch `<4096`。冷启动 draft 包含整个 prompt，不能只按热比例估算。超预算请求等待；单请求无法满足保守峰值时明确失败。
- 初次迁移至少 old full + new hot；后续包括同批 append、全部恢复 cold、串行迁移中的最大 replacement hot。CPU live 预算包含 old/new cold 重叠。源码固定 k/r，不支持层间 overlap；CPU 流水与模型 forward CUDA Graph 使用上述显式开关。
- 只保留一个生产 GPU KV pool；关闭未使用的 legacy 1.5× CPU full buffer。`free_page` 增加池内可用页，**不降低已经预分配的 CUDA tensor/reserved 显存**。降低物理显存需要显式减小 `compact_pool_pages` 并验证峰值。
- `compact_cpu_pages` 限制活跃 cold Tensor slots；PyTorch pinned host cache 可能保留释放的块，不是 OS 锁页/reserved 硬上限。`freed_source_pages` 是累计归还旧源页次数，不是净释放页数。
- public 入口沿用原 `clip_length/max_position_embeddings` 输出预算裁剪；compact 要求非空输入和正的有限输出预算。原 speculative 终止检查可能使实际输出超过请求上限，报告使用实际 token 数；接近模型最大位置时需预留 `k+1` 的尾部空间，未改变原输出终止语义。
- TP setup/allocation errors 在模型 collective 前协调；CUDA fault 仍为致命错误，不能靠 Gloo 自动恢复。

## 验证与性能口径

完整证据见 [实施报告](../../log/kv-offload-native-review-20261014/report_zh.md)，设计与旧原型事实见 [交接文档](../../research/kv_offload_handoff_astra_2026-10-14.md)。`log/` 中大部分产物仅在本机，不能假定 fresh clone 包含全部日志。

Qwen TP1/TP2 在相同 batch、query、KV 内容和顺序下与原生对照 logits 逐位一致，覆盖搬运后 epoch；实际 KV 搬运也做逐位校验。完整前缀分块重算 teacher 在个别 mixed batch 超出原严格阈值，原生对照出现相同差异；前缀反序压力也有严格失败记录。因此不承诺任意重排/批形状的 bitwise 输出一致。

原 RAFT TopK 对固定真实 score 输入重复调用会产生不同排列和并列选择。自由运行的 temperature=0 长轨迹因此也不完全可重复。配对传输计时使用**仅测试用**固定原生有效选择记录；仍执行原 TopK，再复制相同记录到输出，两 compact arm 的完整输出、接受数、批次工作量一致后才计时。生产不使用 trace，也未改 TopK。

AIME 8 条、TP1/BF16、B4/k4/r.1/min128、thinking 开、temperature=0、eager 三轮：原生自由运行 33.294 秒；固定选择 compact resident 36.120 秒；固定选择 compact CPU 40.793 秒。两 compact arm 差 4.673 秒（12.94%）。原生与 compact 长轨迹不同，差值不能解释成纯 attention 或搬运开销。Trace 复制和 host 检查开销包含在测量中，报告另列代表性 microbench，不作精确相减。

同形状原生/compact 的 prefill、draft、verify forward logits 均相同；graph microbench GPU forward 约 17.33/13.23/14.76 ms。该轮图只用于 micro 测量，当时服务端仍为 eager；后续 serving Graph 适配见上节。搬运 host wall 包含分配与隐式等待，eager event 包含提交间隙；二者不能相加，也不能称为纯 kernel 时间。

## 完整 AIME 接受率扫描

`experiments/reproduction/run_compact_acceptance_scan.py` 复用原 r 扫描的 AIME24 全部30题已分词输入、顺序及输出预算，Qwen3-8B、TP2、FP16、k8、温度0.6、thinking 开、最大总长度32768，分别运行 r=.05/.10/.15 各一次。启动脚本限定 GPU 4/5 与 6/7 两组，第三档等待一组完成后启动；不重新运行原生基线。输入文件和旧扫描日志需要已存在于本地，fresh clone 不能直接假定具备这些产物。

扫描保存每轮接受长度、每100步进度、实际 batch 分布、完成原因与原始输出；未单独记录本次 model forward 时间。相较旧扫描关闭了 CUDA Graph 和 async CPU，因此吞吐差异包含执行模式、搬运与等待的差异。输出位于 `log/kv-offload-acceptance-aime-20261014/`；提交启动代码不表示整组实验已经完成。

## 传输等待计时（6d3f02c之后新增）

新启动的compact scheduler每100步及正常结束时写入 `result_dir/transfer-timing.rank{rank}.json`。`transfer_event_wait_fraction` 是本rank在现有restore/repack事件上等待的CPU wall时间，除以同一scheduler.run区间的elapsed_seconds；模型加载不计入。两rank分别报告，不能相加当全局占比。计时不新增CUDA Event或设备同步，原有等待、TP协调和commit顺序保持一致。

`directions.restore/repack` 分别包含提交、首次事件等待、TP状态交换、commit的累计秒数。提交时间包含分配、分区、元数据和kernel提交，也可能有隐式等待；repack事件等待覆盖partition、热KV gather和冷KV D2H，不能解释为纯PCIe传输。commit含原有第二次等待同一个已完成事件及页表更新/释放，其CPU开销另计，不重复计为首次传输等待。`transfer_section_wall_fraction` 汇总上述四阶段，用于观察整个恢复/重排路径的成本，不等于纯传输占比，也不是潜在异步收益上限。

已运行的worker不会热加载此改动，无法追溯此前未记录的等待。扫描脚本给之后启动的worker分别保存实际加载源码哈希；原控制器启动时的source-sha256.json不能代表后来启动的全部worker。当前排队的r=.15将使用新增计时，r=.05/.10已运行部分没有该项实测。当前仍未单独记录各次模型forward耗时。
