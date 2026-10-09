# VeriCache / SparseSpec 研究交接（2026-10-08）

本文汇总本次会话的实际进展，供下一次会话直接承接。优先级高于根目录旧的 `conversation_handoff_zh.md`。本文整理历史结果，没有重新运行 GPU 实验。

## 1. 当前目标与协作方式

用户 Song Shiyi，华中科技大学。使用中文沟通。用户需要自己理解机制、参与设计和写代码；助手负责拆解、解释和协助修改。每轮只推进一个小步骤，先说明机制和源码入口，再共同确定改动，不直接交付一套未经讨论的实现。不主动使用子 agent，不重复已经通过的环境检查和 GPU 测试。

当前倾向以 **SparseSpec main 的 `serve/` 独立引擎**为实验基础，实现 VeriCache 的资源调度，然后比较原始 baseline、VeriCache 调度和用户提出的调度。实验模型、数据集和硬件不强制对齐 VeriCache，但 VeriCache 调度机制需要认真对照论文。

尚未开始实现这些调度变化。旧 vLLM self-spec 开发保留，当前没有继续接通它的执行路径。

## 2. 工作区、仓库及提交

- 工作区：`/data/users/songshiyi/vericache`，也可通过 `/home/songshiyi/data/vericache` 访问。
- 当前研究代码：`external/SparseSpec`。
- 官方远端 `origin`：`https://github.com/sspec-project/SparseSpec.git`。
- 用户远端 `personal`：`git@github.com:shiyisong015-source/my-sparsespec.git`。
- 官方基础提交：`5c9616c9af109ecec0fdf193221d3f5be6f05922`。
- 当前本地及用户远端 `main`：`d5346b25a docs: annotate SparseSpec inference and scheduling flow`。
- 这个提交只包含 5 个文件的阅读注释：`serve/model/model.py`、`serve/request/kv_cache_ptr/pillar.py`、`serve/run.py`、`serve/scheduler/scheduler.py`、`serve/scheduler/spec_scheduler.py`。提交前比较了 Python AST，确认可执行代码未改变。
- 推送成功，当前 main 跟踪 `personal/main`。最初浅克隆推送缺少对象，已用 `git fetch --unshallow origin` 补齐历史解决。
- 工作区根目录目前不是可用 Git 仓库；`artifacts/`、本文等根目录文件不属于这次 SparseSpec 提交。旧 `external/vllm` 是另外一个仓库，有未提交改动，未包含在本次提交中。

三个源码子模块都已补齐，可直接阅读：

| 子模块 | 固定提交 |
| --- | --- |
| `3rdparty/vllm` | `9cdebe5afed2c754c64fead58064ac6ac7a43c6d` |
| `3rdparty/flashinfer` | `fd1c44a4f76431e019f796ba561c786edf5e8e7f` |
| `3rdparty/raft` | `c12777506b609275681b95fb147d8625fc452345` |

重要：跑通的是 `serve/`，不是 `3rdparty/vllm`。子模块能打开不等于 vLLM fork 已运行验证。

## 3. 已完成的 A100 运行验证

已有结果位于 `artifacts/sparsespec-graph/`：

- `run.py`：为测试创建的调用脚本，构造请求、传入配置。
- `run.sh`：设置依赖路径并启动上述脚本。
- `README.md`：环境及依赖说明。
- `result/summary.log`、`result/requests.json`：同步 CPU 调度结果。
- `result-async/summary.log`、`result-async/requests.json`：异步 CPU 调度结果。

配置：单 A100 PCIe 40GB、Qwen3-8B BF16、两个请求，每个输入 916 token，输出目标 64；`kv_cache=pillar`、`chunk_size=256`、`spec_stride=4`、`budget_ratio=0.05`、`num_min_budget=16`、`temperature=0`、CUDA graphs 开启。

结果：同步和异步均完成；实际 graph token 数 2098、graph 使用比例 100%；100/118 个草稿 token 接受，84.75%；两个请求共输出 132 token；同步与异步输出 token IDs 完全相同。

限制：输出目标 64，实际每请求返回 66，严格长度边界未修复。只证明小规模执行路径跑通，不证明加速、与 full-KV 输出等价、长上下文能力、压力卸载能力或远端加载能力。不要用这次耗时直接作为正式性能结论。

隔离环境：`.venv-sparsespec` 共享旧环境的 torch 2.13.0+cu130；`.sparse-native` 提供 RMM 等原生依赖；系统 nvcc 12.8。测试 launcher 仍引用旧工作区的自定义 FlashInfer 源码，补齐新子模块后没有改 launcher。RAFT 使用固定源码头文件并补充安装生成的头文件。完整细节见实验 README。

重跑原有实验：

```bash
bash artifacts/sparsespec-graph/run.sh --async-cpu
```

**最新实验偏好：用户决定先关闭 CUDA graph 分析瓶颈。** 这只是已达成的设置方向，目前脚本仍写着 `cuda_graph=True`，尚未修改或重新测试。`async_cpu` 与 `cuda_graph` 是独立开关，关闭 graph 不等于同步调度。用户还指出 SparseSpec 实验常用温度 0.65；具体论文实验配置尚未在本次逐项核对。我们温度 0 的 smoke 不能代替随机采样正式实验。

## 4. 已读懂的执行流程

调用链：

```text
artifacts 下的实验脚本创建 Request
  → serve/run.py::run_inference
  → 初始化 KV 类型共享缓冲，创建 KVPool，绑定模型
  → graph wrapper（即使不开 graph，也保留 wrapper，走 eager runner）
  → run_spec_scheduler → SpecScheduler → BaseScheduler.run
  → 每轮 admit_requests
  → prepare_forward_inputs
  → register_input → KVPool.pre_attn_fwd → dispatch_attn_plan
  → Transformer.forward → TransformerBlock → Attention / MLP
  → sampler
  → 分数 snapshot 加入 input_metadata
  → update_requests
  → 下一轮
```

用户已理解 QKV、GQA、RMSNorm、RoPE、残差、SwiGLU MLP、合并 QKV/MLP 投影和词表 logits，不需要重新从基础讲起。

### KV 存储与稀疏视图

- `capacity` 是 GPU KV 池物理页数；`page_size=1`，页数等于 token 槽位数。
- GPU KV 形状为 `[层数, 页数, 2, 页大小, KV head数, head_dim]`。
- CPU pinned KV 池预分配约 GPU 页数的 1.5 倍，用于换出和恢复；分配 CPU 池不等于已经发生卸载。
- 请求接纳时持有 `FullCachePtr`；prefill 完成后转换为 `PillarCachePtr`。后者继承 StreamingCachePtr/FullCachePtr，增加稀疏索引，继续管理全量 KV。
- draft 和 verify 通过不同索引访问同一 KVPool，当前稀疏索引选择不会自动释放全量 KV。
- `FullCachePtr.scatter/gather/copy_token_probs` 是公共复制工具，不是把请求切回 FullCachePtr，也不是另建完整 KV。
- 两块 pillar 类共享缓冲存 draft 物理索引、verify 分数快照，分别 int32 / float16。128K 与 256K 是配置上限，2 倍不是算法约束。Qwen3-8B 单 GPU 的这两块合计约 288MiB，不含其他缓冲。

### 输入准备

- NORMAL 每次处理至多一个 prefill chunk。
- DRAFT 每次输入一个尚未 forward 的 token，生成下一个草稿。
- VERIFY 输入草稿数 + 1 个位置；只新分配一个 KV 槽位，因为最后生成的草稿尚未算自身 KV，其余槽位已有，验证时重新计算并写入。
- `initial_idx` 是第一轮推测计数起点，用来错开第一次 verify，不是序列位置。后续验证后归零。
- `token_indptr` / `all_kv_indptr` 是每请求的 token / KV 索引拼接边界。
- `pos_offsets = len_kv_cache - num_tokens`：长度已在 allocate 时增加，减掉输入长度得到本轮片段的序列起点。
- `kv_len_arr` 每请求的本轮 attention KV 长度；`all_kv_len_arr` 每请求的完整 KV 长度。两者都是每请求一项，后者不是按 token 重复展开。
- `probs_src_start_pos`、`probs_token_num_list`、`probs_tgt_tensor_ptrs` 分别记录 draft 概率源行、行数、目标 GPU 地址。概率从 GPU 临时批量输出复制到请求的跨轮 `draft_probs`，仍留在 GPU。
- `torch.frombuffer` 不会自动提供 pinned memory，变量名含 pinned 不能当作证明。

### Attention 和评分

- 各阶段经过同一模型的投影、MLP；差别主要在 attention 读全量还是稀疏 KV。
- 每层先计算 QKV、QK norm（Qwen3）、RoPE，写入 K/V，再执行 attention。
- pillar 的 `dispatch_attn_fwd` 无条件调用 `run_attention` 和 `rematerialize_scores` 两步。
- 自定义 attention kernel 对 DRAFT 使用 `flatten_indices`，NORMAL/VERIFY 使用全量 `kv_indices`。
- 额外评分 kernel 重读 QK、使用第一次 attention 的 lse 归一化，并聚合多个 query / GQA query head 的贡献。
- `3rdparty/flashinfer/include/flashinfer/attention/persistent.cuh` 有 `if (variant.request_type != 2 || packed_qo_start != 0) continue;`，提前跳过非 verify 工作项。因此纯 draft 仍有 launch/遍历开销，但不执行 verify 的完整评分计算。
- 同一 stream 中，评分位于后续层计算之前，会影响混合 batch 完成时间；具体耗时尚未测。
- Top-k 按层、按 KV head 选择；参考的是本轮 verify 的 Q。被拒绝草稿位置的 Q 也可能贡献评分，目前没有确认实现会剔除这部分贡献。
- KV 总预算为 `min(KV长度, max(num_min_budget, ceil(KV长度*budget_ratio)))`；实际 Top-k 再减去最近 token 的预留。源码中 `num_accepted_tokens` 传入的是验证最终输出数量，包含 correction/bonus，不能直接解释成纯接受草稿数。
- 冷启动第一次 draft 使用全量上下文；后续 verify 结果后更新 Top-k。不是 SnapKV/KVzip 已接入。

### update_requests 与异步

用户已理解该函数，不必再逐段展开 rejection sampler：取回 token IDs、scatter draft 概率、收集验证请求、gather 对齐 draft/target 概率、GPU rejection sampling、CPU parse、更新 token / 回退拒绝 KV、判断结束、切换状态、更新 Top-k、记录 prev_idx、整理队列。

- 同步：本轮结果，VERIFY 直接处理；异步：消费上一轮结果，处理 STALL 请求。
- 异步状态：VERIFY → STALL → DRAFT；同步 VERIFY → DRAFT。
- sampler 在 GPU 上；update_requests 的 CPU token 拷贝和结果 parse 仍有同步点。异步只是尝试与本轮 GPU 工作重叠，不是完全不阻塞。
- 异步输入先有 `-233` 占位 token，`update_metadata_device` 直接用上一轮 GPU 结果替换本轮 GPU 输入；随后 host 更新 CPU token。
- prepare 本轮仍收集 STALL 的 KV 映射，但不执行其 forward，供后续处理上一轮 verify 结果。
- 上一轮元数据保存由主循环 `last_gen_input_metadata = input_metadata` 完成，不在 prepare 中。
- 异步 update_requests 接到的 `input_metadata` 是上一轮字典，而 `spec_cache_args` 保存本轮完整 KV 映射。
- `prev_idx` 定位上一轮分数，`idx` 定位本轮 KV 映射。读取旧 prev_idx / 更新 Top-k 之后，才记录本轮 prev_idx，并移除完成请求。
- 例：上一轮 [A,B]，下一轮 A 完成只剩 [B]。B 用 prev_idx=1 查上一轮数组，用 idx=0 查本轮数组；不能混用两轮数组。

## 5. 当前卸载机制与 VeriCache 的差别

SparseSpec `offloading` 是显存不足时的请求换出：从活跃队列移除，KV 分批搬到 CPU，空间足够后恢复，全部提交恢复后再进入活跃队列。它不会让被换出请求持续稀疏 draft。

每批至多 1024 token KV；offload/restore 返回 request 表示还有下一批，返回 None 表示批次全部提交，不表示 DMA 传输此刻已经完成。底层有 non_blocking 拷贝，且索引整理、CPU 索引取回等还可能执行 kernel 或同步；不能宣称搬运全程与计算重叠。

admit_requests 中：先估算活跃请求尚需 KV 的空间；空间负则换出请求；按 KV 长度降序排列再倒序选，所以优先较短请求，跳过 STALL；空间足优先恢复最近换出请求，再接纳新请求。

offloading 接纳策略同时只允许一个 prefilling_request，支持 chunked prefill 与其他请求 draft/verify 混合，不是稀疏 prefill。当前离线入口一次提交全部请求；按 QPS/到达时间释放请求尚未实现。

要实现 VeriCache，还需全量 CPU KV + GPU 稀疏驻留、verify 加载和回收、与 draft 重叠的传输，以及带宽/显存预算调度。不是仅替换一个调度函数。LMCache 可提供优化传输和缓存管理，但 SparseSpec 独立引擎需要适配，没有直接接入；是否更快尚未测。

## 6. 最新研究想法：联合选择 ratio 和 draft length

用户提出：先根据显存占用确定 KV 保留 ratio；固定 ratio 后，draft length 的收益可能先升后降；画最大收益的 80% 水平线，得到近优长度区间，在区间内根据需要结束 draft、转 verify。

用户进一步强调：加速不只来自单请求推测，而来自 **稀疏 KV 驻留节省显存、扩大活跃 batch，以及混合 draft/verify 的批处理**。因此不能仅优化单请求一轮接受率或速度。

当前讨论的建模方向（未实现、未验证）：

1. 显存约束需计入稀疏驻留、新增 KV、verify 全量加载预留、模型及辅助缓冲。
2. ratio 和长度是耦合的；显存首先给 ratio 可行范围，而非证明最大可行 ratio 最优。
3. 优化目标是一定时间内所有请求最终确认的输出 token 数 / 实际耗时，需跨多轮计算，draft 当轮未提交输出不能判为无收益。
4. 每请求长度还取决于其他请求的 draft/verify 状态、显存余量、传输进度；不能只写成 k=f(r)。
5. 80% 应明确度量：如果采用相对 baseline 的净加速收益 G=R/R_baseline-1，最大 1.5× 时，80% 净收益门槛是 1.4×。这只是讨论建议，用户尚未正式确定收益定义。
6. 单峰及连续区间是待测假设；不同负载、上下文和并发下区间可能不同。区间内结束 draft 不等于任意暂停不会损失收益，等待也要计入模型。

三组先保持相同模型、压缩选择和 KV 预算，比较原始 SparseSpec、在 SparseSpec 上实现的 VeriCache 调度、用户调度，便于区分收益来源。用 pillar 替代论文主实验压缩器时必须准确标注，不能称完整主实验复现。

## 7. 下一步应怎么承接

当前用户已完成完整执行链和 update_requests 的理解，最新重点转向联合 ratio / draft length 的系统级建模。

下一小步建议：共同列出一个 draft 请求和一个 verify 请求的 GPU KV 占用，以及混合 batch 一轮的计算、评分、加载/回收成本。先明确资源状态与目标函数，再决定最小改动或测量方案。

不要直接实现大规模动态调度。当前 spec_stride 是统一固定配置（实验 4，源码 CLI 默认 16），第一轮 initial_idx 可导致变长；每请求每轮动态 k 未实现。未来改动要同时检查验证长度、状态转移、EOS 检查、采样缓冲和 kernel 支持上限。

先从项目已有 PDF 查 VeriCache 的精确调度算法，不要把我们的建模想法当成论文原算法：根目录已有 VeriCache PDF 和 SparseSpec PDF。用户不希望重复提供资料。

## 8. 阅读注释中的已知问题

提交保留了用户的阅读笔记，不代表每条注释都已审校。`spec_scheduler.py` 中 `all_kv_len_arr` 上方“按 token 存储”注释不正确，它实际每请求追加一个完整 KV 长度；`allocate_tokens` 上方关于修改长度与分配顺序的注释也不够准确，应以实现为准。后续若整理注释，再与用户沟通小范围修正。
