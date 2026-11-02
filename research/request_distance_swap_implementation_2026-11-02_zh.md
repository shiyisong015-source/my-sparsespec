# 距下一次 Verify 的历史页交换：TP1 实现

本版是独立的 `distance_swap` 策略；保留预约版与旧 on-demand 路径。按 [设计稿](request_distance_swap_design_2026-11-02_zh.md) 中已纠正的下一次 Verify 距离排序，不需要预约时隙、带宽标定或接受长度预测。两项真实 GPU 存储数据测试已通过，Qwen3-8B TP1 推理与三窗采集已跑通；性能结论以预算核对后的正式报告为准。

## 实现入口

| 文件 | 职责 |
|---|---|
| `serve/scheduler/distance_policy.py` | 请求距离/阶段排序，防止同优先级来回替换，未来释放量的唯一 claim |
| `serve/scheduler/distance_swap_scheduler.py` | 下一次 Verify 距离、真实准入、原生计算流水线、selection 和退休交接 |
| `serve/scheduler/distance_swap_engine.py` | 独立工作线程，事件轮询、交换计划、双向分块推进和资源变化后的计划修订 |
| `serve/request/kv_cache_ptr/distance_swap.py` | 稳定 Draft 页、按历史顺序的 CPU backing、分块保存/释放/恢复，以及无空闲热页工作区时的 bootstrap kernel |

## 请求排序

策略是：优先替换距下一次 Verify 更远的冷 KV，保留 Draft 下一步必用的热 KV。距离只衡量“这个请求还要执行多少步 Draft 才进入下一次 Verify”：

```python
distance = 0 if status == VERIFY else spec_stride - cur_spec_idx
```

- VERIFY 距离为 0；DRAFT 按自身剩余 Draft 步数计距，恢复从近到远、卸载从远到近。
- 使用 `cur_spec_idx`，不能用 `k - len_draft_probs` 替代，因为初始 speculation offset 可能不为 0。
- 热 KV（selected + recent/tail）供下一步 Draft 使用，独立保护；距离排序只决定仅供 Verify 使用的历史副本如何替换。它不是对整个请求的热/冷驻留集统一排序。
- 同距离卸载仍保留 VERIFY 优先的平局规则；原生稳定状态下 VERIFY=0、DRAFT>0，因此跨阶段平局通常不出现。同距离同阶段按等待/最近服务顺序和 request_id 确定顺序，防止对等请求反复交换。
- query、metadata、batch 上限和显存容量继续决定实际计算可行性，但不改变这个距离。全局迭代推进、其他请求的计算和传输轮询也不减少该请求的剩余步数。
- NORMAL/STALL 不参与此排序；selection/结果消费完成后重新发布候选。

例如 k=8：A 已进入 Verify，距离 0；B 的 `cur_spec_idx=2`，距离 6；C 的 `cur_spec_idx=6`，距离 2。优先恢复 A，优先替换 B 的历史页，保护 B/C 的 Draft 热页。

传输线程在 H2D 分块完成后重新比较最新候选。如果更近的 Verify 恢复可行，撤销原计划未提交部分并切换；已提交的块继续保活。下传前也再次验证牺牲者优先级，避免其 Draft 已推进到更近位置后仍按旧计划换出。重排中已分配的 H2D 目标计入真实驻留，不能再次要求等量未来释放。

本版不使用接受长度预测或队列中下一次任意 forward 的批次位置作为距离。日志和 timing snapshot 的 `distance_scope` 标记新口径，不能把旧影子批次距离与新距离混用。

## 为什么用了新的页表示

原 compact 的每层/head selection 可以来自不同物理页。不能简单把“一个 cold slot 已保存”当成“对应原物理页已无人使用”，也不能在原 `commit_pack()` 中逐块释放，因为它会整体替换热页。

本版在每次 selection 后为选中的前缀创建独立 GPU hot 副本，保留 recent/tail 的原页；Draft 的这部分视图在历史页下传期间不变化。完整历史前缀按原请求顺序保存到 pinned CPU 分块，每个块覆盖所有层/head，完成事件就绪后才释放该块的原物理页。

代价是 CPU backing 包含被选中前缀的历史副本，GPU 同时保留独立 selected 页；不是旧 compact 的精确 cold-only 搬运量。比较时必须报告真实字节量与 GPU/CPU 页峰值。

没有空间先建 hot 副本时，可先保存历史块，再从 CPU 分块构建 per-head hot 页。该请求在 hot 就绪前暂停计算，其他请求仍可执行。正常情况下 hot 就绪后的历史下传可以和同请求 Draft 重叠。初始 dense Draft 尚未产生稀疏 selection 时，其全部历史被保留为必需视图，不能被误判为可释放的 cold 前缀。

## 交换与未来显存

选中接收者后，先用真实空闲量与选定牺牲者的分块释放量形成计划；即使当前空闲为 0 也能建立交换计划。每个 `(request_id, generation, chunk_offset)` 释放来源只能被一个计划认领。

H2D 和 D2H 使用两条 stream，各自最多保持一个普通块在途。下传一个块完成后，立即释放源页，给上传的后续块使用；无需等待整个牺牲请求下传完。恢复的完整目标不在计划创建时一次性分配，实际分配发生在每块提交时。热页初始化也使用 H2D stream，不与普通 H2D 块交错初始化。

计算追加页有单独 headroom。计算准入、hot 分配或新请求导致资源变化时，计划补充合法牺牲块；不能满足则撤销未提交计划。已提交 chunk 始终保活，不能随取消回收。全量已驻留的 Verify 直接参与计算。

请求恢复后有 service pin；计算封存和 Verify 保护完整视图，直到 CPU rejection/selection 发布下一轮，而不是 GPU event 一完成就解除保护。Draft 租约保护稳定的 hot/recent/tail，允许不再被它读取的历史页继续释放。完整 Verify 必须等所有历史块完成恢复和页表发布。

## 线程与清理

主线程继续负责 token、原生采样/rejection、执行阶段和 selection。后台线程负责选择搬运、提交分块与查询完成事件；运行在自己的 CUDA device/inference context 中。

共享 RLock 串行保护池分配、请求视图和计划；模型 forward 的 CPU 提交不持有该锁。第一版 pinned backing 分配和 GPU 搬运 launch 在锁内进行，锁内不显式等待 CUDA event；这仍可能造成 host 侧锁竞争，不声称完全消除了 CPU 耦合。

计算流水线增加默认空操作 hook：计算提交通知、正常结束排空搬运、异常清理前停止生产者。旧策略默认行为不变。退休等待 compute、control、所有块传输以及旧 hot 页的消费事件；worker 停止并 join 后才同步搬运 stream、关闭请求，避免后台在清理之后继续提交。

首版只支持 TP1 eager。TP2/多 rank 与 CUDA graph 明确拒绝启用；未实现设计中的 leader 广播与独立后台 control group。

## 使用

常规 launcher 在原 compact TP1/async 参数上启用：

```text
--compact_distance_swap --distance_swap_chunk_pages 256
--kv_cache compact --compact_async_transfers --async_cpu
--max_num_batched_tokens 2048 --tp_ranks 1
```

不同时启用 `compact_reservation` 或 `compact_dynamic_admission`，也不传预约带宽参数。单请求配置检查要求其完整 Verify 与独立 Draft 视图能放入池。多请求准入按当前 prefill chunk 的真实空闲页、query、metadata 和 batch 上限决定；已按用户要求移除“所有请求最终热预算之和＋一次最坏完整 Verify”的聚合容量保护，不再为 32768 最终上下文提前压低准入数。后续增长和恢复按真实页与分块交换推进；不靠 replay 或隐式改变 r 获得进展，也不保证任意超额准入组合都能持续前进。

已有固定输入和模型后，组件脚本可直接选择：

```bash
CUDA_VISIBLE_DEVICES=0 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
python experiments/diagnostics/component_profile.py \
  --model /your/models/Qwen3-8B --input .repro/profile-input.json \
  --output log/distance-swap-smoke \
  --policy distance_swap --distance-swap-chunk-pages 256 \
  --requests 128 --batches 32 --pool-pages 98304 \
  --buffer-batch-capacity 128 --query-budget 2048 \
  --output-tokens 32 --warmup-seconds .5 --record-seconds 2 \
  --record-windows 3 --window-gap-seconds 1 \
  --minimum-forwards 1 --window-check-steps 1 --detail-every 1
```

这是开发短测，不是自然 EOS 吞吐。无显存压力时允许没有搬运，不要求发生 restore 才开窗。要覆盖交换路径需另做较小 pool 的压力用例，并记录实际准入数和失败，不能假定 batch 上限等于实际 batch。

`distance-swap.rank0.jsonl` 记录整次运行的计划、未来页来源、分块提交/完成与 CPU 时间戳，附带 `window_id`。`swap_d2h_pages`、`swap_h2d_pages` 等为新策略计数，旧 `migrations/restores` 不适用。组件脚本在采样窗口记录逐块 CUDA Event 和 selected-view gather 耗时；Event 在生产者依赖之后开始，仍是 GPU 操作跨度，不能当作纯 PCIe DMA 耗时。host submit-to-observe 是另一个时钟域。

正式 profile 可使用 `--mem-fraction .9 --runtime-reserve-bytes <本机实测值>`，以 `floor((CUDA 可用总显存 × .9 − 原生模型预算 − 运行缓冲预留)/页字节数)` 定容。预留须覆盖元数据、采样/模型临时张量、allocator 和采集器开销；两侧取同一个较大值，结束后核对实测峰值。固定页数开发短测不等于 0.9 总预算实验。`--windows-only` 在全部窗口完成后安全排空并回收页，未完成请求保持未完成，不伪造 EOS。

`pcie_isolated_exec.py` 按 sysfs PCIe root 加锁并检查整个 root 的占用，绑定本地 CPU 和 NUMA 内存；不同 GPU 编号本身不能保证传输互不争用。正式对照使用不同 root 的 GPU0 与 GPU4，并监测其同域 GPU 的外来进程。

## 已做检查

- 最新相关回归共 88 项：86 项通过，2 项 GPU 数据测试在 CPU 检查中跳过；包括新策略、预约表/策略和原生 async/dynamic 回归。两项 GPU 数据测试此前已在真实 A100 上单独通过。
- 覆盖距离偏移、原生 NumPy bucket 索引、距离不受 query/全局迭代影响、块边界切换、旧牺牲者失效、在途目标页保活与取消聚合准入预留；诊断脚本 14 项 CPU 测试通过。
- 新 bootstrap Triton kernel 已离线编译为 sm_80 PTX；stride 使用运行时参数，物理地址计算使用 64 位，避免不同历史长度反复专门化和大 pool 地址溢出。
- 组件脚本 CLI 的 `distance_swap` 和 chunk 参数解析检查通过；Python 编译与 diff whitespace 检查通过。

两项此前跳过的 GPU 数据测试已在 A100 上通过，覆盖分块搬运的数据保真和无空闲 hot 工作区的 bootstrap。真实模型短测和长窗口运行均完成了池页回收核对；吞吐与重叠效果需结合 CUDA/NVTX、输出计数和真实字节量判断。完整原生 compact CPU suite 的两项已知基线容量测试失败仍见前一次交接记录，未纳入此次通过的定向测试集合。
