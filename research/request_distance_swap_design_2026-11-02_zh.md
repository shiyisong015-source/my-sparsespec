# 请求距离驱动的计算/传输解耦设计

状态：设计已按用户确认纠正为“距下一次 Verify 的剩余 Draft 步数”。TP1 首版已实现，具体覆盖、表示方式调整和未完成项见 [实现记录](request_distance_swap_implementation_2026-11-02_zh.md)。下文保留完整设计目标，不能把其中的 TP、多线程协议等目标都视作已实现。

## 1. 已确定的策略

调度对象是请求中仅供 Verify 使用的历史页。Draft 下一步必用的 selected + recent/tail 热页保留，不纳入可替换候选；在这个前提下比较各请求距下一次 Verify 的距离。

- 恢复：优先距离近的请求。
- 卸载：优先距离远的请求。
- 卸载距离相同：优先 VERIFY，再 DRAFT。
- 同距离同阶段：先卸载最近被服务过的请求；仍相同时按 request_id 决定，保证可复现。
- 恢复同距离：先服务等待更久的请求，再按 request_id；不擅自增加冷热优先或“小请求优先”。

计算、上传、下传分别推进。距离用于选择和替换，不是承诺某个时刻完成的预约。实际数据依赖仍由 CUDA event 和页所有权保证。

## 2. 距离的具体定义

距离统一定义为“该请求还需执行多少步 Draft 才到下一次 Verify”。原稿的影子批次/下一次任意 forward 位置定义已撤销。

| 状态 | Verify 距离 | 页与执行约束 |
|---|---|---|
| DRAFT | `spec_stride - cur_spec_idx` | 热页供当前 Draft 使用；历史副本按未来 Verify 距离替换/预取 |
| VERIFY | 0 | 完整历史恢复并发布后才能计算 |
| STALL | 不生成候选 | 等待 rejection/selection 消费与下一轮发布 |
| NORMAL | 不生成候选 | 沿用真实 prefill 准入限制 |

k=8、`cur_spec_idx=6` 的请求距离为 2，即使它刚执行过一个 Draft 步且 `len_draft_probs=1`，距离也不是 7，因为原生初始 offset 可能为 5。必须依据 `cur_spec_idx`，不改初始 offset、原生 k 边界、采样或 rejection。

只有该请求实际推进 Draft 才会缩短距离；其他请求 forward、CPU 调度空迭代和传输事件不会让它自动倒计时。query/metadata/batch/内存约束只决定实际能否执行或搬运，不再用于构造距离。

同距离优先卸载 VERIFY 的平局规则保留为次级规则。在原生稳定状态下 VERIFY 距离 0、DRAFT 距离大于 0，因此常见平局发生在同阶段请求之间，不能为了产生跨阶段平局而人为改变距离。

热页的立即使用优先于 Verify 历史副本；只有历史副本因下一次 Verify 的远近参与当前交换策略。本版不把已被 compute lease 保护的热页或完整 Verify 视图当作免费页。

## 3. 选上传对象和卸载对象

恢复候选：有有效 CPU backing、缺少下一次 Verify 需要的历史页、未退休、没有冲突事务的请求。全量已驻留的 Verify 直接进入计算，不制造搬运。

```python
restore_key(r) = (distance(r), waiting_since(r), request_id(r))
victim_key(r) = (-distance(r), 0 if phase(r) == VERIFY else 1,
                 -last_service_sequence(r), request_id(r))
```

`waiting_since` 是首次进入本次待服务队列的序号，不在每次重排时重置。Compute lease、控制流尚未消费的结果、CPU 容量以及事务冲突首先决定是否为合法候选，再应用上述排序；这些约束不能通过降低距离绕过。

每次取最近的恢复候选，使用现有空闲页和确定释放量形成计划。不足则按 victim_key 从远到近选择牺牲者，直到覆盖本次恢复与必要工作区；一个上传对象可以对应多个牺牲者，一个牺牲者也可提供多批页。

替换只在接收者比牺牲者更应该驻留时执行：接收者距离更近，或距离相同且接收者 DRAFT、牺牲者 VERIFY。同距离同阶段的稳定排序只用于从多个合法牺牲者中选择，不足以反复交换双方。接收者的当前服务视图取得 service lease，直到一次实际计算完成或该服务被显式撤销，避免刚恢复就按平局规则被卸载。

若所有可选对象都被保护或更应驻留，等待相应真实事件；不得自选一个更近的对象强行换出。按用户后续要求，多请求准入不预留最坏完整 Verify 与所有请求最终热预算，而按当前 prefill chunk 的真实可分配页准入。多 lease 占用、后续增长和 CPU 容量仍是运行时约束，不能声称任意准入组合都不存在无进展情况。

## 4. 未来显存与交换计划

不要求生成计划时已有全部目标页。不再依赖固定 BW/HBM 时隙；改用具体页和事件组成的释放依赖图。

```text
SwapPlan
  plan_id / plan_generation
  receiver_request / receiver_epoch / service_sequence
  required_pages
  victims[(request_id, epoch, chunk_ids)]
  free_page_claims
  future_page_claims[(producer_chunk, page_ids)]
  append_and_workspace_reserve
```

页的未来归属必须唯一。一个 future_page_claim 不允许同时支撑两个上传对象；一次 D2H 完成的释放信用也不重复计入全局空闲页和接收者专属页。

| 未来容量来源 | 能否用于计划 | 能否立即覆盖物理页 |
|---|---|---|
| 当前未占用且未被 claim 的页 | 可以 | 可以 |
| 已提交 D2H，compute fence 已明确的源页 | 可以 | 等该页 D2H 与计算依赖完成 |
| 已选中、已锁定源页及 CPU backing 的待提交 D2H | 可以，计划内条件信用 | 先成功提交并完成 D2H |
| 预计 EOS、预计接受长度、尚未选定的未来卸载 | 不计硬信用 | 不可以 |

未来账本不仅计算总页数，还记录释放依赖。没有时长预测也能判断依赖完成后的容量是否满足；它不保证依赖在某个墙钟时间发生。计划必须有能够启动的首批操作，不能 A 等 B 的页、B 又等 A 的页。发现依赖环时调整牺牲者，或使用专门保留的少量 bootstrap 页。

## 5. 分块交换与 Draft 并行

第一版独立 H2D、D2H stream；是否获得物理双向重叠由设备决定，正确性不依赖双向并发。

```text
D2H:  B.chunk0 -------- B.chunk1 -------- B.chunk2
              event0            event1
H2D:  A.free  A.chunk0 ---------- A.chunk1 --------
Compute:       其他 ready Draft / Verify 持续执行
```

同一物理页遵循 `最后计算读/写完成 -> D2H 完成 -> H2D 覆盖`。不用等待整个 B 下传完才上传 A，但绝不在该页数据尚未保存时覆盖。完成事件驱动下一批操作；不在 CUDA host callback 中调用 CUDA API，用传输工作线程查询事件并唤醒调度。

冷热分类只在请求选定后决定如何形成正确的搬运内容，不参与请求排序：

- DRAFT 请求卸载后，尽量保留其当前 Draft 必需视图，只下传非必需部分。选中的热页保持原物理位置，避免当前 repack 复制热页带来的停顿与临时空间。
- VERIFY 请求若是合法牺牲者，可以下传完整恢复集；若已有同 epoch 的有效 CPU 副本，不重复下传不变页，只补齐未备份的 recent/tail。不能使用过期副本。
- 要进一步卸载 DRAFT 必需视图属于请求整体暂停，需要新的挂起布局与恢复逻辑；首版只在明确支持后开放，不能把保护中的页当成可释放量。
- 同请求 Draft 与非必需页下传可并行。Draft tail 使用独立追加页，迁移仅处理冻结的历史页集合。后续追加不能被旧快照 commit 覆盖。
- 恢复可边进行边 Draft，只要当前 Draft 页未变化；完整 Verify 必须等待所需页全部恢复并发布。

现有 `begin_pack()/commit_pack()` 会复制热页并整体释放旧 indices，不能直接复用来实现该分块协议；`begin_restore()` 的一次性目标分配也要拆开。

## 6. 状态、页表与 epoch

请求的执行阶段与驻留状态分离，不用一个 phase 字符串混合表达二者。

```text
ExecPhase: NORMAL / DRAFT / VERIFY / STALL
Residency: 每个逻辑 KV 页的 GPU 位置、CPU 副本版本、在途状态
Transaction: plan_id / request_epoch / frozen_pages / pending_chunks
Lease: compute read/write / service / transfer source / future destination
```

VERIFY -> rejection -> selection 为新的 epoch 边界。选择、下传、恢复命令携带 epoch 和计划 generation；旧事件仍需释放其资源，但不得发布到新页表。选择器索引 buffer 必须按 epoch 保活，不能在 D2H 尚读取时覆盖。

请求页表使用稳定逻辑索引到物理页映射。Draft 在捕获 metadata 时获取 compute lease 和页表版本；已提交 metadata 指向的页直到对应 fence 完成均不可回收。传输侧允许更新不被 compute 使用的映射；活跃视图切换在下一批 metadata 捕获边界原子发布。任何已恢复完整的 Verify 都必须同时匹配执行 epoch、完整覆盖和无冲突布局变化。

## 7. 两个执行者的所有权

引入 TransferEngine 工作线程，让事务完成后能在 CPU 正提交 forward 时继续选对象、规划和发起下一批搬运。计算线程不等待一次完整交换结束才提交其他 ready 请求。

使用共享 ResidencyManager 作为页分配、claim、lease 和页表的唯一入口；现有无锁 `KVPool._free_gpu` 不能由两个线程直接访问。短临界区串行化账本更新，不在锁内等待 CUDA event、执行 GPU 拷贝或 TP collective。

- 计算线程：执行阶段、token/prob 状态、原生 sampling/rejection/selection；发布带 generation 的不可变候选快照。封存 batch 前通过 manager 获取页租约和追加额度。
- 传输线程：距离候选、SwapPlan、CPU backing、分块提交和完成检查；只通过 manager 管理驻留元数据，不直接修改 Request.status、token_ids 或 draft_probs。
- ResidencyManager：拥有页账本与布局版本。产生不可变 ReadyView，计算线程捕获时验证 generation 并取得 lease。
- 事件队列：发布 forward 完成、结果可消费、新 selection、传输完成、请求退休和失败。距离由已发布的该请求 Draft 进度更新；其他请求的 forward 或传输轮询不推进该距离。读到过期快照时重新规划未提交部分。

经理器锁之外的提交采用 `PREPARED -> SUBMITTED -> COMPLETE` 协议。PREPARED 已取得独占 claim 和源 lease；提交失败时撤销未提交的 claim，已经提交的 chunk 继续保活至完成。不能简单解锁后再次查看同一 request 并假定布局没有变化。

## 8. TP、取消和退出

TP1 先实现，但接口预留单一 leader 规划：TP leader 发出统一 plan/chunk 序列，其他 rank 按同序执行；所有 rank 目标分配成功后才统一提交。完成掩码跨 rank 汇合后推进释放或发布，不能各 rank 按本地事件快慢独立选下一请求。

传输协调使用专用 control group 和固定消息序列，不在后台线程与现有计算线程交错使用同一个 collective 序列。所有 rank 的启动、分配失败、停止和 drain 均须覆盖。不要只把现有 `_poll_transfers()` 移到线程里。

取消未提交计划：撤销其独占信用，并重排依赖它的计划。已提交 chunk：保留 source、destination 和 CPU backing，待事件完成后再回收或交接。退休请求不再接新事务；等 compute、control、H2D/D2H 所有 lease 归零后恰好回收一次。异常退出停止新提交、排空可完成事务、通知其他 rank，保留真实失败信息。

## 9. 一个排序例子

假设 k=8，A 已到 Verify 且缺历史页，距离 0；B 为 Draft、`cur_spec_idx=2`，距离 6；C 为 Draft、`cur_spec_idx=6`，距离 2。

1. 优先恢复 A。
2. 优先选择 B 的历史副本作为卸载来源，其 selected/recent/tail 热页保留供 Draft 使用。
3. B 的可释放量不足时，再考虑 C 的历史副本。
4. 不要求整请求下传完；某块完成后即可为 A 后续上传提供目标页。
5. B 的 Draft 推进后，下一笔搬运重新比较其最新距离；不能继续使用已经失效的牺牲者顺序。
6. 已提交 chunk 保活到完成；更近的请求恢复可行时，在 H2D 块边界切换未提交计划，不因为不可行的新候选反复取消现有可行预取。

## 10. 实施拆分与检查

作为新策略 `distance_swap`，保留现有 reservation 和 on-demand 路径供复跑。每阶段保持 native sampling/rejection 和 k/r 不变。

1. CPU 纯策略模型：DistancePolicy、请求优先级、交换计划、未来页唯一 claim、依赖环检查、lease 与取消。先固定合成事件验证规则，不引入接受长度预测。
2. 页级迁移原语：稳定热页、CPU 版本、分块 D2H/H2D、tail 合并和完整 Verify 发布。先用可控事件和数据校验验证页所有权。
3. TP1 TransferEngine：接入真实计算快照与独立工作线程，检查 Draft 与非必需页下传重叠、零空闲页启动和线程退出。
4. TP 多 rank：leader 统一计划、独立协调 group、不同 rank 事件延迟及分配失败处理。
5. 同条件运行对照，记录实际 D/V/P、有效输出 token/s、交换量和计算等待；不把策略启用本身当成性能证明。

关键回归：距离优先于阶段；同距离优先卸载 VERIFY；无效未来信用不得重复使用；全池占用时通过分块交换推进；冻结页不被覆盖；tail 追加与 selection epoch 正确；新候选到来只重排未提交工作；CPU 容量不足和多 lease 造成的无进展显式暴露；取消、EOS、异常后无泄漏或重复释放；TP 不同完成次序不改变计划和发布顺序。

观察项：候选距离与阶段、每次牺牲者选择原因、plan/chunk lineage、已 claim 的未来页、传输队列长度、event 到下一次提交延迟、compute 等页/传输等页、lease 阻塞、反复交换次数与字节。观察不得每步同步设备或复制大张量。
