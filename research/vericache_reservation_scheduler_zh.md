# VeriCache 风格预约表调度（2026-10-26）

## 目的与入口

在当前 SparseSpec 原生 compact 热冷 KV 实现上，增加可选的 BW/HBM 双环预约策略。默认路径不变；入口为 `serve.run.run_inference(compact_reservation=True, ...)`，CLI 对应 `--compact_reservation`。

参考 [VeriCache §5 与 Algorithm 1](https://arxiv.org/html/2605.17613v1#S5)：围绕目标 draft 长度搜索可行的未来 Verify 窗口，提前发起恢复。这里实现的是在本仓库布局上的策略移植，不是 VeriCache 官方 vLLM/LMCache 系统的完整复现。

## 文件与改动

|文件|职责|
|---|---|
|`serve/scheduler/reservation_table.py`|纯 CPU 双环；窗口搜索、未来释放事件、依赖、物化和级联取消；绝对迭代号防止绕回污染|
|`serve/scheduler/reservation_scheduler.py`|预约准入、提前恢复、到期 Verify、每轮冷 KV 卸载、真实回收确认和预测修正|
|`serve/request/kv_cache_ptr/compact.py`|允许恢复期间追加热尾部；repack 期间仍禁止追加|
|`serve/run.py`|新增独立开关、参数校验、调度器分派|
|`serve/tests/test_reservation_table.py`|双环容量与随机操作测试|
|`serve/tests/test_reservation_scheduler.py`|阶段、恢复期间追加、D2H 预约和临时空间测试|
|`serve/tests/validate_reservation.py`|通过公开入口执行真实 TP2 验证|

## 两张表如何记账

- BW 环的单位是秒，单个窗口预算为 `reservation_iteration_ms / 1000`。
- HBM 环的单位是原始 KV pool 页。模型、索引和工作区不重新分配进该环；它只管理 pool 内额外恢复页，其他容量限制继续生效。
- 单次传输时长为 `实际冷 KV 字节 / 有效每 rank 带宽`，跨度取迭代时长的向上取整。前面的窗口充满，最后一个窗口记剩余时长。
- 预约跨度内保留完整目标页数。当前恢复一次性分配全部目标页，不实现边传边分配。
- 候选顺序为目标、目标-1、目标+1、目标-2、目标+2，限制在 `[传输跨度, horizon-1]`。
- H2D 恢复和本实现每轮 D2H repack 使用同一个串行 BW 环。repack 没有额外 HBM 环占用，源页与新热页临时空间已经在实际驻留/临时空间中计算。
- 表内负编号 `-request_id-1` 专用于 repack；快照带 `kind`，事件日志还原为真实请求编号。
- 成功提交恢复、实际分配目标页后，从 HBM 预约中扣掉对应页，改由实际驻留统计负责，防止重复计数。BW 预约仍然保留。
- 实际页分配器只认已经释放的页；预约表允许借用**已预约 repack 的预计释放容量**，但只有前置释放被确认后，依赖它的恢复才能实际分配页。没有预测未知的 EOS 或尚未确定的 Verify 结果。

## 未来空闲容量（本轮扩展）

对未来逻辑窗口 `t`，表内使用以下页数模型：

```text
预测已用(t) = 当前实际驻留 + 已知增长保护 + repack 工作区保护
             + 未物化的恢复预约(t) - 尚未提交的计划释放(t)
预测空闲(t) = pool 总页数 - 预测已用(t)
```

这里的增长保护来自已知输入与有界 Draft 窗口，不是所有请求的最大输出长度。恢复成功提交后，目标页已经计入实际驻留，相应 HBM 预约扣除；释放提交后，实际驻留已经降低，相应释放预测不再扣第二遍。

一次 repack 在 BW 表中占好窗口后，登记 `release_at = deadline + 1`。未开始时，其预计净释放量为 `原源页 - 新热页`。开始后，新热页已分配并进入实际驻留，此时释放事件改记**整个旧源页数**，避免重复计算临时热副本。只有所有 TP rank 的搬运完成、旧源页真正回收后，才提交释放事件。

恢复预约记录支撑它的释放事件编号。因为当前 H2D 一次分配全部目标页，只能使用预计在恢复**起点前**可用的释放，不能借用传输中途才腾出的页。执行时同时检查依赖提交与真实空闲页数，不超额分配。

例如 pool 为22页，A当前热页5页、待恢复冷页7页，B全量驻留12页。现在只空闲5页，A不能恢复。若B计划变成5页热KV，其 repack 净释放7页，A就能先预约到这个释放之后。B搬运期间源页仍锁定；B提交后A才分配7页开始恢复。这个案例由策略单元测试直接覆盖。

预测修正规则：

- repack 延期：推迟释放时刻，撤销依赖它且尚未开始的恢复，重新搜索窗口。
- repack 取消：级联撤销依赖预约；事件使用单调递增编号，防止下一轮同一请求误用旧释放。
- 实际分配或驻留时间与预测不同：以实际驻留重新检查未来窗口，撤销有冲突的未开始恢复；在途搬运和计算使用的页不回收。
- 已提交的释放保留依赖关系直至不再引用；它不再充当未来容量信用。全部请求完成后表、依赖事件与 pool 占用都必须清空。

这仍是保守的容量可行性预测，不是未来完成时间的保证，也没有新增任意受害请求选择或分块换入。

## 一个请求的执行流程

1. 新请求仅按已知输入和一个最大 Draft 窗口保护 KV 增长，另保护压缩重排临时页与已有恢复需求。不按整个输出上限做聚合准入预留。
2. 多个请求可一起推进 chunk prefill；受总 query budget、元数据容量与 pool 页限制。
3. 第一次 selection 前仍是全量 KV Draft，这是当前 SparseSpec 的启动流程。此时预约的额外恢复字节为零。
4. 有了 selection 后，先预约 repack 的 D2H 窗口，开始后停住该请求，等所有 TP rank 的搬运事件完成，再切换到 hot 布局。
5. 为下一轮预约 H2D/Verify；可利用其他已排入 BW 表的 repack 将来释放的空间。未找到合法窗口的请求保留原状态等待，不继续产生无界的草稿。
6. 到达恢复起点、依赖释放已提交且真实空闲页足够时，分配冷 KV 目标页、发起异步恢复；不会因为预测未来有空位就超额分配。
7. 恢复期间该请求可继续 Draft。冷数据写入新的目标页；Draft 使用不变的热页和独立追加的尾部。repack 期间仍不能执行它。
8. 到达 Verify deadline 或到达原生最大 k 后停止 Draft。受 deadline、原生 query/metadata 限制和全 TP 恢复完成事件约束，才实际执行 Verify。
9. 沿用原生 rejection、修正 token 和 selection。Verify 的异步 STALL 结果处理结束后，取消上一轮预约，再进入下一轮 repack/预约。
10. 请求结束取消剩余预约，GPU 页继续经过现有延迟退休机制回收。

若预测 deadline 超过了最大 `spec_stride`，请求会在原生 k 上限停止 Draft 后等待；不会增加未分配的概率或索引缓冲。若 deadline 较早，使用实际 `len_draft_probs` 调整 `initial_idx`，不丢弃已经生成的草稿。

## 不确定性与安全边界

- `reservation_iteration_ms` 是估计值，混合 batch、CPU 开销、真实链路速度与既有恢复 kernel 干扰均会导致超时。deadline 不是实时完成保证。
- 尚未开始、已超期的预约取消后重新搜索；已经开始的恢复不取消、不重复传输。等事件完成后再执行 Verify，并统计 deadline miss。
- 实际执行始终保持最多一个 KV 事务在途，预测错误不会突破实际链路串行约束。
- 没有计算请求时，调度器可将空的逻辑窗口快进到下一预约事件；统计 `reservation_empty_windows`，不把它解释为真实墙钟传输时间。
- CPU 冷 KV 保留有效副本；本策略不隐式采用 CPU LRU、重计算或热 KV 卸载。CPU 容量不足先阻止 repack/新准入，无法推进时明确失败。
- 任一配置不能推进时不会隐式减 batch、减 r 或回退其他策略。
- 原生 attention、采样/rejection、Graph 和 CPU pipeline 的实现继续复用。没有把之前的 carveout 诊断控制自动并入生产默认项。

## 参数

|参数|含义|
|---|---|
|`compact_reservation=True`|选择此策略；要求 compact KV、CPU async、transfer async|
|`max_num_batched_tokens`|显式 query 总预算，至少 `spec_stride+1`|
|`reservation_bandwidth_gbps`|必填；每 TP rank 的有效十进制 GB/s，需按同时执行的 TP 环境测量|
|`reservation_iteration_ms`|必填；所有 TP rank 使用相同的迭代时长估计|
|`reservation_horizon=64`|两张环形表的窗口数|
|`reservation_target_length=None`|搜索锚点，默认 `spec_stride`，必须在 `[1,spec_stride]`|

不与 `compact_early_verify_length` 混用。`compact_dynamic_admission` 不必开启；本策略独立复用其 token budget 与缓冲计算。

## 日志与验证

输出目录内 `reservations.rankN.jsonl` 记录 reserve、repack_reserve、reload_start、repack_start、release_commit、release_delay、cancel、deadline_miss；包含依赖编号。`compact-metrics.rankN.json` 记录预约、延期、等待、传输与页回收指标；周期 transfer timing 报告同时包含未来空闲页曲线和释放事件快照。

本次验证目录为 `log/vericache-reservation-20261026/`。其中：

- `cpu-tests.log`：新增表与策略测试。
- `regression-tests.log`：原有所有权、动态准入、异步传输相关回归。
- `eager-tp2/`：补入 D2H 预约前的第一次集成检查，仅作开发过程记录。
- `eager-final-tp2/`：最终 eager 路径验证。
- `graph-pressure-tp2/`：最终 Graph 路径，缩小 pool 验证容量受限时推进。

这些短测试使用原始 token 化的四条 LongBench 输入，输出上限128；不改变 prompt，不用于任务判分或正式吞吐比较。4 GB/s、20 ms 是功能测试参数，尚未证明是校准后的最优配置。测试检查完成数、完整卸载恢复循环、两 rank 状态与所有页回收；不宣称不同调度轨迹的文本逐位一致。

CPU 测试命令：

```bash
SSPEC_VENV=/data/users/songshiyi/vericache/.venv-vllm011 source repro/activate.sh
python -m unittest serve.tests.test_reservation_table serve.tests.test_reservation_scheduler -v
```

确认 GPU4/5 空闲后，复跑已保存的 Graph 验证：

```bash
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.run \
  --standalone --nproc-per-node=2 serve/tests/validate_reservation.py \
  --worker-payload log/vericache-reservation-20261026/graph-pressure-tp2/worker-payload.json
```

每份 payload 保存完整参数与输入，`validation.rankN.json` 保存执行结果与源码 hash。正式性能对比前应固定源码与输入版本、校准链路/迭代参数，并保持相同采样条件。

未来容量扩展的验证单独保存在 `log/vericache-future-capacity-20261026/`，不覆盖上述旧版本结果。该目录的 `findings_zh.md` 汇总最终验证与限制；`graph-tp2/` 是修正临时热副本记账前的开发记录。
