# 预约搬运队列：原因 B 的第一版修复

本版沿用 `reservation_iteration_ms` 作为近似 forward 周期，不新增周期稳定性实验，也不修改 k/r、接受长度预测或 resident 强制卸载策略。

预约模式默认开启 `reservation_transfer_queue`。每次准入时提交所有已经到达预约起点、满足释放依赖和真实内存容量的恢复/重排事务。同一 transfer stream 串行执行多笔请求，不再等待下一次调度才提交同槽第二笔事务。只预提交当前已 ready 的事务，不提前执行未来时隙。

每请求仍限一笔布局事务；恢复目标和重排热页工作区均在提交时真实分配，CPU backing 计入在途容量。源页、CPU 数据和搬运计划由每笔 job/transfer 保活。完成检查继续跨 TP rank 协调；仅事件全部完成后发布该请求布局和确认释放信用。空 batch 等待第一笔在途事务后重新 poll。提交失败沿用逐事务协调与回滚，在途其他事务不因预约取消而释放。

这实现搬运执行与 forward 的部分解耦：GPU 可以在 CPU 提交模型期间连续搬运；目标页分配、页表 commit、依赖解除和后续预约仍由调度线程推进。尚不包含后台线程、实时自适应时钟、未来释放前预分配或原因 A 的同请求 repack/Draft 重叠。

## 开关

- `serve/run.py` 默认启用；使用 `--no-reservation_transfer_queue` 恢复原单事务模式。
- `experiments/diagnostics/component_profile.py` 默认启用；使用 `--no-reservation-transfer-queue` 做旧执行器对照。
- profile 配置及 scheduler timing snapshot 记录该开关。现有 `max_transfers_inflight` 包含已排队事务，不代表物理并发拷贝数量。
- 传输 Event 区间不包含前序事务完整排队时间；host submit 到 commit 包含排队、传输与完成观察延迟。

## 回归

新增同槽两笔 repack 一次入队、未启动伪过期消除、完成事件独立检查、每请求重复提交拦截、两笔 restore 的实际目标分配与最终页回收，以及旧模式开关测试。原有释放依赖用例继续约束“未来释放不能代替真实空闲页”。

本机预约表/策略 26 项测试、诊断 6 项测试通过。真实 CUDA/TP 执行和吞吐尚未验证，当前环境无可用 CUDA runtime。扩大原生 compact CPU 测试出现两项容量边界失败，已在 `/tmp` 导出的未修改 HEAD 基线上复现：`test_constructor_rejects_unsafe_recent_and_score_tile` 抛出 `KeyError: max_batch_size`，`test_native_metadata_limits_are_strict_and_checked_before_dispatch` 未抛出预期 ValueError；不将此版声明为全部测试通过。
