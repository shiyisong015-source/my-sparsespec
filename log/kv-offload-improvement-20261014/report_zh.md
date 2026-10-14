# Compact KV 卸载/恢复：修正与验证记录（2026-10-14）

## 1. 当前结果

已修正 verify 独占 forward 导致其他请求暂停、默认 CUDA device 下的 CPU 元数据分配、准入重复计入退休 full KV，以及 growing replacement 引起的显存碎片失败。最终代码完成 TP1/TP2 实模型、KV 专项 memcheck 与公开入口检查。

相同 8 条 AIME 的 CPU 卸载路径中位耗时由 **59.43s 降至 44.15s（降低 25.7%）**。原生 SparseSpec 为 **33.04s**；当前卸载版仍慢 **33.6%**。这一短输入、B=4、显存充足的测试尚未证明扩大可驻留 batch 后的吞吐收益。

## 2. 哪些文件发生了修改

本次相对开始前快照共修改 15 个文件。完整逐行差异见 [changes.patch](/data/users/songshiyi/vericache/external/SparseSpec/log/kv-offload-improvement-20261014/changes.patch)；前后 SHA256 见 [changes.json](/data/users/songshiyi/vericache/external/SparseSpec/log/kv-offload-improvement-20261014/changes.json)。`before/` 保留开始前文件，`source/` 保留最终文件。本次版本保存实现、复跑命令和小体积验证摘要；完整源码快照与大体积运行数据保留在本机 `log/`，可通过复跑命令重新生成。提交和远端状态以 Git 版本记录为准。

| 文件与改动起始位置 | 本次改动 |
|---|---|
| [serve/attention/README_packed_kv.md](/data/users/songshiyi/vericache/external/SparseSpec/serve/attention/README_packed_kv.md:36) | 记录布局、两端分配、容量边界、合批、数值差异和验证入口。 |
| [serve/attention/batched_kv_attention.py](/data/users/songshiyi/vericache/external/SparseSpec/serve/attention/batched_kv_attention.py:236) | 四份 attention 元数据合成一次 pinned CPU→GPU 上传；每次 forward 单次绑定 stream；本轮核心 attention CUDA 数学未改。 |
| [serve/attention/full_forward.py](/data/users/songshiyi/vericache/external/SparseSpec/serve/attention/full_forward.py:17) | 全量临时 KV 从高端分配；批量层准备与单次 stream 登记；传递固定 lease 容量。 |
| [serve/attention/kv_buffer_pool.py](/data/users/songshiyi/vericache/external/SparseSpec/serve/attention/kv_buffer_pool.py:74) | 支持从高端分配 full/restore，从低端分配持久 hot；保留对齐产生的碎片；增加 largest_free_bytes。 |
| [serve/attention/kv_partition.py](/data/users/songshiyi/vericache/external/SparseSpec/serve/attention/kv_partition.py:3) | 增加 hot_capacity/cold_capacity；实际 KV 为整块连续前缀视图，剩余空间在尾部；cold 直接写 pinned CPU、恢复只复制实际 KV；restore 从高端分配。 |
| [serve/attention/packed_batch.py](/data/users/songshiyi/vericache/external/SparseSpec/serve/attention/packed_batch.py:40) | RoPE 元数据使用显式 CPU pinned 连续 buffer，一次上传；整批 Q/K/V 形状检查，减少逐请求切片与重复登记。 |
| [serve/attention/packed_forward.py](/data/users/songshiyi/vericache/external/SparseSpec/serve/attention/packed_forward.py:98) | 批量层准备保留生命周期/stream/写入检查；依赖与 buffer 登记每个 forward 执行一次；传递 replacement 容量。 |
| [serve/scheduler/packed_scheduler.py](/data/users/songshiyi/vericache/external/SparseSpec/serve/scheduler/packed_scheduler.py:101) | 准入覆盖 full→hot 及后续 restore/repack 两个峰值；verify/draft/prefill 合批；受容量阻挡的 ready 请求等待；固定 hot/cold 预留容量；最后一 token 无需 replacement。 |
| [serve/tests/bench_aime_kv_overhead.py](/data/users/songshiyi/vericache/external/SparseSpec/serve/tests/bench_aime_kv_overhead.py:242) | 支持选择计时对照组，仍保留同模型/同预留内存；GPU resident 与 CPU offload 配对校验输出、接受轮次和选择轨迹。 |
| [serve/tests/test_batched_kv_attention.py](/data/users/songshiyi/vericache/external/SparseSpec/serve/tests/test_batched_kv_attention.py:19) | 覆盖默认 CUDA device 下 CPU pinned 元数据、stream 变化拒绝；保留 20K FP64 attention/LSE/score 参考。 |
| [serve/tests/test_kv_buffer_pool.py](/data/users/songshiyi/vericache/external/SparseSpec/serve/tests/test_kv_buffer_pool.py:28) | 覆盖尾端对齐、两端分配、反复扩大 restore 后的区间合并。 |
| [serve/tests/test_kv_partition.py](/data/users/songshiyi/vericache/external/SparseSpec/serve/tests/test_kv_partition.py:8) | 独立参考检查预留容量与实际连续视图；用 poison 验证 CPU slack 不写入、不进入 DMA。 |
| [serve/tests/test_packed_batch.py](/data/users/songshiyi/vericache/external/SparseSpec/serve/tests/test_packed_batch.py:43) | 增加 query-head 几何改变时提前拒绝的回归检查。 |
| [serve/tests/test_packed_scheduler.py](/data/users/songshiyi/vericache/external/SparseSpec/serve/tests/test_packed_scheduler.py:4) | 覆盖合批、容量阻挡、生命周期准入边界；真实 KV 内核测试 20K 输入/512 decode 多次增长；小模型用于内存回归。 |
| [serve/tests/validate_packed_serving.py](/data/users/songshiyi/vericache/external/SparseSpec/serve/tests/validate_packed_serving.py:110) | 加入 20.4K 受限 arena、压力准入与释放检查；输出相异时保存 logits 和 full self-spec 对照；分别报告 token 精确一致与数值误差。 |

## 3. 分配失败的原因与处理

已复现的失败：准入计数满足总容量，但不断增长的 cold/hot replacement 无法放进上一轮留下的较小空洞，出现 “No verify request fits”。原失败日志保留在 `validation-final-tp2.log`。

现在 full KV 和临时 restored cold 从 GPU arena 高端分配；持久 hot 从低端分配。每个请求根据输入长度、输出上限、k、r 固定 hot/cold lease 的最大容量，实际数据重排到 lease 的连续前缀。数据跨层保持连续，所有多余容量在整个 tensor 末尾，不在层间留洞，因此 H2D 仍只传实际 cold KV。CPU 固定容量会占用一些 slack，用空间换取本次已复现的增长场景可运行。

令 N=input+output_limit−1，H=min(N,max(minimum,ceil(rN),k+2))，C=N−min(N,max(minimum,ceil(rN),2))，P=H+k+1；每个 token KV 字节数为 L×2×local_KV_heads×head_dim×dtype_bytes。

| 请求阶段 | GPU 基础预留 | 一个请求 verify/repack 需要的额外预留 | CPU 基础/额外 |
|---|---:|---:|---:|
| full | F=max(input+k+1,P) | max(P,C+2P−F) | C / C |
| packed | P | C+P | C / C |

准入检查为 Σ基础 + max(额外) + 对齐余量。full→packed 使基础减少 F−P，而额外预留最多增加 F−P，所以该转换不会放大这项准入上界。它覆盖 initial full+hot replacement、之后 old hot+cold restore+new hot 两个峰值。多个同时 verify 则按实际 free intervals 模拟，选择能放入的 ready 前缀。仅剩一 output token 的请求完成后无需 replacement。

这是容量上界与本次分配策略的验证，不构成任意请求顺序下连续 arena 永不碎片化的证明。代码保留不可行时报错和 largest_free_bytes；不会隐式减小 k/r/batch 把失败配置标成成功。

## 4. AIME 计时与开销

配置：Qwen3-8B，A100 PCIe 40GB 单卡（GPU 0），BF16，TP1；AIME24 随机 8 条，seed=330，sample ID=89/79/62/61/72/77/60/70；输入平均 109 token；thinking 开启，temperature=0，输出上限 512，B=4，k=4，r=.1，minimum=128，chunk=256。CUDA graph 和异步 CPU 调度关闭。主计时不在每层新增同步，端到端终点等待 GPU 完成；加载、预留 arena、warmup 不计入。配置执行顺序随机化，每组 3 次。

| 路径 | 中位耗时/s | 3 次范围/s | 中位吞吐/token·s⁻¹ |
|---|---:|---:|---:|
| 原生 SparseSpec | 33.04 | 32.89–33.17 | 124.78 |
| 新实现：cold 在 GPU 的常驻对照 | 39.37 | 39.36–39.40 | 104.05 |
| 新实现：cold 在 CPU 的卸载路径 | 44.15 | 44.14–44.18 | 92.77 |

**CPU 传输路径增量：4.79s，较 GPU 常驻对照增加 12.2%。** Resident 对照通过 GPU shadow 将 cold gather/restore 变为 D2D，保持相同稀疏数学、输出、接受轮次和选择轨迹；三次配对均严格一致。该差值包含换用 CPU 传输后的整体路径成本，不等于独立测出的 DMA 时间。所有组保留同模型与相同预留 arena，包括实验 shadow。

原生调度器与新实现的 attention、批次边界和生成轨迹不同，原生输出还可能在边界多生成 token；吞吐按实际 token 数计算。原生与新实现的差值不能全部归因于卸载。优化前后也存在生成/批次轨迹差异，因此 25.7% 为该工作负载的实测端到端改善，不是每次内核调用的固定加速比。

按总运行耗时除以日志 forward 次数，再取三次中位数，原生、新版 GPU 常驻、新版 CPU 卸载分别为 28.93、35.21、39.49 ms/步。新版常驻路径增加约 6.28 ms/步，CPU 路径再增加 4.28 ms/步，合计约 10.56 ms/步。这是包含调度、选择、重排和等待的摊销值；原生和新版的 forward 工作量并非逐步相同。

已有 8 次 verify 窗口中，模型 forward 的 CUDA Event 平均跨度在新版 GPU 常驻和 CPU 卸载之间为 40.47→40.56 ms，差约 0.09 ms。该对照不能回答重写 attention 相对原生的纯 forward 增量；仍缺相同 batch、上下文长度和 query 数下的原生 forward 直接计时。

- 新实现每次测量输出 4096 token，forward 1118 次；实际 batch 均值 3.896，其中 B=4 的 forward 为 1050 次。上一版 forward 为 1264，实际 batch 均值约 3.475。
- verify 请求轮次 873，verify 模型 forward 450；verify+draft 合批 343 次，verify+prefill 2 次。
- 每方向实际 cold KV 载荷 31,140,790,272 bytes（29.00 GiB），migration/restore 各 865 次。GPU arena 已用峰值 356.20 MiB，CPU 峰值 352.55 MiB；它们不是整个模型 CUDA 内存。arena 各预留 2 GiB，回收区间供复用不会降低其全局预留量。
- nominal r=.1，但 minimum=128 导致所选历史 hot 占比按 Σhot/Σhistory 为 **34.3%**，不能把本次结果视为实际仅保留 10% KV。
- `phases-*.json` 为单独诊断轮的 8 个窗口，CUDA Event 在末尾异步收集；parent range 与子范围重叠，stream 上的提交间隙也可能计入，不可直接相加作为 GPU 忙碌时间或传输利用率。

query tile 16/32 的微测保留在 `query-tile-probe.log`：16 在短 draft 中只快约 3%，verify 与 20K verify 更慢，因此保持默认 32。单个 attention grid 同时包含 hot/cold chunks，之后 LSE merge 和 score 是独立 kernel。grid/TBs、CUDA occupancy 与 wave 估计已记录；wave 模型忽略各 TB 时长差异，不作为实测 SM 利用率。微测用 CUDA graph 降低 Python 提交间隙，正式 serving 仍关闭 graph。

原始 9 次计时见 [throughput.csv](/data/users/songshiyi/vericache/external/SparseSpec/log/kv-offload-improvement-20261014/throughput.csv)；结果汇总见 [summary.json](/data/users/songshiyi/vericache/external/SparseSpec/log/kv-offload-improvement-20261014/aime-final/summary.json)；请求/token、每次接受轮次、选择计数见 `aime-final/measured-*.json`。

## 5. 验证结果与保留的差异

| 检查 | 结果 |
|---|---|
| 组件测试 | 73 项，71 项执行通过、2 项 TP 条件跳过；[components-complete.log](/data/users/songshiyi/vericache/external/SparseSpec/log/kv-offload-improvement-20261014/components-complete.log) |
| KV 专项 memcheck | 27 项通过、0 errors；[memcheck-complete.log](/data/users/songshiyi/vericache/external/SparseSpec/log/kv-offload-improvement-20261014/memcheck-complete.log) |
| Qwen3 TP1 | 12 个配置完成，状态 passed_with_numerical_difference；[validation.json](/data/users/songshiyi/vericache/external/SparseSpec/log/kv-offload-improvement-20261014/validation-complete-tp1/validation.json) |
| Qwen3 TP2 | 11 个配置完成，状态 passed_with_numerical_difference；[validation.json](/data/users/songshiyi/vericache/external/SparseSpec/log/kv-offload-improvement-20261014/validation-allocator-tp2/validation.json) |
| 公开 run_inference 入口 | PUBLIC_ENTRY_PASSED；[public-complete-entry.log](/data/users/songshiyi/vericache/external/SparseSpec/log/kv-offload-improvement-20261014/public-complete-entry.log) |
| 20K 输入 / 512 decode 增长 | 小型确定性模型调用真实 KV 内核；完成多次重排与恢复、无残留 lease；[growth-and-live-view.log](/data/users/songshiyi/vericache/external/SparseSpec/log/kv-offload-improvement-20261014/growth-and-live-view.log) |

TP1/TP2 覆盖 k=4/14、r=.05/.25/.1、temperature=.65 可运行性、4.8K 与 20.423K 输入、压力准入。压力用例初始只容纳 3 个 full 请求，首批 pack 后准入第 4 个，实际 batch 达到 4。结束后所有请求 lease 归还，free interval 完整合并。20.423K 真实 Qwen 测试输出 12 token，TP1 GPU arena 约 3.296 GiB，实际已用峰值约 3.087 GiB；这验证无法同时存放两份完整历史时仍能迁移，不是 20K/512 decode 的 Qwen 吞吐结果。

保留两项非 token 精确相等的诊断：

- **TP1 20K：** 首次差异在位置 3，首次全量 verify 中、尚未 pack/offload。dense AR 两个最高 logits 均为 30.375；packed 为 30.375/30.25。full self-spec 对照与 packed 在该位置 logits 完全相同（max error=0），说明这一差异在全量 verify 中也出现。相对 dense AR：max abs=.375，relative RMSE=.01153，cosine=.999968。
- **TP2 k4/r.25：** 请求 1、位置 7。dense AR 两个最高 logits 均为 35.0；packed 为 35.25/35.0，full self-spec 对照选择与 AR 一致。max abs=.28125，relative RMSE=.01193，cosine=.999955。该例不能宣称 packed 与 full self-spec 完全相同。

两个诊断均满足之前已有的误差门限（max abs≤.5，relative RMSE≤.015，cosine≥.9999，top2 gap≤2×max abs）；本次没有放宽门限。报告状态明确保留 numerical_difference，不能据此宣称所有 greedy 输出严格一致。温度为 0，因此这里不是随机采样误差。独立 FP64 attention/LSE/score 与 KV 内容/poison 检查仍作为布局正确性证据。

完整流程在 compute-sanitizer 下会报告 FlashInfer RoPE 的 CUDA API error 400；独立只调用该 API 的程序也复现，日志为 `memcheck-flashinfer-reference.log`。本次零错误结论限于新增 KV 相关内核专项，未声称整个 Qwen/FlashInfer 程序 sanitizer 无错误，未据此修改环境或推断特定驱动根因。

## 6. 复跑与后续范围

复跑脚本：[commands.sh](/data/users/songshiyi/vericache/external/SparseSpec/log/kv-offload-improvement-20261014/commands.sh)。设置空闲的 `KV_SINGLE_GPU` 与两个不同的 `KV_TP_GPUS`，命令串行运行，最多同时使用两卡。脚本写入带时间戳的新目录，保留本次结果。

本轮实现仍为整个模型 forward→pack/搬运完成→下一 forward；不含逐层传输重叠、异步 CPU 调度、动态 k/r 或图执行。k/r 在单次运行内固定，lease 按请求声明的输出上限保守预留。已经修复本次复现的失败并验证两端分配；任意大 batch、不同长度分布与更紧显存预算仍需单独确认，不用本次短 AIME 推断生产收益。

当前测试进程已结束，GPU 0/1 均已释放。更完整机器可读汇总：[analysis.json](/data/users/songshiyi/vericache/external/SparseSpec/log/kv-offload-improvement-20261014/analysis.json)。
