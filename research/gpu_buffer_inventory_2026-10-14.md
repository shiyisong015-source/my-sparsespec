# 原生与compact的GPU池外buffer清单

记录日期：2026-10-14。当前compact源码检查点：`6d3f02c`，已推送personal。本文记录同步/eager版本；异步与新准入策略尚未实施。

配置：Qwen3-8B、FP16、TP=2、k=8、词表151936。所有空间均为每个rank/每张GPU，MiB=2^20 bytes。仅计算有效分配容量，不等于allocator reserved或nvidia-smi占用。

## 原生与当前版对照摘要

| GPU池外buffer | 原生 SparseSpec | 当前compact |
|---|---:|---:|
| 全量页索引（两份） | 16 MiB | 16 MiB |
| Draft读取索引（两份） | 144 MiB | 144 MiB |
| Draft索引汇总 | 72 MiB | 72 MiB |
| Attention选择分数（两份） | 144 MiB | 144 MiB |
| Attention分分快照 | 72 MiB | 72 MiB |
| 小元数据（两套） | 约0.75 MiB | 约0.75 MiB |
| Attention共享浮点workspace | 256 MiB | 256 MiB |
| Attention整数workspace（两份） | 16 MiB | 16 MiB |
| TopK workspace（两份） | 128 MiB | 128 MiB |
| 30请求选择索引 | 101.25 MiB | 101.25 MiB |
| 30请求draft概率 | 173.88 MiB | 173.88 MiB |
| 上述常驻合计 | 1123.88 MiB | 1123.88 MiB |
| Customized通信（另计） | 40 MiB+meta | 40 MiB+meta |
| 新版搬运分区索引及排序（临时） | 无 | 按单请求长度分配，详见下文 |

当前固定主体为37个独立GPU tensor，每个驻留请求再增加2个；B=30时97个。该计数不包含模型权重、KV池、通信、临时tensor及内部库分配。

## 固定常驻GPU buffer（两版共用）

| 名称 | 单份形状 | 类型 | 份数 | 合计 MiB | 用途 |
|---|---|---|---:|---:|---|
| `kv_indices` | `[2097152]` | int32 | 2 | 16.000000 | 全量物理page索引 |
| `batch_indices` | `[32768]` | int32 | 2 | 0.250000 | query对应请求 |
| `positions` | `[32768]` | int32 | 2 | 0.250000 | KV追加位置 |
| `pos_offsets` | `[4096]` | int32 | 2 | 0.031250 | 逻辑位置偏移 |
| `qo_indptr` | `[4097]` | int32 | 2 | 0.031258 | query前缀和 |
| `kv_indptr` | `[4097]` | int32 | 2 | 0.031258 | 本轮可见KV前缀和 |
| `all_kv_indptr` | `[4097]` | int32 | 2 | 0.031258 | 物理页表前缀和 |
| `kv_len_arr` | `[4096]` | int32 | 2 | 0.031250 | 本轮KV长度 |
| `kv_last_page_len` | `[4096]` | int32 | 2 | 0.031250 | 最后page长度 |
| `batch_size_tensor + nnz_tensor` | `各[1]` | int32 | 2 | 0.000015 | batch/query计数；每套两个tensor |
| `request_type` | `[4096]` | int32 | 2 | 0.031250 | 请求阶段 |
| `verify_kv_indptr` | `[4097]` | int32 | 2 | 0.031258 | verify score偏移 |
| `flatten_indices` | `[36,4,131072]` | int32 | 2 | 144.000000 | 供attention读取的分层分head draft索引 |
| `draft_kv_indices_tensor_buf` | `[36,4,131072]` | int32 | 1 | 72.000000 | draft索引汇总缓冲 |
| `dump_logits` | `[36,4,262144]` | FP16 | 2 | 144.000000 | attention score；不是词表logits |
| `dump_logits_snapshot_tensor_buf` | `[36,4,262144]` | FP16 | 1 | 72.000000 | score快照 |
| `attention float_workspace_buffer` | `[268435456]` | uint8 | 1 | 256.000000 | 两套attention wrapper共享同一块 |
| `attention int_workspace_buffer` | `[8388608]` | uint8 | 2 | 16.000000 | 计划/执行整数workspace |
| `TopK float_workspace_buffer` | `[67108864]` | uint8 | 2 | 128.000000 | 两套TopK wrapper各自拥有一块 |

固定部分合计 **848.750046 MiB**，计37个独立常驻GPU tensor。两套阶段元数据即使关闭CUDA Graph也仍分配；共用attention浮点workspace只计一次。

源码位置：`serve/request/kv_cache_ptr/base.py:205`、`pillar.py:51`与`:162`、`serve/model/model.py:108`与`:122`、`serve/attention/backend.py:354`、`3rdparty/flashinfer/flashinfer/attention.py:94`。

## 按已接纳请求分配（两版共用）

| 名称 | 每请求形状 | 类型 | 每请求 MiB | 30请求 MiB |
|---|---|---|---:|---:|
| `selected_indices_buffer_gpu` | `[36,4,6144]` | int32 | 3.375000 | 101.250000 |
| `draft_probs` | `[10,151936]` | FP32 | 5.795898 | 173.876953 |

常驻主体：**848.750 + B × 9.170898 MiB**。B=30时为 **1123.877 MiB，约1.098 GiB**。计37+2B个独立常驻GPU tensor。请求概率MAX_SPEC_LEN当前为10，即使k=8也不只分配8行。

源码：`serve/request/request.py:54`、`streaming.py:41`（原生选择索引）、`compact.py:54`（新版选择索引）。

## TP通信（两版，customized启用时）

`CustomAllreduce` 本rank物理分配：`meta_ptrs` 为16 MiB+`ops.meta_size()`（cudaMalloc），`buffer_ptrs`为16 MiB（cudaMalloc），`rank_data`为8 MiB（torch tensor）。共40 MiB+meta，额外3块。远端IPC映射不重复算成本rank物理分配。NCCL和CUDA context内部显存未在该公式中；graph开启时注册/捕获的内存也另计。

源码：`serve/distribute/communicator.py:41`、`custom_all_reduce.py:116`与`:149`。

## 按轮临时分配（两版）

设Q为本轮实际query token行数，v为需要TopK选择的请求数，D为本轮全部verify的草稿位置数。以下是单tensor大小，不应把不同生命周期的项直接相加当峰值。

| buffer | 形状／大小 | 说明 |
|---|---|---|
| TopK `out_idx` | `[v,36,4,6144]` int32，3.375v MiB | 返回新的选择索引 |
| TopK `out_val` | 同形状FP16，1.6875v MiB | 分配完整选择容量 |
| query/页表/概率搬运元数据 | query IDs、指针、长度等小数组 | 部分由CPU数组复制到GPU；页表通常4×实际pages bytes |
| 汇总词表logits | `[Q,151936]` FP16，0.289795Q MiB | TP all-gather结果；本地shard另有短暂存储 |
| 当前采样概率 `probs` | `[Q,151936]` FP32，0.579590Q MiB | 完整词表，TP不对半 |
| 随机采样 `q` | 同probs大小 | 新版rank0采样，其他rank分配概率接收buffer |
| verify汇总 `draft_probs_tensor` | `[D,151936]` FP32，0.579590D MiB | 从各请求常驻概率复制 |
| verify汇总 `target_probs_tensor` | 同上，0.579590D MiB | 从本轮采样概率复制 |
| rejection随机数矩阵 | `[verify请求数,151936]` FP32 | 当前compact在rank0运行rejection |
| 模型中间激活 | Q/K/V、MLP、attention out、LSE等 | 随Q变化，各层重用；不能乘36后直接当峰值 |

源码：`serve/attention/backend.py:391`、`serve/model/model.py:338`、`serve/sampling/fast_fused_op.py:39`、`serve/sampling/sampler.py:160`、`serve/scheduler/spec_scheduler.py:442`、`serve/sampling/rejection_sampler.py:282`。

## compact新增的临时GPU索引（原生没有）

P为KV池pages，N为正在重新打包的请求完整KV长度，H为热KV数，C=N-H。每层head的页选择不同，故分区索引带36×4维度。

| 名称 | 形状/类型 | 大小 |
|---|---|---|
| `full`、`plan.full_pages` | `[N]` int32，各一份 | 各4N bytes |
| `inverse` | `[P]` int64 | 8P bytes；P=351614约2.683 MiB |
| `positions` | `[36,4,selected_count]` int64 | 1152×selected_count bytes |
| `membership` | `[36,4,N]` bool | 144N bytes；N=20K约2.747 MiB |
| sort keys及`cold_order` | `[36,4,N]` int64 | 每份1152N bytes；N=20K每份约21.973 MiB |
| `plan.cold_pages` | `[36,4,C]` int32 | 576C bytes |
| `plan.hot_pages` | `[36,4,H]` int32 | 576H bytes |
| gather/restore目的页索引 | `[H]`或`[C]` int32 | 4H或4C bytes |
| 物理长度元数据 | `[batch]` int32 | 临时替换本轮append长度 |

`order`为expand视图，不是36×4×N的新物理分配；`recent`同样是视图。但算术中间值、类型转换和sort workspace有额外瞬时分配，以上单项不构成完整临时峰值。当前串行repack，分区临时buffer按正在搬运的单请求分配，并非一次为全部请求分配。

源码：`serve/attention/kv_partition.py:83`、`serve/request/kv_cache_ptr/compact.py:179`、`serve/attention/kv_pack.py:242`。

## 不属于GPU池外buffer的内容

- 新热KV目的pages、恢复冷KV目的pages、原有热KV和tail都在同一个GPU KV池内；不重复计算为池外KV tensor。
- 卸载后的`cold_cpu`是CPU pinned内存，GPU恢复后可暂时仍保留CPU副本；旧/新CPU冷KV在repack时短暂重叠。
- 原生legacy CPU KV池按1.5×GPU池容量预分配，本配置约36.214 GiB/rank；compact关闭这块legacy池，改为按请求分配cold_cpu。CPU张量不计入GPU清单。
- 原生的selected_indices_buffer_cpu和各请求页号数组是CPU侧存储，不计入GPU索引容量。

## 汇总与前图修正

两版共同的已知常驻GPU辅助buffer主体约1.098 GiB（30请求），customized通信再加约0.039 GiB及meta。此前图中约0.96 GiB只覆盖了部分项，漏计TopK 128 MiB和attention int workspace 16 MiB及小元数据，不能称为完整常驻辅助buffer总量。

还需要额外考虑临时激活、概率汇总、TopK输出、新版分区sort、allocator缓存、CUDA/NCCL内部显存，以及旧版graph/async保留的生命周期。以上是源码分配清单，不是实际峰值测量。
