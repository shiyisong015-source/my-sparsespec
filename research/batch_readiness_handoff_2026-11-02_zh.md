# 新服务器交接：修复 ready batch 偏小与预约执行不一致

日期：2026-11-02。本文是交接和实施建议；截至写入本文，生产调度、KV管理和kernel尚未因本次诊断而修改。

## 1. 从哪个版本开始

- 仓库：`https://github.com/shiyisong015-source/my-sparsespec`。
- 分支：`research/sparsespec-reproduction`。
- 被诊断的代码基线：`d8723f703b6e356c744853597dd564fb5d02c522`。
- 本文后续以文档提交形式发布；不要把文档提交误认为性能修复已完成。
- 当前实现是 **VeriCache式预约表 + 我们的compact KV布局**，不能直接声称完整复现了VeriCache论文的所有实现细节。本文定位的问题属于当前实现。

在新服务器已克隆的目录中获取本文：

```bash
cd ~/my-sparsespec
git pull --ff-only origin research/sparsespec-reproduction
```

先在新服务器保存自己的基线提交，再开始修改。硬件变更后在本机重跑基线；旧A100数据用于理解问题，不作为新机器的直接性能对照。

## 2. 这次要解决什么

现象：max batch设为32、始终有32个已准入请求，但实际参与forward的平均请求数只有4.49、3.45、2.48。这里的batch计数是 `D+V+P`，不包含STALL，也不是query token数量。

优先任务：

1. 修复预约资源模型与实际KV事务推进粒度不一致。
2. 验证“全量KV仍驻留、稀疏视图合法”的Draft能否继续计算，避免无显存压力时先等待强制卸载。
3. 用相同输入、k/r、内存容量和计时方式做单项对照，观察实际batch及有效输出吞吐。

本阶段固定k=8、r=.05，不继续做接受长度预测，也不通过同时更改k、r、query预算来混淆收益。尚未证明修完能达到实际batch32。

## 3. 已有实验与证据边界

### 固定条件

| 项目 | 参考机设置 |
|---|---|
| 模型 | Qwen3-8B，FP16，原thinking模板 |
| TP / 执行方式 | TP1，eager；CPU流水与KV传输异步开启 |
| 输入 | Git内固定LCB128，seed330；不是合成等长输入 |
| k / r / 最小热预算 | 8 / .05 / 192 |
| query预算 / chunk | 2048 / 256 |
| batch上限 | 本次证据为32；原计划扫描16/32/48/64/72/128 |
| KV pool | 98304页；每页147456字节，共13.5GiB |
| 元数据buffer容量 | 固定按batch128配置，而非随每档缩小 |
| 请求终止 | 自然EOS，或32768总上下文边界；正式运行无额外输出截断 |
| 采样 | 热身120秒后采三段，每段180秒，间隔180秒；细NVTX约1/8 forward |
| 原预约参数 | 有效带宽6.62567GB/s，逻辑迭代31.2709ms，horizon64 |

带宽来自原生搬运kernel短校准，不是PCIe物理峰值；31.27ms来自短预检，不保证真实长decode迭代耗时。新服务器需要重新校准并记录参数。

### 三个B32窗口

| 指标 | 窗口1 | 窗口2 | 窗口3 |
|---|---:|---:|---:|
| forward数 | 3180 | 3358 | 3256 |
| 已准入请求数 | 32 | 32 | 32 |
| 实际平均batch | 4.493 | 3.455 | 2.476 |
| 平均D / V | 3.994 / .499 | 3.070 / .384 | 2.199 / .276 |
| 最大query token | 16 | 16 | 13 |
| 最少空闲pool页 | 65654 | 49026 | 40039 |
| 平均Verify等待ms | 824.1 | 1793.1 | 2679.9 |
| 平均单次恢复Event ms | 15.99 | 23.95 | 30.55 |
| SM Active窗口均值 | 37.6% | 43.6% | 40.1% |

三窗的Verify等待原因变化快照中，`capacity_insufficient` 和未满足释放依赖的记录均为0。日志只在原因变化时记录，这不是每一瞬间的排除证明，但不支持“query预算或显存容量打满导致batch小”的解释。第三窗最低空闲仍约5.50GiB。

三个窗口起止、JSONL、Attention/MLP 36层标注与传输配对已检查。Nsight三份报告仍带“可能未收全NVTX/CUDA事件”的警告；覆盖检查不等于逐事件零丢失。B32整批自然EOS和最终页回收未确认，不能据此报告整批吞吐。

这三段来自同一运行的不同阶段，不是独立重复实验；上下文随生成增长，不能把窗口间变化都归因于单一变量。

## 4. 原因A：resident Draft 被强制卸载门槛挡住

当前路径：

```text
Verify完成 → 原生rejection/selection → keep_full_resident
           → 为resident请求预约repack
           → 等待repack执行与提交
           → 获得下一轮恢复/Verify预约
           → Draft才重新具备计算资格
```

关键代码：

- [reservation_scheduler.py](../serve/scheduler/reservation_scheduler.py)：`_plan_repacks()`约268行，对每轮有冷选集差额的resident Draft安排卸载，并非只在显存压力下卸载。
- 同文件 `_place_reservations()`约232行：正号请求预约候选包含full/hot；resident只有在 `len(indices)==len_loading_indices` 时进入。有稀疏选集的resident被排除。
- 同文件 `_eligible()`约162行：非NORMAL请求没有正号预约就返回False，因此resident Draft无法进入batch。
- [compact.py](../serve/request/kv_cache_ptr/compact.py)：`keep_full_resident()`约181行，已经支持保留所有GPU页、用选中前缀及recent/tail执行Draft；resident不等于数据未就绪。
- [async_compact_scheduler.py](../serve/scheduler/async_compact_scheduler.py)：`update_requests()`约290行，在selection完成后调用上述resident接续路径。

第三窗口开头的具体状态：

| 32个已准入请求的分组 | 数量 |
|---|---:|
| 进入forward的Draft | 4 |
| 未进入batch的Draft，初始观察为resident | 19 |
| 未进入batch的Verify，初始观察为hot | 9 |

该forward只有4个query token。这里19个resident Draft的阻塞不是因为缺失冷KV，而是当前策略要求它们先走重排/卸载和预约。

### 修改建议与边界

这属于策略调整，需要独立开关和对照，不应与原因B的实现一致性修复混在一个结果中。

- 在无搬运冲突、稀疏视图合法、追加页有容量时，允许resident Draft继续计算。
- 全量KV已驻留的下一次Verify不需要H2D，但仍受k边界、结果消费、元数据和计算预算约束。
- 真正需要释放空间时复用 `prepare_resident_offload()`，它会把selection后新增的tail纳入热后缀。不要重新发明一套页表。
- 如取消既有repack预约，必须处理它的预计释放事件和依赖恢复；不能保留不存在的未来空闲信用。
- 正在repack的请求仍要停住；不能让计算读正在被搬迁/回收的页。
- 不要仅删除 `_eligible()` 的一行判断。还需处理正/负号预约、过期、下一次selection、resident Verify、退役及压力回退。

## 5. 原因B：预约允许同槽多事务，执行器每轮只推进一笔

### 当前代码

- [reservation_table.py](../serve/scheduler/reservation_table.py)：`reserve()`约84行按字节/带宽为时隙记账；小事务可以共享同一时隙。
- `ReservationCompactSpecScheduler.admit_requests()`约354行：每轮开头poll一次传输，随后调用 `_start_reload()` 与 `_start_repack()`。
- `_start_reload()`约325行与 `_start_repack()`约299行：`_transfers`非空就返回；成功提交一笔也立即返回。
- `AsyncCompactSpecScheduler._submit_transfer()`约60行：明确禁止同时存在两笔KV事务。

因此，同一轮调度不会在第一笔完成后继续推进第二笔。一次调度迭代与一次forward并不完全等价：可能有不执行forward的空迭代。

### 具体冲突，可直接写成回归用例

窗口1原始预约日志：

```json
{"iteration":2882,"event":"repack_reserve","request_id":26,"start":2898,"deadline":2898,"cold_pages":694,"reservation_kind":"repack"}
{"iteration":2884,"event":"repack_reserve","request_id":24,"start":2898,"deadline":2898,"cold_pages":609,"reservation_kind":"repack"}
```

两笔共1303页，约192.1MB；该时隙模型允许约207.2MB，因此BW表认为可行。但实际事件为：

```json
{"iteration":2898,"event":"repack_start","request_id":26}
{"iteration":2899,"event":"cancel","request_id":24,"reason":"repack_unstarted_overdue","reservation_kind":"repack"}
```

以上JSON是从原始行摘录相关字段，省略了release event等字段，不是原始文件的逐字节副本。

第一窗有2048次 `repack_unstarted_overdue` 取消；检测到1469对“一笔按时启动、同槽另一笔未启动过期”的组合证据。组合数不是独立请求数。

### 为什么平均batch恰好约4.5

当前k8稳态中，每请求一轮约8个Draft请求步、1个Verify请求步，伴随一次D2H和一次H2D。因此有近似工作量关系：

```text
平均计算batch ≈ 9 × Verify次数 / forward次数
             ≈ (9/2) × KV事务次数 / forward次数
```

| 窗口 | Draft请求步 | Verify请求步 | KV事务数 | 事务/forward | 记账估计batch | 实际batch |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 12701 | 1586 | 3180 | 1.000 | 4.500 | 4.493 |
| 2 | 10310 | 1291 | 2581 | .769 | 3.459 | 3.455 |
| 3 | 7161 | 900 | 1798 | .552 | 2.485 | 2.476 |

这是本配置、当前每轮两事务路径的窗口记账关系，不是self-spec的通用batch上限；边界中未完成轮次也会造成小偏差。第一窗每个forward间隔恰好新提交一笔事务；后两窗存在一个forward间隔内多次调度、提交多笔事务的情况，不能把“每调度迭代一笔”绝对说成“每forward一笔”。

### 后两窗为什么更低

冷KV增长；在原标定带宽与时隙长度下，平均预约跨度由1.12格变为1.61、2.02格。新建/重建预约的平均起点提前量分别约17.6、41.0、59.9个逻辑迭代。达到k上限或Verify阶段的请求会等待，ready集合缩小。

上述预约统计混合重排和恢复、含重建，不是按独立请求去重的估计；原迭代时长标定也不保证墙钟准确。

### 修改建议与边界

先确定执行模型，再让预约表记相同粒度：

- **保守一致性版本**：若坚持每调度迭代只提交一笔，就在表中增加对应事务数/启动预算，避免产生执行器无法兑现的同槽预约。这只能改善可兑现性，不会自动提高链路服务率。
- **性能版本**：支持ready事务的批量提交或独立推进，让一条transfer stream上的多笔事务在CPU进行模型提交时仍能连续执行。多笔排队不要求物理同时拷贝，也不要求立即引入后台线程。
- 若改为多笔已提交事务，必须同时审查 `_submit_transfer()`、`_poll_transfers()`、等待/取消、故障清理、source/destination保活及观察器，不能只去掉单事务断言。
- 同一请求仍只能有一个布局变化事务；多个请求可以排队。提交后的事务不能靠取消表项回收正在使用的页。
- 未来释放不能充当已经存在的GPU目标内存。当前恢复是一次性分配目标页，不能仅凭未来有空位就提前超额分配；分块恢复属于另一项改动。
- 若设计后台推进，GPU页分配、页表发布和请求状态仍需有明确的单一所有者/同步方式，不能让线程无保护地修改scheduler状态。
- TP2下，提交顺序和commit顺序必须跨rank一致，并在所有rank对应事件完成后发布布局。先做TP1，再做TP2。

## 6. 不能破坏的状态与内存约束

| 场景 | 必须满足 |
|---|---|
| resident继续Draft | selection有效、无冲突repack、recent/tail覆盖正确、追加页属于本请求 |
| repack | 等该请求此前计算fence；完成前保留全部源页；完成后才能释放 |
| restore期间Draft | Draft只读不变的热页和合法尾部；冷数据写新分配目标页 |
| Verify | 所需冷KV已完成恢复并发布页表；不得读部分恢复结果 |
| 新selection | 前一轮Verify/rejection已完成；旧预约及释放依赖正确清理 |
| 取消预约 | 已提交事务继续保活；只撤销未物化计划，不制造免费页 |
| 结束或异常 | GPU页恰好回收一次，CPU源与索引保活到事件完成 |

不要改原生采样、rejection、温度或随机数顺序来提高接受率；不同调度可能改变随机调用轨迹，不能仅凭输出不逐位一致判为KV搬运错误。应分别检查搬运数据、页所有权、TP状态一致和采样流程。

## 7. 建议提交顺序与回归测试

每步独立提交，保留开关，使旧基线仍可复跑：

1. **先加失败回归**：同槽两笔小事务，第一笔GPU事件完成后第二笔应按新的执行契约推进，不再出现单纯由提交粒度导致的伪过期。
2. 修复事务推进/预约一致性，保持原强制卸载策略，以隔离原因B。
3. 增加resident Draft接续路径或压力卸载策略，隔离原因A。
4. 最后再处理窗口采集收尾，避免为拿到三窗profile等待整批长输出。

需补充的关键用例：

- resident且稀疏视图合法、无压力：可追加Draft，后续原地Verify与再次selection合法。
- 同一resident请求先追加若干token，再被选为卸载对象：recent/tail计数、页表和所有权正确。
- resident源页仍在被compute使用时，repack不能越过fence回收。
- 两个ready KV事务的推进、延迟、完成、取消与异常清理；长事务占用时不能假设其他任务已经结束。
- 不同请求事件乱序query、TP各rank完成速度不同：只发布满足全rank条件的状态。
- 撤销repack后，依赖其释放的恢复被重排；不泄漏、不重复计入释放信用。
- metadata/query容量不足、GPU页不足、CPU容量不足时保留真实约束，不隐式缩batch/r冒充成功。

已有CPU回归：

```bash
source repro/activate.sh
CUDA_VISIBLE_DEVICES='' python -m unittest \
  serve.tests.test_reservation_table \
  serve.tests.test_reservation_scheduler -v
python -m unittest discover -s experiments/diagnostics -p 'test_*.py' -v
```

基线已通过23项预约表/策略测试与6项profile测试；这些旧用例未覆盖本次暴露的全部问题。

## 8. 新服务器的实验顺序

环境、模型和输入准备见 [迁移说明](../repro/profile-migration_zh.md)。不要直接运行旧机器硬编码路径的sudo shell。

### 先功能短测

假设已生成 `.repro/profile-input.json`，GPU0获准独占，模型路径需自行替换：

```bash
source repro/activate.sh
CUDA_VISIBLE_DEVICES=0 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
python experiments/diagnostics/component_profile.py \
  --model /your/models/Qwen3-8B --input .repro/profile-input.json \
  --output log/handoff-smoke-baseline \
  --policy reservation --requests 128 --batches 32 \
  --pool-pages 98304 --buffer-batch-capacity 128 \
  --output-tokens 32 --warmup-seconds .5 --record-seconds 2 \
  --record-windows 3 --window-gap-seconds 1 \
  --minimum-forwards 1 --window-check-steps 1 --detail-every 1
```

这是有32-token输出上限的开发验证，不能用它报告自然EOS吞吐。修改版使用新目录，不覆盖基线。更大batch及最长边界先预检，失败记录保留。

原reservation测试模式要求出现搬运后才开始采窗；若修改为无压力保留resident，可能没有搬运，此时需显式设置 `--minimum-restores 0` 并在结果标明，避免窗口永远不开始。增加真正的压力测试确保卸载/恢复路径仍被覆盖。

### 再做相同条件下的短窗口对照

先batch32，至少区分旧基线、仅修原因B、再加原因A三个版本；同机器、同输入/顺序、k/r、pool、buffer容量、采样和计时配置一致。标定流程一致，实际参数有变化就记录；不要一边改执行器一边私下调带宽参数。

观察到明确改善后，再扫16/32/48/64/72/128。主报告同时给max batch、实际D/V/P、query、上下文、gate计数和窗口有效输出token/s；不能只给max batch或forward/s。

### 收尾流程单独修改

当前组件脚本三窗采完后仍继续128请求自然EOS，再写observer/coverage和整批completion。旧B16因此拖住后续batch；旧B32中途退出时，原始三窗可读但缺少收尾元数据。

建议增加明确的“仅采窗口后安全结束”模式：每窗关闭即保存真实窗口元数据，最后一窗后排空在途工作、正确回收并退出。未完成请求必须标为未完成，不制造自然EOS或全程吞吐结果。故障退出也应保留已关闭窗口的元数据。

## 9. 建议补充的观测，注意观察开销

现有Verify时间线主要解释Verify等待，对未选中的resident Draft没有逐步完整gate记录。建议按抽样forward记录：

- 已准入请求数、ready D/V/P与STALL数。
- resident Draft等待repack、无预约、等待预约起点、等待恢复、等待结果消费各数量。
- query、draft/verify元数据、GPU pool、CPU容量各门槛；允许同时保留多个限制标志。
- 每轮poll、ready事务数、提交/完成事务数、队列长度、取消及deadline miss原因。
- 单事务GPU区间、host提交到commit、真实目标页分配时刻；区分排队与自己的传输。
- 实际周期时长分布，与用于预约的逻辑时隙估计对照。

避免每步复制大张量或为观察新增全设备同步。当前CPU admission计时包含观察器drain、状态遍历和日志；尚未单独测量扣除，不能全部当生产调度开销。需要定量收益时做相同配置的计时开/关对照。

## 10. 已有组件时间，避免重读时误解

| B32窗口均值，ms/对应调用 | 窗口1 | 窗口2 | 窗口3 |
|---|---:|---:|---:|
| CPU准入与调度（含观察器） | 10.672 | 9.251 | 12.142 |
| CPU元数据准备 | 5.695 | 5.079 | 3.978 |
| 全部已记录forward GPU Event | 38.481 | 36.873 | 36.106 |
| 普通采样GPU Event | .140 | .133 | .130 |
| 拒绝采样GPU Event | .452 | .485 | .710 |
| KV选择索引更新GPU Event | .267 | .206 | .181 |
| repack事务GPU Event | 17.610 | 25.554 | 31.217 |
| restore事务GPU Event | 15.992 | 23.954 | 30.545 |

另有1169个完整细采样forward：Attention core约5.5–7.3ms，MLP约8.5ms，其他forward kernel约5.2–5.9ms。其他项包括QKV/输出投影、Norm、RoPE、KV追加、残差、embedding/lm_head等。Attention core不等于整个Attention模块。

同一细采样集合的kernel工作合计约19.5–21.9ms，对应Event区间约38.3–41.3ms。不能把差额全归CPU：它可能包含提交空隙、依赖与并发事务干扰；细标注也有观察开销。不要混用细采样与全量forward的均值作精确减法。

CPU、计算流、传输流存在重叠，各种调用次数不同，父子项也重叠；上述行不能直接求和。SM Active的补数不是可直接使用的空闲SM数；PCIe事务忙也不等于物理链路饱和。

## 11. 旧服务器证据位置与迁移限制

旧服务器根目录：

```text
/data/users/songshiyi/vericache/external/SparseSpec/log/
  component-profile-verify-waits-tp1-k8-20261102/
    hardware-sweep-20261102-124000/
```

| 相对路径 | 内容 |
|---|---|
| `jobs/b32/b32_w01`等 | batches、components、transfers、reservations、rounds、Verify时间线 |
| `jobs/b32/nsys.1/2/3.nsys-rep` | 三份原始Nsight报告 |
| `jobs/b32/recovery-check/inspection.json` | 覆盖、告警、硬件指标检查 |
| `jobs/b32/recovery-check/kernel-attribution.1/2/3.json` | kernel归属 |
| `offline-analysis/batch-gate-diagnosis_zh.md/.json` | 本次门槛诊断及例子 |
| `offline-analysis/diagnose_batch_gates.py` | 从旧窗口日志复算本次证据的脚本 |
| `offline-analysis/all-components_zh.md/.csv` | 完整已采集组件表与口径 |
| `offline-analysis/window-analysis.json` | B16/B32窗口级统计 |
| `offline-analysis/fine-components.json` | 细采样forward的GPU工作拆分 |
| `offline-analysis/b32-bottlenecks.png` | 初步分析图 |

完整log与Nsight数据库不在Git中。本文已包含编写最小回归所需的例子和汇总；如需复核原始时间线，请单独迁移上述证据目录或在新服务器重新采集。

旧离线 `attribute_kernels()` 在某些未匹配API上反复向前回扫，分析较慢；旧机在log下用 `attribute_fast.py` 线性扫描进行CUDA launch与NVTX关联，并通过小嵌套范围夹具对照。该辅助脚本不在Git中，若需要复算需一并复制或改进正式离线分析器。这只影响离线分析耗时，不是GPU推理瓶颈。

## 12. 环境恢复中已遇到的两个问题

- 新机没有Python3.12：用uv安装后通过 `SSPEC_PYTHON="$(uv python find --managed-python 3.12)" bash repro/bootstrap.sh` 指定解释器。
- 第三方子模块网络不稳定：仍沿用原安装方法，先以HTTP/1.1、串行方式重试，再继续bootstrap：

```bash
git -c http.version=HTTP/1.1 submodule update \
  --init --recursive --jobs 1 3rdparty/flashinfer 3rdparty/raft
```

HTTP/1.1用于绕过此前HTTP/2 framing错误，不能修复无法连接github.com:443的网络出口问题。依赖仍通过子模块和pip下载，本次没有改成离线依赖包。遇到下载失败先解决网络，不要替换固定FlashInfer版本或静默删掉原生依赖检查。

## 13. 完成标准

第一阶段完成意味着：

- 预约表和执行器遵循同一套可兑现的事务推进契约，具体同槽冲突有回归覆盖。
- ready batch偏小的原因可按阶段说明；支持的resident接续路径不再被无必要的搬运门槛阻塞。
- GPU/CPU源与目标、计算fence、页表发布、取消依赖与TP状态检查均通过。
- 同机基线与修改版有可比较窗口；同时报告实际batch、有效输出、等待、搬运量、失败和观察开销边界。
- 没有靠丢弃请求、提前标记EOS、改变采样、隐式缩batch/r或放松恢复完整性得到表面收益。

达到这些条件后再讨论更复杂的k/r调节或传输/计算联合策略。目前不承诺最优batch、吞吐提升幅度或论文创新性。
