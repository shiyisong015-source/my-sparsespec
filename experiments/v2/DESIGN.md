# Qwen3-8B 第二轮实验方案（讨论稿，尚未启动）

## 目标

检验当前 SparseSpec 中不同资源状态/上下文长度下，(k,r) 的有效生成效率是否存在稳定交叉；若存在，构建接近最优收益的配置集合。不能预设一定有中间最优k，也不把接受率高等同于吞吐高。

研究问题：
1. 较长上下文下，更少 KV 访问节省的成本能否抵消接受损失？
2. 固定 r 时，draft length 的高收益集合是否随实际 batch/上下文分布变化？
3. r 的选择能否先由资源与上下文约束缩小范围，再选择 k，且损失足够小？
4. 当前 KV 选择算法是否限制长 draft 的收益？该问题保留为第二阶段的算法对照，避免第一阶段同时修改内核与算法。

## 固定环境与容量

- /data/models/Qwen3-8B，BF16，单GPU TP=1；GPU 0/1/2用于独立配置并行，记录GPU UUID、型号、GPU占用和版本/代码哈希。
- enable_thinking=False；temperature=.65；EOS自然结束，不强迫凑足输出。
- 初次扫描seed=330；正式复测seed=330/331/332，同一seed使用相同prompt顺序，各配置仍可能产生不同生成轨迹。
- cuda_graph=False，async_cpu=False；沿用第一轮，未来优化必须另成实验条件。
- max_mem_per_rank=26 GiB；KV池容量以启动日志为准（此前约78K token）。
- 模型36层、8 KV heads、head_dim=128：2(K/V)*36*8*128*2字节=147456字节/token=144 KiB/token，未含辅助buffer。
- max_model_len=32768；主实验输入（含chat template）<=20480，输出上限2048，k<=14时为边界预留15 token，总长度上界22543。
- pillar选定KV预算必须<6144；ceil(22543*.25)=5636，主网格有容量余量。逐样本预检仍必须执行，不默默clip r。
- 完整KV仍驻留GPU供verify使用；降低r不会同比释放完整KV池。显存压力由准入与offload处理，r主要改变draft访问的KV数量。
- MAX_TOTAL_DRAFT_KV_LEN=128K、MAX_TOTAL_VERIFY_KV_LEN=256K等总buffer也必须检查；模型KV容量并不等于全部显存限制。
- 主实验chunk_size先取256以衔接原实现。12个边界预检请求检查长prefill占比；若占比主导成本，先单独比较256/1024后确定统一值，绝不在主扫描中混用。

## 数据：自然长度分布与受控长度覆盖

建议新增 LongBench v1 的 GovReport、QMSum、MultiNews 三个摘要数据集；LiveCodeBench原100条保留作短上下文参考。
这些数据集覆盖政府报告、会议和多新闻文档，使用官方任务提示。官方平均长度不是Qwen token数，需要本地重新tokenize。
选择摘要任务是为了有较长回答的机会，不保证每个请求会生成很长；记录真实输出分布，不按某个(k,r)的输出长度过滤测试集。

来源：
- LongBench ACL 2024：https://aclanthology.org/2024.acl-long.172/
- 官方任务说明：https://github.com/THUDM/LongBench/blob/main/LongBench/task.md
- 官方数据加载说明：https://github.com/THUDM/LongBench/blob/main/LongBench/README.md

下载/预处理前固定来源revision、prompt模板和校验和。超过输入上限的样本报告排除率，不机械截断或重复文本凑长度。
诊断长度分箱：<2K、2–8K、8–16K、16–20K（Qwen chat模板后token数）。分箱仅用于覆盖/分析，不把每个箱当成新的语义任务。

- 小样本筛选：32条，尽量每长度箱8条、三个来源均有覆盖；若某箱不足，报告真实数量，补原始官方数据或调整上限需另行说明。
- 正式确认：另抽96条、每数据集32条，固定seed，按长度合法条件下均匀随机抽样；不强制均匀长度分布。与筛选集按样本ID去重。
- 主结果保留所有请求；按长度箱给辅助结果。真实请求分布未知，不将三数据集等权混合宣称为生产流量分布。
- 数据不足以覆盖20K时先统计后定，不承诺未检查的数据长度。

## 参数与顺序

A. 计时与可运行性预检
- 新增计时后先用原LCB少量请求核验统计，包含批内prefill/draft/verify混合。
- 三档batch上限4/16/32，最长输入、最大r=.25、最大k=14需覆盖；先每个数据集4条共12条预检。
- 串行对照计时开/关的端到端结果，报告测量开销；CUDA Event异步收集，不每步新增全设备同步。
- GPU间用相同短校准配置检查性能可比性，主任务不固定把某个r永远分给某张GPU。

B. 筛选（32条固定请求）
- batch上限：4/16/32。
- k：4/8/14；r：.05/.10/.15/.25，合计36组。
- 加3组相同batch下标准AR基线（每个batch一组），输出/采样/负载/内存配置一致。
- r=1不是AR；r=1不能直接走当前6144选择buffer路径。第一阶段不使用未经验证的r=1配置。
- 队列持续有请求；报告实际batch、runtimes含prefill与decode，另分离decode指标，不能只报告max_batch。
- OOM/失败保存并归入不可行区域，不通过隐式减batch/r把失败配置标成成功。

C. 正式确认（独立96条）
- 若小样本出现值得确认的收益交叉，完整网格 k=4/6/8/12/14，r=.05/.10/.15/.25，batch上限4/16/32，共60组。
- 加3组AR基线。边界样本先校验，随后再开大扫描。
- 对接近最优的候选与固定k=4,r=.15基线，追加到3次重复，随机化配置执行顺序；每次使用同一请求集合。
- 若筛选仍全面偏向k=4，只复测边界候选与基线确认；优先分析成本/算法，无需为了网格齐全直接重复60组。
- 95%高收益集合按每个可比条件计算，给均值与计时不确定性；集合不连续时不强行用min/max填成区间。

D. 后续扩展门槛
- k=20/30需要定制打分kernel的128行分发、实例化、约简与所有stride相关buffer检查；必须与参考attention分数/选择索引/输出做数值校验，不能只检查不崩溃。
- 当k=12/14已具竞争力或长上下文显示verify摊销机会，再做长k独立分支；不把“必须先赢才能探索”作为普遍定律，预检也可安排少量机制探针。
- r=.25仍居边界且有收益时，再做更大r；长上下文可能先触及6144限制，需明确扩大buffer还是缩小共同可行域。
- 算法对照：在相同KV预算下比较当前选择与例如固定近期窗口，先做机制小实验。基线差异包含选择质量与选择成本，不能只看接受率。
- 若要使r直接对应GPU KV驻留比例，必须另外实现完整KV的分层驻留/恢复，这是另一个系统改动。

## 计时与日志口径

每步至少保存：step_id、CPU开始/结束时间、GPU model-forward elapsed、sampling/verification与KV选择/搬运可分离阶段计时、实际活动请求数、query token总数、各状态请求数、各请求ID/k/实际draft数/r/kv_len/draft_kv_len、KV池占用、offload/restore事件、实际提交输出token数。

- CPU step wall time用于端到端成本；CUDA event是GPU stream上的执行区间，不等于请求延迟或全部系统成本。
- 用event.query或已有同步点批量收集已完成事件，避免每步synchronize改变原执行行为。
- batch混合时model-forward是整批耗时，不能把它平均除以请求数后当成真实每请求耗时。
- 给rounds.jsonl增加step_id以关联接受结果；请求draft开始快照与verify发生时快照语义分开记录。
- acceptance继续排除cold_start统计，但总时间、总吞吐仍包含冷启动成本；同时给warm decode窗口结果。
- forward时间不是全部收益成本：采样、verify/rejection、top-k选择、CPU调度和offload都应计入调度目标。
- 窗口收益用总提交token/总wall time。不能对每轮tok/s直接不加权平均。
- 若计算TTFT/ITL，记录请求入队、首token提交以及各次token burst的时间；verify一次提交多个token，不把整批耗时伪装成单token latency。

## 主要产物

1. 按实际batch与长度分布分组的 k-r 吞吐曲线/热图，标记95%候选集合。
2. draft/verify相关forward、KV选择、调度与搬运成本分解；混合batch单列。
3. 接受长度分布与r的边际收益，按请求配对bootstrap；不同配置生成轨迹不完全对齐，不能当成同一状态的反事实。
4. 最佳静态配置与固定原基线比较，再判断是否值得做在线策略。
5. 若开展在线策略，比较静态最佳、仅调r、仅调k、联合/分层调度，并使用相同到达序列；原离线日志不能直接证明在线加速。

## 拟改代码位置（需实现前逐项告知）

- serve/scheduler/scheduler.py：forward CUDA events、step wall time、batch状态、实际提交量与资源事件日志。
- serve/scheduler/spec_scheduler.py：verify记录增加step关联，补充选择阶段计时。
- serve/run.py：可选计时日志路径与开关、AR运行路径验证、统一配置记录。
- scripts/run_acceptance_case.py：输入数据集/输出长度/计时配置，不再默认硬编码LCB语义；保留按metadata数量检查完成数。
- 新增长上下文样本准备与第二轮扫描脚本，独立输出目录，保持第一轮结果可复查。

本文件是实验设计，未改上述运行代码、未下载新数据、未启动新GPU实验。
