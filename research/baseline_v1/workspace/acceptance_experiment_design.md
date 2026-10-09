# SparseSpec acceptance 扫描设计

当前约定（2026-10-08）：

- 模型：Qwen3-8B，后续扩展到更大的 Qwen 模型；关闭 thinking。
- 横轴 draft length：4、8、10、12、14。
- 每条曲线的 KV 保留比例：0.05、0.10、0.15。
- 数据来源：WildChat-1M、LMSYS-Chat-1M、SWE-smith 官方 expert trajectories。
- 每个数据集固定随机抽取 100 个请求；所有参数点使用相同请求。
- 请求包含生成前的完整历史；抽样种子及数据 revision 必须保存。
- 并发为 1；生成上限 1024 token，EOS 正常终止。
- 输入加目标输出上限为 32768；记录超限排除的数量和比例。
- acceptance = 接受的 draft token 总数 / 实际 draft token 总数；主图排除全量 KV 的冷启动轮次。
- 同时保存逐请求输出长度、验证轮数及逐轮 KV 长度和实际预算。

下一步：检查官方数据格式及访问条件，准备样本并统计长度；先用少量请求检查长 decode 程度与耗时，再做完整扫描。

已修正：cold_start 随每次 KV cache 重建设置，后续 verify 更新预算后清除。

本地首轮扫描：LiveCodeBench 固定随机种子 330，随机 100 个请求，并发 1，15 个参数点，不设两小时截止时间。样本输入为 158–1629 token，生成上限 1024。每个参数点单独保存并支持续跑，失败重试一次后停止，保留日志。按请求 bootstrap 2000 次计算 95% 置信区间。

扫描前修正：BatchTopKAttention 的分数 kernel 对 packed query 长度大于 32 时选择 64 行 tile，覆盖 k=8、10 的 36、44 行。

Qwen3 试跑环境：`.venv-sparsespec` 的 Transformers 已从 4.45.2 更新为 4.51.3，tokenizers 更新为 0.21.4，以支持 Qwen3 配置。控制环境未修改。

来源：

- WildChat：https://arxiv.org/abs/2405.01470
- LMSYS-Chat-1M：https://arxiv.org/abs/2309.11998
- SWE-smith：https://arxiv.org/abs/2504.21798
- SWE-smith 官方轨迹资产：https://swesmith.com/getting_started/assets/
