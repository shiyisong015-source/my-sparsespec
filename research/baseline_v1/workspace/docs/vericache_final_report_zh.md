# VeriCache A100 复现报告

> 状态：待填写
> 最终截止日期：2026 年 10 月 31 日

## 摘要

说明复现目标、完成范围、主要结果，以及是否达到预先定义的验收标准。
明确区分“核心机制复现”与“论文级性能复现”。

## 1. 复现范围

- 已实现功能：
- 未实现功能：
- 与原论文实现的主要差异：

## 2. 实验环境

记录 GPU、驱动、CUDA、操作系统、Python、PyTorch、vLLM、LMCache、模型、
源码 commit 与环境锁文件。

## 3. 实现方法

### 3.1 Self-spec draft/verify 状态机

说明同一目标模型如何完成 draft、verify、accept/reject 和回滚。

### 3.2 压缩 KV 视图

说明 A100 上采用的 sink-plus-recent block 选择方式、压缩比例和显存所有权。

### 3.3 Full-KV 卸载与恢复

说明 pinned CPU 与 LMCache 后端的实现状态、数据传输和生命周期。

### 3.4 调度与异步传输

说明 CUDA stream、同步点、NVTX 区间，以及实际实现的重叠范围。

## 4. 正确性验证

报告固定 prompt 数量、greedy token 逐项对比、不同 draft length、EOS、拒绝、
bonus token 和非完整 KV block 等测试结果。

## 5. LongBench 实验

分别报告 2K、4K、8K、16K–18K 和 30K–40K 配置。若某项受 A100 容量限制，
记录失败条件和能够通过的最大配置，不隐藏失败结果。

## 6. 性能结果

分别列出 baseline、self-spec 无卸载和 self-spec 有卸载的：

- 端到端延迟与吞吐；
- draft/verify 时间；
- accepted tokens 与 acceptance rate；
- D2H/H2D 传输时间和暴露等待时间；
- 峰值 GPU 显存与 CPU 内存。

## 7. 结果分析

解释性能变化来自计算、KV 压缩、PCIe 传输还是调度开销；不得仅凭端到端数字
推断原因。

## 8. 已知限制

说明 A100 与论文硬件差异、未实现特性、LMCache 集成状态、可扩展性与测量误差。

## 9. 复现步骤

提供从干净 shell 开始的环境激活、smoke test、正确性测试、LongBench 和 profiling
命令，并标注每个命令的预期输出及结果目录。

## 10. 结论

依据计划中的 Definition of done，逐项给出通过、未通过或受限结论。
