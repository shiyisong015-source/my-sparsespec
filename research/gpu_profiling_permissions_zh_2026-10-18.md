# GPU 资源测量：权限申请与采集口径

日期：2026-10-18。本文记录只读检查结果及管理员申请项，未更改系统权限、驱动参数或运行时代码。

## 1. 本机已确认的情况

| 项目 | 当前结果 |
|---|---|
| GPU | NVIDIA A100-PCIE-40GB，8 张 |
| 驱动 | 580.178.04 |
| 运行用户 | songshiyi，普通用户 |
| 驱动性能计数器限制 | `/proc/driver/nvidia/params` 中 `RmProfilingAdminOnly: 1` |
| Nsight Systems GPU Metrics 查询 | 8 张卡均返回 `Insufficient privilege` |
| 普通 PCIe 遥测 | `nvidia-smi dmon -s t -c 1` 可读取 Rx/Tx MB/s；本次空闲采样为 0，无 N/A 或权限错误 |
| Nsight Systems | 项目本地已有 2026.5.1，可执行文件路径见下文 |
| Nsight Compute | `/usr/local/cuda-12.8/bin/ncu`，2025.1.1；另有 CUDA 13.4 目录中的版本 |
| CPU perf 限制 | `kernel.perf_event_paranoid=4`；本次 GPU 指标目标不要求首先开放 CPU perf |

普通查询在 agent 沙箱外可以访问设备；Nsight GPU Metrics 在相同执行环境仍报告权限不足。这是驱动计数器权限问题，不能仅靠调整 `CUDA_VISIBLE_DEVICES` 解决。

## 2. 可直接提交给管理员的申请文字

> 申请在本机使用 NVIDIA Nsight Systems / Nsight Compute 采集 GPU 性能计数器，用于研究 LLM attention 与 KV 搬运的资源竞争。当前为 A100、驱动 580.178.04，用户 songshiyi；驱动设置 RmProfilingAdminOnly=1，Nsight Systems GPU Metrics 返回 Insufficient privilege。
>
> 申请允许指定实验进程树访问 GPU 性能计数器，可提供受控的管理员启动方式，或在维护窗口设置 NVreg_RestrictProfilingToAdminUsers=0，使普通用户可使用计数器。GPU 分配仍按现有实验授权执行。
>
> 目标指标为 PCIe Rx/Tx、SM Active、SM Issue、warp occupancy、Tensor 活跃度及 HBM 吞吐；先做工具支持性核验，再采集短窗口。希望安排一对 TP2 GPU 的独占测量窗口，并协调已有 DCGM/其他 profiler 对计数器的占用。若同一 PCIe 上游有其他设备流量，也请允许记录该流量或安排安静窗口。

### 管理员可选择的生效方式

1. **受控启动 profiler 及被测进程树**：以 root 或带 `CAP_SYS_ADMIN` 的受控启动环境运行。NVIDIA 文档也列出 R565 起支持 `CAP_PERFMON`，但有 secure-execution 条件；本机授权方式由管理员验证。不要只给一个外层脚本授权而假定所有子进程均继承成功。
2. **开放普通用户访问**：管理员在 `/etc/modprobe.d/` 的配置文件中设置：

   ```text
   options nvidia NVreg_RestrictProfilingToAdminUsers=0
   ```

   这会开放普通用户计数器访问。Ubuntu 上可能需要更新 initramfs，并在维护窗口重启后生效。当前存在 GPU 任务时，不执行卸载驱动模块的操作。
3. **容器场景**：需要主机驱动允许访问，或由管理员配置相应容器权限。当前申请无需把容器权限或用户权限扩大为不受控的通用管理权限。

本机驱动是 R580，不能照搬 NVIDIA 新文档中仅适用于 R610+ 的 `profiler-device` 设备节点授权流程。

权限依据：[NVIDIA 性能计数器权限说明](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)。

## 3. 授权后的验收

使用管理员确定的执行环境运行：

```bash
/data/users/songshiyi/vericache/external/SparseSpec/log/reproduction/resource-motivation/tools/nsys-cli/opt/nvidia/nsight-systems-cli/2026.5.1/target-linux-x64/nsys profile --gpu-metrics-devices=help
```

验收标准：目标 GPU 被列为支持 GPU Metrics，不再出现 `Insufficient privilege`。之后查询可用 metric set，并用短时自有工作负载确认导出的 SQLite 含真实 GPU_METRICS 样本。仅修改配置文件、仅看到 GPU 名称，均不算完成验收。

若目标 GPU 仍不能采集，需要区分设备访问、驱动权限、工具版本和其他 profiler 占用。Nsight Systems GPU Metrics、Nsight Compute 和 DCGM profiling 可能竞争同一计数器；暂停 DCGM profiling 属管理员协调事项，不由本实验擅自全机暂停。

## 4. 采什么、如何画图

| 目标 | 测量方法 | 图表与口径 |
|---|---|---|
| 不同 QPS 下的 PCIe 趋势 | 全程 NVML/nvidia-smi Rx/Tx 遥测，记录时间戳 | 每卡、每方向的 GB/s 随时间变化；结合队列、请求到达、KV 容量 |
| 传输突发与 attention 的关系 | Nsight Systems CUDA/NVTX/Graph 节点轨迹，配合 GPU Metrics | 同一时间轴展示 attention/MLP/搬运 kernel、PCIe、SM 和 HBM 指标 |
| attention 执行时 SM 活跃情况 | Nsight GPU Metrics 的 SM Active、SM Issue、warp occupancy，及可用的 Tensor/HBM 指标 | 分 Draft/Verify、batch/上下文长度，给出时间曲线和分布 |
| 单个 attention kernel 的资源约束 | Nsight Compute 的代表性 kernel 采样 | occupancy、寄存器/共享内存等驻留约束、调度和访存指标；用于解释原因 |
| 余量是否可用于搬运 | attention 单独执行与 attention+受控搬运的对照 | 同时报告隐藏的有效搬运字节、完成时间、attention 延迟及竞争变化 |

Nsight Systems 短窗先尝试默认 10 kHz，即约 100 微秒一采样点；检查实际频率、丢样和工具开销后再决定是否提高。比采样窗口还短的 attention kernel，不能由单个混合采样点精确归因，需要代表性重复或 Nsight Compute 补充。

具体可用指标集需要解锁后由工具枚举确认，不能预先保证所有指标能在同一遍采集中获得。

PCIe 硬件遥测包含设备上的其他 PCIe 流量，例如 TP 通信，不能全部标为 KV 传输。当前 SM 发起的 host-memory 搬运也不应仅通过 CUDA memcpy 事件统计；需要搬运 kernel/NVTX 与硬件流量联合判断。硬件接口计数与有效 KV 字节的统计口径也需分别保留。

## 5. 如何解释“可用 SM”

- SM Active 表示平均每个 SM 有至少一个驻留 warp 的周期比例；等待访存的 warp 也可计为 active。
- 50% SM Active 既可能是部分 SM 长期空闲，也可能是全部 SM 交替忙闲。不能直接断言某一时刻恰好有一半 SM 可分配给搬运。
- SM Issue、warp occupancy、HBM 吞吐可以帮助判断计算、驻留或访存方面的余量；这些仍不能单独保证另一个 kernel 可以无干扰并行。
- Tensor Active 低并不说明整数/访存型搬运所需资源一定空闲。
- 最终用受控并发搬运对照测量“attention 窗口里完成多少额外有效搬运”，并同时保留 attention 延迟变化。

Nsight Compute 默认采集可能串行化和重放 kernel，因此完整系统中的并发与时间关系以 Nsight Systems 轨迹和未受其扰动的对照计时为依据。Nsight Compute 单独用于代表性 kernel 的深入诊断，TP/NCCL 的必要并发还需专门处理。

参考：

- [Nsight Systems GPU Metrics 与指标定义](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#gpu-metrics)
- [DCGM profiling 指标与权限](https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/profiling.html)
- [Nsight Compute 的重复性与串行化](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#reproducibility)
