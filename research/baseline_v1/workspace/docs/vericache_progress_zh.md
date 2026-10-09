# VeriCache A100 复现进度

## 2026-09-22：干净基线完成

### 固定版本

- GPU：NVIDIA A100-PCIE-40GB，SM80；实验固定使用逻辑 GPU 0。
- 驱动：595.58.03。
- control：官方 wheel `vLLM 0.28.0`、`LMCache 0.5.5`。
- vLLM 源码：官方 tag `v0.28.0`，commit
  `2cf0a6915ce544dc493a0990f2ea38d81601128a`。
- LMCache 源码：官方 tag `v0.5.5`，commit
  `05a013b29da78cf2321b9b46ec5039dde2fb0bb0`。
- Python 3.12.14、PyTorch 2.13.0+cu130、CuPy 14.2.0。

### 环境隔离

- `.conda/envs/control`：只使用发布 wheel 的对照环境。
- `.conda/envs/dev`：从官方 vLLM v0.28.0 源码导入的 editable 环境。
- dev 环境的 CUDA/C++ 扩展来自同版本官方 vLLM wheel；未混入 Vegas 或
  SparseSpec 代码。
- control 与 dev 使用不同的 vLLM/FlashInfer cache 目录。

### 已通过检查

1. control 环境版本、wheel 来源、CUDA、LMCache connector 和 CUDA 扩展检查通过。
2. Qwen3-4B 的 vLLM + LMCache MP smoke test 通过。
3. 第二次共享前缀请求命中 1,792 个 KV token。
4. 日志未出现 double-unpin、负 pin count 或 traceback。
5. LongBench 16K–18K fixture 含 1 条记录，LongBench-v2 30K–40K fixture
   含 20 条记录；原始和处理后缓存均可在 offline 模式定位。
6. 官方 wheel 与未修改 editable 源码对两条固定 prompt 的 greedy 输出逐 token
   一致。
7. vLLM 和 LMCache 源码工作树均保持干净。

### 复查命令

```bash
cd /home/songshiyi/vericache

source scripts/activate_vericache_control.sh
python scripts/check_vericache_control.py
python scripts/check_longbench_offline.py
bash scripts/run_vericache_control.sh --model /data/models/Qwen3-4B

python scripts/compare_baselines.py \
  artifacts/baseline_control.json \
  artifacts/baseline_dev.json
```

### 下一阶段

在 full KV 仍常驻 GPU 的条件下实现最小 same-model self-spec 状态机。第一版只支持：

- 单请求；
- greedy sampling；
- 固定 draft length 4；
- draft、verify、accept/reject 和 rollback；
- 使用完整 KV 的正确性参考路径，暂不加入压缩 KV 或 KV 卸载。

此阶段的退出条件是固定 prompt 的 self-spec 最终 token IDs 与 baseline 完全一致，
并能记录每轮 draft token、accepted length 和 correction token。
