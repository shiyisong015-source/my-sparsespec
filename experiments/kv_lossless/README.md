# 真实冷 KV 无损压缩诊断

只增加实验脚本；不修改 `serve/`。采集文件和结果放在 `log/kv-lossless-20261021/`。

## 数据与边界

- Qwen3-8B，TP=2，r=.05，沿用当前动态准入、异步 CPU 和 CUDA Graph。
- 从固定 LongBench 确认集选最接近 2K、8K、16K 的三个真实输入；每条生成最多 40 token。
- 在首次 verify 后的原生 KV 选择完成时，取得未选中的冷 KV。只采集第 0、11、23、35 层，两个 TP rank 均采集。
- 使用现有分区、搬运 kernel；再逐层、逐 head 直接读取原 pool，逐位比较。保存 FP16 的原始 uint16 位模式。
- CPU 分别测 K、V；GPU 测合并 KV。不同实验的布局、块大小会影响压缩率，不能当成同一种配置直接比较。
- 这是离线 codec 基准，未将压缩接入在线卸载路径，也未证明与 attention 重叠时的收益。

## 环境与命令

依赖安装到 `.repro/kv-codec-packages`，不替换推理环境原有包：

```bash
export SSPEC_VENV=/data/users/songshiyi/vericache/.venv-vllm011
source repro/activate.sh
uv pip install --python "$SSPEC_VENV/bin/python" \
  --target .repro/kv-codec-packages --no-deps \
  -r experiments/kv_lossless/requirements.txt
python experiments/kv_lossless/capture.py --prepare log/kv-lossless-20261021/capture-new
# 必须先确认 GPU 空闲，下面编号只是示例。
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 \
  experiments/kv_lossless/capture.py \
  --worker-payload log/kv-lossless-20261021/capture-new/worker-payload.json
OMP_NUM_THREADS=1 python experiments/kv_lossless/benchmark.py \
  --capture log/kv-lossless-20261021/capture-new \
  --output log/kv-lossless-20261021/benchmark --mode cpu
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 python experiments/kv_lossless/benchmark.py \
  --capture log/kv-lossless-20261021/capture-new \
  --output log/kv-lossless-20261021/benchmark --mode gpu
```

## 口径

- `stored_fraction = 压缩字节数 / 原始字节数`；`factor` 为其倒数。
- CPU LZ4、Zstd 为单线程，64 KiB / 1 MiB 块；计入块边界 offset 表。
- `byte_shuffle` 将每个 FP16 的低、高字节分别连续排列，是可逆排列；完整恢复时间包含反排列。
- GPU codec 绑定当前 Torch stream；CUDA Event 计时，解压配置及输出缓冲预先建立，另报头解析和 host wall 时间。
- GPU 压缩大小包括 nvCOMP bitstream metadata；压缩上传到同一预分配缓冲后解压，和原始 pinned H2D 分别比较。
- DLPack 暴露压缩分配容量；有效长度使用 `Array.buffer_size`，只传输有效前缀，并清零无效尾部后验证恢复。
- ANS FP16、256 KiB 配置额外做 7 对交替 H2D 计时，减小原始/压缩计时的带宽波动。
- GPU 计时每配置 7 次，CPU 解压 5 次，报告中位数和最大值。GB/s 使用十进制原始字节数。
- 每个成功配置必须逐字节恢复；失败配置保留异常，不以有损结果冒充无损。
- 解压速度不包含模型 forward、在线调度与初次建立压缩归档的成本。GPU 在独占、离线条件下测试。
