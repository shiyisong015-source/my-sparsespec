#!/usr/bin/env bash
# Recheck commands. Choose free GPU IDs before running; at most two are used.
set -euo pipefail
cd /data/users/songshiyi/vericache/external/SparseSpec
source /data/users/songshiyi/vericache/scripts/activate_reproduction.sh
KV_RUN_DIR="log/kv-offload-improved-recheck-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$KV_RUN_DIR"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 RUN_PACKED_KV_GPU=1 RUN_KV_PARTITION_GPU=1
KV_SINGLE_GPU="${KV_SINGLE_GPU:-0}"
KV_TP_GPUS="${KV_TP_GPUS:-0,1}"
# Run serially; the TP check uses exactly two distinct physical devices.
if [[ ! "$KV_SINGLE_GPU" =~ ^[0-9]+$ || ! "$KV_TP_GPUS" =~ ^([0-9]+),([0-9]+)$ ]]; then
  echo 'Choose one GPU for KV_SINGLE_GPU and two GPUs for KV_TP_GPUS.' >&2
  exit 2
fi
if [[ "${BASH_REMATCH[1]}" == "${BASH_REMATCH[2]}" ]]; then
  echo 'KV_TP_GPUS must contain two distinct GPUs.' >&2
  exit 2
fi

CUDA_VISIBLE_DEVICES="$KV_SINGLE_GPU" python -m unittest \
  serve.tests.test_kv_pack serve.tests.test_kv_partition \
  serve.tests.test_kv_buffer_pool serve.tests.test_packed_attention \
  serve.tests.test_packed_batch serve.tests.test_full_forward \
  serve.tests.test_packed_forward serve.tests.test_batched_kv_attention \
  serve.tests.test_packed_scheduler -v > "$KV_RUN_DIR/components.log" 2>&1

CUDA_VISIBLE_DEVICES="$KV_SINGLE_GPU" /usr/local/cuda-12.8/bin/compute-sanitizer \
  --tool memcheck --error-exitcode 99 python -m unittest \
  serve.tests.test_batched_kv_attention \
  serve.tests.test_kv_partition.PackedKVRegionTest \
  serve.tests.test_kv_buffer_pool \
  serve.tests.test_packed_scheduler.AdmissionPeakTest \
  serve.tests.test_packed_scheduler.MixedVerifySchedulingTest \
  serve.tests.test_packed_scheduler.VerifyCapacityTest.test_gpu_headroom_limits_group_to_two_without_allocations \
  serve.tests.test_packed_scheduler.VerifyCapacityTest.test_cpu_double_backing_limits_group_to_one \
  serve.tests.test_packed_scheduler.VerifyCapacityTest.test_infeasible_configuration_fails_before_restore \
  serve.tests.test_packed_scheduler.VerifyCapacityTest.test_final_round_needs_restore_but_no_replacement \
  -v > "$KV_RUN_DIR/memcheck.log" 2>&1

CUDA_VISIBLE_DEVICES="$KV_SINGLE_GPU" python -m serve.tests.validate_packed_serving \
  --output "$KV_RUN_DIR/validation-tp1" > "$KV_RUN_DIR/validation-tp1.log" 2>&1
CUDA_VISIBLE_DEVICES="$KV_SINGLE_GPU" python -m serve.tests.validate_packed_serving \
  --public-entry --output "$KV_RUN_DIR/public-entry" > "$KV_RUN_DIR/public-entry.log" 2>&1

CUDA_VISIBLE_DEVICES="$KV_SINGLE_GPU" python -m serve.tests.bench_aime_kv_overhead \
  --output "$KV_RUN_DIR/aime-overhead-thinking" --samples 8 --output-length 512 \
  --batch 4 --k 4 --r .1 --repetitions 3 \
  --modes original_sparse resident_sparse packed_offload > "$KV_RUN_DIR/aime-benchmark.log" 2>&1

CUDA_VISIBLE_DEVICES="$KV_TP_GPUS" torchrun --standalone --nproc_per_node=2 \
  -m serve.tests.test_kv_pack > "$KV_RUN_DIR/components-pack-tp2.log" 2>&1
CUDA_VISIBLE_DEVICES="$KV_TP_GPUS" torchrun --standalone --nproc_per_node=2 \
  -m serve.tests.test_packed_forward > "$KV_RUN_DIR/components-forward-tp2.log" 2>&1
CUDA_VISIBLE_DEVICES="$KV_TP_GPUS" torchrun --standalone --nproc_per_node=2 \
  -m serve.tests.validate_packed_serving --output "$KV_RUN_DIR/validation-tp2" \
  > "$KV_RUN_DIR/validation-tp2.log" 2>&1
