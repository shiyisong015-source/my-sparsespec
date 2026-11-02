#!/usr/bin/env bash
set -euo pipefail

component_repo=/data/users/songshiyi/vericache/external/SparseSpec
component_venv=/data/users/songshiyi/vericache/.venv-vllm011
component_nsys="$component_repo/log/reproduction/resource-motivation/tools/nsys-cli/opt/nvidia/nsight-systems-cli/2026.5.1/target-linux-x64/nsys"
if [[ ${1:-} == --sweep ]]; then
    shift
    exec "$component_venv/bin/python" "$component_repo/experiments/diagnostics/run_component_profile_sweep.py" "$@"
fi
if [[ $(id -u) -ne 0 || -z ${SUDO_USER:-} ]]; then
    echo 'Run this script using sudo from your normal user terminal; only GPU2 is targeted (TP=1).' >&2
    exit 1
fi
component_user="$SUDO_USER"
component_gpu=${COMPONENT_PROFILE_GPU:-2}
component_batch_text=${COMPONENT_PROFILE_BATCHES:-16 32 48 64 72 128}
component_windows=${COMPONENT_PROFILE_WINDOWS:-3}
component_record_seconds=${COMPONENT_PROFILE_RECORD_SECONDS:-180}
component_window_gap=${COMPONENT_PROFILE_WINDOW_GAP_SECONDS:-180}
read -r -a component_batches <<< "$component_batch_text"
if [[ ! $component_gpu =~ ^[0-9]+$ || ${#component_batches[@]} -eq 0 ]]; then
    echo 'Invalid GPU or batch configuration' >&2; exit 1
fi
for component_batch in "${component_batches[@]}"; do
    [[ $component_batch =~ ^[0-9]+$ ]] && ((component_batch>=1 && component_batch<=128)) || exit 1
done
component_root=${COMPONENT_PROFILE_ROOT:-"$component_repo/log/component-profile-verify-waits-tp1-k8-20261102"}
component_stamp=$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)
component_output=${COMPONENT_PROFILE_OUTPUT:-"$component_root/hardware-$component_stamp"}
component_preflight=${COMPONENT_PROFILE_PREFLIGHT:-"$component_output/preflight"}
component_cache="$component_repo/.repro/component-profile-cache"
runuser -u "$component_user" -- mkdir -p "$component_root" "$component_output"
runuser -u "$component_user" -- mkdir -p \
    "$component_cache/triton" "$component_cache/cuda" \
    "$component_cache/torch_extensions" "$component_cache/xdg" "$component_cache/matplotlib"
runuser -u "$component_user" -- mkdir -p "$component_repo/log/runtime/component-profile-locks"
exec 9> "$component_repo/log/runtime/component-profile-locks/gpu$component_gpu.lock"
flock -n 9 || { echo 'Another component capture is active; retry when this GPU is free.' >&2; exit 75; }
trap 'chown -R -- "$SUDO_UID:$SUDO_GID" "$component_output"' EXIT
trap 'echo "Component profile failed; inspect $component_output/preflight-runtime.log and $component_output/runtime.log" >&2' ERR

# Inspect only the target GPU. Do not stop anyone else's workload,
# collectors, or change any driver/module settings.
runuser -u "$component_user" -- "$component_venv/bin/python" - "$component_gpu" "$component_output" <<'PY'
import json,os,sys
from pathlib import Path
import pynvml as n
n.nvmlInit()
try:
    for index in (int(sys.argv[1]),):
        handle=n.nvmlDeviceGetHandleByIndex(index)
        tasks=n.nvmlDeviceGetComputeRunningProcesses(handle)
        graphics=n.nvmlDeviceGetGraphicsRunningProcesses(handle)
        # The v1 API includes driver-reserved memory in used. The v2 API
        # reports it separately, matching nvidia-smi's application usage.
        memory=n.nvmlDeviceGetMemoryInfo(handle,version=n.nvmlMemory_v2)
        pids=sorted({t.pid for t in (*tasks,*graphics)})
        status=(f'GPU{index}: used={memory.used/1024**2:.1f} MiB, '
                f'driver_reserved={memory.reserved/1024**2:.1f} MiB, PIDs={pids}')
        print(status,flush=True)
        if pids or memory.used>128*1024**2:
            print(f'{status}; GPU is occupied; no capture started.',file=sys.stderr,flush=True)
            raise SystemExit(75)
        masks=n.nvmlDeviceGetCpuAffinity(handle,(os.cpu_count()+63)//64)
        allowed=os.sched_getaffinity(0)
        cpus=[i for i in allowed if masks[i//64]&(1<<(i%64))]
        if not cpus:raise SystemExit('No allowed CPUs local to selected GPU')
        Path(sys.argv[2],'cpu-list.txt').write_text(','.join(map(str,sorted(cpus)))+'\n')
        Path(sys.argv[2],'gpu-placement.json').write_text(json.dumps(dict(
            gpu=index,cpu_affinity=sorted(cpus),pci_bus_id=str(n.nvmlDeviceGetPciInfo(handle).busId)),indent=2)+'\n')
finally:
    n.nvmlShutdown()
PY
component_cpu_list=$(cat "$component_output/cpu-list.txt")

printf '%s\n' "$component_output" > "$component_root/latest-hardware-path.txt"
nvidia-smi topo -m > "$component_output/topology.txt"
nvidia-smi --query-gpu=index,pci.bus_id,memory.used,utilization.gpu --format=csv \
    > "$component_output/gpu-inventory-before.csv"
echo "Long-decode reservation profile on GPU$component_gpu, TP=1, batch=$component_batch_text, KV pool=13.5 GiB; output: $component_output"
echo "Progress: tail -f $component_output/runtime.log"
if [[ ! -f "$component_preflight/manifest.rank0.json" ]]; then
echo "First running a separate B128 preflight (32-token development cap): $component_preflight"
runuser -u "$component_user" -- env CUDA_VISIBLE_DEVICES="$component_gpu" RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
    TRITON_CACHE_DIR="$component_cache/triton" \
    CUDA_CACHE_PATH="$component_cache/cuda" \
    TORCH_EXTENSIONS_DIR="$component_cache/torch_extensions" \
    XDG_CACHE_HOME="$component_cache/xdg" \
    MPLCONFIGDIR="$component_cache/matplotlib" taskset -c "$component_cpu_list" bash -c '
        cd "$1"
        SSPEC_VENV="$2" source repro/activate.sh
        exec python experiments/diagnostics/component_profile.py --output "$3" \
          --pool-pages 98304 --buffer-batch-capacity 128 \
          --policy reservation --requests 128 --batches 128 --output-tokens 32 \
          --capture --warmup-seconds 0.5 --record-seconds 2 --record-windows 3 --window-gap-seconds 1 \
          --minimum-forwards 1 --window-check-steps 1 --detail-every 1
    ' -- "$component_repo" "$component_venv" "$component_preflight" \
    > "$component_output/preflight-runtime.log" 2>&1
fi
runuser -u "$component_user" -- env MPLCONFIGDIR="$component_cache/matplotlib" \
    "$component_venv/bin/python" "$component_repo/experiments/diagnostics/analyze_component_profile.py" \
    --root "$component_preflight" > "$component_output/preflight-analysis.log" 2>&1
runuser -u "$component_user" -- "$component_venv/bin/python" \
    "$component_repo/experiments/diagnostics/validate_component_preflight.py" \
    --root "$component_preflight" --gpu "$component_gpu" --repo "$component_repo"
component_iteration_ms=$(cat "$component_preflight/iteration-ms.txt")
"$component_nsys" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --gpu-metrics-devices="$component_gpu" --gpu-metrics-frequency=1000 \
    --session-new="component-gpu$component_gpu-$component_stamp" \
    --run-as="$component_user" --capture-range=cudaProfilerApi --capture-range-end=repeat \
    --force-overwrite=false --output="$component_output/nsys" \
    env CUDA_VISIBLE_DEVICES="$component_gpu" RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
        TRITON_CACHE_DIR="$component_cache/triton" \
        CUDA_CACHE_PATH="$component_cache/cuda" \
        TORCH_EXTENSIONS_DIR="$component_cache/torch_extensions" \
        XDG_CACHE_HOME="$component_cache/xdg" \
        MPLCONFIGDIR="$component_cache/matplotlib" taskset -c "$component_cpu_list" bash -c '
        cd "$1"
        SSPEC_VENV="$2" source repro/activate.sh
        component_job_output="$3"
        component_job_iteration="$4"
        component_job_windows="$5"
        component_job_seconds="$6"
        component_job_gap="$7"
        shift 7
        exec python experiments/diagnostics/component_profile.py --output "$component_job_output" --capture \
          --pool-pages 98304 --buffer-batch-capacity 128 --reservation-iteration-ms "$component_job_iteration" \
          --reservation-iteration-source "Single-GPU B128 short preflight median CUDA-event forward time; short contexts, not a long-decode timing guarantee" \
          --policy reservation --batches "$@" \
          --warmup-seconds 120 --record-seconds "$component_job_seconds" \
          --record-windows "$component_job_windows" --window-gap-seconds "$component_job_gap" --detail-every 8
    ' -- "$component_repo" "$component_venv" "$component_output" "$component_iteration_ms" \
         "$component_windows" "$component_record_seconds" "$component_window_gap" "${component_batches[@]}" \
    > "$component_output/runtime.log" 2>&1

component_reports=0
for component_report in "$component_output"/nsys*.nsys-rep; do
    [[ -e "$component_report" ]] || continue
    component_reports=$((component_reports+1))
    "$component_nsys" export --type=sqlite --output="${component_report%.nsys-rep}.sqlite" \
        "$component_report" >> "$component_output/export.log" 2>&1
done
if [[ $component_reports -eq 0 ]]; then
    runuser -u "$component_user" -- "$component_venv/bin/python" - "$component_output" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
if any(json.loads(path.read_text())['capture_started'] for path in root.glob('b*/observer.rank0.json')):
    raise SystemExit('Capture windows started but Nsight produced no reports; inspect runtime.log')
print('Requests ended before recording eligibility; no hardware window was collected.',flush=True)
PY
fi
runuser -u "$component_user" -- env MPLCONFIGDIR=/tmp/sparsespec-component-mpl \
    "$component_venv/bin/python" "$component_repo/experiments/diagnostics/analyze_component_profile.py" \
    --root "$component_output" | tee "$component_output/analysis.log"
echo "Completed: $component_output"
