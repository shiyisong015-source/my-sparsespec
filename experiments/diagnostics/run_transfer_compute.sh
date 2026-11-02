#!/usr/bin/env bash
set -euo pipefail
umask 0002
profile_repo=/data/users/songshiyi/vericache/external/SparseSpec
cd "$profile_repo"
export SSPEC_VENV=/data/users/songshiyi/vericache/.venv-vllm011
source repro/activate.sh
export CUDA_VISIBLE_DEVICES=4,5 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
profile_nsys="$profile_repo/log/reproduction/resource-motivation/tools/nsys-cli/opt/nvidia/nsight-systems-cli/2026.5.1/target-linux-x64/nsys"
profile_root="$profile_repo/log/transfer-compute-current-20261021"
profile_case="$profile_root/$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)"
mkdir -p "$profile_root"
exec 9> "$profile_root/capture.lock"
flock -n 9 || { echo '本项目采集正在运行'; exit 1; }
profile_user="${SUDO_USER:-songshiyi}"
profile_as_user=()
if [[ $(id -u) -eq 0 ]]; then
    profile_as_user=(runuser -u "$profile_user" --)
fi
"${profile_as_user[@]}" "$VIRTUAL_ENV/bin/python" experiments/diagnostics/transfer_compute.py --prepare "$profile_case"
chmod g+w "$profile_case"
trap 'if [[ $(id -u) -eq 0 && -n ${SUDO_UID:-} && -n ${SUDO_GID:-} ]]; then chown -R -- "$SUDO_UID:$SUDO_GID" "$profile_case"; fi' EXIT
export TRITON_CACHE_DIR="$profile_case/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="$profile_case/torchinductor-cache"
export CUDA_CACHE_PATH="$profile_case/cuda-cache"
"${profile_as_user[@]}" "$VIRTUAL_ENV/bin/python" - "$profile_case" <<'PY'
import json,os,sys
from pathlib import Path
root=Path(sys.argv[1])
for directory in [root, *[Path(os.environ[n]) for n in ('TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','CUDA_CACHE_PATH')]]:
    directory.mkdir(parents=True,exist_ok=True)
    check=directory/f'.write-check-{os.getpid()}'
    check.write_text('ok');check.unlink()
(root/'worker-permission-check.json').write_text(json.dumps(dict(uid=os.getuid(),gid=os.getgid(),successful=True))+'\n')
PY
printf '%s\n' "$profile_case" > "$profile_root/latest-path.txt"
profile_metrics=()
if [[ $(id -u) -eq 0 ]]; then
    profile_metrics=(--gpu-metrics-devices=4,5 --gpu-metrics-frequency=10000 --run-as="$profile_user")
fi
python experiments/diagnostics/transfer_compute.py --guard "$profile_case/guard.jsonl" -- \
    "$profile_nsys" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop \
    "${profile_metrics[@]}" --force-overwrite=false --output="$profile_case/nsys" \
    python -m torch.distributed.run --standalone --nproc_per_node=2 \
    experiments/diagnostics/transfer_compute.py --worker-payload "$profile_case/worker-payload.json" \
    > "$profile_case/runtime.log" 2>&1
"$profile_nsys" export --type=sqlite --output="$profile_case/nsys.sqlite" "$profile_case/nsys.nsys-rep" \
    > "$profile_case/export.log" 2>&1
python experiments/diagnostics/analyze_transfer_compute.py --case "$profile_case"
echo "完成：$profile_case"
