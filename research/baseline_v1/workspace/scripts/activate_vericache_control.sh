#!/usr/bin/env bash
# Official vLLM + LMCache control for the clean VeriCache workspace.
# Usage: source scripts/activate_vericache_control.sh

_vericache_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_vericache_conda_base="$(conda info --base)"
_vericache_env="${_vericache_root}/.conda/envs/control"

if [[ ! -d "${_vericache_env}" ]]; then
  echo "Missing environment: ${_vericache_env}" >&2
  return 1
fi

# shellcheck disable=SC1091
source "${_vericache_conda_base}/etc/profile.d/conda.sh"
conda activate "${_vericache_env}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TORCH_CUDA_ARCH_LIST="8.0"
export HF_HOME="${_vericache_root}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export VLLM_CACHE_ROOT="${_vericache_root}/.cache/vllm-control"
export FLASHINFER_WORKSPACE_BASE="${_vericache_root}/.cache/flashinfer-control"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LD_LIBRARY_PATH="${_vericache_env}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

mkdir -p "${HF_DATASETS_CACHE}" "${VLLM_CACHE_ROOT}" \
  "${FLASHINFER_WORKSPACE_BASE}"

unset _vericache_env
unset _vericache_conda_base
unset _vericache_root
