#!/usr/bin/env bash
# Usage: source scripts/activate_sparsespec.sh

_ss_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_ss_repo="${_ss_root}/external/SparseSpec"

if [[ ! -f "${_ss_root}/.venv-sparsespec/bin/activate" ]]; then
  echo "Missing SparseSpec environment: ${_ss_root}/.venv-sparsespec" >&2
  return 1
fi

# shellcheck disable=SC1091
source "${_ss_root}/.venv-sparsespec/bin/activate"
export CUDA_HOME="${_ss_root}/.cuda12-env"
export FLASHINFER_WORKSPACE_BASE="${_ss_root}/.cache/flashinfer-sparsespec"
export HF_HOME="${_ss_root}/.cache/huggingface"
export TORCH_CUDA_ARCH_LIST=8.0
export PYTHONNOUSERSITE=1
export PYTHONPATH="${_ss_repo}/3rdparty/flashinfer:${_ss_repo}${PYTHONPATH:+:${PYTHONPATH}}"
export INSTALL_PREFIX="${_ss_root}/.sparse-native"
export LD_LIBRARY_PATH="${INSTALL_PREFIX}/lib:${CUDA_HOME}/lib64:${_ss_root}/.venv-sparsespec/lib/python3.12/site-packages/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
mkdir -p "${FLASHINFER_WORKSPACE_BASE}" "${HF_HOME}"

unset _ss_repo _ss_root
