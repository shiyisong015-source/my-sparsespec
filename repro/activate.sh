#!/usr/bin/env bash
# Usage: source repro/activate.sh
_ss_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_ss_venv="${SSPEC_VENV:-${_ss_repo}/.repro/venv}"
if [[ ! -f "${_ss_venv}/bin/activate" ]]; then
  echo "Missing Python environment: ${_ss_venv}. Run bash repro/bootstrap.sh first." >&2
  return 1
fi
source "${_ss_venv}/bin/activate"
if [[ -z "${CUDA_HOME:-}" ]]; then
  if [[ -x "${_ss_repo}/.repro/cuda12/bin/nvcc" ]]; then
    export CUDA_HOME="${_ss_repo}/.repro/cuda12"
  elif [[ -x /usr/local/cuda-12.8/bin/nvcc ]]; then
    export CUDA_HOME=/usr/local/cuda-12.8
  else
    echo 'Set CUDA_HOME to a CUDA 12.8 toolkit, or install repro/environment/cuda12-linux-64-explicit.txt.' >&2
    return 1
  fi
fi
export INSTALL_PREFIX="${SSPEC_NATIVE_PREFIX:-${_ss_repo}/.repro/native}"
export FLASHINFER_WORKSPACE_BASE="${SSPEC_FLASHINFER_CACHE:-${_ss_repo}/.repro/flashinfer-cache}"
export HF_HOME="${HF_HOME:-${_ss_repo}/.repro/huggingface}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${_ss_repo}/3rdparty/flashinfer:${_ss_repo}${PYTHONPATH:+:${PYTHONPATH}}"
_ss_site="$("${_ss_venv}/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export LD_LIBRARY_PATH="${INSTALL_PREFIX}/lib:${CUDA_HOME}/lib64:${CUDA_HOME}/targets/x86_64-linux/lib:${_ss_site}/nvidia/cuda_runtime/lib:${_ss_site}/nvidia/nccl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CPATH="${_ss_site}/nvidia/curand/include:${CUDA_HOME}/include:${CUDA_HOME}/targets/x86_64-linux/include${CPATH:+:${CPATH}}"
export ENABLE_INTRA_NODE_COMM=1
export MAX_JOBS="${MAX_JOBS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
mkdir -p "${FLASHINFER_WORKSPACE_BASE}" "${HF_HOME}"
unset _ss_repo _ss_venv _ss_site
