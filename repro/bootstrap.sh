#!/usr/bin/env bash
# Install into the repository's private .repro directory.
set -euo pipefail
_ss_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_ss_python="${SSPEC_PYTHON:-python3.12}"
_ss_venv="${SSPEC_VENV:-${_ss_repo}/.repro/venv}"
_ss_native="${SSPEC_NATIVE_PREFIX:-${_ss_repo}/.repro/native}"
"${_ss_python}" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 is required"'
git -C "${_ss_repo}" submodule update --init --recursive 3rdparty/flashinfer 3rdparty/raft
if [[ ! -x "${_ss_venv}/bin/python" ]]; then
  "${_ss_python}" -m venv "${_ss_venv}"
fi
"${_ss_venv}/bin/python" -m pip install --upgrade pip
# Explicitly choose CUDA 12.8 wheels, then restore the complete observed set.
"${_ss_venv}/bin/python" -m pip install --no-deps \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
"${_ss_venv}/bin/python" -m pip install --no-deps \
  -r "${_ss_repo}/repro/environment/requirements-lock.txt"
"${_ss_venv}/bin/python" - "${_ss_repo}" "${_ss_native}" <<'PY'
import hashlib, json, platform, sys, tarfile
from pathlib import Path
repo, target = map(Path, sys.argv[1:])
if platform.system() != 'Linux' or platform.machine() != 'x86_64':
    raise SystemExit('Bundled native libraries require Linux x86_64; build RAFT/RMM from source for other platforms.')
env = repo / 'repro/environment'
manifest = json.loads((env / 'native-manifest.json').read_text())
archive = env / 'native-linux-x86_64.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest['archive_sha256']
target.mkdir(parents=True, exist_ok=True)
with tarfile.open(archive) as tar:
    tar.extractall(target, filter='data')
for name, digest in manifest['files'].items():
    assert hashlib.sha256((target / name).read_bytes()).hexdigest() == digest, name
print('Native headers/libraries restored:', target)
PY
printf 'Environment installed. Run: source %q\n' "${_ss_repo}/repro/activate.sh"
