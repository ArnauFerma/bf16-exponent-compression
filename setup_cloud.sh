#!/usr/bin/env bash
# Setup on a RENTED GPU (RunPod / Vast.ai / similar).
#
# Differences from setup_linux.sh: in a rented container you are usually
# root, CUDA and Python are already there, and sudo does not exist. This
# script assumes neither and adapts.
#
# Usage:  bash setup_cloud.sh
set -uo pipefail
cd "$(dirname "$0")"

say() { printf '\n\033[36m== %s ==\033[0m\n' "$1"; }

say "GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || {
    echo "nvidia-smi does not respond: the instance has no visible GPU"; exit 1; }

say "python"
PY=python3
$PY --version || { echo "no python3"; exit 1; }

# venv is cleanest, but many containers lack the module and have no sudo to
# install it. In that case install into the system python, which in a
# throwaway container has no downside.
if $PY -m venv --help >/dev/null 2>&1; then
    [ -d .venv ] || $PY -m venv .venv
    PY=./.venv/bin/python
    echo "using venv"
else
    echo "no venv module; installing into the system python (ephemeral container)"
fi

say "dependencies"
$PY -m pip install --upgrade pip -q
$PY -m pip install numpy "cupy-cuda12x[ctk]" -q || {
    echo "cupy install failed; retrying without the [ctk] extra"
    $PY -m pip install numpy cupy-cuda12x -q; }

say "cupy/NVRTC check"
$PY - <<'PYEOF'
import cupy as cp
p = cp.cuda.runtime.getDeviceProperties(0)
res = p['multiProcessorCount'] * p['maxThreadsPerMultiProcessor']
print('GPU:', p['name'].decode(), ' SMs:', p['multiProcessorCount'])
print('compute capability:', cp.cuda.Device(0).compute_capability)
print('L2:', p['l2CacheSize'] // 1024, 'KiB')
print(f'resident threads: {res:,}   L2 per resident thread: {p["l2CacheSize"]/res:.0f} B')
k = cp.RawKernel('extern "C" __global__ void s(int*o){o[0]=__clz(1u);}', 's', backend='nvrtc')
o = cp.zeros(1, dtype=cp.int32); k((1,), (1,), (o,)); cp.cuda.Stream.null.synchronize()
print('NVRTC compiles and runs:', int(o[0]) == 31)
PYEOF

say "Qwen3-0.6B model (1.4 GB)"
mkdir -p outputs real_model
if [ ! -f real_model/model.safetensors ]; then
    (command -v curl >/dev/null && curl -L --fail -o real_model/model.safetensors \
        https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors) || \
    wget -O real_model/model.safetensors \
        https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors
else
    echo "already downloaded"
fi

say "weight extraction"
[ -f outputs/real_exp_counts.npy ] || $PY extract_real_weights.py

say "kernel correctness"
$PY verify_kernels.py 256 32000000 || {
    echo "VERIFICATION FAILED: do not continue, the timings would be worthless"; exit 1; }

# run_all.sh assumes ./.venv/bin/python; without a venv, leave it a symlink
if [ ! -x ./.venv/bin/python ]; then
    mkdir -p .venv/bin
    ln -sf "$(command -v python3)" .venv/bin/python
fi

printf '\n\033[32mDone. Now:  bash run_all.sh\033[0m\n'
