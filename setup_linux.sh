#!/usr/bin/env bash
# Setup on Linux with an NVIDIA GPU.   Usage:  bash setup_linux.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo -e "\n== system dependencies =="
if ! python3 -m venv --help >/dev/null 2>&1; then
    echo "python3-venv missing; installing (asks for sudo)"
    sudo apt-get update -qq && sudo apt-get install -y python3-venv python3-dev
fi

echo -e "\n== python environment =="
[ -d .venv ] || python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip -q
./.venv/bin/python -m pip install numpy "cupy-cuda12x[ctk]" -q
echo "installed"

echo -e "\n== cupy/NVRTC check =="
./.venv/bin/python - <<'PY'
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
PY

echo -e "\n== Qwen3-0.6B model (1.4 GB) =="
mkdir -p outputs real_model
if [ ! -f real_model/model.safetensors ]; then
    curl -L --fail -o real_model/model.safetensors \
        https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors
else
    echo "already downloaded"
fi

echo -e "\n== weight extraction =="
[ -f outputs/real_exp_counts.npy ] || ./.venv/bin/python extract_real_weights.py

echo -e "\n== kernel correctness =="
./.venv/bin/python verify_kernels.py 256 32000000

echo -e "\nDone. Now:  bash run_all.sh"
