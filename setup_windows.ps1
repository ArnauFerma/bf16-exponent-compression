# Environment setup on a Windows machine with an NVIDIA GPU.
# Uso:  powershell -ExecutionPolicy Bypass -File setup_windows.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "== GPU ==" -ForegroundColor Cyan
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

Write-Host "`n== venv ==" -ForegroundColor Cyan
if (-not (Test-Path ".venv")) { python -m venv .venv }
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip -q
& $py -m pip install numpy "cupy-cuda12x[ctk]" -q
Write-Host "dependencies installed"

Write-Host "`n== cupy/NVRTC check ==" -ForegroundColor Cyan
& $py -c @"
import cupy as cp, numpy as np
p = cp.cuda.runtime.getDeviceProperties(0)
print('GPU:', p['name'].decode(), ' SMs:', p['multiProcessorCount'])
print('compute capability:', cp.cuda.Device(0).compute_capability)
print('L2:', p['l2CacheSize']//1024, 'KiB   max threads/SM:', p['maxThreadsPerMultiProcessor'])
res = p['multiProcessorCount']*p['maxThreadsPerMultiProcessor']
print(f'resident threads: {res:,}   L2 per resident thread: {p[\"l2CacheSize\"]/res:.0f} B')
k = cp.RawKernel('extern \"C\" __global__ void s(int*o){o[0]=__clz(1u);}','s',backend='nvrtc')
o = cp.zeros(1, dtype=cp.int32); k((1,),(1,),(o,)); cp.cuda.Stream.null.synchronize()
print('NVRTC compiles and runs:', int(o[0]) == 31)
"@

Write-Host "`n== Qwen3-0.6B model (1.4 GB) ==" -ForegroundColor Cyan
New-Item -ItemType Directory -Force outputs, real_model | Out-Null
if (-not (Test-Path "real_model\model.safetensors")) {
    curl.exe -L --fail -o real_model\model.safetensors `
        https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors
} else { Write-Host "already downloaded" }

Write-Host "`n== weight extraction ==" -ForegroundColor Cyan
if (-not (Test-Path "outputs\real_exp_counts.npy")) { & $py extract_real_weights.py }
else { Write-Host "already extracted" }

Write-Host "`n== kernel correctness ==" -ForegroundColor Cyan
& $py verify_kernels.py 256 32000000

Write-Host "`nDone. Now:" -ForegroundColor Green
Write-Host "  powershell -ExecutionPolicy Bypass -File run_all.ps1"
