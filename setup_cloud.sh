#!/usr/bin/env bash
# Preparacion en una GPU ALQUILADA (RunPod / Vast.ai / similar).
#
# Diferencias con setup_linux.sh: en un contenedor alquilado normalmente eres
# root, ya hay CUDA y Python, y no existe sudo. Este script no asume ninguna
# de las dos cosas y se adapta.
#
# Uso:  bash setup_cloud.sh
set -uo pipefail
cd "$(dirname "$0")"

say() { printf '\n\033[36m== %s ==\033[0m\n' "$1"; }

say "GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || {
    echo "nvidia-smi no responde: la instancia no tiene GPU visible"; exit 1; }

say "python"
PY=python3
$PY --version || { echo "sin python3"; exit 1; }

# venv es lo mas limpio, pero en muchos contenedores falta el modulo y no hay
# sudo para instalarlo. En ese caso se instala en el python del sistema, que
# en un contenedor de usar y tirar no tiene ningun inconveniente.
if $PY -m venv --help >/dev/null 2>&1; then
    [ -d .venv ] || $PY -m venv .venv
    PY=./.venv/bin/python
    echo "usando venv"
else
    echo "sin modulo venv; instalando en el python del sistema (contenedor efimero)"
fi

say "dependencias"
$PY -m pip install --upgrade pip -q
$PY -m pip install numpy "cupy-cuda12x[ctk]" -q || {
    echo "fallo instalando cupy; probando sin el extra [ctk]"
    $PY -m pip install numpy cupy-cuda12x -q; }

say "comprobacion de cupy/NVRTC"
$PY - <<'PYEOF'
import cupy as cp
p = cp.cuda.runtime.getDeviceProperties(0)
res = p['multiProcessorCount'] * p['maxThreadsPerMultiProcessor']
print('GPU:', p['name'].decode(), ' SMs:', p['multiProcessorCount'])
print('capacidad de computo:', cp.cuda.Device(0).compute_capability)
print('L2:', p['l2CacheSize'] // 1024, 'KiB')
print(f'hilos residentes: {res:,}   L2 por hilo residente: {p["l2CacheSize"]/res:.0f} B')
k = cp.RawKernel('extern "C" __global__ void s(int*o){o[0]=__clz(1u);}', 's', backend='nvrtc')
o = cp.zeros(1, dtype=cp.int32); k((1,), (1,), (o,)); cp.cuda.Stream.null.synchronize()
print('NVRTC compila y ejecuta:', int(o[0]) == 31)
PYEOF

say "modelo Qwen3-0.6B (1,4 GB)"
mkdir -p outputs real_model
if [ ! -f real_model/model.safetensors ]; then
    (command -v curl >/dev/null && curl -L --fail -o real_model/model.safetensors \
        https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors) || \
    wget -O real_model/model.safetensors \
        https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors
else
    echo "ya descargado"
fi

say "extraccion de pesos"
[ -f outputs/real_exp_counts.npy ] || $PY extract_real_weights.py

say "correccion de los kernels"
$PY verify_kernels.py 256 32000000 || {
    echo "LA VERIFICACION HA FALLADO: no sigas, los tiempos no valdrian nada"; exit 1; }

# run_all.sh asume ./.venv/bin/python; si no hay venv, se le deja un enlace
if [ ! -x ./.venv/bin/python ]; then
    mkdir -p .venv/bin
    ln -sf "$(command -v python3)" .venv/bin/python
fi

printf '\n\033[32mListo. Ahora:  bash run_all.sh\033[0m\n'
