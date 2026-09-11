#!/usr/bin/env bash
# Todas las medidas + perfilado. Uso:  bash run_all.sh
set -uo pipefail
cd "$(dirname "$0")"
PY=./.venv/bin/python
OUT=resultados
mkdir -p $OUT

echo "== intentando fijar relojes de la GPU =="
# En Linux esto suele funcionar en tarjetas consumer (en Windows/WDDM no).
# Elimina el ruido de DVFS por completo. Si falla, no pasa nada: los scripts
# ya compensan con calentamiento y mediana.
if sudo -n true 2>/dev/null; then
    sudo nvidia-smi -pm 1 || true
    MAXCLK=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)
    sudo nvidia-smi -lgc "${MAXCLK},${MAXCLK}" || echo "  (no soportado, seguimos)"
else
    echo "  sin sudo sin password; saltando (opcional)"
fi
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm --format=csv | tee $OUT/gpu_info.txt

echo -e "\n== 1/4  barrido base (Huffman vs escalera, kernel original) =="
$PY bench_gpu.py 64000000 2>&1 | tee $OUT/1_bench_base.txt | tail -20

echo -e "\n== 2/4  variantes del patron de acceso (atribucion) =="
$PY bench_opt.py 64000000 2>&1 | tee $OUT/2_bench_opt.txt | tail -25

echo -e "\n== 3/4  Huffman vs escalera CON optimizacion =="
$PY bench_head2head.py 64000000 2>&1 | tee $OUT/3_head2head.txt | tail -25

echo -e "\n== 4/4  perfilado con Nsight Compute =="
if command -v ncu >/dev/null 2>&1; then
    bash profile_ncu.sh 2>&1 | tail -20
    cp -r ncu_out $OUT/ 2>/dev/null || true
else
    echo "ncu no instalado; saltando (ver INSTRUCCIONES_LINUX.md paso 2)"
fi

cp outputs/gpu_bench.json $OUT/ 2>/dev/null || true
echo -e "\n== recogiendo resultados =="
tar czf resultados.tar.gz $OUT
echo "LISTO -> resultados.tar.gz  (mandar este fichero)"

if sudo -n true 2>/dev/null; then sudo nvidia-smi -rgc || true; fi
