#!/usr/bin/env bash
# Perfilado con Nsight Compute.  Uso:  bash profile_ncu.sh
set -uo pipefail
cd "$(dirname "$0")"
PY=./.venv/bin/python
mkdir -p ncu_out

# lts__t_sector_hit_rate es LA metrica decisiva: si el precipicio de BLOCK es
# por L2, el acierto de L2 debe hundirse cuando el working set la desborda.
METRICS="lts__t_sector_hit_rate.pct,\
l1tex__t_sector_hit_rate.pct,\
dram__bytes.sum.per_second,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__thread_inst_executed_per_inst_executed.ratio,\
gpu__time_duration.sum"

NCU=ncu
# Los contadores requieren permisos: o el modulo con
# NVreg_RestrictProfilingToAdminUsers=0, o sudo.
if ! $NCU --version >/dev/null 2>&1; then echo "ncu no encontrado"; exit 1; fi
if ! $NCU --metrics gpu__time_duration.sum $PY -c "pass" >/dev/null 2>&1; then
    echo "sin permisos de contadores -> usando sudo"
    NCU="sudo $NCU"
fi

for block in 64 128 256 512 1024; do
  for codec in huffman ladder floor; do
    echo "-- BLOCK=$block codec=$codec"
    $NCU --csv --target-processes all \
        --kernel-name "regex:decode_|mem_floor" \
        --launch-skip 1 --launch-count 1 \
        --metrics "$METRICS" \
        $PY profile_run.py --block $block --threads 128 --codec $codec \
        > "ncu_out/b${block}_${codec}.csv" 2>&1
  done
done
echo "perfilado en ncu_out/"
