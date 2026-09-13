#!/usr/bin/env bash
# All measurements + profiling. Usage:  bash run_all.sh
set -uo pipefail
cd "$(dirname "$0")"
PY=./.venv/bin/python
# results/<gpu-name>/, e.g. results/gtx1050ti/, results/a100/
SLUG=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 \
       | sed -E 's/NVIDIA |GeForce |Tesla //g; s/[^A-Za-z0-9]//g' | tr 'A-Z' 'a-z')
OUT=results/${SLUG:-unknown-gpu}
mkdir -p $OUT

echo "== environment =="
$PY env_info.py $OUT/env_info.json

echo "== trying to lock the GPU clocks =="
# On Linux this usually works on consumer cards (not under Windows/WDDM).
# It removes DVFS noise entirely. If it fails, no harm done: the scripts
# already compensate with warm-up and medians.
if sudo -n true 2>/dev/null; then
    sudo nvidia-smi -pm 1 || true
    MAXCLK=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)
    sudo nvidia-smi -lgc "${MAXCLK},${MAXCLK}" || echo "  (not supported, continuing)"
else
    echo "  no passwordless sudo; skipping (optional)"
fi
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm --format=csv | tee $OUT/gpu_info.txt

echo -e "\n== 1/5  base sweep (Huffman vs ladder, original kernel) =="
$PY bench_gpu.py 64000000 2>&1 | tee $OUT/1_bench_base.txt | tail -20

echo -e "\n== 2/5  access-pattern variants (attribution) =="
$PY bench_opt.py 64000000 2>&1 | tee $OUT/2_bench_opt.txt | tail -25

echo -e "\n== 3/5  Huffman vs ladder WITH the optimisation =="
$PY bench_head2head.py 64000000 2>&1 | tee $OUT/3_head2head.txt | tail -25

echo -e "\n== 4/5  8-bit index + prefix-sum =="
$PY bench_idx8.py 64000000 2>&1 | tee $OUT/4_bench_idx8.txt | tail -20

echo -e "\n== 5/5  Nsight Compute profiling =="
if command -v ncu >/dev/null 2>&1; then
    bash profile_ncu.sh 2>&1 | tail -20
    cp -r ncu_out $OUT/ 2>/dev/null || true
else
    echo "ncu not installed; skipping (see OPERATOR_LINUX.md step 2)"
fi

cp outputs/gpu_bench.json $OUT/ 2>/dev/null || true
echo -e "\n== collecting results =="
tar czf results.tar.gz -C results "$(basename $OUT)"
echo "DONE -> results.tar.gz  (send this file)"

if sudo -n true 2>/dev/null; then sudo nvidia-smi -rgc || true; fi
