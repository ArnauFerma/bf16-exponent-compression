#!/usr/bin/env python3
"""
Un solo lanzamiento de kernel, para perfilar con Nsight Compute (ncu).

bench_gpu.py hace cientos de lanzamientos (calentamiento + 11 repeticiones x
45 configuraciones): inviable bajo ncu, que reproduce cada kernel varias veces
para leer los contadores. Aqui se hace exactamente 1 calentamiento + 1 medido,
y se perfila el segundo con --launch-skip 1 --launch-count 1.

N por defecto es 64M y NO conviene bajarlo: con BLOCK=1024 son 62.500 hilos,
justo por encima de los 43.008 residentes de una RTX 3060. Con N mas pequeno
no se llena la GPU y la ocupacion medida no significa nada.

    python profile_run.py --block 256 --threads 128 --codec ladder
"""
import argparse, os
import numpy as np, cupy as cp
import bitpack as bp, gpu_kernels as gk
from bench_gpu import _floor_k, to_u32_index

ap = argparse.ArgumentParser()
ap.add_argument("--block", type=int, default=256)
ap.add_argument("--threads", type=int, default=128)
ap.add_argument("--codec", choices=("huffman", "ladder", "floor"), default="ladder")
ap.add_argument("--n", type=int, default=64_000_000)
a = ap.parse_args()

mm = np.memmap("outputs/real_weights_bf16.bin", dtype=np.uint16, mode="r")
counts = np.load("outputs/real_exp_counts.npy")
expo = ((np.asarray(mm[:a.n]) >> 7) & 0xFF).astype(np.uint8)

src = "ladder" if a.codec == "floor" else a.codec
if src == "huffman":
    code_of, len_of, lengths, codes = bp.huffman_code_arrays(counts)
    t = gk.build_huffman_gpu_tables(lengths, codes)
    t.update({f"d_{k}": cp.asarray(t[k]) for k in
              ("lut", "first_code", "first_index", "cnt", "sorted_syms")})
else:
    code_of, len_of, slots, escape = bp.ladder_code_arrays(counts)
    d_slots = cp.asarray(gk.ladder_slots_flat(slots))

# Cache en disco: profile_ncu.ps1 llama a este script 15 veces y codificar
# 64M simbolos en numpy tarda ~1 min. Sin cache serian ~15 min tirados.
cache = f"outputs/cache_{a.codec if a.codec != 'floor' else 'ladder'}_{a.n}.npz"
if os.path.exists(cache):
    z = np.load(cache)
    packed, lens = z["packed"], z["lens"]
    print(f"cache: {cache}")
else:
    packed, _, _ = bp.encode_stream(expo, code_of, len_of, a.n)
    lens = len_of[expo]                      # longitudes <= 30, caben en uint8
    np.savez(cache, packed=packed, lens=lens)
    print(f"cache escrito: {cache}")

lens = lens.astype(np.int64)
starts = np.cumsum(lens) - lens
total_bits = int(starts[-1] + lens[-1])
d_words = cp.asarray(bp.to_words(packed, total_bits))
offs = to_u32_index(starts[::a.block], total_bits)
d_offs = cp.asarray(offs)
d_out = cp.zeros(expo.size, dtype=cp.uint8)
nb = offs.size
grid = (nb + a.threads - 1) // a.threads

if a.codec == "huffman":
    k, args = gk.huffman_kernel(), (
        d_words, d_offs, d_out, t["d_lut"], t["d_first_code"], t["d_first_index"],
        t["d_cnt"], t["d_sorted_syms"], np.int32(t["maxlen"]), np.int32(nb),
        np.int32(a.block), np.int64(expo.size))
elif a.codec == "ladder":
    k, args = gk.ladder_kernel(), (
        d_words, d_offs, d_out, d_slots, np.int32(nb), np.int32(a.block),
        np.int64(expo.size))
else:
    k, args = _floor_k, (
        d_words, d_offs, d_out, np.int32(nb), np.int32(a.block),
        np.int64(expo.size), np.int64(total_bits))

for _ in range(2):                    # 1 calentamiento + 1 perfilado
    k((grid,), (a.threads,), args)
    cp.cuda.Stream.null.synchronize()

print(f"ok codec={a.codec} BLOCK={a.block} threads={a.threads} "
      f"grid={grid} hilos={nb:,}")
