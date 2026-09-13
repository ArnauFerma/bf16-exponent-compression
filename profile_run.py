#!/usr/bin/env python3
"""
A single kernel launch, for profiling with Nsight Compute (ncu).

bench_gpu.py does hundreds of launches (warm-up + 11 repetitions x 45
configurations): unworkable under ncu, which replays each kernel several
times to read the counters. Here it is exactly 1 warm-up + 1 measured, and
the second is profiled with --launch-skip 1 --launch-count 1.

N defaults to 64M and should NOT be lowered: at BLOCK=1024 that is 62,500
threads, just above the 43,008 resident on an RTX 3060. With a smaller N the
GPU is not full and the measured occupancy means nothing.

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

# On-disk cache: profile_ncu.ps1 calls this script 15 times and encoding
# 64M symbols in numpy takes ~1 min. Without it, ~15 min wasted.
cache = f"outputs/cache_{a.codec if a.codec != 'floor' else 'ladder'}_{a.n}.npz"
if os.path.exists(cache):
    z = np.load(cache)
    packed, lens = z["packed"], z["lens"]
    print(f"cache: {cache}")
else:
    packed, _, _ = bp.encode_stream(expo, code_of, len_of, a.n)
    lens = len_of[expo]                      # lengths <= 30, fit in uint8
    np.savez(cache, packed=packed, lens=lens)
    print(f"cache written: {cache}")

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

for _ in range(2):                    # 1 warm-up + 1 profiled
    k((grid,), (a.threads,), args)
    cp.cuda.Stream.null.synchronize()

print(f"ok codec={a.codec} BLOCK={a.block} threads={a.threads} "
      f"grid={grid} threads={nb:,}")
