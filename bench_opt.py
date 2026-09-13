#!/usr/bin/env python3
"""Correctness + timing of the 4 variants of the optimised kernel."""
import sys, time
import numpy as np, cupy as cp
import bitpack as bp, gpu_kernels as gk, kernel_opt as ko
from bench_gpu import _floor_k, to_u32_index, measure

N = int(sys.argv[1]) if len(sys.argv) > 1 else 64_000_000
SMEM_LIMIT = 48 * 1024

mm = np.memmap("outputs/real_weights_bf16.bin", dtype=np.uint16, mode="r")
counts = np.load("outputs/real_exp_counts.npy")
expo = ((np.asarray(mm[:N]) >> 7) & 0xFF).astype(np.uint8)
print(f"symbols: {expo.size:,}\n")

code_of, len_of, slots, escape = bp.ladder_code_arrays(counts)
d_slots = cp.asarray(gk.ladder_slots_flat(slots))
lens = len_of[expo].astype(np.int64)
starts = np.cumsum(lens) - lens
total_bits = int(starts[-1] + lens[-1])
packed, _, _ = bp.encode_stream(expo, code_of, len_of, N)
d_words = cp.asarray(bp.to_words(packed, total_bits))
d_out = cp.zeros(expo.size, dtype=cp.uint8)

hdr = f"{'BLOCK':>6}{'thr':>5} {'variant':>22} {'shared':>8} {'ms':>9} {'GB/s':>7} {'vs base':>8}  ok"
print(hdr); print("-" * len(hdr))

for block in (64, 128, 256, 512, 1024):
    for threads in (32, 64, 128, 256):
        offs = to_u32_index(starts[::block], total_bits)
        d_offs = cp.asarray(offs)
        nb = offs.size
        grid = (nb + threads - 1) // threads
        smw = ko.max_words_per_cudablock(offs, threads, total_bits)
        moved = total_bits / 8 + expo.size + offs.nbytes

        # baseline: the current kernel from gpu_kernels.py
        base_ms, _, _ = measure(lambda: gk.ladder_kernel()(
            (grid,), (threads,), (d_words, d_offs, d_out, d_slots,
             np.int32(nb), np.int32(block), np.int64(expo.size))))
        d_out.fill(0)
        gk.ladder_kernel()((grid,), (threads,), (d_words, d_offs, d_out, d_slots,
                            np.int32(nb), np.int32(block), np.int64(expo.size)))
        cp.cuda.Stream.null.synchronize()
        ok = bool((cp.asnumpy(d_out) == expo).all())
        print(f"{block:6d}{threads:5d} {'base (global)':>22} {'-':>8} "
              f"{base_ms:8.2f} {moved/(base_ms*1e-3)/1e9:7.2f} {'1.00':>8}  {'yes' if ok else 'NO'}")

        for si, so in ((1, 0), (0, 1), (1, 1)):
            nbytes = (smw * 4 if si else 0) + (threads * block if so else 0)
            name = f"smem in={si} out={so}"
            if nbytes > SMEM_LIMIT:
                print(f"{block:6d}{threads:5d} {name:>22} {nbytes:7d}B "
                      f"{'--- exceeds 48 KiB ---':>26}")
                continue
            d_out.fill(0)
            ko.launch(d_words, d_offs, d_out, d_slots, nb, block, expo.size,
                      total_bits, threads, si, so, smw)
            cp.cuda.Stream.null.synchronize()
            ok = bool((cp.asnumpy(d_out) == expo).all())
            ms, _, _ = measure(lambda: ko.launch(
                d_words, d_offs, d_out, d_slots, nb, block, expo.size,
                total_bits, threads, si, so, smw))
            print(f"{block:6d}{threads:5d} {name:>22} {nbytes:7d}B "
                  f"{ms:8.2f} {moved/(ms*1e-3)/1e9:7.2f} {base_ms/ms:7.2f}x  "
                  f"{'yes' if ok else 'NO'}")
    print()
