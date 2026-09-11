#!/usr/bin/env python3
"""Indice uint32 vs indice de 8 bits + prefix-sum: correccion, velocidad y
compresion. Ambos con la salida en shared (la variante ganadora)."""
import sys
import numpy as np, cupy as cp
import bitpack as bp, gpu_kernels as gk, kernel_opt as ko, kernel_idx8 as k8
from bench_gpu import to_u32_index, measure

N = int(sys.argv[1]) if len(sys.argv) > 1 else 64_000_000
mm = np.memmap("outputs/real_weights_bf16.bin", dtype=np.uint16, mode="r")
counts = np.load("outputs/real_exp_counts.npy")
expo = ((np.asarray(mm[:N]) >> 7) & 0xFF).astype(np.uint8)
d_out = cp.zeros(expo.size, dtype=cp.uint8)
print(f"simbolos: {expo.size:,}\n")

co_h, lo_h, lengths, codes = bp.huffman_code_arrays(counts)
th = gk.build_huffman_gpu_tables(lengths, codes)
th.update({f"d_{k}": cp.asarray(th[k]) for k in
           ("lut", "first_code", "first_index", "cnt", "sorted_syms")})
co_l, lo_l, slots, escape = bp.ladder_code_arrays(counts)
d_slots = cp.asarray(gk.ladder_slots_flat(slots))

streams = {}
for name, (co, lo) in (("huffman", (co_h, lo_h)), ("ladder", (co_l, lo_l))):
    lens = lo[expo].astype(np.int64)
    starts = np.cumsum(lens) - lens
    tb = int(starts[-1] + lens[-1])
    packed, _, _ = bp.encode_stream(expo, co, lo, N)
    streams[name] = (cp.asarray(bp.to_words(packed, tb)), starts, tb)

def reduction(avg_bits, block, idx_b):
    return 100 * (1 - (1.0 + avg_bits/8 + idx_b/block)/2)

hdr = (f"{'BLOCK':>6}{'thr':>5}{'codec':>9}{'indice':>10} | {'ms':>8}{'GB/s':>8} | "
       f"{'B/bloque':>9}{'compresion':>11} | ok")
print(hdr); print("-" * len(hdr))
for block in (64, 128, 256):
    for threads in (128,):
        for name in ("huffman", "ladder"):
            d_words, starts, tb = streams[name]
            avg_bits = tb / expo.size
            offs32 = to_u32_index(starts[::block], tb)
            d_offs = cp.asarray(offs32); nb = offs32.size
            grid = (nb + threads - 1) // threads
            smw = ko.max_words_per_cudablock(offs32, threads, tb)
            moved32 = tb/8 + expo.size + offs32.nbytes

            # --- referencia: uint32 + salida en shared
            if name == "huffman":
                f32 = lambda: ko.launch_h(d_words, d_offs, d_out, th, nb, block,
                                          expo.size, tb, threads, 0, 1, smw)
            else:
                f32 = lambda: ko.launch(d_words, d_offs, d_out, d_slots, nb, block,
                                        expo.size, tb, threads, 0, 1, smw)
            d_out.fill(0); f32(); cp.cuda.Stream.null.synchronize()
            ok32 = bool((cp.asnumpy(d_out) == expo).all())
            ms32, _, _ = measure(f32)
            print(f"{block:6d}{threads:5d}{name:>9}{'uint32':>10} | {ms32:7.2f} "
                  f"{moved32/(ms32*1e-3)/1e9:7.2f} | {4.0:9.3f}"
                  f"{reduction(avg_bits, block, 4.0):10.2f}% | {'si' if ok32 else 'NO'}")

            # --- nuevo: 8 bits + prefix-sum
            try:
                lc, sb, minlen, nb2, idx_b = k8.build_index8(starts, block, tb, threads)
            except ValueError as e:
                print(f"{block:6d}{threads:5d}{name:>9}{'uint8':>10} |  {e}")
                continue
            d_lc, d_sb = cp.asarray(lc), cp.asarray(sb)
            shared = threads * block
            if name == "huffman":
                f8 = lambda: k8.huffman_idx8()((grid,), (threads,), (
                    d_words, d_sb, d_lc, d_out, th["d_lut"], th["d_first_code"],
                    th["d_first_index"], th["d_cnt"], th["d_sorted_syms"],
                    np.int32(th["maxlen"]), np.int32(minlen), np.int32(nb),
                    np.int32(block), np.int64(expo.size)), shared_mem=shared)
            else:
                f8 = lambda: k8.ladder_idx8()((grid,), (threads,), (
                    d_words, d_sb, d_lc, d_out, d_slots, np.int32(minlen),
                    np.int32(nb), np.int32(block), np.int64(expo.size)),
                    shared_mem=shared)
            d_out.fill(0); f8(); cp.cuda.Stream.null.synchronize()
            ok8 = bool((cp.asnumpy(d_out) == expo).all())
            ms8, _, _ = measure(f8)
            moved8 = tb/8 + expo.size + lc.nbytes + sb.nbytes
            print(f"{block:6d}{threads:5d}{name:>9}{'uint8+ps':>10} | {ms8:7.2f} "
                  f"{moved8/(ms8*1e-3)/1e9:7.2f} | {idx_b:9.3f}"
                  f"{reduction(avg_bits, block, idx_b):10.2f}% | {'si' if ok8 else 'NO'}"
                  f"   ({ms32/ms8:.3f}x tiempo, "
                  f"{reduction(avg_bits,block,idx_b)-reduction(avg_bits,block,4.0):+.2f} pts)")
    print()
