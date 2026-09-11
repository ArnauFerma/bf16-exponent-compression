#!/usr/bin/env python3
"""Huffman vs escalera CON la optimizacion del patron de acceso.

El resultado de la Fase 2 (la escalera no aporta nada) se midio con el kernel
lento. Al mover el cuello de botella, el ranking puede cambiar: hay que
rehacer la comparacion, no extrapolarla.
"""
import sys
import numpy as np, cupy as cp
import bitpack as bp, gpu_kernels as gk, kernel_opt as ko
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

hdr = f"{'BLOCK':>6}{'thr':>5} {'variante':>12} | {'huff ms':>9}{'ladd ms':>9} | {'huff GB/s':>10}{'ladd GB/s':>10} | {'l/h':>7}  ok"
print(hdr); print("-" * len(hdr))
for block in (64, 128, 256, 512, 1024):
    for threads in (32, 64, 128):
        for si, so in ((0, 1), (1, 0)):
            row, okall = {}, True
            for name in ("huffman", "ladder"):
                d_words, starts, tb = streams[name]
                offs = to_u32_index(starts[::block], tb)
                d_offs = cp.asarray(offs); nb = offs.size
                smw = ko.max_words_per_cudablock(offs, threads, tb)
                nbytes = (smw*4 if si else 0) + (threads*block if so else 0)
                if nbytes > 48*1024:
                    okall = None; break
                if name == "huffman":
                    fn = lambda: ko.launch_h(d_words, d_offs, d_out, th, nb, block,
                                             expo.size, tb, threads, si, so, smw)
                else:
                    fn = lambda: ko.launch(d_words, d_offs, d_out, d_slots, nb, block,
                                           expo.size, tb, threads, si, so, smw)
                d_out.fill(0); fn(); cp.cuda.Stream.null.synchronize()
                okall &= bool((cp.asnumpy(d_out) == expo).all())
                ms, _, _ = measure(fn)
                row[name] = (ms, (tb/8 + expo.size + offs.nbytes)/(ms*1e-3)/1e9)
            if okall is None:
                continue
            h, l = row["huffman"], row["ladder"]
            print(f"{block:6d}{threads:5d} {f'in={si} out={so}':>12} | "
                  f"{h[0]:8.2f} {l[0]:8.2f} | {h[1]:10.2f}{l[1]:10.2f} | "
                  f"{l[0]/h[0]:7.3f}  {'si' if okall else 'NO'}")
    print()
