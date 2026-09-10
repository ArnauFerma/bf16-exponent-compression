#!/usr/bin/env python3
"""Correccion de los dos kernels contra el array de exponentes original."""
import numpy as np, cupy as cp, sys
import bitpack as bp, gpu_kernels as gk

BLOCK = int(sys.argv[1]) if len(sys.argv) > 1 else 256
N = int(sys.argv[2]) if len(sys.argv) > 2 else 32_000_000

mm = np.memmap("outputs/real_weights_bf16.bin", dtype=np.uint16, mode="r")
counts = np.load("outputs/real_exp_counts.npy")          # distribucion COMPLETA
chunk = np.asarray(mm[:N])
expo = ((chunk >> 7) & 0xFF).astype(np.uint8)
print(f"simbolos: {expo.size:,}   BLOCK={BLOCK}   (tablas de la distribucion completa)")

def check(name, code_of, len_of, launch):
    packed, offs, nbits = bp.encode_stream(expo, code_of, len_of, BLOCK)
    words = bp.to_words(packed, nbits)
    n_blocks = offs.size
    assert nbits < 2**32, 'indice uint32 desbordado'
    d_words = cp.asarray(words); d_offs = cp.asarray(offs.astype(np.uint32))
    d_out = cp.zeros(expo.size, dtype=cp.uint8)
    launch(d_words, d_offs, d_out, n_blocks)
    cp.cuda.Stream.null.synchronize()
    got = cp.asnumpy(d_out)
    ok = np.array_equal(got, expo)
    print(f"  {name:9} bits={nbits:,}  bloques={n_blocks:,}  "
          f"roundtrip GPU bit a bit: {'SI' if ok else 'NO'}")
    if not ok:
        bad = np.flatnonzero(got != expo)
        print(f"    fallos: {bad.size:,}  primeros: {bad[:5]}  "
              f"esperado {expo[bad[:5]]} obtenido {got[bad[:5]]}")
    return ok

# --- Huffman ---
code_of, len_of, lengths, codes = bp.huffman_code_arrays(counts)
t = gk.build_huffman_gpu_tables(lengths, codes)
t.update({f"d_{k}": cp.asarray(t[k]) for k in
          ("lut", "first_code", "first_index", "cnt", "sorted_syms")})
print(f"  Huffman maxlen={t['maxlen']}  LUT={1<<gk.LUT_BITS} entradas "
      f"({(1<<gk.LUT_BITS)*2} B en SRAM)")
ok_h = check("huffman", code_of, len_of,
             lambda w, o, out, nb: gk.run_huffman(w, o, out, t, nb, BLOCK, expo.size))

# --- Escalera ---
code_of2, len_of2, slots, escape = bp.ladder_code_arrays(counts)
flat = gk.ladder_slots_flat(slots)
d_slots = cp.asarray(flat)
n_esc = int(np.isin(expo, list(escape)).sum())
print(f"  Escalera tabla={sum(len(c) for c in slots)} B  "
      f"escape tocado {n_esc:,} veces ({100*n_esc/expo.size:.3f}%)")
ok_l = check("escalera", code_of2, len_of2,
             lambda w, o, out, nb: gk.run_ladder(w, o, out, d_slots, nb, BLOCK, expo.size))

print("\nRESULTADO:", "ambos kernels correctos" if (ok_h and ok_l) else "FALLO")
sys.exit(0 if (ok_h and ok_l) else 1)
