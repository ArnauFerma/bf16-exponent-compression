#!/usr/bin/env python3
"""
Banco de pruebas: Huffman canonico vs escalera, sobre pesos sinteticos y
sobre pesos reales de un modelo (Qwen3-0.6B). Fase 1 del HANDOFF.

Para el fichero real (751M pesos) NO se construye el bitstream completo en
Python puro (demasiado lento / no es el objetivo aqui). El tamano
comprimido se calcula analiticamente a partir de las longitudes de codigo
y los conteos exactos por exponente (exacto, no estimado). La
verificacion de roundtrip bit a bit se hace sobre una muestra aleatoria
grande usando la MISMA tabla de codigos derivada de la distribucion
completa -- un codigo de prefijo es "memoryless" por simbolo, asi que
probar la muestra prueba la correccion del codec igual de bien que
probar el fichero entero.
"""
import json, sys, heapq
import numpy as np

sys.path.insert(0, ".")
from df11_reference import canonical_huffman
import ladder_codec as lc

BLOCK_DEFAULT = 256
IDX_WIDTH_BYTES = 4   # offset de bloque como uint32 (como en la spec del handoff)


def huffman_analytic(counts):
    lengths, _codes = canonical_huffman(counts)
    bits = sum(int(counts[s]) * l for s, l in lengths.items())
    return lengths, bits


def index_overhead_bytes(n, block, width=IDX_WIDTH_BYTES):
    n_blocks = (n + block - 1) // block
    return n_blocks * width, n_blocks


def report_file(label, expo_counts, n_weights, blocks=(64, 128, 256, 512, 1024),
                 shapes=None):
    if shapes is None:
        shapes = {
            "(1,1,1,2)": [1, 1, 1, 2],
            "(1,1,2,2)": [1, 1, 2, 2],
            "(1,2,2,3)": [1, 2, 2, 3],
        }

    total = int(expo_counts.sum())
    p = expo_counts[expo_counts > 0].astype(np.float64) / total
    H = float(-(p * np.log2(p)).sum())
    n_distinct = int((expo_counts > 0).sum())

    print(f"\n{'='*70}\n{label}\n{'='*70}")
    print(f"pesos: {n_weights:,}   simbolos distintos de exponente: {n_distinct}")
    print(f"entropia del exponente: {H:.4f} bits")

    sm_bytes = n_weights   # signo+mantisa: 1 byte/peso, sin comprimir

    results = {}

    # --- Huffman canonico, BLOCK=256 ---
    h_lengths, h_bits = huffman_analytic(expo_counts)
    idx_bytes, n_blocks = index_overhead_bytes(n_weights, BLOCK_DEFAULT)
    h_exp_bytes = (h_bits + 7) // 8
    h_table_bytes = 256  # 1 byte de longitud por simbolo (peor caso)
    h_total = h_exp_bytes + sm_bytes + idx_bytes + h_table_bytes
    h_reduction = 100 * (1 - h_total / (n_weights * 2))
    print(f"\nHuffman canonico (BLOCK={BLOCK_DEFAULT}):")
    print(f"  avg bits/exponente : {h_bits/total:.4f}")
    print(f"  exponentes         : {h_exp_bytes:>12,} B")
    print(f"  signo+mantisa      : {sm_bytes:>12,} B")
    print(f"  indice bloques     : {idx_bytes:>12,} B  ({100*idx_bytes/h_total:.3f}%)")
    print(f"  tabla longitudes   : {h_table_bytes:>12,} B")
    print(f"  TOTAL              : {h_total:>12,} B   reduccion {h_reduction:.2f}%")
    results["huffman"] = dict(avg_bits=h_bits/total, total_bytes=h_total, reduction_pct=h_reduction)

    # --- Escalera, varias formas, BLOCK=256 ---
    results["ladder"] = {}
    for shape_name, rung_bits in shapes.items():
        lengths, slots, escape = lc.build_ladder(expo_counts, rung_bits)
        bits = lc.analytic_bits(expo_counts, lengths)
        kraft = lc.kraft_sum(lengths)
        hdr_bytes = lc.header_bytes(slots, rung_bits)
        exp_bytes = (bits + 7) // 8
        total_l = exp_bytes + sm_bytes + idx_bytes + hdr_bytes
        red = 100 * (1 - total_l / (n_weights * 2))
        gap = h_reduction - red
        print(f"\nEscalera {shape_name} (BLOCK={BLOCK_DEFAULT}):")
        print(f"  avg bits/exponente : {bits/total:.4f}   Kraft={kraft:.4f}")
        print(f"  exponentes         : {exp_bytes:>12,} B")
        print(f"  cabecera (tabla)   : {hdr_bytes:>12,} B")
        print(f"  TOTAL              : {total_l:>12,} B   reduccion {red:.2f}%   "
              f"({'+' if gap<=0 else '-'}{abs(gap):.2f} pts vs Huffman)")
        results["ladder"][shape_name] = dict(avg_bits=bits/total, total_bytes=total_l,
                                              reduction_pct=red, kraft=kraft,
                                              header_bytes=hdr_bytes)

    # --- barrido de BLOCK, forma por defecto (1,1,1,2) ---
    print(f"\nBarrido de BLOCK (escalera (1,1,1,2), indice uint32):")
    lengths11, slots11, escape11 = lc.build_ladder(expo_counts, [1, 1, 1, 2])
    bits11 = lc.analytic_bits(expo_counts, lengths11)
    exp_bytes11 = (bits11 + 7) // 8
    hdr11 = lc.header_bytes(slots11, [1, 1, 1, 2])
    results["block_sweep"] = {}
    for blk in blocks:
        idx_b, n_blk = index_overhead_bytes(n_weights, blk)
        tot = exp_bytes11 + sm_bytes + idx_b + hdr11
        red = 100 * (1 - tot / (n_weights * 2))
        print(f"  BLOCK={blk:5d}  bloques={n_blk:9,}  indice={idx_b:10,} B "
              f"({100*idx_b/tot:.3f}%)  TOTAL={tot:12,} B  reduccion={red:.2f}%")
        results["block_sweep"][blk] = dict(index_bytes=idx_b, total_bytes=tot, reduction_pct=red)

    return results


def roundtrip_check(bf16_array_or_memmap, sample_size, expo_counts, label,
                     rung_bits=(1, 1, 1, 2), block=256, seed=0):
    """Verifica roundtrip bit a bit de Huffman y escalera sobre una muestra
    aleatoria grande, usando la tabla de codigos derivada de la
    distribucion COMPLETA (expo_counts)."""
    n = len(bf16_array_or_memmap)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=sample_size)
    sample = np.asarray(bf16_array_or_memmap[idx])   # copia pequena a RAM
    expo_sample = ((sample >> 7) & 0xFF).astype(np.int64)

    print(f"\n--- roundtrip check ({label}, muestra={sample_size:,}) ---")

    # Huffman
    h_lengths, h_codes = canonical_huffman(expo_counts)
    w = lc.BitWriter()
    offsets = []
    for i, e in enumerate(expo_sample):
        if i % block == 0:
            offsets.append(w.bitpos())
        w.write(h_codes[int(e)], h_lengths[int(e)])
    data = w.flush()
    # decode
    table = {(h_lengths[s], h_codes[s]): s for s in h_codes}
    maxlen = max(h_lengths.values())
    out = np.empty(len(expo_sample), dtype=np.int64)
    for b in range(len(offsets)):
        r = lc.BitReader(data, offsets[b])
        start, end = b * block, min(b * block + block, len(expo_sample))
        for i in range(start, end):
            code, ln = 0, 0
            while True:
                code = (code << 1) | r.read_bit(); ln += 1
                if (ln, code) in table:
                    out[i] = table[(ln, code)]; break
    ok_h = np.array_equal(out, expo_sample)
    print(f"  Huffman  roundtrip lossless: {'SI' if ok_h else 'NO'}")

    # Escalera
    lengths, slots, escape = lc.build_ladder(expo_counts, list(rung_bits))
    encode_of = lc.make_code_tables(slots, list(rung_bits), escape)
    data_l, offsets_l = lc.encode_symbols(expo_sample, encode_of, block=block)
    rec_l = lc.decode_symbols(data_l, offsets_l, len(expo_sample), slots,
                               list(rung_bits), lc.RAW_BITS, block=block)
    ok_l = np.array_equal(rec_l, expo_sample)
    n_escape_hit = sum(1 for e in expo_sample if int(e) in escape)
    print(f"  Escalera roundtrip lossless: {'SI' if ok_l else 'NO'}  "
          f"(muestra toco el escape {n_escape_hit} veces)")

    return ok_h, ok_l


if __name__ == "__main__":
    all_results = {}

    # ---------- sintetico ----------
    syn = np.fromfile("outputs/weights_bf16.bin", dtype=np.uint16)
    syn_expo = (syn >> 7) & 0xFF
    syn_counts = np.bincount(syn_expo, minlength=256)
    all_results["synthetic"] = report_file("SINTETICO (gen_weights_bf16.py, seed 1234)",
                                            syn_counts, len(syn))
    roundtrip_check(syn, min(2_000_000, len(syn)), syn_counts, "sintetico")

    # ---------- real ----------
    real_counts = np.load("outputs/real_exp_counts.npy")
    real_n = int(real_counts.sum())
    all_results["real_qwen3_0.6b"] = report_file("REAL: Qwen/Qwen3-0.6B (model.safetensors, todos los tensores BF16)",
                                                   real_counts, real_n)
    real_mm = np.memmap("outputs/real_weights_bf16.bin", dtype=np.uint16, mode="r")
    roundtrip_check(real_mm, 2_000_000, real_counts, "real Qwen3-0.6B")

    with open("outputs/bench_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nResultados guardados en outputs/bench_results.json")
