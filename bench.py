#!/usr/bin/env python3
"""
Test bench: canonical Huffman vs ladder, on synthetic weights and on real
model weights (Qwen3-0.6B). Phase 1 of the HANDOFF.

For the real file (~600M weights) the full bitstream is NOT built in pure
Python (too slow, and not the point here). The compressed size is computed
analytically from the code lengths and the exact per-exponent counts (exact,
not estimated). The bit-exact roundtrip check runs on a large random sample
using the SAME code table derived from the full distribution -- a prefix
code is memoryless per symbol, so testing the sample tests the codec's
correctness as well as encoding the whole file would.
"""
import json, sys, heapq
import numpy as np

sys.path.insert(0, ".")
from df11_reference import canonical_huffman
import ladder_codec as lc

BLOCK_DEFAULT = 256
IDX_WIDTH_BYTES = 4   # block offset as uint32 (as in the handoff's spec)


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
    print(f"weights: {n_weights:,}   distinct exponent symbols: {n_distinct}")
    print(f"exponent entropy: {H:.4f} bits")

    sm_bytes = n_weights   # sign+mantissa: 1 byte/weight, uncompressed

    results = {}

    # --- canonical Huffman, BLOCK=256 ---
    h_lengths, h_bits = huffman_analytic(expo_counts)
    idx_bytes, n_blocks = index_overhead_bytes(n_weights, BLOCK_DEFAULT)
    h_exp_bytes = (h_bits + 7) // 8
    h_table_bytes = 256  # 1 length byte per symbol (worst case)
    h_total = h_exp_bytes + sm_bytes + idx_bytes + h_table_bytes
    h_reduction = 100 * (1 - h_total / (n_weights * 2))
    print(f"\nCanonical Huffman (BLOCK={BLOCK_DEFAULT}):")
    print(f"  avg bits/exponent  : {h_bits/total:.4f}")
    print(f"  exponents          : {h_exp_bytes:>12,} B")
    print(f"  sign+mantissa      : {sm_bytes:>12,} B")
    print(f"  block index        : {idx_bytes:>12,} B  ({100*idx_bytes/h_total:.3f}%)")
    print(f"  length table       : {h_table_bytes:>12,} B")
    print(f"  TOTAL              : {h_total:>12,} B   reduction {h_reduction:.2f}%")
    results["huffman"] = dict(avg_bits=h_bits/total, total_bytes=h_total, reduction_pct=h_reduction)

    # --- Ladder, several shapes, BLOCK=256 ---
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
        print(f"\nLadder {shape_name} (BLOCK={BLOCK_DEFAULT}):")
        print(f"  avg bits/exponent  : {bits/total:.4f}   Kraft={kraft:.4f}")
        print(f"  exponents          : {exp_bytes:>12,} B")
        print(f"  header (table)     : {hdr_bytes:>12,} B")
        print(f"  TOTAL              : {total_l:>12,} B   reduction {red:.2f}%   "
              f"({'+' if gap<=0 else '-'}{abs(gap):.2f} pts vs Huffman)")
        results["ladder"][shape_name] = dict(avg_bits=bits/total, total_bytes=total_l,
                                              reduction_pct=red, kraft=kraft,
                                              header_bytes=hdr_bytes)

    # --- BLOCK sweep, default shape (1,1,1,2) ---
    print(f"\nBLOCK sweep (ladder (1,1,1,2), uint32 index):")
    lengths11, slots11, escape11 = lc.build_ladder(expo_counts, [1, 1, 1, 2])
    bits11 = lc.analytic_bits(expo_counts, lengths11)
    exp_bytes11 = (bits11 + 7) // 8
    hdr11 = lc.header_bytes(slots11, [1, 1, 1, 2])
    results["block_sweep"] = {}
    for blk in blocks:
        idx_b, n_blk = index_overhead_bytes(n_weights, blk)
        tot = exp_bytes11 + sm_bytes + idx_b + hdr11
        red = 100 * (1 - tot / (n_weights * 2))
        print(f"  BLOCK={blk:5d}  blocks={n_blk:9,}  index={idx_b:10,} B "
              f"({100*idx_b/tot:.3f}%)  TOTAL={tot:12,} B  reduction={red:.2f}%")
        results["block_sweep"][blk] = dict(index_bytes=idx_b, total_bytes=tot, reduction_pct=red)

    return results


def roundtrip_check(bf16_array_or_memmap, sample_size, expo_counts, label,
                     rung_bits=(1, 1, 1, 2), block=256, seed=0):
    """Bit-exact roundtrip of Huffman and ladder on a large random sample,
    using the code table derived from the FULL distribution (expo_counts)."""
    n = len(bf16_array_or_memmap)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=sample_size)
    sample = np.asarray(bf16_array_or_memmap[idx])   # small copy into RAM
    expo_sample = ((sample >> 7) & 0xFF).astype(np.int64)

    print(f"\n--- roundtrip check ({label}, sample={sample_size:,}) ---")

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
    print(f"  Huffman  roundtrip lossless: {'YES' if ok_h else 'NO'}")

    # Ladder
    lengths, slots, escape = lc.build_ladder(expo_counts, list(rung_bits))
    encode_of = lc.make_code_tables(slots, list(rung_bits), escape)
    data_l, offsets_l = lc.encode_symbols(expo_sample, encode_of, block=block)
    rec_l = lc.decode_symbols(data_l, offsets_l, len(expo_sample), slots,
                               list(rung_bits), lc.RAW_BITS, block=block)
    ok_l = np.array_equal(rec_l, expo_sample)
    n_escape_hit = sum(1 for e in expo_sample if int(e) in escape)
    print(f"  Ladder   roundtrip lossless: {'YES' if ok_l else 'NO'}  "
          f"(sample hit the escape {n_escape_hit} times)")

    return ok_h, ok_l


if __name__ == "__main__":
    all_results = {}

    # ---------- synthetic ----------
    syn = np.fromfile("outputs/weights_bf16.bin", dtype=np.uint16)
    syn_expo = (syn >> 7) & 0xFF
    syn_counts = np.bincount(syn_expo, minlength=256)
    all_results["synthetic"] = report_file("SYNTHETIC (gen_weights_bf16.py, seed 1234)",
                                            syn_counts, len(syn))
    roundtrip_check(syn, min(2_000_000, len(syn)), syn_counts, "synthetic")

    # ---------- real ----------
    real_counts = np.load("outputs/real_exp_counts.npy")
    real_n = int(real_counts.sum())
    all_results["real_qwen3_0.6b"] = report_file("REAL: Qwen/Qwen3-0.6B (model.safetensors, all unique BF16 tensors)",
                                                   real_counts, real_n)
    real_mm = np.memmap("outputs/real_weights_bf16.bin", dtype=np.uint16, mode="r")
    roundtrip_check(real_mm, 2_000_000, real_counts, "real Qwen3-0.6B")

    with open("outputs/bench_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nResults saved to outputs/bench_results.json")
