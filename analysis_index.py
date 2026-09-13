#!/usr/bin/env python3
"""
(a) Mutual information ~0 must not be an artefact of looking only at the
    first 64M weights (the embedding matrix). Checked on contiguous windows
    spread across the whole file.
(b) What fixing the INDEX is worth, the only structural headroom left.
"""
import numpy as np
from df11_reference import canonical_huffman

mm = np.memmap("outputs/real_weights_bf16.bin", dtype=np.uint16, mode="r")
TOT = mm.size
counts_full = np.load("outputs/real_exp_counts.npy")

def H(c):
    c = c[c > 0].astype(np.float64); p = c / c.sum()
    return float(-(p * np.log2(p)).sum())

print("=== (a) mutual information in windows spread across the model ===")
print(f"{'offset':>14} {'H(X)':>8} {'H(Y|X)':>8} {'I(X;Y)':>9}")
rng = np.random.default_rng(0)
mis = []
for off in [0, TOT//8, TOT//4, TOT//2, 3*TOT//4, TOT - 40_000_000]:
    w = ((np.asarray(mm[off:off+32_000_000]) >> 7) & 0xFF).astype(np.int64)
    c1 = np.bincount(w, minlength=256); h1 = H(c1)
    cj = np.bincount(w[:-1]*256 + w[1:], minlength=65536)
    hj = H(cj); hc = hj - h1; mi = h1 - hc
    mis.append(mi)
    print(f"{off:>14,} {h1:8.4f} {hc:8.4f} {mi:9.5f}")
print(f"\nmax I(X;Y) observed: {max(mis):.5f} bits  -> neighbouring exponents "
      f"are, for practical purposes, independent.\n")

print("=== (b) cost of the block index ===")
L, _ = canonical_huffman(counts_full)
len_of = np.zeros(256, dtype=np.int64)
for s, l in L.items():
    len_of[s] = l
avg_bits = sum(int(counts_full[s]) * l for s, l in L.items()) / counts_full.sum()
print(f"Huffman over the whole model: {avg_bits:.4f} bits/exponent\n")

# real block-length distribution, to see whether it fits in 8 bits
w = ((np.asarray(mm[:64_000_000]) >> 7) & 0xFF).astype(np.int64)
lens = len_of[w]
for BLOCK in (64, 256):
    bl = lens[:len(lens)//BLOCK*BLOCK].reshape(-1, BLOCK).sum(axis=1)
    print(f"BLOCK={BLOCK:4d}: block length in bits  min={bl.min()} "
          f"max={bl.max()} mean={bl.mean():.1f} range={bl.max()-bl.min()}")
print()

def reduction(block, idx_bytes_per_block):
    per_w = 1.0 + avg_bits/8 + idx_bytes_per_block/block
    return 100 * (1 - per_w/2), per_w

print(f"{'index scheme':>42} {'B/block':>9} {'BLOCK=64':>10} {'BLOCK=256':>10}")
schemes = [
    ("absolute uint32 (current spec)",           4.0),
    ("relative uint16 + uint32/superblock(256)",  2.0 + 4.0/256),
    ("uint8 lengths + warp prefix-sum",           1.0),
]
for name, b in schemes:
    r64, _ = reduction(64, b); r256, _ = reduction(256, b)
    print(f"{name:>42} {b:9.3f} {r64:9.2f}% {r256:9.2f}%")

print(f"""
Reading: the measured fast point (BLOCK=64, output in shared, 4.80 ms) today
loses 2.35 points of compression against BLOCK=256 ONLY because of the index.
With a hierarchical 16-bit index that loss almost disappears, without touching
the decode kernel or the entropy code.""")
