#!/usr/bin/env python3
"""
Can vector (pairwise) coding gain anything on the exponent?

Two very different possible sources of gain:

  1. Code redundancy. Scalar Huffman already sits at 2.678 bits against an
     entropy of 2.645: only 0.033 bits of headroom. No entropy coder,
     arithmetic or vector, can beat that IF the symbols are independent.

  2. Correlation between neighbouring exponents. If it exists, H(X,Y) < 2*H(X)
     and pairwise (or context) coding captures the difference. This is NOT
     bounded by the 0.033 bits: it is new headroom.

2 is an empirical question. It is measured here.
"""
import numpy as np
from df11_reference import canonical_huffman
import ladder_codec as lc

N = 64_000_000
mm = np.memmap("outputs/real_weights_bf16.bin", dtype=np.uint16, mode="r")
expo = ((np.asarray(mm[:N]) >> 7) & 0xFF).astype(np.int64)

def H(counts):
    c = counts[counts > 0].astype(np.float64)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())

# ---- marginal
c1 = np.bincount(expo, minlength=256)
H1 = H(c1)

# ---- NON-overlapping adjacent pairs (what would actually be coded)
a, b = expo[0:2*(N//2):2], expo[1:2*(N//2):2]
cp = np.bincount(a * 256 + b, minlength=65536)
Hpair = H(cp)

# ---- overlapping pairs: measures the real correlation of the sequence
co = np.bincount(expo[:-1] * 256 + expo[1:], minlength=65536)
Hjoint = H(co)
Hcond = Hjoint - H1                    # H(Y|X)
MI = H1 - Hcond                        # I(X;Y)

print(f"symbols analysed        : {expo.size:,}")
print(f"H(X)  marginal          : {H1:.4f} bits")
print(f"H(X,Y) joint (overlap.) : {Hjoint:.4f} bits  -> {Hjoint/2:.4f} per symbol")
print(f"H(Y|X) conditional      : {Hcond:.4f} bits")
print(f"I(X;Y) mutual info      : {MI:.4f} bits   <-- the NEW headroom")
print()
print(f"scalar Huffman          : ", end="")
L1, _ = canonical_huffman(c1)
huff1 = sum(int(c1[s]) * l for s, l in L1.items()) / c1.sum()
print(f"{huff1:.4f} bits/symbol   (redundancy {huff1-H1:.4f})")

# ---- Huffman over pairs
L2, _ = canonical_huffman(cp)
huff2 = sum(int(cp[s]) * l for s, l in L2.items()) / cp.sum() / 2
n_pairs = int((cp > 0).sum())
print(f"Huffman over pairs      : {huff2:.4f} bits/symbol   "
      f"({n_pairs:,} distinct pairs, maxlen={max(L2.values())})")
print(f"  gain vs scalar        : {huff1-huff2:+.4f} bits/symbol "
      f"({100*(huff1-huff2)/huff1:+.2f}%)")

# ---- ladder over pairs: how many rungs are needed
print()
print("Ladder over pairs (rungs have to be widened):")
for shape in ([1,1,1,2], [2,2,3,4], [4,4,5,6], [5,5,6,7], [6,6,7,8]):
    lens, slots, esc = lc.build_ladder(cp, shape, raw_bits=16)
    bits = lc.analytic_bits(cp, lens) / cp.sum() / 2
    tab = sum(len(s) for s in slots) * 2      # 2 bytes per pair
    print(f"  {str(tuple(shape)):>16}  {bits:.4f} bits/symbol  "
          f"table={tab:>6,} B  ({bits-huff2:+.4f} vs Huffman-pairs)")

# ---- what a decode LUT for pairs would cost
print()
print(f"Decode table in SRAM:")
print(f"  scalar Huffman  : LUT 2^11 x 2 B            = {2048*2:,} B")
print(f"  Huffman pairs   : LUT 2^11 x 4 B (16b syms) = {2048*4:,} B  (plus fallback)")
print(f"  scalar ladder   : 10 symbols x 1 B          = 10 B")

# ---- context modelling (order 1): one table per previous symbol
print()
print("CONTEXT modelling (order 1): one code table per previous symbol")
top = np.argsort(-c1)[:8]
tot_ctx_bits, n_ctx = 0.0, 0
for ctx in range(256):
    if c1[ctx] == 0:
        continue
    sub = co[ctx*256:(ctx+1)*256]
    if sub.sum() == 0:
        continue
    Lc, _ = canonical_huffman(sub)
    tot_ctx_bits += sum(int(sub[s]) * l for s, l in Lc.items())
    n_ctx += 1
ctx_bits = tot_ctx_bits / co.sum()
print(f"  contexts with data    : {n_ctx}")
print(f"  Huffman with context  : {ctx_bits:.4f} bits/symbol "
      f"({huff1-ctx_bits:+.4f} vs scalar)")
print(f"  SRAM table cost       : Huffman {n_ctx*2048*2/1024:>8,.0f} KiB   "
      f"ladder {n_ctx*10:>6,} B")
