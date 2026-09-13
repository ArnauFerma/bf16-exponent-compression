#!/usr/bin/env python3
"""
How much rate does coding the BF16 fields separately give away?

The format codes the exponent and stores sign+mantissa raw. A coder over the
whole 16-bit symbol can never do worse on rate, because joint entropy is
subadditive:

    H(sign, exp, mant) <= H(sign) + H(exp) + H(mant)

The gap is exactly the mutual information between the fields. This measures
it, exactly, over every unique weight of the model: the full alphabet has
65,536 symbols, so the joint histogram is a single bincount.

Also reports what canonical Huffman achieves on each alphabet, so the
comparison is between achieved rates and not only between entropies.
"""
import sys
import numpy as np
from df11_reference import canonical_huffman

SRC = sys.argv[1] if len(sys.argv) > 1 else "outputs/real_weights_bf16.bin"
CHUNK = 32_000_000

mm = np.memmap(SRC, dtype=np.uint16, mode="r")
n = mm.size
c16 = np.zeros(65536, dtype=np.int64)
for i in range(0, n, CHUNK):
    c16 += np.bincount(np.asarray(mm[i:i + CHUNK]), minlength=65536)
assert int(c16.sum()) == n

# marginals and pairs from the full joint (no second pass needed)
v = np.arange(65536, dtype=np.int64)
sign, expo, mant = (v >> 15) & 1, (v >> 7) & 0xFF, v & 0x7F
def marg(key, size):
    return np.bincount(key, weights=c16, minlength=size).astype(np.int64)
c_s, c_e, c_m = marg(sign, 2), marg(expo, 256), marg(mant, 128)
c_em, c_se, c_sm = marg(expo * 128 + mant, 32768), marg(sign * 256 + expo, 512), marg(sign * 128 + mant, 256)

def H(c):
    c = c[c > 0].astype(np.float64); p = c / c.sum()
    return float(-(p * np.log2(p)).sum())

def huff(c):
    L, _ = canonical_huffman(c)
    return sum(int(c[s]) * l for s, l in L.items()) / c.sum(), max(L.values()), len(L)

Hs, He, Hm, Hem, Hse, Hsm, Hw = map(H, (c_s, c_e, c_m, c_em, c_se, c_sm, c16))

print(f"weights: {n:,}   distinct 16-bit values: {int((c16 > 0).sum()):,} of 65,536\n")
print("entropies (bits)")
print(f"  H(sign)             {Hs:8.4f}")
print(f"  H(exp)              {He:8.4f}")
print(f"  H(mant)             {Hm:8.4f}")
print(f"  H(exp, mant)        {Hem:8.4f}")
print(f"  H(sign, exp)        {Hse:8.4f}")
print(f"  H(sign, mant)       {Hsm:8.4f}")
print(f"  H(sign, exp, mant)  {Hw:8.4f}   (= H of the 16-bit value)")
print(f"  sum of marginals    {Hs+He+Hm:8.4f}\n")

print("mutual information (bits/weight)")
print(f"  I(exp; mant)        {He+Hm-Hem:8.4f}")
print(f"  I(sign; exp)        {Hs+He-Hse:8.4f}")
print(f"  I(sign; mant)       {Hs+Hm-Hsm:8.4f}")
print(f"  field-split gap     {Hs+He+Hm-Hw:8.4f}   = H(sign)+H(exp)+H(mant) - H(sign,exp,mant)\n")

he, me, ne = huff(c_e)
hw, mw, nw = huff(c16)
split_ach = 1 + 7 + he
print("achieved with canonical Huffman (bits/weight, no index)")
print(f"  field split: 1 + 7 + Huffman(exp)   {split_ach:8.4f}   (exp: {he:.4f} bits, {ne} symbols, maxlen {me})")
print(f"  full 16-bit alphabet Huffman        {hw:8.4f}   ({nw:,} symbols, maxlen {mw})")
print(f"  difference                          {split_ach-hw:8.4f}   ({100*(split_ach-hw)/16:.2f} points of the 16 bits)\n")

print("where the difference comes from (bits/weight)")
print(f"  true floor H(sign, exp, mant)                 {Hw:8.4f}")
print(f"  + mutual information between fields           {Hs+He+Hm-Hw:8.4f}")
print(f"  + sign and mantissa stored raw above entropy  {8-Hs-Hm:8.4f}")
print(f"  + Huffman redundancy on the exponent          {he-He:8.4f}")
print(f"  = field split achieved                        {split_ach:8.4f}")
print(f"  full alphabet: floor + Huffman redundancy     {Hw:8.4f} + {hw-Hw:.4f} = {hw:.4f}")

# where the exp-mant dependence lives, if anywhere: conditional mantissa
# entropy per exponent, weighted, for the exponents that carry most mass
print("\nmantissa entropy conditional on exponent (top exponents by mass)")
top = np.argsort(-c_e)[:8]
for e in top:
    sub = c_em[e * 128:(e + 1) * 128]
    print(f"  exp={e:3d}  mass={100*c_e[e]/n:6.2f}%   H(mant | exp)={H(sub):.4f}")
zero = int(c16[0] + c16[0x8000])
print(f"\nexact zeros (+0 and -0): {zero:,}  ({100*zero/n:.4f}% of weights)")
