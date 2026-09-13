#!/usr/bin/env python3
"""
Test bench: DFloat11-style lossless compression of BF16 weights.

- CANONICAL Huffman over the exponent field (prefix-free by construction).
- Sign + mantissa left raw (their entropy is already ~maximal).
- COARSE index: one offset per block of N symbols, not per weight.
  -> parallelisable at block level, ~0.05% overhead.

Usage:
    python3 df11_reference.py weights_bf16.bin
"""
import sys, heapq, struct
import numpy as np

BLOCK = 256          # symbols per block (unit of parallelism)

# ---------------------------------------------------------------- Huffman
def canonical_huffman(counts):
    """Returns (lengths, codes) for a canonical prefix-free code."""
    syms = [(int(c), i) for i, c in enumerate(counts) if c > 0]
    if len(syms) == 1:                       # degenerate case
        return {syms[0][1]: 1}, {syms[0][1]: 0}

    heap = [(c, 1, [s]) for c, s in syms]    # (weight, max_depth, symbols)
    heapq.heapify(heap)
    depth = {s: 0 for _, s in syms}
    while len(heap) > 1:
        c1, _, s1 = heapq.heappop(heap)
        c2, _, s2 = heapq.heappop(heap)
        for s in s1 + s2:
            depth[s] += 1
        heapq.heappush(heap, (c1 + c2, 0, s1 + s2))

    lengths = {s: depth[s] for _, s in syms}

    # canonical assignment: sort by (length, symbol) and count up
    order = sorted(lengths, key=lambda s: (lengths[s], s))
    codes, code, prev_len = {}, 0, lengths[order[0]]
    for s in order:
        code <<= (lengths[s] - prev_len)
        codes[s] = code
        code += 1
        prev_len = lengths[s]
    return lengths, codes

# ---------------------------------------------------------------- bitstream
class BitWriter:
    def __init__(self): self.buf = bytearray(); self.acc = 0; self.n = 0
    def write(self, value, nbits):
        self.acc = (self.acc << nbits) | value; self.n += nbits
        while self.n >= 8:
            self.n -= 8
            self.buf.append((self.acc >> self.n) & 0xFF)
        self.acc &= (1 << self.n) - 1
    def bitpos(self): return len(self.buf) * 8 + self.n
    def flush(self):
        if self.n: self.buf.append((self.acc << (8 - self.n)) & 0xFF); self.n = 0
        return bytes(self.buf)

class BitReader:
    def __init__(self, data, bitpos=0): self.d = data; self.p = bitpos
    def read(self, nbits):
        v = 0
        for _ in range(nbits):
            byte = self.d[self.p >> 3]
            v = (v << 1) | ((byte >> (7 - (self.p & 7))) & 1)
            self.p += 1
        return v
    def read_bit(self):
        byte = self.d[self.p >> 3]
        b = (byte >> (7 - (self.p & 7))) & 1
        self.p += 1
        return b

# ---------------------------------------------------------------- codec
def encode(bf16):
    expo = ((bf16 >> 7) & 0xFF).astype(np.int64)
    sign_mant = (((bf16 >> 15) & 1).astype(np.uint16) << 7) | (bf16 & 0x7F)

    counts = np.bincount(expo, minlength=256)
    lengths, codes = canonical_huffman(counts)

    w = BitWriter()
    block_offsets = []
    for i, e in enumerate(expo):
        if i % BLOCK == 0:
            block_offsets.append(w.bitpos())
        w.write(codes[int(e)], lengths[int(e)])
    encoded = w.flush()

    return {
        "encoded": encoded,
        "block_offsets": np.array(block_offsets, dtype=np.uint64),
        "lengths": lengths,
        "sign_mant": sign_mant.astype(np.uint8),   # 8 bits: sign + 7 mantissa
        "n": len(expo),
    }

def decode(blob):
    lengths = blob["lengths"]
    _, codes = None, None
    # rebuild the canonical code from the lengths alone (that is what is transmitted)
    order = sorted(lengths, key=lambda s: (lengths[s], s))
    codes, code, prev = {}, 0, lengths[order[0]]
    for s in order:
        code <<= (lengths[s] - prev); codes[s] = code; code += 1; prev = lengths[s]
    # inverse table (length, code) -> symbol
    table = {(lengths[s], codes[s]): s for s in codes}
    maxlen = max(lengths.values())

    out = np.empty(blob["n"], dtype=np.int64)
    n_blocks = len(blob["block_offsets"])

    # >>> This loop over blocks is what runs in parallel on the GPU. <<<
    for b in range(n_blocks):
        r = BitReader(blob["encoded"], int(blob["block_offsets"][b]))
        start = b * BLOCK
        end = min(start + BLOCK, blob["n"])
        for i in range(start, end):
            code, ln = 0, 0
            while True:
                code = (code << 1) | r.read_bit(); ln += 1
                if (ln, code) in table:
                    out[i] = table[(ln, code)]; break
                if ln > maxlen: raise ValueError("invalid code")
    sm = blob["sign_mant"].astype(np.uint16)
    return (((sm >> 7) & 1) << 15) | (out.astype(np.uint16) << 7) | (sm & 0x7F)

# ---------------------------------------------------------------- main
if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "weights_bf16.bin"
    bf16 = np.fromfile(path, dtype=np.uint16)
    print(f"File:    {path}")
    print(f"Weights: {bf16.size:,}   ({bf16.nbytes:,} bytes)\n")

    blob = encode(bf16)
    exp_bytes  = len(blob["encoded"])
    sm_bytes   = blob["sign_mant"].nbytes
    idx_bytes  = blob["block_offsets"].nbytes
    tab_bytes  = 256                       # 1 length byte per symbol
    total      = exp_bytes + sm_bytes + idx_bytes + tab_bytes

    print(f"  Huffman exponents  : {exp_bytes:>10,} bytes")
    print(f"  sign+mantissa      : {sm_bytes:>10,} bytes")
    print(f"  block index        : {idx_bytes:>10,} bytes  ({100*idx_bytes/total:.3f}%)")
    print(f"  table (lengths)    : {tab_bytes:>10,} bytes")
    print(f"  ---------------------------------------")
    print(f"  TOTAL              : {total:>10,} bytes")
    print(f"  ratio              : {total/bf16.nbytes:.4f}")
    print(f"  reduction          : {100*(1-total/bf16.nbytes):.2f}%")
    print(f"  bits per weight    : {8*total/bf16.size:.3f}\n")

    rec = decode(blob)
    ok = np.array_equal(rec, bf16)
    print(f"  roundtrip lossless : {'YES (bit-exact)' if ok else 'NO'}")
    print(f"  parallel blocks    : {len(blob['block_offsets']):,} (of {BLOCK} symbols)")
