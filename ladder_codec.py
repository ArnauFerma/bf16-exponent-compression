#!/usr/bin/env python3
"""
"Ladder" codec for the BF16 exponent field.

Prefix code organised in rungs: the most frequent symbol goes in the shortest
rung. Rung r has the prefix "r ones followed by a zero" and then rung_bits[r]
index bits (2**rung_bits[r] symbols fit in that rung). Anything that fits in
no rung uses an escape code: n_rungs ones (no trailing zero) followed by the
8 raw bits of the exponent itself, so that case needs no table.

The code is complete by construction, whatever rung_bits is: rung r
contributes 2**rung_bits[r] symbols of length (r+1+rung_bits[r]), i.e.
2**-(r+1) of mass, and the escape keeps the remaining 2**-n_rungs. Note that
kraft_sum() below sums only over the symbols actually present, so it comes
out below 1 when not every escape codeword is used.
"""
import numpy as np

DEFAULT_RUNG_BITS = [1, 1, 1, 2]   # the (1,1,1,2) shape from the handoff
RAW_BITS = 8                        # BF16 exponent = 8 bits


def build_ladder(counts, rung_bits=DEFAULT_RUNG_BITS, raw_bits=RAW_BITS):
    """counts: array of 256 frequencies (index = exponent value).

    Returns:
      lengths: dict symbol -> length in bits
      slots:   list of lists, slots[r] = symbols assigned to rung r
                (order = index within the rung)
      escape:  set of symbols that go through the escape path
    """
    n_rungs = len(rung_bits)
    cap = [1 << b for b in rung_bits]
    total_slots = sum(cap)

    present = [(int(c), int(s)) for s, c in enumerate(counts) if c > 0]
    present.sort(key=lambda x: (-x[0], x[1]))   # frequency desc, symbol asc
    ordered_syms = [s for _, s in present]

    ladder_syms = ordered_syms[:total_slots]
    escape_syms = set(ordered_syms[total_slots:])

    slots = []
    lengths = {}
    i = 0
    for r in range(n_rungs):
        length = (r + 1) + rung_bits[r]
        chunk = ladder_syms[i:i + cap[r]]
        slots.append(chunk)
        for sym in chunk:
            lengths[sym] = length
        i += cap[r]

    escape_length = n_rungs + raw_bits
    for sym in escape_syms:
        lengths[sym] = escape_length

    return lengths, slots, escape_syms


def analytic_bits(counts, lengths):
    return sum(int(counts[s]) * lengths[s] for s in lengths if counts[s] > 0)


def kraft_sum(lengths):
    return sum(2.0 ** -l for l in lengths.values())


def header_bytes(slots, rung_bits):
    """Minimal header: for each rung, the list of symbols it holds (1 byte
    per symbol; the order IS the index). rung_bits need not be stored if the
    shape is fixed and known to both sides."""
    return sum(len(s) for s in slots)


# ---------------------------------------------------------------- bitstream
class BitWriter:
    def __init__(self):
        self.buf = bytearray(); self.acc = 0; self.n = 0

    def write(self, value, nbits):
        if nbits == 0:
            return
        self.acc = (self.acc << nbits) | value; self.n += nbits
        while self.n >= 8:
            self.n -= 8
            self.buf.append((self.acc >> self.n) & 0xFF)
        self.acc &= (1 << self.n) - 1

    def bitpos(self):
        return len(self.buf) * 8 + self.n

    def flush(self):
        if self.n:
            self.buf.append((self.acc << (8 - self.n)) & 0xFF); self.n = 0
        return bytes(self.buf)


class BitReader:
    def __init__(self, data, bitpos=0):
        self.d = data; self.p = bitpos

    def read_bit(self):
        byte = self.d[self.p >> 3]
        b = (byte >> (7 - (self.p & 7))) & 1
        self.p += 1
        return b

    def read(self, nbits):
        v = 0
        for _ in range(nbits):
            v = (v << 1) | self.read_bit()
        return v


def make_code_tables(slots, rung_bits, escape_syms, raw_bits=RAW_BITS):
    """sym -> (prefix_bits_value, prefix_len, index_or_raw, index_len)"""
    n_rungs = len(rung_bits)
    encode_of = {}
    for r, chunk in enumerate(slots):
        prefix = ((1 << r) - 1) << 1          # r ones followed by a 0
        plen = r + 1
        for idx, sym in enumerate(chunk):
            encode_of[sym] = (prefix, plen, idx, rung_bits[r])
    escape_prefix = (1 << n_rungs) - 1        # n_rungs ones
    for sym in escape_syms:
        encode_of[sym] = (escape_prefix, n_rungs, sym, raw_bits)
    return encode_of


def encode_symbols(expo_array, encode_of, block=256):
    w = BitWriter()
    block_offsets = []
    for i, e in enumerate(expo_array):
        if i % block == 0:
            block_offsets.append(w.bitpos())
        prefix, plen, idx, ilen = encode_of[int(e)]
        w.write(prefix, plen)
        w.write(idx, ilen)
    return w.flush(), np.array(block_offsets, dtype=np.uint64)


def decode_symbols(data, block_offsets, n, slots, rung_bits, raw_bits, block=256):
    n_rungs = len(rung_bits)
    out = np.empty(n, dtype=np.int64)
    n_blocks = len(block_offsets)
    for b in range(n_blocks):
        r = BitReader(data, int(block_offsets[b]))
        start = b * block
        end = min(start + block, n)
        for i in range(start, end):
            ones = 0
            while ones < n_rungs and r.read_bit() == 1:
                ones += 1
            if ones < n_rungs:
                idx = r.read(rung_bits[ones])
                out[i] = slots[ones][idx]
            else:
                out[i] = r.read(raw_bits)
    return out


if __name__ == "__main__":
    # quick self-test on random data with a skewed distribution
    rng = np.random.default_rng(0)
    probs = np.array([0.28, 0.22, 0.20, 0.11, 0.08, 0.06, 0.03, 0.01, 0.005, 0.005])
    probs = probs / probs.sum()
    syms = rng.choice(len(probs), size=20000, p=probs)
    counts = np.bincount(syms, minlength=256)

    lengths, slots, escape = build_ladder(counts)
    print("Kraft:", kraft_sum(lengths))
    encode_of = make_code_tables(slots, DEFAULT_RUNG_BITS, escape)
    data, offsets = encode_symbols(syms, encode_of, block=256)
    rec = decode_symbols(data, offsets, len(syms), slots, DEFAULT_RUNG_BITS, RAW_BITS, block=256)
    print("roundtrip OK:", np.array_equal(rec, syms))
    print("bits:", analytic_bits(counts, lengths), "vs raw", len(syms)*8)
