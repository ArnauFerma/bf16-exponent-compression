#!/usr/bin/env python3
"""
Vectorised bitstream encoder (numpy) for canonical Huffman and the ladder.

The BitWriter in df11_reference/ladder_codec is a per-symbol Python loop:
correct, but unusable at the scale the kernels need (tens of millions of
symbols). This produces EXACTLY the same bitstream (MSB-first within each
byte, like np.packbits with bitorder='big') in a few vectorised passes: one
per code length present.

Also returns the bit offset of each block, which is the coarse index that
makes decoding parallelisable.
"""
import numpy as np

from df11_reference import canonical_huffman
import ladder_codec as lc


# ------------------------------------------------------------------ tables
def huffman_code_arrays(counts):
    """-> code_of[256] uint32, len_of[256] uint8, (lengths, codes) dicts"""
    lengths, codes = canonical_huffman(counts)
    code_of = np.zeros(256, dtype=np.uint32)
    len_of = np.zeros(256, dtype=np.uint8)
    for s, L in lengths.items():
        code_of[s] = codes[s]
        len_of[s] = L
    return code_of, len_of, lengths, codes


def ladder_code_arrays(counts, rung_bits=(1, 1, 1, 2)):
    """-> code_of[256] uint32, len_of[256] uint8, slots, escape"""
    rung_bits = list(rung_bits)
    _lengths, slots, escape = lc.build_ladder(counts, rung_bits)
    encode_of = lc.make_code_tables(slots, rung_bits, escape)
    code_of = np.zeros(256, dtype=np.uint32)
    len_of = np.zeros(256, dtype=np.uint8)
    for s, (prefix, plen, idx, ilen) in encode_of.items():
        code_of[s] = (prefix << ilen) | idx
        len_of[s] = plen + ilen
    return code_of, len_of, slots, escape


# --------------------------------------------------------------- bitstream
def encode_stream(syms, code_of, len_of, block):
    """syms: array of symbols (integers 0..255).

    Returns (packed_bytes, block_bitpos int64[n_blocks], total_bits).
    packed_bytes is byte-for-byte identical to what BitWriter produces.
    """
    syms = np.asarray(syms)
    lens = len_of[syms].astype(np.int64)
    codes = code_of[syms]

    ends = np.cumsum(lens)
    starts = ends - lens
    total_bits = int(ends[-1]) if syms.size else 0

    bits = np.zeros(total_bits, dtype=np.uint8)
    maxlen = int(lens.max()) if syms.size else 0
    for L in range(1, maxlen + 1):
        sel = np.flatnonzero(lens == L)
        if sel.size == 0:
            continue
        c = codes[sel]
        st = starts[sel]
        for j in range(L):
            bits[st + j] = (c >> np.uint32(L - 1 - j)) & np.uint32(1)

    packed = np.packbits(bits)              # bitorder='big' == BitWriter
    block_bitpos = starts[::block].copy().astype(np.int64)
    return packed, block_bitpos, total_bits


def to_words(packed, total_bits):
    """bytes -> big-endian uint32 (bit p lives in word p>>5, bit 31-(p&31)),
    with 2 guard words so peek32 can always read w[i+1]."""
    n_words = (total_bits + 31) // 32 + 2
    buf = np.zeros(n_words * 4, dtype=np.uint8)
    buf[:packed.size] = packed
    return np.frombuffer(buf.tobytes(), dtype=">u4").astype(np.uint32)


# ------------------------------------------------------------------ selftest
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    probs = np.array([.28, .22, .20, .11, .08, .06, .03, .01, .005, .005, .002])
    probs /= probs.sum()
    syms = rng.choice(len(probs), size=100_000, p=probs).astype(np.int64)
    counts = np.bincount(syms, minlength=256)

    # Huffman: compare against the original BitWriter, byte for byte
    code_of, len_of, lengths, codes = huffman_code_arrays(counts)
    packed, offs, nbits = encode_stream(syms, code_of, len_of, 256)

    w = lc.BitWriter()
    ref_offs = []
    for i, e in enumerate(syms):
        if i % 256 == 0:
            ref_offs.append(w.bitpos())
        w.write(codes[int(e)], lengths[int(e)])
    ref = w.flush()
    print("huffman  bytes identical :", packed.tobytes() == ref)
    print("huffman  offsets equal   :", np.array_equal(offs, np.array(ref_offs)))

    # Ladder: compare against the original encode_symbols
    code_of2, len_of2, slots, escape = ladder_code_arrays(counts)
    packed2, offs2, nbits2 = encode_stream(syms, code_of2, len_of2, 256)
    encode_of = lc.make_code_tables(slots, [1, 1, 1, 2], escape)
    ref2, ref_offs2 = lc.encode_symbols(syms, encode_of, block=256)
    print("ladder   bytes identical :", packed2.tobytes() == ref2)
    print("ladder   offsets equal   :", np.array_equal(offs2, ref_offs2.astype(np.int64)))
