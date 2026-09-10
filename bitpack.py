#!/usr/bin/env python3
"""
Encoder vectorizado de bitstream (numpy) para Huffman canonico y escalera.

El BitWriter de df11_reference/ladder_codec es un bucle Python por simbolo:
correcto pero inviable a la escala que necesitan los kernels (decenas de
millones de simbolos). Aqui se produce EXACTAMENTE el mismo bitstream
(MSB-first dentro de cada byte, igual que np.packbits con bitorder='big')
en unas pocas pasadas vectorizadas: una por longitud de codigo presente.

Devuelve tambien los offsets en bits de cada bloque, que son el indice
grueso que hace paralelizable la decodificacion.
"""
import numpy as np

from df11_reference import canonical_huffman
import ladder_codec as lc


# ------------------------------------------------------------------ tablas
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
    """syms: array de simbolos (enteros 0..255).

    Devuelve (packed_bytes, block_bitpos int64[n_blocks], total_bits).
    packed_bytes es identico byte a byte a lo que produce BitWriter.
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
    """bytes -> uint32 big-endian (bit p vive en word p>>5, bit 31-(p&31)),
    con 2 words de guarda para que peek32 pueda leer siempre w[i+1]."""
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

    # Huffman: comparar contra el BitWriter original, byte a byte
    code_of, len_of, lengths, codes = huffman_code_arrays(counts)
    packed, offs, nbits = encode_stream(syms, code_of, len_of, 256)

    w = lc.BitWriter()
    ref_offs = []
    for i, e in enumerate(syms):
        if i % 256 == 0:
            ref_offs.append(w.bitpos())
        w.write(codes[int(e)], lengths[int(e)])
    ref = w.flush()
    print("huffman  bytes identicos :", packed.tobytes() == ref)
    print("huffman  offsets iguales :", np.array_equal(offs, np.array(ref_offs)))

    # Escalera: comparar contra encode_symbols original
    code_of2, len_of2, slots, escape = ladder_code_arrays(counts)
    packed2, offs2, nbits2 = encode_stream(syms, code_of2, len_of2, 256)
    encode_of = lc.make_code_tables(slots, [1, 1, 1, 2], escape)
    ref2, ref_offs2 = lc.encode_symbols(syms, encode_of, block=256)
    print("escalera bytes identicos :", packed2.tobytes() == ref2)
    print("escalera offsets iguales :", np.array_equal(offs2, ref_offs2.astype(np.int64)))
