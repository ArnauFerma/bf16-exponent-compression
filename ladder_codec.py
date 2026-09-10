#!/usr/bin/env python3
"""
Codec "escalera" (ladder code) para el campo exponente de BF16.

Codigo de prefijo por "rungs" (escalones): el simbolo mas frecuente entra
en el escalon mas corto. Cada escalon r tiene prefijo "r unos + un cero"
seguido de rung_bits[r] bits de indice (2**rung_bits[r] simbolos caben en
ese escalon). Lo que no cabe en ningun escalon usa un codigo de escape:
n_rungs unos (sin cero final) + 8 bits crudos = el propio valor del
exponente, sin necesidad de tabla para ese caso.

Kraft se cumple exactamente por construccion, sea cual sea rung_bits:
cada escalon r aporta 2**rung_bits[r] simbolos de longitud (r+1+rung_bits[r]),
que suma 2**-(r+1) de masa; el escape se queda con el resto 2**-n_rungs.
"""
import numpy as np

DEFAULT_RUNG_BITS = [1, 1, 1, 2]   # forma (1,1,1,2) del handoff
RAW_BITS = 8                        # exponente BF16 = 8 bits


def build_ladder(counts, rung_bits=DEFAULT_RUNG_BITS, raw_bits=RAW_BITS):
    """counts: array de 256 frecuencias (indice = valor de exponente).

    Devuelve:
      lengths: dict simbolo -> longitud en bits
      slots:   list de listas, slots[r] = simbolos asignados al escalon r
                (orden = indice dentro del escalon)
      escape:  set de simbolos que van por el camino de escape
    """
    n_rungs = len(rung_bits)
    cap = [1 << b for b in rung_bits]
    total_slots = sum(cap)

    present = [(int(c), int(s)) for s, c in enumerate(counts) if c > 0]
    present.sort(key=lambda x: (-x[0], x[1]))   # frecuencia desc, simbolo asc
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
    """Cabecera minima: para cada escalon, la lista de simbolos que ocupa
    (1 byte por simbolo, el orden ES el indice). No hace falta guardar
    rung_bits si la forma es fija y conocida por ambos lados."""
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
        prefix = ((1 << r) - 1) << 1          # r unos seguidos de un 0
        plen = r + 1
        for idx, sym in enumerate(chunk):
            encode_of[sym] = (prefix, plen, idx, rung_bits[r])
    escape_prefix = (1 << n_rungs) - 1        # n_rungs unos
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
    # auto-test rapido sobre datos aleatorios con distribucion sesgada
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
