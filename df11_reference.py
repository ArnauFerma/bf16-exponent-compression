#!/usr/bin/env python3
"""
Banco de pruebas: compresion lossless de pesos BF16 estilo DFloat11.

- Huffman CANONICO sobre el campo exponente (prefijo-libre garantizado).
- Signo + mantisa se dejan crudos (su entropia ya es ~maxima).
- Indice GRUESO: un offset por bloque de N simbolos, no por peso.
  -> paralelizable a nivel de bloque, overhead ~0.05%.

Uso:
    python3 df11_reference.py weights_bf16.bin
"""
import sys, heapq, struct
import numpy as np

BLOCK = 256          # simbolos por bloque (unidad de paralelismo)

# ---------------------------------------------------------------- Huffman
def canonical_huffman(counts):
    """Devuelve (lengths, codes) con codigo canonico prefijo-libre."""
    syms = [(int(c), i) for i, c in enumerate(counts) if c > 0]
    if len(syms) == 1:                       # caso degenerado
        return {syms[0][1]: 1}, {syms[0][1]: 0}

    heap = [(c, 1, [s]) for c, s in syms]    # (peso, profundidad_max, simbolos)
    heapq.heapify(heap)
    depth = {s: 0 for _, s in syms}
    while len(heap) > 1:
        c1, _, s1 = heapq.heappop(heap)
        c2, _, s2 = heapq.heappop(heap)
        for s in s1 + s2:
            depth[s] += 1
        heapq.heappush(heap, (c1 + c2, 0, s1 + s2))

    lengths = {s: depth[s] for _, s in syms}

    # asignacion canonica: ordenar por (longitud, simbolo) y contar
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
        "sign_mant": sign_mant.astype(np.uint8),   # 8 bits: signo + 7 mantisa
        "n": len(expo),
    }

def decode(blob):
    lengths = blob["lengths"]
    _, codes = None, None
    # reconstruir codigo canonico solo desde las longitudes (asi se transmite)
    order = sorted(lengths, key=lambda s: (lengths[s], s))
    codes, code, prev = {}, 0, lengths[order[0]]
    for s in order:
        code <<= (lengths[s] - prev); codes[s] = code; code += 1; prev = lengths[s]
    # tabla inversa (longitud, codigo) -> simbolo
    table = {(lengths[s], codes[s]): s for s in codes}
    maxlen = max(lengths.values())

    out = np.empty(blob["n"], dtype=np.int64)
    n_blocks = len(blob["block_offsets"])

    # >>> Este bucle sobre bloques es lo que en GPU va en paralelo. <<<
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
                if ln > maxlen: raise ValueError("codigo invalido")
    sm = blob["sign_mant"].astype(np.uint16)
    return (((sm >> 7) & 1) << 15) | (out.astype(np.uint16) << 7) | (sm & 0x7F)

# ---------------------------------------------------------------- main
if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "weights_bf16.bin"
    bf16 = np.fromfile(path, dtype=np.uint16)
    print(f"Fichero: {path}")
    print(f"Pesos:   {bf16.size:,}   ({bf16.nbytes:,} bytes)\n")

    blob = encode(bf16)
    exp_bytes  = len(blob["encoded"])
    sm_bytes   = blob["sign_mant"].nbytes
    idx_bytes  = blob["block_offsets"].nbytes
    tab_bytes  = 256                       # 1 byte de longitud por simbolo
    total      = exp_bytes + sm_bytes + idx_bytes + tab_bytes

    print(f"  exponentes Huffman : {exp_bytes:>10,} bytes")
    print(f"  signo+mantisa      : {sm_bytes:>10,} bytes")
    print(f"  indice de bloques  : {idx_bytes:>10,} bytes  ({100*idx_bytes/total:.3f}%)")
    print(f"  tabla (longitudes) : {tab_bytes:>10,} bytes")
    print(f"  ---------------------------------------")
    print(f"  TOTAL              : {total:>10,} bytes")
    print(f"  ratio              : {total/bf16.nbytes:.4f}")
    print(f"  reduccion          : {100*(1-total/bf16.nbytes):.2f}%")
    print(f"  bits por peso      : {8*total/bf16.size:.3f}\n")

    rec = decode(blob)
    ok = np.array_equal(rec, bf16)
    print(f"  roundtrip lossless : {'SI (bit a bit)' if ok else 'NO'}")
    print(f"  bloques paralelos  : {len(blob['block_offsets']):,} (de {BLOCK} simbolos)")
