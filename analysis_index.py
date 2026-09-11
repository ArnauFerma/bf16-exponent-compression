#!/usr/bin/env python3
"""
(a) La informacion mutua ~0 no puede ser un artefacto de haber mirado solo
    los primeros 64M pesos (que son embed_tokens). Se comprueba en ventanas
    contiguas repartidas por todo el fichero.
(b) Cuanto vale arreglar el INDICE, que es el unico margen estructural que
    queda.
"""
import numpy as np
from df11_reference import canonical_huffman

mm = np.memmap("outputs/real_weights_bf16.bin", dtype=np.uint16, mode="r")
TOT = mm.size
counts_full = np.load("outputs/real_exp_counts.npy")

def H(c):
    c = c[c > 0].astype(np.float64); p = c / c.sum()
    return float(-(p * np.log2(p)).sum())

print("=== (a) informacion mutua en ventanas repartidas por el modelo ===")
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
print(f"\nI(X;Y) maxima observada: {max(mis):.5f} bits  -> los exponentes "
      f"vecinos son, a efectos practicos, independientes.\n")

print("=== (b) coste del indice de bloques ===")
L, _ = canonical_huffman(counts_full)
len_of = np.zeros(256, dtype=np.int64)
for s, l in L.items():
    len_of[s] = l
avg_bits = sum(int(counts_full[s]) * l for s, l in L.items()) / counts_full.sum()
print(f"Huffman sobre el modelo completo: {avg_bits:.4f} bits/exponente\n")

# distribucion real de longitud de bloque, para ver si cabe en 8 bits
w = ((np.asarray(mm[:64_000_000]) >> 7) & 0xFF).astype(np.int64)
lens = len_of[w]
for BLOCK in (64, 256):
    bl = lens[:len(lens)//BLOCK*BLOCK].reshape(-1, BLOCK).sum(axis=1)
    print(f"BLOCK={BLOCK:4d}: longitud de bloque en bits  min={bl.min()} "
          f"max={bl.max()} media={bl.mean():.1f} rango={bl.max()-bl.min()}")
print()

def reduction(block, idx_bytes_per_block):
    per_w = 1.0 + avg_bits/8 + idx_bytes_per_block/block
    return 100 * (1 - per_w/2), per_w

print(f"{'esquema de indice':>42} {'B/bloque':>9} {'BLOCK=64':>10} {'BLOCK=256':>10}")
schemes = [
    ("uint32 absoluto (spec actual)",            4.0),
    ("uint16 relativo + uint32/superbloque(256)", 2.0 + 4.0/256),
    ("uint8 longitudes + prefix-sum de warp",     1.0),
]
for name, b in schemes:
    r64, _ = reduction(64, b); r256, _ = reduction(256, b)
    print(f"{name:>42} {b:9.3f} {r64:9.2f}% {r256:9.2f}%")

print(f"""
Lectura: el punto rapido medido (BLOCK=64, salida en shared, 4,80 ms) pierde
hoy 2,35 puntos de compresion frente a BLOCK=256 SOLO por el indice. Con
indice jerarquico de 16 bits esa perdida casi desaparece, sin tocar el
kernel de decodificacion ni el codigo de entropia.""")
