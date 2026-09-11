#!/usr/bin/env python3
"""
Puede ganar algo la codificacion vectorial (por pares) sobre el exponente?

Dos fuentes de ganancia posibles, muy distintas:

  1. Redundancia del codigo. Huffman escalar ya esta a 2,6650 bits contra una
     entropia de 2,6340: solo 0,031 bits de margen. Ningun codificador de
     entropia, ni aritmetico ni vectorial, puede ganar mas que eso SI los
     simbolos son independientes.

  2. Correlacion entre exponentes vecinos. Si existe, H(X,Y) < 2*H(X) y la
     codificacion por pares (o con contexto) captura esa diferencia. Esto NO
     esta acotado por los 0,031 bits: es margen nuevo.

La 2 es una pregunta empirica. Se mide aqui.
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

# ---- pares adyacentes NO solapados (que es lo que se codificaria)
a, b = expo[0:2*(N//2):2], expo[1:2*(N//2):2]
cp = np.bincount(a * 256 + b, minlength=65536)
Hpair = H(cp)

# ---- pares solapados: mide la correlacion real de la secuencia
co = np.bincount(expo[:-1] * 256 + expo[1:], minlength=65536)
Hjoint = H(co)
Hcond = Hjoint - H1                    # H(Y|X)
MI = H1 - Hcond                        # I(X;Y)

print(f"simbolos analizados     : {expo.size:,}")
print(f"H(X)  marginal          : {H1:.4f} bits")
print(f"H(X,Y) conjunta (solap.): {Hjoint:.4f} bits  -> {Hjoint/2:.4f} por simbolo")
print(f"H(Y|X) condicional      : {Hcond:.4f} bits")
print(f"I(X;Y) informacion mutua: {MI:.4f} bits   <-- el margen NUEVO")
print()
print(f"Huffman escalar         : ", end="")
L1, _ = canonical_huffman(c1)
huff1 = sum(int(c1[s]) * l for s, l in L1.items()) / c1.sum()
print(f"{huff1:.4f} bits/simbolo   (redundancia {huff1-H1:.4f})")

# ---- Huffman sobre pares
L2, _ = canonical_huffman(cp)
huff2 = sum(int(cp[s]) * l for s, l in L2.items()) / cp.sum() / 2
n_pairs = int((cp > 0).sum())
print(f"Huffman sobre pares     : {huff2:.4f} bits/simbolo   "
      f"({n_pairs:,} pares distintos, maxlen={max(L2.values())})")
print(f"  ganancia vs escalar   : {huff1-huff2:+.4f} bits/simbolo "
      f"({100*(huff1-huff2)/huff1:+.2f}%)")

# ---- escalera sobre pares: cuantos escalones hacen falta
print()
print("Escalera sobre pares (hay que ampliar los escalones):")
for shape in ([1,1,1,2], [2,2,3,4], [4,4,5,6], [5,5,6,7], [6,6,7,8]):
    lens, slots, esc = lc.build_ladder(cp, shape, raw_bits=16)
    bits = lc.analytic_bits(cp, lens) / cp.sum() / 2
    tab = sum(len(s) for s in slots) * 2      # 2 bytes por par
    print(f"  {str(tuple(shape)):>16}  {bits:.4f} bits/simbolo  "
          f"tabla={tab:>6,} B  ({bits-huff2:+.4f} vs Huffman-pares)")

# ---- que costaria una LUT de decodificacion para pares
print()
print(f"Tabla de decodificacion en SRAM:")
print(f"  Huffman escalar : LUT 2^11 x 2 B            = {2048*2:,} B")
print(f"  Huffman pares   : LUT 2^11 x 4 B (simb 16b) = {2048*4:,} B  (mas fallback)")
print(f"  Escalera escalar: 10 simbolos x 1 B         = 10 B")

# ---- modelado por contexto (orden 1): una tabla por simbolo anterior
print()
print("Modelado por CONTEXTO (orden 1): una tabla de codigos por simbolo previo")
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
print(f"  contextos con datos   : {n_ctx}")
print(f"  Huffman con contexto  : {ctx_bits:.4f} bits/simbolo "
      f"({huff1-ctx_bits:+.4f} vs escalar)")
print(f"  coste en tablas SRAM  : Huffman {n_ctx*2048*2/1024:>8,.0f} KiB   "
      f"escalera {n_ctx*10:>6,} B")
