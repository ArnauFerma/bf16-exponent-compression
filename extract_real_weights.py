#!/usr/bin/env python3
"""
Extrae los pesos BF16 crudos de un .safetensors a un .bin plano de uint16,
acumulando de paso los conteos exactos por exponente.

No necesita torch: el formato safetensors es
    [8 bytes: len(header) uint64 LE][header JSON][buffer de datos]
y el header da dtype, shape y (start, end) de cada tensor dentro del buffer.
Como solo interesan los patrones de bits, se leen como uint16 sin convertir
(numpy no tiene dtype bfloat16).

Streaming tensor a tensor: nunca carga el fichero entero en RAM.

Los tensores cuyo contenido es byte a byte identico a uno ya escrito se
omiten (p.ej. lm_head.weight cuando es una copia de embed_tokens.weight con
tie_word_embeddings). Un peso que existe una vez en el modelo se cuenta una
vez.
"""
import json, sys, os, hashlib
import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else "real_model/model.safetensors"
DST = sys.argv[2] if len(sys.argv) > 2 else "outputs/real_weights_bf16.bin"
CNT = sys.argv[3] if len(sys.argv) > 3 else "outputs/real_exp_counts.npy"

os.makedirs(os.path.dirname(DST) or ".", exist_ok=True)

with open(SRC, "rb") as f:
    hdr_len = int.from_bytes(f.read(8), "little")
    header = json.loads(f.read(hdr_len))
    data_start = 8 + hdr_len

    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    dtypes = sorted({v["dtype"] for v in tensors.values()})
    print(f"tensores: {len(tensors)}   dtypes presentes: {dtypes}")

    bf16 = {k: v for k, v in tensors.items() if v["dtype"] == "BF16"}
    skipped = {k: v["dtype"] for k, v in tensors.items() if v["dtype"] != "BF16"}
    if skipped:
        print(f"omitidos (no BF16): {len(skipped)}  -> {sorted(set(skipped.values()))}")

    # orden por offset = lectura secuencial en disco
    order = sorted(bf16.items(), key=lambda kv: kv[1]["data_offsets"][0])

    counts = np.zeros(256, dtype=np.int64)
    total = 0
    seen = {}            # sha256 del contenido -> nombre del primer tensor
    duplicates = []
    with open(DST, "wb") as out:
        for name, meta in order:
            s, e = meta["data_offsets"]
            nbytes = e - s
            assert nbytes % 2 == 0, f"{name}: {nbytes} bytes no es par"
            f.seek(data_start + s)
            raw = f.read(nbytes)
            digest = hashlib.sha256(raw).hexdigest()
            if digest in seen:
                duplicates.append((name, seen[digest], nbytes // 2))
                continue
            seen[digest] = name
            arr = np.frombuffer(raw, dtype=np.uint16)
            arr.tofile(out)
            counts += np.bincount((arr >> 7) & 0xFF, minlength=256)
            total += arr.size

if duplicates:
    print(f"\ntensores omitidos por ser copia exacta de otro ({len(duplicates)}):")
    for name, orig, n in duplicates:
        print(f"  {name}  ==  {orig}   ({n:,} pesos)")

print(f"\npesos BF16 extraidos : {total:,}   ({total*2:,} bytes)")
assert int(counts.sum()) == total, "conteos no cuadran con el total"
np.save(CNT, counts)

p = counts[counts > 0].astype(np.float64) / total
H = float(-(p * np.log2(p)).sum())
print(f"exponentes distintos : {int((counts>0).sum())}")
print(f"entropia exponente   : {H:.4f} bits")
print(f"\nescrito: {DST}  y  {CNT}")
