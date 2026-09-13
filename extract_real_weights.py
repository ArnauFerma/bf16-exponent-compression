#!/usr/bin/env python3
"""
Extracts the raw BF16 weights of a .safetensors into a flat uint16 .bin,
accumulating the exact per-exponent counts along the way.

No torch needed: the safetensors format is
    [8 bytes: len(header) uint64 LE][JSON header][data buffer]
and the header gives dtype, shape and (start, end) of each tensor in the
buffer. Only the bit patterns matter, so they are read as uint16 without
conversion (numpy has no bfloat16 dtype).

Streams tensor by tensor: never loads the whole file into RAM.

Tensors whose content is byte-for-byte identical to one already written are
skipped (e.g. lm_head.weight when it is a copy of embed_tokens.weight under
tie_word_embeddings). A weight that exists once in the model is counted once.
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
    print(f"tensors: {len(tensors)}   dtypes present: {dtypes}")

    bf16 = {k: v for k, v in tensors.items() if v["dtype"] == "BF16"}
    skipped = {k: v["dtype"] for k, v in tensors.items() if v["dtype"] != "BF16"}
    if skipped:
        print(f"skipped (not BF16): {len(skipped)}  -> {sorted(set(skipped.values()))}")

    # order by offset = sequential read from disk
    order = sorted(bf16.items(), key=lambda kv: kv[1]["data_offsets"][0])

    counts = np.zeros(256, dtype=np.int64)
    total = 0
    seen = {}            # sha256 of content -> name of the first tensor
    duplicates = []
    with open(DST, "wb") as out:
        for name, meta in order:
            s, e = meta["data_offsets"]
            nbytes = e - s
            assert nbytes % 2 == 0, f"{name}: {nbytes} bytes is odd"
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
    print(f"\ntensors skipped as exact copies of another ({len(duplicates)}):")
    for name, orig, n in duplicates:
        print(f"  {name}  ==  {orig}   ({n:,} weights)")

print(f"\nBF16 weights extracted: {total:,}   ({total*2:,} bytes)")
assert int(counts.sum()) == total, "counts do not add up to the total"
np.save(CNT, counts)

p = counts[counts > 0].astype(np.float64) / total
H = float(-(p * np.log2(p)).sum())
print(f"distinct exponents    : {int((counts>0).sum())}")
print(f"exponent entropy      : {H:.4f} bits")
print(f"\nwritten: {DST}  and  {CNT}")
