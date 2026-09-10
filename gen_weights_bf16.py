import numpy as np, struct, math, json

rng = np.random.default_rng(1234)

# Simula varios "tensores" de un transformer: cada uno Gaussiano con su
# propia escala, mas una pequena componente de cola pesada (outliers reales).
tensors = []
specs = [
    ("layer0.attn.q_proj", 262144, 0.020),
    ("layer0.attn.k_proj", 262144, 0.018),
    ("layer0.mlp.up_proj", 393216, 0.012),
    ("layer0.mlp.down_proj", 393216, 0.009),
    ("layer1.attn.q_proj", 262144, 0.025),
    ("layer1.mlp.up_proj", 393216, 0.011),
    ("embed_tokens",       196608, 0.030),
]
for name, n, std in specs:
    core = rng.normal(0.0, std, n)
    # ~0.3% de outliers (los modelos reales los tienen)
    mask = rng.random(n) < 0.003
    core[mask] = rng.normal(0.0, std*8, mask.sum())
    tensors.append((name, core.astype(np.float32)))

flat32 = np.concatenate([t[1] for t in tensors])

# float32 -> bfloat16 con redondeo round-to-nearest-even
u32 = flat32.view(np.uint32)
rounding = ((u32 >> 16) & 1) + 0x7FFF
bf16 = ((u32 + rounding) >> 16).astype(np.uint16)

import os
os.makedirs("outputs", exist_ok=True)
bf16.tofile("outputs/weights_bf16.bin")

# --- Estadisticas de entropia por campo ---
sign = (bf16 >> 15) & 0x1
expo = (bf16 >> 7) & 0xFF
mant = bf16 & 0x7F

def entropy(arr, nbits):
    counts = np.bincount(arr, minlength=1 << nbits).astype(np.float64)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())

H_sign = entropy(sign, 1)
H_exp  = entropy(expo, 8)
H_mant = entropy(mant, 7)

# Longitud media optima con Huffman canonico sobre el exponente
counts = np.bincount(expo, minlength=256)
symbols = [(int(c), i) for i, c in enumerate(counts) if c > 0]

import heapq
h = [(c, i, None, None) for c, i in symbols]
heapq.heapify(h)
nodes = {}
nxt = 256
while len(h) > 1:
    a = heapq.heappop(h); b = heapq.heappop(h)
    nodes[nxt] = (a, b)
    heapq.heappush(h, (a[0]+b[0], nxt, a, b))
    nxt += 1
root = h[0]

lengths = {}
def walk(node, d):
    c, sym, l, r = node
    if l is None and r is None:
        lengths[sym] = max(d, 1); return
    walk(l, d+1); walk(r, d+1)
walk(root, 0)

total = int(counts.sum())
avg_exp_bits = sum(lengths[s]*counts[s] for s in lengths) / total

orig_bits_per_w = 16.0
new_bits_per_w = 1 + 7 + avg_exp_bits   # signo + mantisa sin tocar + exponente comprimido

stats = {
    "n_weights": int(bf16.size),
    "file_bytes": int(bf16.size * 2),
    "entropy_sign_bits": round(H_sign, 4),
    "entropy_exponent_bits": round(H_exp, 4),
    "entropy_mantissa_bits": round(H_mant, 4),
    "huffman_avg_exponent_bits": round(avg_exp_bits, 4),
    "distinct_exponents": len(symbols),
    "max_code_length": max(lengths.values()),
    "bits_per_weight_original": orig_bits_per_w,
    "bits_per_weight_compressed": round(new_bits_per_w, 4),
    "compression_ratio": round(new_bits_per_w / orig_bits_per_w, 4),
    "size_reduction_pct": round(100*(1 - new_bits_per_w/orig_bits_per_w), 2),
}
print(json.dumps(stats, indent=2))

# Top exponentes
order = np.argsort(-counts)[:12]
print("\nTop exponentes (valor, ocurrencias, %, long. Huffman):")
for e in order:
    if counts[e] == 0: continue
    print(f"  {e:3d}  {counts[e]:8d}  {100*counts[e]/total:6.2f}%   {lengths[int(e)]} bits")
