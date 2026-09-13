# Lossless BF16 weight compression — handoff

Status document. Last updated: **2026-09-13**.

For the chronological log with all the tables, see [RESULTS.md](RESULTS.md).
The original version of this handoff (before any GPU was available) is in git
history, commit `098cf3b`.

---

## 0. Summary in one sentence

The original hypothesis — that a **ladder code** with a 10 B table would beat
Huffman's 4 KiB hierarchical LUT — **is refuted**: it loses on both axes. But
by attacking the memory access pattern and the block index, the codec went from
9.70 ms / 30.73% to **4.82 ms / 32.97%**.

---

## 1. Status of the claims

This matters more than anything else: it separates what is measured from what
is speculated.

### Measured and reproducible

Unless stated otherwise, on real **Qwen3-0.6B** (596,049,920 unique BF16
weights; the duplicated `lm_head` is dropped, see METHODOLOGY.md) and,
for the kernels, a sample of 64M exponents on a **GTX 1050 Ti**.

| Claim | Value | How it was verified |
|---|---|---|
| Exponent entropy | 2.645 bits | Exact count over 596.0M weights |
| Canonical Huffman | 2.678 bits/exp -> 32.48% | Codec implemented, bit-exact roundtrip |
| Ladder (1,1,1,2) | 2.815 bits/exp -> 31.62% | Same, −0.86 points |
| Both GPU kernels | bit-exact roundtrip correct | 32M real symbols, 20 BLOCK x threads combinations |
| **The decoder is NOT compute-bound** | 8.2% of peak bandwidth | "Memory floor" kernel with the same traffic |
| Coalescing the **output** | **1.92x** | 4 variants compiled to attribute the gain |
| Coalescing the **input** | 1.01x | Never was the bottleneck |
| BLOCK cliff = L2 effect | crossover between 128 and 256 | Matches the working-set overflow, and the point where staging input in shared jumps to 4.38x |
| **Ladder slower than Huffman** | 5% to 43% | With the optimized kernel, in every configuration |
| Mutual information between neighbouring exponents | **0.0002–0.0100 bits** | 6 windows spread across the model |
| Order-1 context modelling | **+0.0000 bits/symbol** | No correlation to exploit |
| Ladder over pairs | 2.754 bits, worse than scalar (2.709) | With a 25x larger table |
| **8-bit index + prefix-sum** | **+2.25 points, zero cost** | 4.82 vs 4.81 ms |
| Mutual information between the BF16 fields | I(exp; mant) = **0.040 bits/weight**; sign independent of both | Exact joint histogram over all 596M unique weights |
| Field split vs full-alphabet Huffman | **0.076 bits/weight = 0.47 points** given away | Both rates computed analytically from exact counts (Phase 2d) |

### Not measured — still open

| Question | Why it matters |
|---|---|
| What happens on a GPU with a large L2 | Everything above is Pascal with 1 MiB of L2. See section 4. |
| Real occupancy, warp stalls, divergence | Nsight Compute **does not support Pascal**; could not be measured here |
| Comparison against DFloat11's published kernels | Only compared against our own implementation |
| End-to-end tokens/s (Phase 3) | "Faster kernel" is not the same as "faster inference" |
| Anomaly: at BLOCK=128 with input staged in shared, Huffman is 2x faster than the ladder | Far more than the 4.8% stream difference justifies. Unexplained. |

### Discarded (dead ends already explored)

- **Compressing the raw bitstream in groups of 3 or 4 bits.** Between −18% and
  +12.6%. Far below attacking the exponent.
- **Per-weight offset index.** Costs ~800% of the file.
- **Global prefix-sum of offsets.** Chicken-and-egg problem; the solution is
  the coarse per-block index.
- **Non-prefix-free codes.** Any new table must satisfy Kraft <= 1.
- **Vector / pairwise coding.** Measured: there is no correlation to exploit
  (I < 0.01 bits) and for the ladder it is worse than the scalar code.
- **The ladder as a performance route.** It loses on compression and on speed.
- **Computing directly on compressed data, with no decode step.** Reasoned,
  not measured. Entropy codes are **not homomorphic**: a codeword is a
  frequency-chosen symbol index with no arithmetic relation to the value it
  denotes, so no operation on codewords corresponds to multiplying values. The
  homomorphic route does exist, but it requires a scale-factor representation
  — `Σ (s_a·a_i)(s_b·b_i) = s_a·s_b·Σ a_i·b_i`, the scales factor out of the
  sum — which is precisely what INT8/INT4 quantization is, and it is lossy.
  Related trilemma, worth keeping in mind: *fixed symbols per block* gives
  variable bits and needs an index (what we do); *fixed bits per block* gives
  O(1) addressing but you no longer know which weights live in a block;
  *fixed in both* is a fixed-rate code, i.e. no entropy coding at all (GPU
  texture formats like BC/ASTC are the existence proof — and they are lossy).
  Pick two. Losslessness closes the third corner, so the index is not
  avoidable. The reachable goal is making the decoder never touch DRAM (GEMM
  fusion), not deleting it.

---

## 2. The format, as it stands

BF16 = `[1 sign][8 exponent][7 mantissa]`. Sign and mantissa are stored raw,
packed into 1 byte per weight. Only the exponent is coded.

```
header          : magic, version, n_weights, BLOCK, code table
sign_mantissa   : n_weights bytes, raw
exponents       : canonical Huffman bitstream
index           : see below
```

**Index (changed in Phase 2c).** Instead of one absolute uint32 offset per
block (4 B/block):

- one **uint32 per superblock** of 32 blocks (= one warp) -> 0.125 B/block
- one **uint8 per block** holding its length in bits, minus a global minimum
  -> 1.0 B/block

Total **1.125 B/block**. Each block's offset is recovered with an exclusive
prefix-sum within the warp (5 steps of `__shfl_up_sync`, cost not measurable).
It fits in 8 bits because block length has little spread: at BLOCK=64, from
130 to 273 bits.

**Limitation:** it only reaches BLOCK=128. At BLOCK=256 the spread is 409 and
does not fit; that would need the relative 16-bit variant (2.016 B/block).

---

## 3. What the project learned

Worth stating explicitly, because it is the opposite of what was expected:

> The two **structural** changes (access pattern and index) are worth **2x in
> speed and +2.25 points**. The **entropy code** question, around which the
> entire project was built, is worth 0.86 points — and in the wrong direction.

Entropy coding is finished *within the field split*: Huffman lands 0.033
bits from the exponent-only floor and neighbouring exponents are independent.
Phase 2d bounds what the split itself gives away: 0.076 bits/weight
(0.47 points) against a full-alphabet Huffman, almost all of it in the two
largest binades. Everything beyond that is structural.

The doubt the original handoff already raised turned out to be the right one:

> *"Canonical Huffman uses a LUT: one SRAM access. The ladder uses `clz` +
> branch + shift + mask. On instruction count it may lose."*

It loses.

---

## 4. What is next, in order of value

### 4.1 Measure on a GPU with a large L2 — BLOCKING for publishing anything

Everything measured is Pascal with 1 MiB of L2. The figure that governs the
BLOCK cliff is **L2 per resident thread**:

| card | L2 | SMs | resident threads | **L2 / thread** |
|---|---|---|---|---|
| GTX 1050 Ti (the home one) | 1 MB | 6 | 12,288 | 85 B |
| RTX 3060 | 3 MB | 28 | 43,008 | 73 B |
| A100 80GB | 40 MB | 108 | 221,184 | 190 B |
| H100 SXM | 50 MB | 132 | 270,336 | 194 B |
| RTX 4090 | 72 MB | 128 | 196,608 | 384 B |
| RTX 4070 | 36 MB | 46 | 70,656 | 534 B |

> **Prediction recorded before measuring:** on a card with a large L2 the BLOCK
> cliff **should disappear**; at BLOCK=1024 the resident working set is 24.5 MB
> (fits in a 4070's 36 MB, does not fit in 1 MB). Staging input in shared
> should stop mattering, and BLOCK=512-1024 should become viable: the first
> configuration where the best compression and good speed coincide.
>
> **If the cliff is still there, the L2 explanation is false** and Phase 2b has
> to be rewritten before any of this leaves the repo.

How to do it: [RENT_A_GPU.md](RENT_A_GPU.md) (~1.40 USD, under an hour) or
the operator guides for a borrowed machine.

### 4.2 Cross-field mutual information — DONE (Phase 2d, 2026-09-13)

Measured exactly over all unique weights: **I(exp; mant) = 0.040
bits/weight**, sign independent of both. A full-alphabet canonical Huffman
achieves 10.602 bits/weight against 10.678 for the field split: **0.076
bits/weight, 0.47 points, is what the split gives away** on this model. The
dependence sits entirely in the two largest binades (exponents 122 and 123),
where the Gaussian tail decays within the binade. Details and the exact
decomposition in RESULTS.md, Phase 2d.

Consequence for claims: the design is 0.033 bits from the *exponent-only*
floor and 0.100 bits from the *true* floor (10.578). The 0.47 points is the
ceiling on any rate improvement that stays lossless and per-weight; it is
also the bound on 2606.15789's rate advantage before any credit for rANS.
The cheapest way to close most of it would be a joint exp+mant code for
exponents 122–123 only; not measured.

### 4.3 Fused BF16 output

Today the kernel emits one exponent byte per weight and a separate pass is
needed to merge sign+mantissa: 64 MB + 64 MB read and 128 MB written =
**256 MB, almost 3x the decode kernel itself (88.7 MB)**. Fusing it drops total
pipeline traffic from ~345 MB to ~217 MB (−37%) and removes an entire launch.
Needed for Phase 3 regardless.

### 4.4 16-bit index for BLOCK >= 256

The 8-bit one does not reach. The relative 16-bit variant gives 2.016 B/block
(+1.94 points at BLOCK=256). Useful if 4.1 confirms that large BLOCK is viable.

### 4.5 Huffman over pairs

Halves the iterations of the serial chain, which is the real bottleneck, and is
also 0.0164 bits/symbol smaller. Cost: an 8 KiB LUT. Speculative but cheap to
try.

### 4.6 Compare against DFloat11 — and against the real frontier

It is published and ships kernels. It is the comparison any reviewer would
demand, and the one that costs the most work.

**Warning (2026-09-13):** DFloat11 is no longer the state of the art. arXiv
2606.15789 beats it by up to 11x by fusing rANS decompression inside the GEMM.
Beating DFloat11 is no longer enough to publish; see section 6.

---

## 5. Traps already hit, so as not to repeat them

- A code without Kraft <= 1 is undecodable even if it appears to work on some
  test cases.
- Extrapolating a saving percentage from one data type to another is invalid.
- A flat table wastes the skew of the distribution.
- **An early `return` before `__syncthreads()` is undefined behaviour.** It
  cost non-deterministic incorrect output in the ladder at BLOCK=512/1024.
  Every thread in the block must reach the barrier.
- **Verify bit-exactness BEFORE timing, never after.** A kernel that writes
  outside its shared memory can give almost-correct results and a flattering
  time.
- **Do not combine optimizations without measuring them separately.** Input +
  output in shared is *worse* than output alone: the input spends shared memory
  and a barrier without buying anything.
- Consumer GPU clocks cannot be locked under Windows/WDDM. Compensate with
  sustained warm-up, median, and randomized order — and close everything else
  using the GPU.
- Nsight Compute does not support Pascal (dropped in 2020.1). A 1050 Ti cannot
  be profiled with any current version.

---

## 6. State of the art

Reviewed **2026-09-13**. The field has moved: **DFloat11 is no longer the
frontier**, which changes the premise of section 4.6.

### DFloat11 — the direct reference

NeurIPS 2025, arXiv 2504.11651, github.com/LeanModels/DFloat11. Does the same
thing as here: Huffman over BF16 exponents, sign and mantissa untouched, ~30%
bit-exact reduction. Hierarchical LUTs in SRAM, a two-phase kernel, and a
*gaps* array holding each thread's bit offset — equivalent to the block index
here, and with exactly the structure the 8-bit index improves on.

Cost they report: **~40% to 2x slower than BF16 at batch 1**, parity (1.02x) at
batch 128, because decompression is constant per forward pass.

### The current frontier

**Approaching Shannon Bound with Lossless LLM Weight Compression**
(arXiv 2606.15789, June 2026). Uses **rANS, not Huffman**, with decompression
*fused inside the GEMM pipeline*: compression tiles match the GEMM geometry and
are decoded into shared memory while computation proceeds.

| | |
|---|---|
| vs DFloat11 | **up to 11x more throughput** |
| BF16 | to ~11-12 bits |
| INT8 / INT4-FP4 | ~4-5 bits / within 0.01-0.1 bits of the Shannon limit |
| End-to-end | Qwen-14B 1.1-1.2x; Mixtral-176B 1.6x (batch 20 -> 95) |

The decisive part: compression goes from being a **tax** to being a net
**gain**.

Their **only** stated objection to Huffman is the rate floor: *"Huffman coding
is fast but limited by integer-length codes, leaving nontrivial gaps to the
Shannon limit."* They do **not** claim Huffman cannot do tile-granular random
access, and it would be wrong if they did — see below.

**Random access is not the differentiator.** Their tiles get random access the
same way our blocks do: independent units plus an offset table. *"Each tile is
then entropy-encoded using ANS with an independently initialized state while
sharing the same per-layer codebook"*, with *"a compact offset entry [...] in
the tile index table"*. Raw rANS is in fact LIFO — encoded forward, decoded
backward — so it has **less** inherent seekability than a Huffman bitstream.
Our 8-bit delta + warp prefix-sum (1.125 B/block) is a tighter version of the
same idea.

What genuinely differs, in order of relevance to us:

1. **Interleaving buys coalescing in the format.** *"Because the compressed
   streams are interleaved across lanes, these renormalization loads are
   naturally coalesced in global memory."* Interleaved rANS round-robins N
   lanes through one buffer, so a warp's loads are adjacent by construction —
   the problem we fixed in the *kernel* with shared-memory staging for 1.92x.
2. **Per-unit state cost.** An rANS decoder state is one register; Huffman
   needs a resident 4 KiB LUT. At tile granularity with per-tile distributions
   this scales badly — the same wall found in Phase 2c (31 x 4 KiB = 124 KiB,
   over the 48 KiB shared budget).
3. **The rate floor**, which is weak for our data (98.8% efficiency).

### Rate comparison — we are NOT ahead of them

Tempting misreading to avoid. They report BF16 *"effective entropy of only
10-12 bits"* and *"about 4-5 bits of redundancy, corresponding to a potential
1.5x reduction"*. Ours is 10.818 bits/weight achieved (8 raw + 2.678 exponent +
0.141 index). Those brackets overlap, but the comparison does not hold:

- **Different models.** Ours is Qwen3-0.6B; they test Qwen-1.5B through
  Llama-405B. A 0.6B model being more compressible is unremarkable.
- **Entropy vs achieved.** Their 10-12 bits is a floor; our 10.818 is a result.
- **Decisive: they treat BF16 as a monolithic 16-bit alphabet**, we code fields
  separately. By subadditivity a full-alphabet coder can never be worse on
  rate, so on the same model they would land at or below our floor. **Their
  approach dominates ours on rate by construction.** Phase 2d measured how
  much: **0.076 bits/weight (0.47 points)** on Qwen3-0.6B, of which 0.040 is
  mutual information between the fields and the rest is the raw mantissa and
  Huffman granularity.

What can honestly be claimed is narrower and still worth stating: we are
**0.033 bits from the theoretical floor of exponent-only BF16 coding**, and
**0.100 bits from the true floor of the 16-bit symbol**, with the gap now
measured rather than assumed. That is a completeness result about this
approach, not a state-of-the-art claim. And rate was never the contested axis
— they are up to 11x faster.

**Float8@2bits / EntQuant** (arXiv 2601.22787, January 2026). ANS via nvCOMP,
1.5-2x slower than BF16, i.e. **matching NF4's speed**. Observes that entropy
coding was historically seen as "a passive storage optimization, applied
offline", not as part of the inference pipeline.

**EntroLLM** (arXiv 2505.02380). Huffman over already-quantized weights, for
edge devices. Decodes **once per sequence on CPU** (1.66 s for uint4) and
amortizes it. A different deployment model; not per-forward-pass decompression.

### Why production quantization does not use entropy codes

GGUF, GPTQ, AWQ and NF4 are all **fixed-width**. The reason usually cited is
"Huffman is slow", and it is imprecise. The real one: **variable-length codes
destroy random access**. With fixed width you address weight N directly and
fuse dequantization inside the GEMM, in-register and on the fly. With variable
length you cannot know where symbol N starts without decoding everything before
it, which forces you to decompress to memory and multiply afterwards — a full
extra round trip.

**Our own measurements confirm this from the other side**: the decoder is
memory-bound (8.2% of peak), coalescing the output was worth 1.92x, and the
choice of entropy code was worth almost nothing.

### What this implies for this project

1. The objection "Huffman is not used because it is slow" is about **lossy
   fixed-width quantization**. What is done here is **lossless** compression,
   where the entropy code is the whole game and Huffman is the incumbent.
2. The main finding here — access pattern and index dominate, the entropy code
   does not — is **independently corroborated** by 2606.15789, which is exactly
   what they exploit.
3. The structural ceiling of the current design is that it is a **standalone
   decompression kernel**. Without GEMM fusion you pay the full memory round
   trip, and that is the difference between 2.01x and 11x.
4. **The 11x is not the entropy coder.** They attribute it to fusion:
   *"eliminates global-memory materialization of decompressed layers and
   overlaps decompression with tensor-core computation"*, with tile-alignment
   worth x3.3-8.2 and double-buffering on top (x4.0-10.1 total over naive).
   ANS makes tile granularity affordable; it is not what makes it fast.
5. So the case for switching to ANS **here** is weak. Our exponents are 2.645
   bits of entropy and Huffman delivers 2.678 — **98.8% efficiency** — so the
   Shannon-gap argument buys almost nothing (it is strong for INT4/FP4, where
   gaps run 6-10x). The plausible move is to **keep canonical Huffman and go
   after fusion**, optionally borrowing the interleaved-stream layout for
   coalescing. That would test the same structural hypothesis at far lower
   cost than a codec rewrite.

### Historical context and others

- **Deep Compression** (Han et al., arXiv 1510.00149, 2015). The classic:
  pruning + quantization + Huffman, 20-30% extra. CNN era, aimed at storage and
  not at inference speed.
- **MLX issue #3043** (January 2026). Open request for rANS quantization: there
  is demand from implementers.
- **Recoil** (arXiv 2306.12141). Parallel rANS decoding, useful if the ANS
  route is explored.
