# Phase 1 — CPU validation: results

Run on real **Qwen/Qwen3-0.6B** weights (`model.safetensors`, `torch_dtype:
bfloat16` confirmed in `config.json`). The file holds 751,632,384 BF16 values,
but `lm_head.weight` is a bit-identical copy of `model.embed_tokens.weight`
(tied embeddings, stored twice); the extractor drops the duplicate and every
number below is over the **596,049,920 unique weights**. All Phase 1 criteria
from the HANDOFF are met.

> **Re-run 2026-09-13.** The first version of this phase counted the
> duplicated tensor (751.6M weights). The whole-model figures moved by at most
> 0.1 points (entropy 2.634 -> 2.645, Huffman 32.56% -> 32.48%, ladder gap
> 0.85 -> 0.86); the conclusions did not. The original tables are in git
> history before this date. The GPU phases below are unaffected: their timing
> sample (the first 64M weights) is the embedding matrix in both layouts, and
> the Huffman table change alters that sample's stream by 843 bits out of
> 165 million.

## Criteria (Section 5, Phase 1)

| Step | Criterion | Result |
|---|---|---|
| 2. Exponent entropy | between 2.4 and 3.0 bits | **2.645 bits** ✓ |
| 3. Canonical Huffman (`df11_reference.py`) | >=28% saving, exact roundtrip | **32.48%** saving, bit-exact roundtrip **YES** ✓ |
| 4. Reassign ladder to the real distribution | — | done, 43 distinct symbols (vs 25 synthetic) |
| 5. Ladder codec | exact roundtrip, <1 point from Huffman | roundtrip **YES**, gap = **0.86 points** ✓ |

## Huffman vs ladder comparison (BLOCK=256, uint32 index)

| | Synthetic (seed 1234) | Real (Qwen3-0.6B) |
|---|---|---|
| weights | 2,162,688 | 596,049,920 |
| distinct exponents | 25 | 43 |
| exponent entropy | 2.7182 bits | 2.6450 bits |
| canonical Huffman | 31.98% | **32.48%** |
| Ladder (1,1,1,2) | 31.28% (−0.69 pts) | **31.62%** (−0.86 pts) |
| Ladder (1,1,2,2) | 30.92% (−1.05 pts) | 31.42% (−1.06 pts) |
| Ladder (1,2,2,3) | 29.95% (−2.03 pts) | 30.07% (−2.41 pts) |

The **(1,1,1,2)** shape from the handoff is still the best of the three tried,
on both synthetic and real data. Real weights are *more* concentrated (lower
entropy, but more distinct long-tail symbols) than the synthetic ones — the
design generalizes without shape changes; only which exponent occupies each
slot changes (the 10-byte header).

## `BLOCK` sweep (real data, ladder (1,1,1,2))

| BLOCK | blocks | index | % overhead | total reduction |
|---|---|---|---|---|
| 64 | 9,313,280 | 35.5 MB | 4.42% | 29.28% |
| 128 | 4,656,640 | 17.8 MB | 2.26% | 30.84% |
| 256 | 2,328,320 | 8.9 MB | 1.14% | 31.62% |
| 512 | 1,164,160 | 4.4 MB | 0.58% | 32.01% |
| 1024 | 582,080 | 2.2 MB | 0.29% | 32.21% |

Confirms the handoff's trade-off: raising `BLOCK` reduces index overhead but
lengthens the serial decode chain within each block (more symbols to decode in
sequence per thread/warp).

## Methodology (important, to avoid repeating work)

- **Compressed size**: computed **analytically** (`Σ counts[s] · length[s]`)
  over exact exponent counts for the whole file — not an estimate, it is
  exact, but it avoids building the 596M-symbol bitstream in pure Python
  (infeasible in reasonable time without GPU/Cython).
- **Bit-exact roundtrip**: verified on a **random sample of 2,000,000 symbols**
  (with replacement, `outputs/real_weights_bf16.bin` via `np.memmap` to avoid
  loading 1.2 GB into RAM), using the code table derived from the **complete**
  distribution. A prefix code is memoryless per symbol, so this proves codec
  correctness just as well as encoding the entire file — the sample hit the
  escape path 10,165 times, so that branch is covered too.
- `real_exp_counts.npy` stores the exact counts for all 596.0M weights so the
  `.safetensors` (1.4 GB) does not have to be re-read on later runs.

## Discarded / observed

- The environment has little RAM (3.5 GiB + 3.5 GiB swap). Loading the whole
  `.safetensors` (a 1.5 GB `f.read()`) or concatenating a 596M `uint16` array
  in one go kills the process (OOM, exit 137). Extraction had to stream tensor
  by tensor, writing straight to disk and accumulating counts incrementally.

## Blocked — requires GPU

> **Resolved in Phase 2** (see below): run on a GTX 1050 Ti. This section
> describes the Phase 1 environment, not the current one.

This environment has no GPU (`nvidia-smi` does not exist, no CUDA-capable
`torch` installable). **Phase 2 (GPU kernel, Nsight Compute) and Phase 3
(matmul integration, tokens/s) cannot be run here.** The question that decides
the whole project — "is the exponent kernel memory-bound or compute-bound?" —
remains unanswered. To continue, a machine with a GPU is needed (or a direct
comparison against the published DFloat11 implementation at
github.com/LeanModels/DFloat11, which does ship kernels).

## New files in this directory

| File | What it is |
|---|---|
| `ladder_codec.py` | Generic ladder codec (any `rung_bits`), with real bitstream encode/decode and Kraft verification. Includes a self-test. |
| `bench.py` | Test harness: Huffman vs ladder, synthetic and real, shape and `BLOCK` sweeps, sampled roundtrip. Produces `outputs/bench_results.json`. |
| `outputs/real_weights_bf16.bin` | 596,049,920 raw BF16 weights extracted from Qwen3-0.6B (all tensors, duplicate `lm_head` dropped, 1.2 GB). |
| `outputs/real_exp_counts.npy` | Exact per-exponent counts (256,) for the above file. |
| `outputs/bench_results.json` | Complete numeric results from `bench.py`. Committed copy: `results/cpu/`. |
| `outputs/bench_log.txt` | Console log of the last `bench.py` run. Committed copy: `results/cpu/`, with the extractor log and `env_info.json`. |
| `real_model/` | `model.safetensors` + `config.json` for Qwen3-0.6B, downloaded from Hugging Face. |

---

# Phase 2 — GPU kernels: results

Run on an **NVIDIA GeForce GTX 1050 Ti** (Pascal, SM 6.1, 6 SMs, 4 GB, 1 MiB
L2, theoretical peak 112.1 GB/s), over the same real Qwen3-0.6B weights
(sample of 64,000,000 exponents).

## What was implemented

| File | What it is |
|---|---|
| `gpu_kernels.py` | The two kernels: `decode_huffman` (4 KiB hierarchical LUT in shared) and `decode_ladder` (clz + switch, 10 B table). |
| `bitpack.py` | Vectorized numpy encoder. Produces a bitstream **byte-for-byte identical** to the original `BitWriter` (verified against both codecs, offsets included); the Python loop was infeasible at kernel scale. |
| `verify_kernels.py` | Correctness: both kernels decode 32M real exponents **bit-exactly**. |
| `bench_gpu.py` | Measurement, including the `mem_floor` kernel. |

Both kernels share the coarse index, the one-thread-per-block mapping, the
`peek32` window reader and the output layout. **The only thing that differs is
the symbol decoder.**

Resource usage (from `kernel.attributes`):

| | registers/thread | shared |
|---|---|---|
| `decode_huffman` | 21 | 4096 B |
| `decode_ladder` | 15 | 16 B |

## Methodology

- **Memory floor kernel (`mem_floor`)**: moves exactly the same traffic (reads
  the same bitstream words, writes the same bytes) but **does not decode**. It
  is an empirical performance ceiling: getting close to it means memory
  dominates and the entropy code is irrelevant.
- This GPU **does not allow locking clocks** (consumer under WDDM, and it also
  drives the desktop: it swings between 139 and 1923 MHz). Compensated with
  sustained warm-up per configuration, clock sampling, randomized configuration
  order and the median of 11 repetitions. With the desktop cleared the IQR
  stays at 0-3 ms.
- The block index is **uint32**, as the format spec says.

## Results (64M symbols, median of 11)

| BLOCK | threads | huffman ms | ladder ms | floor ms | h/floor | l/floor | **l/h** |
|---|---|---|---|---|---|---|---|
| **64** | 128 | **9.70** | **9.73** | 9.57 | 1.01 | 1.02 | **1.003** |
| 128 | 128 | 51.08 | 34.55 | 12.77 | 4.00 | 2.71 | 0.676 |
| 256 | 128 | 82.75 | 56.95 | 13.31 | 6.22 | 4.28 | 0.688 |
| 512 | 128 | 118.63 | 89.84 | 16.06 | 7.38 | 5.59 | 0.757 |
| 1024 | 128 | 153.77 | 119.27 | 19.52 | 7.88 | 6.11 | 0.776 |

## The answer to the question that decides everything

**At the only operating point you would actually choose, the kernel is
MEMORY-bound, and the ladder contributes nothing.**

There are two regimes and they point in opposite directions:

- **BLOCK=64 is the fastest configuration, by 3-16x.** There both decoders are
  within **1-2% of the memory floor** and within **0.3-0.9% of each other**.
  The entropy code is irrelevant. Worse: the ladder is systematically the
  **slower** one (1.003-1.009x), which is exactly what you would expect — in a
  memory-bound regime its 4.8% larger bitstream costs time and buys nothing.
- **BLOCK>=128** the ladder wins by 5-32%, confirming the compute-bound
  hypothesis... but **all those configurations are 3-13x slower in absolute
  terms**. Nobody would use them.

In other words: **risk #1 from the handoff materialized.** On this hardware the
ladder ends up a curiosity, 0.86 points worse on compression with no speed
compensation. The handoff said explicitly that this would be a valid result; it
is.

## Three caveats that bound the conclusion

**1. "Memory-bound" here means bound by a bad access pattern, not by the
hardware.** Peak is 112.1 GB/s; the best configuration achieves **9.1 GB/s,
8.2% of peak**. Each thread walks its own region, so a warp scatters into 32
independent streams. There is ~10x of headroom on the table, available to
**both** codecs, by attacking the access pattern (warp-level cooperative
decoding, vectorized loads). That prize is far larger than the 0.86 points
separating the two codes.

**2. Output dominates the traffic.** At BLOCK=64, 88.7 MB moves, of which
**64 MB (71%) is the exponent output**. The two bitstreams differ by less than
1 MB. Structurally the entropy code can only influence ~23% of the traffic, and
fusing the sign+mantissa merge to emit BF16 directly would shrink that fraction
further. That is why the two codecs converge, and this is **not** specific to
this GPU.

**3. The BLOCK cliff looks like an L2 effect, and therefore does not
transfer.** L2 is 1024 KiB, and the resident working set crosses it between
BLOCK=128 (520 KiB) and BLOCK=256 (1040 KiB) — exactly where performance
collapses. This is **consistent, not proven**: BLOCK=128 already degrades even
though it still fits. An H100 has ~50 MB of L2 and an RTX 4090 ~72 MB, so the
cliff shifts far to the right, large BLOCK becomes viable, and **that is
precisely the regime where the ladder wins**. This result argues against the
ladder on Pascal; it **does not settle the question on a modern card**.

## Not measurable here

**Nsight Compute does not support this GPU.** NVIDIA dropped Pascal (SM 6.x)
support in version 2020.1; no current version profiles a 1050 Ti. The only
profiler with Pascal support is legacy `nvprof`, which only ships with CUDA
Toolkit <=11.x (~3 GB install) and needs admin privileges for the counters. We
decided not to install it: the floor kernel answers the central question more
directly than the counters would, and handoff risk #4 (warp divergence on the
escape branch) is moot when compute does not dominate at the operating point.

So the following remain **unmeasured**: real occupancy, warp stall reasons and
divergence.

## Suggested next steps

1. **Local, free:** attack the access pattern (8.2% of peak). If even half of
   that 10x is recovered, every later conclusion changes.
2. **~5 USD, half a day:** rent an RTX 4090 or L40S. Redo the BLOCK sweep with
   a large L2 and use Nsight Compute for stalls and divergence.
3. **Only if 2 looks promising:** A100/H100 to compare head-to-head against
   DFloat11's published kernels on their target hardware.

---

# Phase 2b — Access pattern optimization

Phase 2 left the kernel at **8.2% of peak** bandwidth. That is not
"memory-bound" in the useful sense: the floor kernel has the same bad access
pattern, so it measured the ceiling *of that layout*, not of the hardware.
This phase attacks the layout.

## Two separable problems

- **Input**: each thread walks its own bit region -> a warp scatters into 32
  streams. But the regions of all threads in a CUDA block are contiguous: they
  can be loaded coalesced into shared.
- **Output**: 71% of the traffic and it behaves worse. Each thread writes
  `BLOCK` consecutive bytes, so a warp writes 32 bytes spaced `BLOCK` apart:
  32 distinct sectors for 32 bytes of data. Same solution.

**4 variants** (input x output) were compiled so the improvement could be
attributed rather than guessed. All verified bit-exactly.

## Result (BLOCK=64, 128 threads, 64M symbols)

| variant | shared | ms | GB/s | vs base |
|---|---|---|---|---|
| base (global) | — | 9.74 | 9.2 | 1.00 |
| smem input | 3.7 KB | 9.66 | 9.3 | 1.01x |
| **smem output** | 8.2 KB | **5.07** | **17.7** | **1.92x** |
| both | 11.9 KB | 6.59 | 13.6 | 1.48x |

**Output alone nearly doubles performance**: 8.2% -> 15.8% of peak. Coalescing
the input is worth ~1%: it was never the bottleneck.

**Doing both is WORSE than output alone** (6.59 vs 5.07). Staging the input in
shared spends shared memory and adds a barrier without buying anything, and
that is paid for in occupancy. Combining optimizations made it worse.

## At BLOCK=256 it inverts — and that confirms the L2 story

| BLOCK=256, 64 threads | ms | vs base |
|---|---|---|
| base | 57.14 | 1.00 |
| **smem input** | **13.05** | **4.38x** |
| smem output | 15.60 | 3.66x |

At BLOCK=64/128 the input working set fits in the 1 MiB L2, so staging it in
shared is redundant (~1%). At BLOCK=256 it is 1040 KiB and no longer **fits**:
staging it in shared is worth 4.38x. The crossover falls exactly where Phase
2's performance cliff fell.

That is now **two independent lines of evidence** for the same explanation.

## Effect on the compression trade-off

| BLOCK | best ms | compression |
|---|---|---|
| 64 | 5.07 | 29.28% |
| 128 | 9.26 | 30.84% |
| 256 | 13.05 | 31.62% |

(Compression is the whole-model ladder figure from Phase 1, uint32 index.)

Tension remains, but far less brutal: BLOCK=256 used to be 6x slower than
BLOCK=64, now it is 2.6x in exchange for 2.34 more points of compression.

## Huffman vs ladder, redone with the optimized kernel

Phase 2's conclusion was measured with the slow kernel. Moving the bottleneck
means the comparison must be **redone**, not extrapolated.

| BLOCK | threads | variant | huffman ms | ladder ms | l/h |
|---|---|---|---|---|---|
| 64 | 64 | output | 5.23 | **5.07** | 0.970 |
| **64** | **128** | **output** | **4.80** | 5.07 | **1.055** |
| 128 | 128 | output | 8.75 | 9.73 | 1.112 |
| 256 | 128 | output | 12.73 | 18.20 | 1.430 |
| 256 | 64 | input | 13.04 | 13.05 | 1.001 |

**Global optimum: Huffman, BLOCK=64, 128 threads, output in shared: 4.80 ms.**
2.02x faster than Phase 2's best (9.70 ms).

The optimization **reinforces** Phase 2's conclusion rather than overturning
it. The ladder now loses on **both axes**: 0.86 points worse compression *and*
5% to 43% slower. Before, it at least won in the compute-bound regime.

And this is the regime where it should have won: at 16.5% of peak, compute
weighs more than before, and it still loses. The doubt the HANDOFF itself
raised turns out to be the right one:

> *"Canonical Huffman uses a LUT: **one** SRAM access. The ladder uses `clz` +
> branch + shift + mask. On instruction count it may lose."*

It loses. One shared-memory access beats a dependent clz/branch/shift/mask
chain.

## Bug found and fixed

Extending the sweep to BLOCK=512/1024 produced **incorrect, non-deterministic**
output in the base ladder kernel. Cause: the early `return` sat **before**
`__syncthreads()`. If `n_blocks` is not a multiple of `blockDim.x`, some
threads in the block exit while the rest wait at a barrier they can no longer
all reach: undefined behaviour. `decode_huffman` synchronized before exiting,
which is why only the ladder failed.

Fixed; all 20 BLOCK x threads combinations verify bit-exactly. The Phase 2
sweep was **rerun** with the corrected kernel and the numbers do not move
(9.71/9.74 against 9.70/9.73 at BLOCK=64), so the published conclusions stand.

## Unexplained

At BLOCK=128 with the input staged in shared, Huffman is **2x** faster than the
ladder (12.83 vs 26.14 ms; 12.92 vs 31.04 in the 2026-09-13 replication), far
more than the 4.8% stream size difference justifies. Neither registers (21 vs 15) nor shared memory predict it. Recorded
as an anomaly, not explained: it is not on the optimal path, but it is the kind
of thing that sometimes hides a bug.

## Pending on hardware with a large L2

All of this is Pascal with 1 MiB of L2. On an RTX 4070 (36 MB L2, 534 B per
resident thread, 6.3x better) the working set at BLOCK=1024 is 24.5 MB and
**fits**. Prediction: the cliff should disappear, staging the input in shared
should stop mattering, and BLOCK=512-1024 would become viable — the first
configuration where the best compression (32.3%) and good speed coincide.

---

# Phase 2c — Vector coding (no) and 8-bit index (yes)

## Question: does vector / pairwise coding help?

The intuition is reasonable: Huffman buys compression by paying in dictionary
size, and the ladder exists precisely to avoid paying that. Scaling to pairs
should be where it shows most. This was measured rather than argued.

The possible gain has **two sources** worth separating:

**(a) Code redundancy.** On the 64M timing sample, Huffman sits at 2.5832 bits
against an entropy of 2.5520: **0.0312 bits/symbol** of headroom (on the whole
model the pair is 2.678 vs 2.645, a 0.033 gap; here Huffman is built on the
sample's own histogram, hence 2.5832 rather than the 2.5838 the full-model
table gives on this sample). No entropy coder — vector,
arithmetic, ANS — can beat that bound if the symbols are independent.

**(b) Correlation between neighbouring exponents.** This is **not** bounded by
(a): it would be new headroom. And it is an empirical question.

Measured over six windows spread across the whole model (not just the first 64M
weights, which are the embedding matrix and could mislead; offsets refer to
the file as extracted before the duplicate `lm_head` was dropped):

| offset | H(X) | H(Y given X) | I(X;Y) |
|---|---|---|---|
| 0 | 2.5494 | 2.5491 | 0.00024 |
| 93,954,048 | 2.5591 | 2.5589 | 0.00026 |
| 187,908,096 | 2.5545 | 2.5543 | 0.00020 |
| 375,816,192 | 2.6535 | 2.6435 | 0.00998 |
| 563,724,288 | 2.6493 | 2.6406 | 0.00870 |
| 711,632,384 | 2.6436 | 2.6347 | 0.00884 |

**Maximum mutual information: 0.00998 bits.** Neighbouring exponents are
independent for practical purposes. Order-1 context modelling confirms it:
**+0.0000 bits/symbol**.

So only (a) remains, and Huffman over pairs captures half of it: 2.5668 vs
2.5832, i.e. 0.0164 bits/symbol — **0.1% of the total file**.

**And for the ladder it is worse than useless:**

| ladder over pairs | bits/symbol | table |
|---|---|---|
| (1,1,1,2) | 5.1172 | 20 B |
| (2,2,3,4) | 3.1712 | 64 B |
| **(4,4,5,6)** | **2.7542** | 256 B |
| (5,5,6,7) | 3.0699 | 512 B |
| (6,6,7,8) | 3.5106 | 842 B |

The best ladder over pairs (2.7542) is **worse than the scalar ladder**
(2.7089) with a 25x larger table. The rung geometry works because three symbols
concentrate 69% of the mass; spreading over 421 observed pairs flattens the
distribution and power-of-two rungs no longer follow it. Vector coding is
precisely the technique that inflates tables, which is the one thing the ladder
exists to avoid.

**Where the intuition is right:** with context modelling, Huffman would need
31 x 4 KiB = **124 KiB of tables, which do not fit in 48 KiB of shared**; the
ladder would need **310 B**. If correlation existed, the ladder would be the
only viable GPU option. The structural argument is correct; what does not exist
is the redundancy to exploit.

**One angle does survive, but for Huffman:** coding pairs halves the number of
decode iterations, and therefore the serial chain of dependent loads, which is
the real bottleneck. Huffman over pairs is simultaneously 0.0164 bits/symbol
smaller and half the iterations, in exchange for an 8 KiB LUT. Yet to be tried.

## 8-bit index + warp prefix-sum

Block length has little spread: at BLOCK=64, between 130 and 273 bits (range
143 < 256). It fits in 8 bits. So instead of one absolute uint32 offset per
block, store one uint32 per superblock of 32 blocks (a warp) plus **one uint8
per block holding its length**, and recover the offset with an exclusive
prefix-sum within the warp (5 steps of `__shfl_up_sync`).

| BLOCK | codec | index | ms | B/block | compression | |
|---|---|---|---|---|---|---|
| 64 | huffman | uint32 | 4.81 | 4.000 | 30.73% | |
| **64** | **huffman** | **uint8+ps** | **4.82** | **1.125** | **32.97%** | **+2.25 pts** |
| 64 | ladder | uint32 | 5.07 | 4.000 | 29.94% | |
| 64 | ladder | uint8+ps | 5.11 | 1.125 | 32.19% | +2.25 pts |
| 128 | huffman | uint32 | 8.74 | 4.000 | 32.29% | |
| 128 | huffman | uint8+ps | 8.77 | 1.125 | 33.41% | +1.12 pts |

Compression percentages are for the 64M sample; the "pts" column is computed
from unrounded values, which is why the Huffman row shows +2.25 while the two
rounded percentages differ by 2.24.

**The prefix-sum costs nothing measurable** (4.82 vs 4.81 ms, within noise;
the 2026-09-13 replication in `results/gtx1050ti/` gives 4.85 vs 5.11, i.e.
the 8-bit index 5% *faster* — both runs support "not slower", neither
supports a precise delta):
five `__shfl_up_sync` are invisible next to a chain of 64 dependent loads.
Index traffic drops from 4 MB to 1.125 MB.

### The speed/compression tension disappears

This was the structural problem of the whole phase. Projected to the full
model:

| | fastest config | best compression |
|---|---|---|
| before | BLOCK=64 -> 30.14% | BLOCK=1024 -> 33.07%, but **32x slower** |
| after | BLOCK=64 -> **32.39%** | BLOCK=1024 -> 33.07% |

The distance between "fast" and "compresses well" goes from 2.93 points to
0.68. There is no longer a choice to make.

**Best global configuration: Huffman, BLOCK=64, 128 threads, output in shared,
8-bit index — 4.82 ms and 32.97%.** Against the Phase 2 reference (9.70 ms,
30.73%): **2.01x faster and +2.25 points**, from two changes that touch neither
the entropy code nor the decode loop.

### Limitation

The 8-bit index **only reaches BLOCK=128**. At BLOCK=256 the block-length range
is 409, above the 255 that fit in an 8-bit delta; the builder raises an
exception rather than truncating silently. Larger blocks need the relative
16-bit variant (2.016 B/block, +1.94 points at BLOCK=256).

## Taken together

The two **structural** changes are worth 2.25 points and 2x in speed. The
entropy-code question around which the whole project was built is worth 0.86
points, and in the wrong direction. The index and the memory layout mattered
enormously more than Huffman-versus-ladder.

Entropy coding is, for practical purposes, finished: Huffman lands 0.033 bits
from the theoretical floor and there is no correlation to exploit. All
remaining headroom is structural.

## Pending

- **Fused BF16 output**: today the kernel emits one exponent byte per weight
  and a separate pass is needed to merge sign+mantissa (64 + 64 MB read, 128 MB
  written = 256 MB, almost 3x the kernel itself). Fusing it drops total
  pipeline traffic from ~345 MB to ~217 MB (-37%) and removes an entire launch.
  Needed for Phase 3 regardless.
- **Huffman over pairs**: half the iterations in the serial chain.

## New files

| File | What it is |
|---|---|
| `kernel_opt.py` | Kernels with input/output staging in shared (4 variants, for attribution). |
| `kernel_idx8.py` | 8-bit index + warp prefix-sum, for Huffman and ladder. |
| `bench_opt.py` | Attribution of the access-pattern improvement. |
| `bench_head2head.py` | Huffman vs ladder with the optimized kernel. |
| `bench_idx8.py` | uint32 vs uint8+prefix-sum: time and compression. |
| `analysis_vector.py` | Joint entropy, mutual information, Huffman/ladder over pairs. |
| `analysis_index.py` | Mutual information by window and cost of each index scheme. |

---

# Phase 2d — Cross-field mutual information (CPU)

## Question

The format codes the exponent and stores sign and mantissa raw. A coder over
the whole 16-bit symbol can never be worse on rate, because joint entropy is
subadditive: `H(sign, exp, mant) <= H(sign) + H(exp) + H(mant)`. The gap is
the mutual information between the fields, and until now it had not been
measured — HANDOFF 4.2 flagged it as the honest ceiling on any rate claim.
arXiv 2606.15789 codes the full 16-bit alphabet, so this is also the exact
amount by which their approach dominates this one on rate, before any
difference in entropy coder.

## Method

Exact, not sampled: the full alphabet has 65,536 symbols, so one `bincount`
over all 596,049,920 unique weights gives the joint histogram, and every
marginal and pair histogram is derived from it (`analysis_fields.py`, 6 s on
the dev machine; log in `results/cpu/analysis_fields.txt`).

## Entropies and mutual information

| | bits/weight |
|---|---|
| H(sign) | 1.0000 |
| H(exp) | 2.6450 |
| H(mant) | 6.9729 |
| H(sign) + H(exp) + H(mant) | 10.6180 |
| **H(sign, exp, mant)** — the true floor | **10.5781** |
| I(exp; mant) | **0.0398** |
| I(sign; exp) | 0.0001 |
| I(sign; mant) | 0.0000 |

Only 7,506 of the 65,536 possible 16-bit values occur.

**Where the dependence lives.** Mantissa entropy conditional on the exponent,
for the exponents carrying most of the mass:

| exp | mass | H(mant \| exp) |
|---|---|---|
| 121 | 29.4% | 6.981 |
| 122 | 21.2% | **6.840** |
| 120 | 21.2% | 6.998 |
| 119 | 12.0% | 6.999 |
| 118 | 6.3% | 6.999 |
| 123 | 3.5% | **6.255** |
| 117 | 3.2% | 7.000 |

The mantissa is uniform (7.00 bits) everywhere except in the two largest
binades, 122 and 123, where the Gaussian tail decays *within* the binade and
small mantissas are more likely. That is the entire 0.04 bits. Below the
mode the mantissa carries no information about the exponent at all.

## Achieved rates, and the exact decomposition

| canonical Huffman, no index | bits/weight |
|---|---|
| field split: 1 + 7 + Huffman(exp), 43 symbols, maxlen 30 | 10.6777 |
| full 16-bit alphabet, 7,506 symbols, maxlen 29 | **10.6021** |
| difference | **0.0756 = 0.47 points** |

The difference decomposes exactly into three parts:

| | bits/weight |
|---|---|
| true floor H(sign, exp, mant) | 10.5781 |
| + mutual information between fields | 0.0399 |
| + sign and mantissa stored raw, above their entropy | 0.0271 |
| + Huffman redundancy on the exponent alphabet | 0.0327 |
| = field split achieved | 10.6777 |
| full alphabet: floor + Huffman redundancy (0.0240) | 10.6021 |

## What this settles

- **The honest rate claim** for this design is: **0.033 bits/weight above
  the exponent-only floor, 0.100 bits/weight above the true floor of the
  16-bit symbol**, of which 0.076 (0.47 points of compression) a
  full-alphabet canonical Huffman recovers on this model.
- **Entropy coding is finished *within the field split***; it is not
  finished in absolute terms. Crossing to the full alphabet is worth 0.47
  points — more than half the ladder-vs-Huffman gap the project was built
  around — at the price of a 7,506-symbol code table instead of 43, which
  changes the decoder's LUT budget and is a different design.
- **2606.15789's rate advantage over this approach is bounded at 0.076
  bits/weight on this model** before any credit for rANS over Huffman;
  their remaining advantage is the fusion, as HANDOFF section 6 argued.
- The 0.040 bits of exp–mant dependence is real but concentrated: a coder
  that treated only exponents 122 and 123 jointly with their mantissa, and
  everything else as now, would capture nearly all of it with a small table.
  Not measured; recorded as the cheapest way to close most of the gap.

---

# Phase 2e — The L2 prediction tested on an A100 and an RTX 4090

## Question

Every GPU number before this phase came from one Pascal card with 1 MiB of
L2. Phase 2 and 2b attributed the collapse of performance at large `BLOCK`
to the resident working set overflowing L2, and HANDOFF 4.1 recorded the
prediction before measuring: on a card with a large L2 per resident thread
the cliff should disappear, staging the input in shared should stop
mattering at BLOCK=256, and BLOCK=512–1024 should become viable. This phase
tests it on two rented cards that bracket the prediction on the L2-per-thread
axis (`RENT_A_GPU.md`; ~1.7 USD in total).

## Cards

| | GTX 1050 Ti | A100-SXM4-80GB | RTX 4090 |
|---|---|---|---|
| architecture | Pascal, SM 6.1 | Ampere, SM 8.0 | Ada, SM 8.9 |
| SMs / L2 | 6 / 1 MiB | 108 / 40 MiB | 128 / 72 MiB |
| **L2 per resident thread** | **85 B** | **190 B** | **384 B** |
| peak bandwidth | 112 GB/s | 2039 GB/s | 1008 GB/s |
| SM clock during the run | 139–1923 MHz, unlocked | 1140 MHz | 2520 MHz |
| where | home PC, Windows | RunPod Secure Cloud container | RunPod Secure Cloud container |
| clocks locked | no (WDDM) | no (no permission in container) | no (same) |
| IQR over 11 repetitions | 0–3 ms | 0 | 0 |
| Nsight Compute | unsupported | blocked by host | blocked by host |

Same code, same 64M-symbol sample, same tables, same `run_all.sh`. Logs and
`env_info.json` in `results/a100sxm480gb/` and `results/rtx4090/`.

## The predictions, one by one

**1. "The BLOCK cliff should disappear."** Base kernel, Huffman, 128 threads:

| BLOCK | 1050 Ti | A100 | 4090 |
|---|---|---|---|
| 64 | 9.70 ms | 1.26 ms | 1.24 ms |
| 128 | 52.11 | 1.88 | 1.76 |
| 256 | 84.86 | 3.02 | 2.02 |
| 512 | 121.08 | 3.03 | 2.26 |
| 1024 | 157.16 | 2.46 | 2.13 |
| **1024 / 64** | **16.2x** | **1.96x** | **1.71x** |

**Confirmed.** The 16x collapse becomes 2x and 1.7x. What remains is not an
L2 effect: the memory-floor kernel rises the same way (A100 1.28 -> 1.72 ms,
4090 1.24 -> 1.64 ms), and at BLOCK=1024 there are only 62,500 threads for
221,184 (A100) or 196,608 (4090) resident slots. The residual is
under-occupancy of the card, and it would affect any one-thread-per-block
decoder.

**2. "Staging the input in shared should stop mattering at BLOCK=256."**
Attribution run, 64 threads:

| input staged, BLOCK=256 | 1050 Ti | A100 | 4090 |
|---|---|---|---|
| speed-up vs base | **4.38x** (4.46x in the replication) | **2.11x** | **1.20x** |

**Not confirmed as stated — and more informative than a yes.** The gain does
not switch off when the working set fits; it decays smoothly with L2 per
thread: 85 B -> 4.4x, 190 B -> 2.0x, 384 B -> 1.2x. Coalescing the input
helps even when every byte is an L2 hit, because 32 scattered streams per
warp are served more slowly from L2 than one contiguous load. The
L2-per-thread figure predicts the *size* of the effect, not its presence.
Phase 2b's explanation ("input staging is redundant when it fits in L2") is
corrected accordingly.

**3. "BLOCK=512–1024 should become viable."** Optimised kernel (output
staged), Huffman, best thread count per row:

| BLOCK | A100 | 4090 |
|---|---|---|
| 64 | **0.44 ms** | **0.20 ms** |
| 256 | 0.97 | 0.39 |
| 1024 | 1.52 | 1.21 |

**Moot rather than wrong.** Large BLOCK is 3–6x slower than BLOCK=64 on both
cards even without a cliff, because the serial chain inside each block is
16x longer and nothing hides it. The reason to want large BLOCK — index
overhead — was removed by the 8-bit index in Phase 2c, which gives BLOCK=64
32.97% against 33.07% for BLOCK=256 with a uint32 index. There is no
configuration on any of the three cards where large BLOCK is the right
choice.

## What the two new cards changed beyond the prediction

**Output staging scales up with the card.** BLOCK=64, 128 threads:

| | 1050 Ti | A100 | 4090 |
|---|---|---|---|
| output staged vs base | 1.92x | 2.61x | 4.55x |
| input + output staged vs base | 1.48x (**worse** than output alone) | 2.91x (better) | 5.49x (better) |

The Phase 2b lesson "combining the two optimisations is worse than output
alone" was Pascal-specific. On Ampere and Ada the combined variant is the
best one. Trap list updated: attribution results do not transfer across
architectures either.

**The optimised kernel tracks SM clock, not bandwidth.** Best Huffman time
for 64M symbols: A100 0.44 ms, 4090 0.20 ms — the 4090 is **2.2x faster
with half the bandwidth**, and 2520 / 1140 MHz = 2.2x. The base kernel does
not show this (1.26 vs 1.24 ms: latency-bound on the scattered pattern,
clock-insensitive). Once the access pattern is fixed, the decoder is bound by
the latency of its 64-step chain of dependent loads, and that chain runs at
SM clock. Two cards is not a proof; it is recorded as the reading most
consistent with the data. Achieved bandwidth confirms the decoder is nowhere
near the memory roof on the datacenter card:

| best configuration | GB/s | % of peak |
|---|---|---|
| 1050 Ti | 18.3 | 16% |
| A100 | 204 | 10% |
| 4090 | 453 | 45% |

**Huffman vs ladder, optimised kernel, at the operating point (BLOCK=64):**

| | 1050 Ti | A100 | 4090 |
|---|---|---|---|
| ladder / Huffman time | 1.055 | 0.97–1.08 | **1.26–1.36** |

The ladder ties on the A100 and loses by a third on the 4090. On the *base*
kernel at BLOCK >= 512 the ladder does beat Huffman on the A100 (l/h
0.79–0.95) — the compute-bound regime the original hypothesis predicted —
but those configurations are 2x slower than BLOCK=64 in absolute terms, as
on Pascal. Three architectures, same verdict: the ladder never wins where
you would run it.

**8-bit index + prefix-sum:** 1.006x (A100) and 0.996x (4090) the time of
the uint32 index, +2.25 points. Free on every card.

## Taken together

- The L2 explanation survives as a *trend* (the cliff and the input-staging
  gain both shrink monotonically with L2 per thread) and is corrected as a
  *threshold* (nothing switches off when the data fits).
- On datacenter hardware the standalone decoder sits at 10–45% of peak
  bandwidth and its time follows SM clock. The remaining headroom is not in
  the entropy code, not in the index, and not in bandwidth: it is in the
  serial chain, which is exactly what GEMM fusion or a wider-than-one-thread
  decode would attack (HANDOFF section 6).
- Best measured configuration on each card, same design throughout:
  Huffman, BLOCK=64, output staged (plus input on Ampere/Ada), 8-bit index.
  Decoding the 64M-symbol sample takes 4.8 ms / 0.44 ms / 0.20 ms.

