# Phase 1 — CPU validation: results

Run on real **Qwen/Qwen3-0.6B** weights (`model.safetensors`, 751,632,384
native BF16 weights, `torch_dtype: bfloat16` confirmed in `config.json`). All
Phase 1 criteria from the HANDOFF are met.

## Criteria (Section 5, Phase 1)

| Step | Criterion | Result |
|---|---|---|
| 2. Exponent entropy | between 2.4 and 3.0 bits | **2.634 bits** ✓ |
| 3. Canonical Huffman (`df11_reference.py`) | >=28% saving, exact roundtrip | **32.56%** saving, bit-exact roundtrip **YES** ✓ |
| 4. Reassign ladder to the real distribution | — | done, 43 distinct symbols (vs 25 synthetic) |
| 5. Ladder codec | exact roundtrip, <1 point from Huffman | roundtrip **YES**, gap = **0.85 points** ✓ |

## Huffman vs ladder comparison (BLOCK=256, uint32 index)

| | Synthetic (seed 1234) | Real (Qwen3-0.6B) |
|---|---|---|
| weights | 2,162,688 | 751,632,384 |
| distinct exponents | 25 | 43 |
| exponent entropy | 2.7182 bits | 2.6340 bits |
| canonical Huffman | 31.98% | **32.56%** |
| Ladder (1,1,1,2) | 31.28% (−0.69 pts) | **31.72%** (−0.85 pts) |
| Ladder (1,1,2,2) | 30.92% (−1.05 pts) | 31.52% (−1.05 pts) |
| Ladder (1,2,2,3) | 29.95% (−2.03 pts) | 30.18% (−2.38 pts) |

The **(1,1,1,2)** shape from the handoff is still the best of the three tried,
on both synthetic and real data. Real weights are *more* concentrated (lower
entropy, but more distinct long-tail symbols) than the synthetic ones — the
design generalizes without shape changes; only which exponent occupies each
slot changes (the 10-byte header).

## `BLOCK` sweep (real data, ladder (1,1,1,2))

| BLOCK | blocks | index | % overhead | total reduction |
|---|---|---|---|---|
| 64 | 11,744,256 | 44.8 MB | 4.43% | 29.37% |
| 128 | 5,872,128 | 22.4 MB | 2.26% | 30.93% |
| 256 | 2,936,064 | 11.2 MB | 1.14% | 31.72% |
| 512 | 1,468,032 | 5.6 MB | 0.58% | 32.11% |
| 1024 | 734,016 | 2.8 MB | 0.29% | 32.30% |

Confirms the handoff's trade-off: raising `BLOCK` reduces index overhead but
lengthens the serial decode chain within each block (more symbols to decode in
sequence per thread/warp).

## Methodology (important, to avoid repeating work)

- **Compressed size**: computed **analytically** (`Σ counts[s] · length[s]`)
  over exact exponent counts for the whole file — not an estimate, it is
  exact, but it avoids building the 751M-symbol bitstream in pure Python
  (infeasible in reasonable time without GPU/Cython).
- **Bit-exact roundtrip**: verified on a **random sample of 2,000,000 symbols**
  (with replacement, `outputs/real_weights_bf16.bin` via `np.memmap` to avoid
  loading 1.5 GB into RAM), using the code table derived from the **complete**
  distribution. A prefix code is memoryless per symbol, so this proves codec
  correctness just as well as encoding the entire file — the sample hit the
  escape path 9,652 times, so that branch is covered too.
- `real_exp_counts.npy` stores the exact counts for all 751.6M weights so the
  `.safetensors` (1.4 GB) does not have to be re-read on later runs.

## Discarded / observed

- The environment has little RAM (3.5 GiB + 3.5 GiB swap). Loading the whole
  `.safetensors` (a 1.5 GB `f.read()`) or concatenating a 751M `uint16` array
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
| `outputs/real_weights_bf16.bin` | 751,632,384 raw BF16 weights extracted from Qwen3-0.6B (all tensors, 1.4 GB). |
| `outputs/real_exp_counts.npy` | Exact per-exponent counts (256,) for the above file. |
| `outputs/bench_results.json` | Complete numeric results from `bench.py`. |
| `outputs/bench_log.txt` | Console log of the last `bench.py` run. |
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
ladder ends up a curiosity, 0.85 points worse on compression with no speed
compensation. The handoff said explicitly that this would be a valid result; it
is.

## Three caveats that bound the conclusion

**1. "Memory-bound" here means bound by a bad access pattern, not by the
hardware.** Peak is 112.1 GB/s; the best configuration achieves **9.1 GB/s,
8.2% of peak**. Each thread walks its own region, so a warp scatters into 32
independent streams. There is ~10x of headroom on the table, available to
**both** codecs, by attacking the access pattern (warp-level cooperative
decoding, vectorized loads). That prize is far larger than the 0.85 points
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
| 64 | 5.07 | 29.37% |
| 128 | 9.26 | 30.93% |
| 256 | 13.05 | 31.72% |

Tension remains, but far less brutal: BLOCK=256 used to be 6x slower than
BLOCK=64, now it is 2.6x in exchange for 2.35 more points of compression.

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
it. The ladder now loses on **both axes**: 0.85 points worse compression *and*
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
ladder (12.83 vs 26.14 ms), far more than the 4.8% stream size difference
justifies. Neither registers (21 vs 15) nor shared memory predict it. Recorded
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

**(a) Code redundancy.** Huffman sits at 2.5832 bits against an entropy of
2.5520: **0.0312 bits/symbol** of headroom. No entropy coder — vector,
arithmetic, ANS — can beat that bound if the symbols are independent.

**(b) Correlation between neighbouring exponents.** This is **not** bounded by
(a): it would be new headroom. And it is an empirical question.

Measured over six windows spread across the whole model (not just the first 64M
weights, which are `embed_tokens` and could mislead):

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

**The prefix-sum costs nothing measurable** (4.82 vs 4.81 ms, within noise):
five `__shfl_up_sync` are invisible next to a chain of 64 dependent loads.
Index traffic drops from 4 MB to 1.125 MB.

### The speed/compression tension disappears

This was the structural problem of the whole phase. Projected to the full
model:

| | fastest config | best compression |
|---|---|---|
| before | BLOCK=64 -> 30.22% | BLOCK=1024 -> 33.15%, but **32x slower** |
| after | BLOCK=64 -> **32.46%** | BLOCK=1024 -> 33.15% |

The distance between "fast" and "compresses well" goes from 2.93 points to
0.69. There is no longer a choice to make.

**Best global configuration: Huffman, BLOCK=64, 128 threads, output in shared,
8-bit index — 4.82 ms and 32.97%.** Against the Phase 2 reference (9.70 ms,
30.73%): **2.01x faster and +2.24 points**, from two changes that touch neither
the entropy code nor the decode loop.

### Limitation

The 8-bit index **only reaches BLOCK=128**. At BLOCK=256 the block-length range
is 409, above the 255 that fit in an 8-bit delta; the builder raises an
exception rather than truncating silently. Larger blocks need the relative
16-bit variant (2.016 B/block, +1.94 points at BLOCK=256).

## Taken together

The two **structural** changes are worth 2.24 points and 2x in speed. The
entropy-code question around which the whole project was built is worth 0.85
points, and in the wrong direction. The index and the memory layout mattered
enormously more than Huffman-versus-ladder.

Entropy coding is, for practical purposes, finished: Huffman lands 0.031 bits
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
