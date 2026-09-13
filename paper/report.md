# Entropy Code or Memory Layout? A Measured Refutation in Lossless BF16 Weight Compression

**Arnau Ferrerons Manich**
Independent researcher · ORCID 0009-0002-7245-7221
Code, data and logs: https://github.com/ArnauFerma/bf16-exponent-compression

## Abstract

Lossless compression of BF16 model weights by entropy-coding the exponent field, as in DFloat11, removes about a third of the bytes at the cost of a decompression kernel on every forward pass. This report tests a specific hypothesis about that kernel: that a fixed-shape prefix code with a 10-byte table (a "ladder" code) would decode faster on a GPU than canonical Huffman with a 4 KiB lookup table, at the cost of less than one point of compression. The hypothesis is refuted on three GPU architectures (Pascal, Ampere, Ada): the ladder code loses 0.86 points of compression and is never faster at any configuration one would actually run. Along the way, two changes that do not touch the entropy code at all — coalescing the decoder's output through shared memory, and replacing the per-block uint32 offset index with 8-bit block lengths recovered by a warp prefix-sum — improved the reference decoder by 2.0x in time and 2.25 points of compression on the same hardware, and the same design decodes 64M symbols in 0.20 ms on an RTX 4090. Exact measurements on Qwen3-0.6B place canonical Huffman 0.033 bits/weight from the exponent-only entropy floor and 0.100 bits/weight from the true floor of the 16-bit symbol; the field split itself gives away 0.076 bits/weight (0.47 points) against a full-alphabet coder, almost all of it in the two largest binades. A performance collapse at large block sizes on the Pascal card is shown to be an L2-capacity effect whose size decays with L2 per resident thread across the three cards, and the optimised decoder's time is found to track SM clock rather than memory bandwidth, consistent with a decoder bound by its serial chain of dependent loads. Every number is traceable to a committed log; the methodology, raw data and the pre-registered predictions are published with the code.

## 1. Introduction

Large language models are distributed and served in bfloat16. The 16 bits of a BF16 value are not equally informative: the sign and the 7-bit mantissa of trained weights are close to incompressible, while the 8-bit exponent of a roughly Gaussian weight distribution carries under 3 bits of entropy. DFloat11 [1] exploits this by Huffman-coding the exponents, storing sign and mantissa raw, and decompressing on the GPU before each matrix multiply, obtaining bit-exact outputs with roughly 30% fewer bytes. The cost is decode time: DFloat11 reports 1.4–2x slower inference than BF16 at batch size 1.

Because the decoder runs on every forward pass, its speed matters as much as the rate, and the choice of entropy code seemed the natural place to look for speed. Canonical Huffman decoding needs a table — DFloat11 uses hierarchical lookup tables in shared memory — and each decoded symbol costs at least one shared-memory access. A prefix code with a *fixed shape* whose codewords can be decoded arithmetically (count leading ones, then read a small index) needs no table beyond a handful of symbol bytes, and decodes with a `clz`, a branch and a shift. The hypothesis this project set out to test, recorded before any measurement, was that such a code would decode faster on a GPU than canonical Huffman, giving up less than one point of compression in exchange.

This report describes that test and its outcome. The contribution is not the ladder code, which loses; it is what the test revealed about where a standalone decompression kernel's time actually goes, with measurements that are reproducible from the published repository:

1. **A refutation on three architectures.** The ladder code is 0.86 points worse in rate and, once the decoder's memory access pattern is fixed, between tied and 1.36x slower than Huffman at every configuration one would run, on a GTX 1050 Ti, an A100 and an RTX 4090 (§5.2, §5.7).
2. **Where the headroom was.** Two structural changes worth 2.0x in time and +2.25 points on the Pascal card: staging the decoder's output through shared memory so that warps write coalesced (1.9x on Pascal, 4.6x on Ada), and an 8-bit block index with a warp prefix-sum that costs nothing measurable (§5.3, §5.4).
3. **The rate floor of the field-split design, measured exactly.** Neighbouring exponents are independent (mutual information < 0.01 bits); the exponent and mantissa of the same weight share 0.040 bits; the field split gives away 0.076 bits/weight against a full-alphabet Huffman code, all of it in the two largest binades (§5.5, §5.6).
4. **An L2 explanation tested across cards.** The 16x collapse at large block sizes on Pascal shrinks to 2x and 1.7x on cards with 2.2x and 4.5x the L2 per resident thread; the benefit of staging the *input* decays with L2 per thread rather than switching off; and the optimised decoder's time tracks SM clock, not bandwidth (§5.7).

All predictions were written down before the measurements that tested them and are kept in the repository as written; the ones that failed are reported as failures.

## 2. Background

**BF16 fields.** A BF16 value is `[1 sign][8 exponent][7 mantissa]`. On the 596,049,920 unique weights of Qwen3-0.6B (§4.1) the sign has 1.000 bits of entropy, the mantissa 6.973 of 7, and the exponent 2.645 of 8. Only 43 distinct exponent values occur; three of them carry 72% of the mass. Everything compressible is in the exponent, and the field-split design of DFloat11 codes only that field.

**Canonical Huffman and the block index.** A Huffman code over the exponent alphabet is prefix-free and, in canonical form, is fully specified by the code length of each symbol, so the table costs at most 256 bytes to transmit. Decoding on a GPU uses a primary lookup table indexed by the next *k* bits of the stream (here *k* = 11, 2048 entries of 2 bytes = 4 KiB in shared memory) with a canonical search for the rare codes longer than *k*. Variable-length codes destroy random access: to decode symbol *n* one must know where it starts. The standard remedy, used by DFloat11 in the form of a per-thread "gaps" array, is a coarse index: an absolute bit offset per block of `BLOCK` symbols. Blocks then decode in parallel, one thread each, and the decode within a block is serial. `BLOCK` trades index overhead against the length of that serial chain.

**The ladder code.** Read ones until the first zero; the count *r* selects a rung; then read `rung_bits[r]` index bits into a small per-rung table. With rung bits (1, 1, 1, 2), the code has lengths 2, 3, 4, 6 for the 10 most frequent symbols and an escape (`1111` plus the 8 raw exponent bits, 12 bits) for the rest. Kraft's sum is exactly 1 by construction, so the code is complete. The shape is fixed; only which symbol occupies which slot changes between models, so the table is 10 bytes. Decoding is `__clz(~w)`, a four-way branch, a shift and a mask — no memory access beyond the 10-byte table.

**State of the art at the time of writing.** DFloat11 [1] (NeurIPS 2025) is the direct reference and uses the same field split with Huffman. In June 2026, arXiv:2606.15789 [2] reported up to 11x higher throughput than DFloat11 by coding the whole 16-bit symbol with rANS and fusing decompression into the GEMM pipeline, so that decompressed tiles never touch global memory. That paper's only stated objection to Huffman is its rate gap to the Shannon bound from integer-length codes; it does not claim, and it would be wrong to claim, that Huffman cannot support tile-granular random access — its tiles obtain random access exactly as blocks do here, via independent streams and an offset table. The relevance of [2] to this report is discussed in §6.

## 3. Hypothesis and pre-registered predictions

The hypothesis, recorded in the repository's first commit before any GPU was available:

> A ladder prefix code with a 10-byte table decodes faster on a GPU than canonical Huffman with a 4 KiB hierarchical LUT, at a cost of under 1 point of compression.

The same document recorded why it might fail: "Canonical Huffman uses a LUT: one SRAM access. The ladder uses `clz` + branch + shift + mask. On instruction count it may lose", and listed as the most likely outcome that "the ladder does not gain any speed and ends up a curiosity 0.8% worse than Huffman". It also stated the question that would decide the matter: whether the decode kernel is memory-bound or compute-bound, since in a memory-bound kernel the entropy code cannot matter.

Later predictions, each written before the measurement that tested it, are quoted in §5 where they apply.

## 4. Experimental setup

A complete description, including everything not repeated here, is in `METHODOLOGY.md` in the repository. This section gives what is needed to read the results.

### 4.1 Data

Qwen/Qwen3-0.6B from Hugging Face, `model.safetensors` (SHA-256 `f47f7117…6874b`), all tensors BF16. The file stores `lm_head.weight` and `model.embed_tokens.weight` as separate, bit-identical tensors (tied embeddings, stored twice); the extractor drops exact-duplicate tensors, leaving **596,049,920 unique weights**. All whole-model figures use this set. Weights are read as raw `uint16` bit patterns and never converted to float.

Two samples are used besides the whole model: a random sample of 2M weights, with replacement, for the CPU bit-exact roundtrip of each codec (the code tables always come from the whole-model histogram; a prefix code is memoryless per symbol, so a sample exercises the same code paths, including the escape); and the **first 64M weights** of the file for every GPU measurement. The latter lie inside the embedding matrix, whose exponent distribution is slightly more concentrated than the model's (entropy 2.552 vs 2.645 bits; Huffman rate 2.584 vs 2.678 bits/symbol with the whole-model table). Compression figures reported from GPU runs are therefore sample figures, and the whole-model projection is given alongside; timing conclusions do not depend on the distribution.

### 4.2 Hardware

| | GTX 1050 Ti | A100-SXM4-80GB | RTX 4090 |
|---|---|---|---|
| architecture | Pascal, SM 6.1 | Ampere, SM 8.0 | Ada, SM 8.9 |
| SMs / L2 | 6 / 1 MiB | 108 / 40 MiB | 128 / 72 MiB |
| L2 per resident thread | 85 B | 190 B | 384 B |
| peak bandwidth | 112 GB/s | 2039 GB/s | 1008 GB/s |
| SM clock during runs | 139–1923 MHz, unlocked (Windows, WDDM) | 1140 MHz | 2520 MHz |
| where | author's PC | rented container | rented container |

The design was developed and the attribution experiments were first run on the 1050 Ti; the two rented cards ran the identical code and sample unattended, after the predictions of §5.7 were recorded. Nsight Compute does not support Pascal and its counters were blocked by the host in both containers, so occupancy, stall reasons and divergence are unmeasured on every card. Software: Python 3.12, NumPy 2.5.3, CuPy 14.2.0 with CUDA 12.9; kernels are CUDA C compiled at run time through NVRTC.

### 4.3 Protocol

**Correctness before timing.** The vectorised encoder is verified byte-for-byte against the reference bit writer, including block offsets. Every kernel is verified bit-exact against the source exponent array — 32M symbols, all 20 combinations of `BLOCK` ∈ {64, …, 1024} × threads — before any timing, and the setup scripts refuse to proceed otherwise; the benchmark harnesses additionally check every configuration they time and print the result beside it. Every row reported here verified.

**Timing.** CUDA events around a single kernel launch; 0.6 s of sustained warm-up per configuration; 11 repetitions, median reported, interquartile range published beside it; configuration order shuffled with a fixed seed so that clock drift does not correlate with configuration; SM clock sampled around each measurement. On the unlocked 1050 Ti the IQR was 0–3 ms with the desktop idle; on the rented cards it was 0 on every row. A full re-run of every stage on the 1050 Ti reproduced every ratio and every conclusion with absolute times moving 1–6%.

**The memory-floor kernel.** A kernel that reads exactly the words each decoder thread would read and writes the same output bytes, but does no decoding. It is an empirical ceiling for *that access pattern*, not for the hardware.

**Attribution.** When two optimisations are candidates, all combinations are compiled and measured separately. When an optimisation moves the bottleneck, earlier comparisons are re-run, not extrapolated.

**Compression accounting.** Sizes are computed exactly from symbol counts: per weight, `8 + avg_exponent_bits + 8 · index_bytes_per_block / BLOCK` bits, plus a code table of 10 B (ladder) or 256 B (Huffman). The fixed file header is excluded.

## 5. Results

### 5.1 Rate

| whole model, canonical code, BLOCK=256, uint32 index | bits/exponent | reduction |
|---|---|---|
| exponent entropy | 2.645 | — |
| canonical Huffman | 2.678 | **32.48%** |
| ladder (1,1,1,2) | 2.815 | 31.62% (−0.86 pts) |
| ladder (1,1,2,2) | 2.847 | 31.42% |
| ladder (1,2,2,3) | 3.063 | 30.07% |

The (1,1,1,2) shape, chosen on synthetic Gaussian weights, remains the best of the three on the real model, and the gap to Huffman is within the pre-registered budget of one point. The rate half of the hypothesis holds. Both codecs roundtrip bit-exactly on CPU.

### 5.2 GPU decode: is it memory-bound?

Two kernels with an identical interface — same coarse index, same one-thread-per-block mapping, same 32-bit window reader, same output layout — differing only in the symbol decoder, plus the memory-floor kernel. GTX 1050 Ti, 64M symbols, base kernel, 128 threads per CUDA block:

| BLOCK | Huffman | ladder | floor | ladder/Huffman |
|---|---|---|---|---|
| 64 | **9.70 ms** | 9.75 | 9.57 | 1.006 |
| 128 | 52.11 | 34.96 | 12.77 | 0.671 |
| 256 | 84.86 | 58.22 | 13.35 | 0.686 |
| 512 | 121.08 | 91.19 | 16.42 | 0.753 |
| 1024 | 157.16 | 122.29 | 20.36 | 0.778 |

At `BLOCK=64`, the only operating point one would choose, both decoders sit within 2% of the memory floor and within 1% of each other: the entropy code is irrelevant, and the ladder is marginally slower because its stream is 4.8% larger. At `BLOCK≥128` the ladder wins by up to 33% — the compute-bound regime the hypothesis anticipated — but those configurations are 3–16x slower in absolute terms.

Three caveats bound this. First, "memory-bound" here means bound by a bad access pattern: the best configuration reaches 9 GB/s, 8% of the card's peak. Second, output dominates: of the 88.7 MB moved at `BLOCK=64`, 64 MB (71%) is the one-byte-per-symbol output, so the entropy code can influence at most 23% of the traffic. Third, the collapse between `BLOCK=128` and `256` coincides with the resident working set (520 KiB → 1040 KiB) crossing the card's 1 MiB L2. This was recorded as consistent, not proven, and became the prediction tested in §5.7.

### 5.3 Attribution: input versus output

Each thread walks its own bit region, so a warp reads 32 scattered streams; and each thread writes `BLOCK` consecutive bytes, so a warp's 32 stores land in 32 distinct sectors. Both are fixable in the same way, because the regions of the threads in one CUDA block are contiguous: stage through shared memory and load or store coalesced. Four variants were compiled so the gain could be attributed rather than guessed. `BLOCK=64`, 128 threads:

| variant | 1050 Ti | A100 | 4090 |
|---|---|---|---|
| base | 1.00 | 1.00 | 1.00 |
| input staged | 1.01x | 1.00x | 1.02x |
| output staged | **1.92x** | 2.61x | 4.55x |
| both staged | 1.48x | **2.91x** | **5.49x** |

Coalescing the output nearly doubles throughput on Pascal and is worth 4.6x on Ada; coalescing the input alone is worth nothing at this block size on any card. Combining the two is *worse* than output alone on Pascal — the input staging spends shared memory and a barrier and buys nothing, and pays in occupancy — but is the best variant on Ampere and Ada. The Pascal lesson "do not combine" did not transfer; it is recorded as architecture-specific.

At `BLOCK=256` on Pascal, the picture inverts: input staging is worth 4.4x, because the input working set no longer fits in L2. That crossover, at the same block size as the collapse in §5.2, was the second line of evidence for the L2 explanation.

With the optimised kernel the Huffman–ladder comparison was redone rather than extrapolated. On the 1050 Ti the best configuration is Huffman, `BLOCK=64`, 128 threads, output staged: **4.80 ms**, 2.02x faster than §5.2's best, with the ladder at 5.07 ms (1.055x). The ladder now loses on both axes: this is the regime where compute weighs more, and one shared-memory lookup still beats a dependent `clz`/branch/shift/mask chain.

### 5.4 The index

Block length has little spread: at `BLOCK=64`, from 130 to 273 bits on this data. So instead of one absolute uint32 offset per block (4 B/block), store one uint32 per superblock of 32 blocks — one warp — and one uint8 per block holding its length minus a global minimum, and recover each block's offset with an exclusive prefix-sum within the warp (five `__shfl_up_sync` steps). The index costs 1.125 B/block. The builder raises an error rather than truncate if the spread exceeds 255, which happens at `BLOCK≥256` on this data.

| `BLOCK=64`, 128 threads, output staged | 1050 Ti | A100 | 4090 | compression (sample) |
|---|---|---|---|---|
| Huffman, uint32 index | 4.81 ms | 0.47 | 0.19 | 30.73% |
| Huffman, 8-bit index + prefix-sum | 4.82 ms | 0.47 | 0.19 | **32.97%** |

The prefix-sum is invisible next to a chain of 64 dependent loads: 1.00x, 1.006x and 0.996x the time on the three cards, for **+2.25 points**. Index traffic falls from 4 MB to 1.1 MB. This also dissolves the tension that motivated large block sizes: the whole-model rate at the fast configuration rises from 30.14% to 32.39%, within 0.68 points of the best any block size achieves.

Against the Phase 2 starting point on the same card (9.70 ms, 30.73%), the reference decoder is now 2.01x faster and 2.25 points smaller from two changes that touch neither the entropy code nor the decode loop.

### 5.5 Is there correlation to exploit between exponents?

Mutual information between adjacent exponents, in six 32M-symbol windows spread across the file: maximum **0.0100 bits**, typically 0.0002. An order-1 context model gains **0.0000 bits/symbol**. Huffman over pairs of exponents captures half of the code's own redundancy (0.016 bits/symbol, 0.1% of the file) at the cost of an 8 KiB table; the ladder over pairs is *worse* than the scalar ladder even with a 25x larger table, because spreading the mass over 421 observed pairs flattens the distribution that power-of-two rungs rely on. Neighbouring exponents are, for practical purposes, independent.

### 5.6 What the field split gives away

Joint entropy is subadditive, `H(sign, exp, mant) ≤ H(sign) + H(exp) + H(mant)`, so a coder over the whole 16-bit symbol — as in [2] — can never do worse on rate than the field split, and does better by exactly the mutual information between the fields. Measured exactly on all 596M unique weights (the full alphabet has 65,536 symbols, of which 7,506 occur):

| | bits/weight |
|---|---|
| H(sign), H(exp), H(mant) | 1.000, 2.645, 6.973 |
| sum of marginals | 10.618 |
| **H(sign, exp, mant)** | **10.578** |
| I(exp; mant) | **0.040** |
| I(sign; exp), I(sign; mant) | 0.0001, 0.0000 |

The dependence between exponent and mantissa is concentrated: conditional mantissa entropy is 7.00 bits for every exponent except the two largest binades, 122 and 123 (mass 21% and 3.5%), where it falls to 6.84 and 6.26 bits — the Gaussian tail decays *within* the top binade, so small mantissas are more likely there. Achieved rates with canonical Huffman and no index:

| | bits/weight |
|---|---|
| true floor H(sign, exp, mant) | 10.578 |
| + mutual information between fields | 0.040 |
| + sign and mantissa stored raw, above their entropy | 0.027 |
| + Huffman redundancy on the exponent alphabet | 0.033 |
| = field split, achieved | **10.678** |
| full 16-bit alphabet, achieved (floor + 0.024 redundancy) | **10.602** |

The field split gives away **0.076 bits/weight, 0.47 points**, on this model. Canonical Huffman on the exponent is 0.033 bits from its own floor; the design as a whole is 0.100 bits from the true floor of the symbol. That is the honest ceiling on any rate claim for exponent-only coding, and it bounds the rate advantage of a full-alphabet coder before any credit for a better entropy coder than Huffman.

### 5.7 The L2 prediction on two more cards

Written before renting the cards: "on a card with a large L2 the BLOCK cliff should disappear; staging the input in shared should stop mattering at BLOCK=256; BLOCK=512–1024 should become viable. If the cliff is still there, the L2 explanation is false." Base kernel, Huffman, 128 threads:

| BLOCK | 1050 Ti (85 B/thread) | A100 (190 B) | 4090 (384 B) |
|---|---|---|---|
| 64 | 9.70 ms | 1.26 ms | 1.24 ms |
| 256 | 84.86 | 3.02 | 2.02 |
| 1024 | 157.16 | 2.46 | 2.13 |
| **1024 / 64** | **16.2x** | **1.96x** | **1.71x** |

**The cliff prediction held.** What remains is not L2: the floor kernel rises the same way, and at `BLOCK=1024` there are only 62,500 threads for 221,184 (A100) or 196,608 (4090) resident slots — under-occupancy that would affect any one-thread-per-block decoder.

**The input-staging prediction failed, informatively.** Its gain at `BLOCK=256` is 4.4x, 2.0x and 1.2x on the three cards. It does not switch off when the working set fits in L2; it decays smoothly with L2 per resident thread, because 32 scattered streams per warp are served more slowly from L2 than one contiguous load. The L2-per-thread figure predicts the size of the effect, not its presence, and the earlier explanation is corrected accordingly.

**"Large BLOCK becomes viable" is moot.** With the optimised kernel, `BLOCK=1024` is 3.5x (A100) and 6x (4090) slower than `BLOCK=64`, because the serial chain is 16x longer and nothing hides it; and §5.4 removed the reason to want it.

**Beyond the prediction.** The best Huffman time for 64M symbols is 0.44 ms on the A100 and 0.20 ms on the 4090: the 4090 is 2.2x faster with half the bandwidth, and its SM clock is 2.2x higher (2520 vs 1140 MHz). The base kernel does not show this (1.26 vs 1.24 ms: latency-bound on the scattered pattern, clock-insensitive). Achieved bandwidth at the best configuration is 16%, 10% and 45% of peak on the three cards. Two cards do not prove it, but the reading most consistent with the data is that once the access pattern is fixed the decoder is bound by the latency of its 64-step chain of dependent loads, which runs at SM clock. Finally, the ladder-to-Huffman time ratio at the operating point is 1.05, ~1.0 and 1.3 on the three cards; on no card does the ladder win where it would be run.

## 6. Discussion

**What the project learned.** The two structural changes were worth 2x in speed and 2.25 points; the entropy-code question the project was built around was worth 0.86 points, in the wrong direction. Entropy coding is finished *within the field split* — 0.033 bits from its floor, no neighbour correlation to exploit — and the split itself is now measured to cost 0.47 points, almost all of it in two binades. Everything else is structural: the access pattern, the index, and the serial chain.

**Relation to GEMM-fused decompression [2].** The 11x throughput gain reported in [2] over DFloat11 is attributed by its authors to fusion: eliminating global-memory materialisation of the decompressed layer and overlapping decompression with tensor-core work, with tile alignment worth 3.3–8.2x and double-buffering on top. It is not attributed to rANS over Huffman, and on this data the Shannon-gap argument for rANS is weak: Huffman is at 98.8% efficiency on the exponent alphabet. Our measurements corroborate the structural reading from the other side. The standalone decoder is at 10–45% of peak bandwidth after every fix we found, its time follows SM clock, and its output — one byte per weight, then a second pass to merge sign and mantissa — is the majority of its traffic. Those are the costs fusion removes. The implication for this design is not to change the entropy code but to keep canonical Huffman and fuse; the interleaved-stream layout of [2], which obtains coalescing in the format rather than in the kernel, is the one idea from that paper that maps directly onto §5.3. The 8-bit index with a warp prefix-sum applies unchanged to DFloat11's gaps array and to [2]'s tile offset table.

**Why production quantisation formats do not use entropy codes.** GGUF, GPTQ, AWQ and NF4 are fixed-width. The usual reason given, "Huffman is slow", is imprecise; the real one is that variable-length codes destroy random access, which forces a decompress-to-memory round trip instead of in-register dequantisation inside the GEMM. Losslessness closes off the fixed-rate corner: fixed symbols per block requires an index (what is done here); fixed bits per block loses track of which weights are where; fixed in both is a fixed-rate code, i.e. no entropy coding, which is what lossy formats do. The index is therefore not avoidable; making the decoder never touch DRAM is the reachable goal.

**Threats to validity.** One model, and a small one; rates on other models will differ. One card of each architecture, in rented containers whose clocks could not be locked (though IQRs were zero). The GPU timing sample is the embedding matrix, which compresses slightly better than the model average; both figures are given. No comparison against DFloat11's shipped kernels — every "Huffman" number is this repository's implementation of the same design. No end-to-end tokens-per-second measurement; kernel time is not inference time. No profiler data on any card. One unexplained anomaly remains on Pascal: at `BLOCK=128` with input staged, Huffman is 2–2.4x faster than the ladder, far more than the stream-size difference justifies; it is off the optimal path and reproduces, and it is recorded rather than explained. The full list, with the reasons each item could not be closed, is in `METHODOLOGY.md`.

## 7. Conclusion

A table-free prefix code does not decode BF16 exponents faster than canonical Huffman on a GPU, on any of three architectures, and costs 0.86 points of compression; the question that motivated the work has a clear negative answer. The decoder's time was never in the entropy code. It was in a memory access pattern that scattered each warp's loads and stores, in an index four times larger than it needed to be, and — once those are fixed — in the serial chain of dependent loads inside each block, which runs at SM clock and which only fusion with the consumer of the weights can hide. Canonical Huffman on the exponent field is 0.033 bits from its floor and the field split is 0.076 bits from the floor of the whole symbol; there is no rate left to find inside this design, and the next step, if one is taken, is structural.

## Reproducibility and authorship

The repository contains the code, the setup scripts that download the model and refuse to proceed unless the kernels verify bit-exact, the raw logs and environment records for all four machines under `results/`, and `METHODOLOGY.md`, which documents data provenance, the sample used by each experiment, protocols and threats to validity. Every table in this report is transcribed from a committed log. The two rented GPU runs cost about 1.7 USD in total and can be repeated by following `RENT_A_GPU.md`.

Claude Code (Anthropic) was used extensively in this project — to implement the codecs, kernels and harnesses, to run and interpret the CPU analysis, to review the literature, to drive the rented GPU runs, and to draft and revise the documentation and this report — under the author's direction. It was also used to audit the repository against its own evidence before publication; that audit found and corrected a duplicated tensor in the whole-model figures and a literature claim that a paper summary asserted and the paper does not. Every commit carries a `Co-Authored-By` trailer and a session link. The hypothesis, the decisions about what to measure and what to conclude, and any remaining errors are the author's.

## References

[1] DFloat11: Lossless compression of LLM weights by Huffman coding the BF16 exponent. NeurIPS 2025. arXiv:2504.11651. https://github.com/LeanModels/DFloat11

[2] Approaching Shannon Bound with Lossless LLM Weight Compression. arXiv:2606.15789, June 2026.

[3] Float8@2bits / EntQuant. arXiv:2601.22787, January 2026.

[4] EntroLLM. arXiv:2505.02380.

[5] S. Han, H. Mao, W. J. Dally. Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding. ICLR 2016. arXiv:1510.00149.

[6] Recoil: Parallel rANS Decoding with Decoder-Adaptive Scalability. arXiv:2306.12141.

[7] D. A. Huffman. A Method for the Construction of Minimum-Redundancy Codes. Proceedings of the IRE, 1952.

[8] E. S. Schwartz, B. Kallick. Generating a canonical prefix encoding. Communications of the ACM, 1964.

[9] Qwen Team. Qwen3-0.6B. https://huggingface.co/Qwen/Qwen3-0.6B

*Author lists and exact titles for [1]–[4] and [6] are to be verified against the arXiv records before submission; they are reproduced here as cited in the repository's literature review.*
