# Methodology

How every number in this repository was produced, so that it can be checked,
repeated or refuted. If a claim in [RESULTS.md](RESULTS.md) or
[HANDOFF.md](HANDOFF.md) is not covered by something on this page, treat the
claim as unsupported.

---

## 1. The question, and the rule for answering it

**Hypothesis under test (recorded before any measurement, commit `098cf3b`):**
a "ladder" prefix code over the BF16 exponent, with a 10-byte table, decodes
faster on GPU than canonical Huffman with a 4 KiB LUT, at a cost of <1 point of
compression.

**Rule:** every prediction is written down *before* the measurement that tests
it, in the document that describes the measurement. The prediction is then
left in place whether it survived or not. Examples: HANDOFF section 4.1, the
"Prediction, written before measuring" blocks in the operator guides, the
"Risks" section of the original handoff (risk #1 is what happened).

The hypothesis was refuted. What was found instead — that the memory access
pattern and the block index dominate the decoder and the entropy code does
not — was not hypothesised in advance; it emerged from the measurements and
is presented as such.

## 2. Data

| | |
|---|---|
| Model | `Qwen/Qwen3-0.6B`, from Hugging Face |
| File | `model.safetensors`, 1,503,300,328 bytes |
| SHA-256 | `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b` |
| `config.json` | `torch_dtype: bfloat16`, `tie_word_embeddings: true`, 28 layers, hidden 1024, vocab 151,936 |
| Weights used | every tensor with dtype `BF16` whose content is not a byte-identical copy of one already written: **596,049,920 values** (the file holds 751,632,384; see below). Non-BF16 tensors, if any, are skipped and reported by the extractor. |
| Extraction | `extract_real_weights.py`: parses the safetensors header directly (no torch), streams tensor by tensor, writes raw `uint16` bit patterns to `outputs/real_weights_bf16.bin` in file order, and accumulates the exact exponent histogram into `outputs/real_exp_counts.npy` (SHA-256 `ff2d6ab40744dc486d0b8d5c0ef17b3e53b6f1a35446b11c8be72d4677ea11e1`). |

The exponent is bits 7–14 of each `uint16` (`(w >> 7) & 0xFF`); sign is bit 15
and mantissa bits 0–6. Nothing is converted to float at any point.

**The synthetic file** (`gen_weights_bf16.py`, seed 1234, 2,162,688 weights:
seven Gaussian tensors with scales 0.009–0.030 plus 0.3% outliers at 8σ) was
used only in Phase 1 to check that the design transfers from synthetic to real
data. No conclusion rests on it.

### Which subset of the weights each experiment used

This matters, and it is the first thing a reader should know:

| Experiment | Symbols | Which ones | Why |
|---|---|---|---|
| Entropy, code tables, compression ratios (Phase 1) | 596,049,920 | all unique | Exact histogram; sizes computed analytically from it |
| CPU bit-exact roundtrip (Phase 1) | 2,000,000 | random with replacement, over the whole file | Pure-Python bitstream of 596M symbols is infeasible on the dev machine; a prefix code is memoryless per symbol, so a sample with the full-distribution table tests the same code paths (the escape branch was hit 10,165 times) |
| GPU kernel verification (`verify_kernels.py`) | 32,000,000 | the **first** 32M | Contiguous chunk is what the kernels consume |
| GPU timing (all `bench_*.py`) | 64,000,000 | the **first** 64M | Same |
| Mutual information (Phase 2c) | 6 windows | spread across the file at fixed offsets | Explicitly to avoid the bias below |

**A duplicated tensor, and what was done about it.** `config.json` says
`tie_word_embeddings: true`, but the safetensors file stores both
`lm_head.weight` and `model.embed_tokens.weight` (151,936 x 1024 = 155,582,464
values each), and they are **bit-identical** (SHA-256 of the raw bytes, and
`np.array_equal` over the extracted file). The first version of the extractor
kept every BF16 tensor, so all Phase 1 figures up to 2026-09-13 were over
751,632,384 values with the embedding matrix counted twice. The extractor now
drops any tensor whose bytes match one already written and reports it; Phase
1 was re-run on the 596,049,920 unique weights and every whole-model figure in
the repository was updated. Size of the correction:

| | with duplicate (751.6M) | unique weights (596.0M) |
|---|---|---|
| exponent entropy | 2.634 bits | **2.645 bits** |
| canonical Huffman | 2.665 bits/sym, 32.56% | **2.678 bits/sym, 32.48%** |
| ladder (1,1,1,2) gap | 0.85 points | **0.86 points** |
| best config, whole model | 32.46% | **32.39%** |

The GPU phases did not need re-running: the timing sample is the same bytes
in both layouts (see next paragraph), the ladder tables are identical, and
the Huffman table changes only for seven symbols with code lengths 16–30
bits, altering the 64M-sample stream by 843 bits out of 165,364,797. The GPU
runs on record used the pre-correction table; the sample compression figures
they report are unchanged to two decimals.

**Where the timing sample comes from.** The first 64M weights of the
extracted file are the start of the embedding matrix (`lm_head.weight`, the
first tensor by data offset, whose content is the embedding; after the fix
it is the only copy). Their exponent distribution is slightly more
concentrated than the model average:

| | first 64M (timing sample) | whole model |
|---|---|---|
| exponent entropy | 2.552 bits | 2.645 bits |
| canonical Huffman, full-model table | 2.584 bits/sym | 2.678 bits/sym |

Consequences: (a) any **compression percentage measured in a `bench_*` run**
(e.g. 32.97%) is a sample figure; the whole-model projection with the same
configuration is lower (32.39% for the best configuration, computed from the
full histogram). Both numbers are reported where they appear. (b) The
**timing** conclusions do not depend on the distribution: the access-pattern
and index effects are structural, and the two bitstreams being compared differ
by <1 MB either way. (c) All Huffman and ladder **tables** are always built
from the full histogram, never from the sample.

## 3. Hardware and software

### Where things ran

| Machine | Role | GPU | Notes |
|---|---|---|---|
| Dev machine (Linux) | Phase 1, all CPU analysis, all writing | none | 3.5 GiB RAM + 3.5 GiB swap. This is why extraction streams and why the CPU roundtrip is sampled. |
| Measurement machine (Windows, WDDM driver model) | Phases 2, 2b, 2c and their replication | NVIDIA GeForce GTX 1050 Ti: Pascal, SM 6.1, 6 SMs, 4 GB GDDR5, 1 MiB L2, theoretical peak 112.1 GB/s (computed from `memoryClockRate` and `memoryBusWidth`) | Also drives the desktop. Clocks cannot be locked under WDDM. |
| RunPod Secure Cloud container (Linux, Ubuntu 24.04 image `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`) | Phase 2e | NVIDIA A100-SXM4-80GB: Ampere, SM 8.0, 108 SMs, 40 MiB L2, peak 2039 GB/s | Rented 2026-09-13, ~46 min. Clock locking refused by the host; SM clock read 1140 MHz. |
| RunPod Secure Cloud container (same image) | Phase 2e | NVIDIA GeForce RTX 4090: Ada, SM 8.9, 128 SMs, 72 MiB L2, peak 1008 GB/s | Rented 2026-09-13, ~40 min. Clock locking refused; SM clock read 2520 MHz. |

The rented runs follow `RENT_A_GPU.md` exactly, with the repository uploaded
as a `git archive` of the commit in use instead of cloned (the repository
was private at the time). Each pod ran `setup_cloud.sh` (model download,
extraction with the duplicate dropped, bit-exact verification) and
`run_all.sh` unattended; both verified `RESULT: both kernels correct` before
timing. The operator guides for a borrowed RTX 4070 describe a run that has
not happened.

### Software versions

`env_info.py` records them without anything that identifies the machine or
the account; `run_all` calls it first and stores `env_info.json` next to the
logs.

| Component | Dev machine | GTX 1050 Ti | A100 pod | RTX 4090 pod |
|---|---|---|---|---|
| OS | Linux 6.8.0, glibc 2.39 | Windows 10, build 19045 | Linux 6.8.0-138 (host), Ubuntu 24.04 image | Linux 6.8.0-85 (host), same image |
| Python | 3.12.3 | 3.12.10 | 3.12.3 | 3.12.3 |
| NumPy | 2.5.3 | 2.5.3 | 2.5.3 | 2.5.3 |
| CuPy | — | 14.2.0 (`cupy-cuda12x[ctk]`) | 14.2.0 | 14.2.0 |
| CUDA runtime / driver API | — | 12.9 / 12.6 | 12.9 / 13.2 | 12.9 / 12.8 |
| NVIDIA driver | — | 560.94 | 595.91.07 | 570.195.03 |

Each column is the `env_info.json` in the corresponding `results/` directory.

The interactive runs from which the RESULTS.md tables were transcribed predate
`env_info.py`; the column above was recorded on 2026-09-13 on the same
machine and environment, during the replication run described in section 10.
No package was updated between the two.

Kernels are CUDA C compiled at runtime through NVRTC via `cupy.RawKernel`. No
compiler flags beyond CuPy's defaults.

## 4. Correctness protocol

Every kernel is verified **bit-exact against the original exponent array
before any timing is taken**, never after. Concretely:

1. **Encoder equivalence.** `bitpack.py` (vectorised numpy) produces a
   bitstream verified **byte-for-byte identical** to the reference Python
   `BitWriter` in `ladder_codec.py` / `df11_reference.py`, including the block
   offsets. So the GPU decoders are tested against the same stream the
   reference codecs would produce.
2. **Kraft inequality** is checked for every code table (`ladder_codec.py`).
   A code with Kraft > 1 is undecodable; this was learned the hard way before
   the repository existed.
3. **GPU roundtrip.** `verify_kernels.py BLOCK N` decodes N symbols on the GPU
   and compares with `np.array_equal` against the source array. On failure it
   prints the first mismatching positions. The kernel variants added in Phases
   2b and 2c are verified the same way inside their own harnesses (the `ok`
   column in `bench_idx8.py`, the checks in `bench_opt.py`), and all 20
   BLOCK x threads combinations were re-verified after the barrier bug in
   Phase 2b.
4. **Setup scripts abort on failure.** `setup_*.{sh,ps1}` run
   `verify_kernels.py 256 32000000` and exit non-zero if either kernel is
   wrong, so a remote operator cannot produce timings from a broken decoder.

## 5. Timing protocol

Defined in `bench_gpu.py::measure` and reused by every later harness:

- Timing is by CUDA events (`cupy.cuda.Event`) around a single kernel launch;
  no host-side work is inside the interval.
- **Warm-up:** the kernel is launched repeatedly for 0.6 s with a
  synchronisation after each launch, so the GPU is at boost clock and no
  launches are queued when timing starts.
- **Repetitions:** 11. Reported value is the **median**; the IQR is reported
  next to it and is the noise estimate.
- **Order:** the list of (BLOCK, threads) configurations is shuffled with a
  fixed seed (`random.Random(0)`) so that clock drift over the run does not
  correlate with configuration.
- **Clock:** `nvidia-smi --query-gpu=clocks.sm` is sampled before and after
  each configuration and stored alongside the time. On the 1050 Ti it ranges
  139–1923 MHz; the rented A100 and 4090 sat at 1140 and 2520 MHz throughout
  (locking was refused by the host, but the IQR was 0 on every row). A "SM cycles per symbol" figure (ms x MHz x SMs / symbols) is
  computed as a clock-invariant view.
- **Environment:** everything else using the GPU is closed. On the
  measurement machine this took the IQR from ~5 ms to 0–3 ms. Under
  Linux, `run_all.sh` additionally tries `nvidia-smi -lgc` to lock clocks;
  this was not available on the measurement machine.
- **Throughput (GB/s)** is `bytes moved / time`, where bytes moved = compressed
  stream + output bytes (one per symbol) + index bytes. It is a
  *useful-traffic* figure, not a hardware-counter figure.

### The memory-floor kernel

`mem_floor` (in `bench_gpu.py`) reads exactly the same 32-bit words each
decoder thread would read and writes the same output bytes, but performs no
decoding. It is an empirical ceiling **for that access pattern**, not for the
hardware: Phase 2b showed the pattern itself was the bottleneck. Note that
`mem_floor` is fed the **ladder** bitstream; the Huffman stream is ~4.8%
smaller, so the floor is very slightly pessimistic for Huffman.

### Attribution discipline

- When two optimisations are candidates, **all combinations are compiled and
  measured separately** (Phase 2b: 4 variants of input x output staging). The
  result that "both" is worse than "output alone" would have been invisible
  otherwise.
- When an optimisation moves the bottleneck, **earlier comparisons are re-run,
  not extrapolated** (Phase 2b redid the Huffman-vs-ladder comparison with the
  optimised kernel; Phase 2b also re-ran the Phase 2 sweep after the barrier
  fix and reported that the numbers did not move).
- Nsight Compute could not be used: it dropped Pascal support in 2020.1.
  Occupancy, stall reasons and divergence are therefore **unmeasured**, and
  the docs say so wherever it matters.

## 6. Compression accounting

All sizes are computed **analytically** from exact symbol counts:

```
exponent bits   = Σ_s count[s] · length[s]
index bytes     = n_blocks · bytes_per_block        (4 for uint32; 1.125 for uint8 + superblock)
total bytes     = n_weights · 1 (sign+mantissa) + exponent_bits/8 + index bytes + table
reduction (%)   = 100 · (1 − total / (2 · n_weights))
```

Equivalently, per weight: `bits/weight = 8 + avg_exp_bits + 8·bytes_per_block/BLOCK`.
The best configuration on the whole model is `8 + 2.678 + 0.141 = 10.818`
bits/weight, i.e. 32.39%.

Included: exponent stream, sign+mantissa bytes, block index, code table (10 B
ladder, 256 B Huffman lengths). Excluded: the fixed header (magic, version,
counts), which is a few dozen bytes on a 1.5 GB file.

For a code, `avg bits/symbol` is exact (it is a weighted sum); it is not
estimated from a sample.

## 7. Information-theoretic measurements (Phase 2c)

- Exponent entropy `H(X)` from the exact histogram.
- Mutual information between adjacent exponents `I(X;Y) = H(X) + H(Y) − H(X,Y)`
  from the joint histogram of consecutive pairs, in six windows at offsets
  0, 93,954,048, 187,908,096, 375,816,192, 563,724,288 and 711,632,384 —
  offsets into the file **as extracted before the duplicate `lm_head` was
  dropped**; the windows are real weight data either way, but to reproduce
  them exactly, extract with the duplicate kept.
- Order-1 context model: `H(Y | X)` from the same joint histogram.
- "Huffman over pairs" rate: canonical Huffman built on the pair alphabet
  (421 pairs observed in the sample), rate computed analytically.
- **Cross-field mutual information (Phase 2d):** one exact histogram of the
  full 16-bit value over all 596,049,920 unique weights (65,536 bins); every
  marginal and pair histogram (sign, exp, mant, exp×mant, sign×exp,
  sign×mant) is derived from it by re-binning, so all entropies and mutual
  informations are exact. Canonical Huffman rates on the exponent alphabet
  and on the full 16-bit alphabet are computed analytically from the same
  counts.

Scripts: `analysis_vector.py`, `analysis_index.py`, `analysis_fields.py`.

## 8. Threats to validity

Stated here in one place. Each is also flagged where the affected number
appears.

1. **Three GPUs, three architectures, but one sample of each.** Phases 2–2c
   and the design decisions were made on a single Pascal card; Phase 2e
   repeats every kernel measurement on one A100 and one RTX 4090, both in
   rented containers. The L2 explanation is now supported as a trend across
   three points and corrected as a threshold (RESULTS Phase 2e). The
   "decoder time follows SM clock" reading rests on two cards and is stated
   as consistent-with, not shown.
2. **Clocks not locked.** Mitigated as described in section 5; the IQR is
   published next to every median. The full replication run (section 10)
   puts the run-to-run spread of absolute times at 1–6%; ratios between
   variants measured in the same run are stable to about 1%.
3. **The timing sample is the embedding matrix.** See section 2.
4. **The GPU runs used a Huffman table built with the embedding counted
   twice.** Shown in section 2 to be an 843-bit difference on the sample.
5. **One model, and a small one (0.6B).** Rates on other models will differ;
   nothing here claims otherwise.
6. **No comparison against DFloat11's shipped kernels.** Every "vs Huffman"
   number is against *this repository's* Huffman implementation, which follows
   the same design (hierarchical LUT, per-thread bit offsets) but is not their
   code.
7. **No end-to-end inference measurement.** Kernel time is not tokens/s.
8. **No profiler data.** See section 5.
9. **Software versions on the measurement machine not recorded.** See
   section 3.
10. **Analytic sizes.** They are exact for the counted symbols, but a real file
   would carry a header and alignment padding not included here.
11. **One unexplained anomaly** (HANDOFF section 1, "Not measured"): Huffman
    2x faster than the ladder at BLOCK=128 with input staged in shared. It is
    off the optimal path and has not been explained.

## 9. How to reproduce

```bash
bash setup_cloud.sh   # or setup_linux.sh; setup_windows.ps1 on Windows
bash run_all.sh       # ~20 minutes on the reference hardware
```

`setup_*` downloads the model, extracts the weights, builds the histogram and
**refuses to continue unless both kernels verify bit-exactly**. `run_all`
executes, in order: `bench_gpu.py` (Phase 2 sweep + floor), `bench_opt.py`
(Phase 2b attribution), `bench_head2head.py` (Phase 2b comparison),
`bench_idx8.py` (Phase 2c index), and `profile_ncu.sh` if `ncu` is present. It
leaves `resultados.tar.gz` with one text log per stage plus
`outputs/gpu_bench.json`.

CPU-only parts (`bench.py`, `analysis_*.py`) need only the extracted weights
and run on any machine with ~2 GB free RAM.

What to compare against: the tables in RESULTS.md, one per script, with the
IQR column as the tolerance. A different GPU is *expected* to give different
absolute times; the claims that should transfer are the ratios (output
staging ≈ 1.9x, index ≈ free, ladder ≥ Huffman time) and the BLOCK cliff
prediction in HANDOFF 4.1.

Expected first check on any new card: `L2 per resident thread` printed by the
setup script should match the table in HANDOFF 4.1 for that card; if it does
not, the machine is not what it claims to be.

## 10. Raw data

Committed under `results/`, one directory per machine, so every table in
RESULTS.md can be checked against the log it was transcribed from:

- `results/cpu/` — Phase 1: `bench_results.json`, `bench_log.txt`, and the
  extractor's log, all from the deduplicated re-run; `analysis_fields.txt`
  is the Phase 2d log. `*.pre-dedup.*` are the
  same outputs from the original run with `lm_head` counted twice, kept so
  the pre-correction tables in git history can be audited too. Produced on
  the dev machine.
- `results/a100sxm480gb/`, `results/rtx4090/` — Phase 2e, same file set,
  produced unattended by `run_all.sh` in RunPod containers on 2026-09-13.
  These *are* the source of the Phase 2e tables.
- `results/gtx1050ti/` — Phases 2/2b/2c: `gpu_bench.json`, the four per-stage
  text logs, `gpu_info.txt` and `env_info.json`. **These are from a complete
  `run_all.ps1` execution on 2026-09-13**, not from the interactive runs the
  RESULTS.md tables were transcribed from (whose console output was not
  saved). They are therefore a same-hardware replication rather than the
  source of the tables. How they compare:

  | quantity | RESULTS.md | replication |
  |---|---|---|
  | Phase 2 base, BLOCK=64/128 thr, Huffman / ladder / floor | 9.70 / 9.73 / 9.57 ms | 9.70 / 9.75 / 9.57 ms |
  | Phase 2 base, BLOCK=1024/128 thr, Huffman / ladder | 153.77 / 119.27 ms | 157.16 / 122.29 ms |
  | 2b, output staged in shared, BLOCK=64/128 thr | 5.07 ms, 1.92x | 5.11 ms, 1.92x |
  | 2b, both staged (worse than output alone) | 6.59 ms | 6.64 ms |
  | 2b, input staged, BLOCK=256/64 thr | 13.05 ms, 4.38x | 13.16 ms, 4.46x |
  | 2b, head-to-head best, Huffman vs ladder | 4.80 vs 5.07 ms, 1.055 | 4.85 vs 5.11 ms, 1.054 |
  | 2b, unexplained anomaly, BLOCK=128/128 thr input staged | 12.83 vs 26.14 ms, 2.0x | 12.92 vs 31.04 ms, 2.4x |
  | 2c, Huffman uint32 vs uint8+prefix-sum, BLOCK=64 | 4.81 vs 4.82 ms | 5.11 vs 4.85 ms |
  | 2c, compression, every configuration | identical | identical |

  Every ratio and every conclusion reproduces. Absolute times move by 1–6%,
  which is the unlocked-clock noise floor on this card. The one pair to read
  with that in mind is the 2c index comparison: in the transcribed run the
  8-bit index cost nothing (4.82 vs 4.81); in the replication it is 5%
  faster (4.85 vs 5.11), with the uint32 reference landing 6% above the
  same kernel's time in the head-to-head stage of the same run. The claim
  supported by both runs is "not slower", not a precise delta.

  The logs predate the translation of the code's printed strings, so their
  column headers are Spanish: *suelo* = floor, *simbolos* = symbols,
  *variante* = variant, *indice* = index, *bloque* = block, *si* = yes,
  *escalera* = ladder, *excede* = exceeds. The numbers are what they are.

  The text logs begin with PowerShell `NativeCommandError` noise: CuPy prints
  a `CUDA_PATH` warning to stderr and PowerShell reports it as an error. It is
  harmless. The `<redacted>` tokens replace filesystem paths that the
  privacy scan removed.

`run_all.sh` / `run_all.ps1` write into `results/<gpu-name>/` directly, and
begin each run with `env_info.py`, which records the versions in section 3.
The logs contain GPU model, clocks and timings only; nothing that identifies
a machine, an account or a person. The one exception is Nsight Compute's CSV
output, whose process column carries the interpreter's path (which can
include a user name): `results/*/ncu_out/` is git-ignored and must be scrubbed
by hand before being added.

The 1.5 GB weight file and the model are not committed; they are regenerated
by the setup scripts from the Hugging Face source and can be checked against
the SHA-256 above.
