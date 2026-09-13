# Lossless BF16 weight compression

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22736348.svg)](https://doi.org/10.5281/zenodo.22736348)

Lossless compression of AI model weights by entropy-coding only the BF16
**exponent** field. Same family as [DFloat11](https://github.com/LeanModels/DFloat11)
(NeurIPS 2025). Sign and mantissa are stored raw because their entropy is
already close to maximal; all the headroom is in the exponent (2.65 bits of 8).

## Status in one line

The project's original hypothesis — a **ladder code** with a 10 B table
instead of Huffman's 4 KiB hierarchical LUT — **has been refuted**: it is
0.86 points worse on compression *and* 5% to 43% slower. But the work produced
a considerably better codec by another route.

## Best measured configuration

**Huffman, BLOCK=64, 128 threads, output staged in shared, 8-bit index**

| | Phase 2 (starting point) | now |
|---|---|---|
| time (64M symbols, GTX 1050 Ti) | 9.70 ms | **4.82 ms** |
| compression, on the 64M timing sample | 30.73% | **32.97%** |
| compression, projected to the whole model | 30.14% | **32.39%** |

2.01x faster and +2.25 points, from two changes that **do not touch the
entropy code**: coalescing the output write, and replacing the uint32 offset
index with 8-bit lengths plus a warp prefix-sum.

The design was found on a GTX 1050 Ti (Pascal, 1 MiB L2) and then measured
unchanged on a rented A100 and RTX 4090 (`results/`):

| best configuration, 64M symbols | GTX 1050 Ti | A100 | RTX 4090 |
|---|---|---|---|
| time | 4.8 ms | 0.44 ms | 0.20 ms |
| ladder / Huffman | 1.05 | ~1.0 | 1.3 |
| 8-bit index cost | none | none | none |

The Pascal "BLOCK cliff" that shaped Phase 2 is an L2 effect: 16x on the
1050 Ti, 2x on the A100, 1.7x on the 4090. The timing sample is the model's
embedding matrix, which compresses slightly better than the model average;
both figures are given above and the difference is explained in
[METHODOLOGY.md](METHODOLOGY.md).

## The documents

| File | What it contains |
|---|---|
| **[HANDOFF.md](HANDOFF.md)** | **Start here.** Current status, what is measured and what is not, and what to do next. |
| [RESULTS.md](RESULTS.md) | Chronological log: Phase 1 (CPU), Phase 2 (kernels), 2b (access pattern), 2c (vector coding and index), 2d (cross-field mutual information), 2e (A100 and RTX 4090). All the number tables. |
| [METHODOLOGY.md](METHODOLOGY.md) | How every number was produced: data and its provenance, which sample each experiment used, hardware, timing and correctness protocols, and the known threats to validity. Read this before citing anything. |
| [RENT_A_GPU.md](RENT_A_GPU.md) | How and where to rent a GPU by the hour for the missing measurements, with the prediction to check against. |
| [OPERATOR_WINDOWS.md](OPERATOR_WINDOWS.md) | Copy-and-paste guide for someone lending a Windows machine with an NVIDIA GPU (Turing or newer). |
| [OPERATOR_LINUX.md](OPERATOR_LINUX.md) | Same for Linux. |

## Reproduce from scratch

```bash
# Linux / rented GPU
bash setup_cloud.sh      # or setup_linux.sh on your own machine
bash run_all.sh
```

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File setup_windows.ps1
powershell -ExecutionPolicy Bypass -File run_all.ps1
```

`setup_*` installs dependencies, downloads Qwen3-0.6B, extracts the weights and
**verifies that the kernels decompress bit-exactly**. If that verification
fails the script exits with an error on purpose: timings from an incorrect
decompressor are worthless.

`run_all` records the environment, runs the five measurement stages into
`results/<gpu-name>/` and packages them as `results.tar.gz` / `results.zip`.

## Code

| File | What it is |
|---|---|
| `ladder_codec.py`, `df11_reference.py` | CPU reference codecs (ladder and canonical Huffman). |
| `bitpack.py` | Vectorized encoder; produces a bitstream **byte-for-byte identical** to the reference. |
| `extract_real_weights.py` | Extracts raw BF16 weights from a `.safetensors` without needing torch. |
| `gpu_kernels.py` | The two baseline kernels. |
| `kernel_opt.py` | Variants that stage input/output in shared memory. |
| `kernel_idx8.py` | 8-bit index + warp prefix-sum. |
| `bench_*.py` | Measurement harnesses. |
| `analysis_*.py` | Joint entropy, mutual information between neighbours and between fields, cost of each index scheme. |

## Requirements

- NVIDIA GPU with compute capability >= 6.1 (profiling with Nsight Compute
  needs >= 7.0: **Pascal will not work**)
- Python 3.10+
- ~5 GB of disk

## Authorship and tools

**Author: Arnau Ferrerons Manich** ([ORCID 0009-0002-7245-7221](https://orcid.org/0009-0002-7245-7221)). The hypothesis, the direction of the
work, and the decisions about what to measure and what to conclude are the
author's. The GPU measurements were run on the author's own hardware and,
for the rented cards, on the author's account by Claude Code under the
author's direction.

**Claude Code (Anthropic) was used extensively throughout this project**: to
implement the codecs, the CUDA kernels and the measurement harnesses, to run
and interpret the CPU-side analysis, to review the literature, and to draft
and revise every document in this repository, including this one. Every
commit carries a `Co-Authored-By` trailer and a link to the session in which
it was produced, so the division of labour can be audited from the git
history rather than taken on trust.

**Verification of what is written here.** Before publication, Claude Code
was also used to audit the repository against its own evidence: every number
in the documents was traced to the raw log or histogram it came from
(`results/`), the whole-model figures were recomputed from the exact counts,
the transcribed GPU tables were checked against a fresh replication on the
same card, and every literature claim that supports a conclusion was checked
against the primary text of the paper rather than a summary of it. That
process found and corrected two errors of its own making — a duplicated
tensor counted twice in every whole-model figure, and a claim about a
competing method that a paper summary asserted and the paper does not — and
both corrections are recorded where they apply. The intent is that nothing
stated as measured is assumed, and nothing attributed to a source is
paraphrased from memory; where a number could not be traced, the documents
say so instead.

Any remaining error in the code, the numbers or the conclusions is the
author's responsibility. Nothing here has been peer-reviewed. The
measurements have not yet been reproduced by anyone else;
[METHODOLOGY.md](METHODOLOGY.md) exists so that they can be.

## License

Code (`*.py`, `*.sh`, `*.ps1`) is released under the MIT License
([LICENSE](LICENSE)). Documentation, figures and measurement data are released
under CC BY 4.0 ([LICENSE-docs](LICENSE-docs)). If you reuse the results,
please cite as in [CITATION.cff](CITATION.cff); the archived record is
[doi:10.5281/zenodo.22736348](https://doi.org/10.5281/zenodo.22736348).
