# Draft: GitHub issue on LeanModels/DFloat11

Post at: https://github.com/LeanModels/DFloat11/issues/new
Label, if the form asks: discussion / question (not bug).

---

**Title:** Measured comparison of the gaps index against a fixed-symbols-per-block 8-bit index, plus three-architecture decoder timings

Hi — I've spent the last few weeks on the same field-split design as DFloat11 (Huffman over BF16 exponents, sign and mantissa raw) as an independent study, and a few of the measurements are directly about design choices in `decode.cu`. Sharing them here in case they're useful; everything below is reproducible from the repo, with raw logs committed.

Repo: https://github.com/ArnauFerma/bf16-exponent-compression · report (PDF): `paper/report.pdf` · archived: doi:10.5281/zenodo.22736348

**1. Index cost.** Your kernel gives each thread a fixed 8 bytes of stream and a 5-bit `gap`, so the index is 5 bits per 64 compressed bits — on Qwen3-0.6B (2.68 bits/exponent) that's about **0.21 bits/weight**. The alternative I ended up with fixes symbols per block instead: one `uint32` per superblock of 32 blocks plus one `uint8` block length, offsets recovered with a 5-step `__shfl_up_sync` exclusive prefix-sum in the warp. That's 1.125 B/block = **0.14 bits/weight at BLOCK=64, 0.07 at BLOCK=128**, and it decodes in one pass (no counting pass, since output positions are implicit). Measured cost of the prefix-sum vs a uint32 absolute index: 1.00x / 1.006x / 0.996x on a GTX 1050 Ti, an A100 and an RTX 4090. So it's about a third smaller than the gaps index on this model. The trade is the counting pass on your side vs the input-side prefix-sum on mine; I have **not** run `decode.cu`, so I can't say which is faster end to end — that's the obvious next measurement and I'd be glad to do it if you'd like a comparison number.

**2. Where the standalone decoder's time goes.** With the output staged through shared memory (which your kernel already does), the same Huffman decoder takes 4.8 ms / 0.44 ms / 0.20 ms for 64M exponents on the three cards. The 4090 is 2.2x faster than the A100 at half the bandwidth — the same ratio as their SM clocks (2520 vs 1140 MHz) — and achieved bandwidth is 16% / 10% / 45% of peak. My reading is that once the access pattern is fixed, the decoder is bound by the serial chain of dependent loads inside each thread, not by memory. That would explain why Tan et al. (ISCA 2026, arXiv:2606.15789) get their gains from fusing decode into the GEMM rather than from the entropy coder.

**3. The rate floor of the field split, exact.** Over all 596M unique weights of Qwen3-0.6B: I(exp; mant) = 0.040 bits/weight, sign independent of both; a canonical Huffman over the full 16-bit alphabet achieves 10.602 bits/weight vs 10.678 for the field split — the split gives away 0.076 bits/weight, almost all of it in the two largest binades (122, 123), where the mantissa entropy drops to 6.84 and 6.26 bits. Might be a cheap +0.4 points if you ever want it: a joint exp+mant code for those two exponents only.

One negative result, for completeness: a fixed-shape prefix code with a 10-byte table (clz + branch, no LUT) does **not** beat canonical Huffman on decode time on any of the three cards, and loses 0.86 points of rate. The LUT is not the bottleneck.

Not a bug report and not asking for anything — if any of this is wrong about `decode.cu`, I'd rather hear it here than leave it in the report. Thanks for publishing the kernel; reading it corrected one of my own claims before it went out.

---

## Notes for posting (not part of the issue)

- Point 1 reads their code correctly as of `master` today (`BYTES_PER_THREAD 8`, 5-bit gaps at `gaps[global_thread_id * 5 / 8]`, `position_offsets` per thread block, two decode loops with a shared-memory scan between). If they push changes before you post, re-check.
- The offer in point 1 (run their kernel head-to-head) is real work: it needs their format's encoder. Only offer it if you're willing to do it or to let me do it on a rented card (~0.5 USD).
- Post it as-is or trim; the three numbered points stand alone.
