# arXiv submission — fields to paste

Upload file: `paper/report-arxiv-source.tar.gz` (one file, `report.tex`, no figures, bibliography inline; compiles with pdflatex, `\pdfoutput=1` set).

## Step 1 — Start new submission

- **License:** CC BY 4.0 (matches the repository's documentation license; arXiv's default "non-exclusive license to distribute" would let people read but not reuse).
- **Archive / primary category:** Computer Science → **cs.PF** (Performance)
- **Cross-lists:** **cs.LG** (Machine Learning), **cs.AR** (Hardware Architecture)

## Step 2 — Metadata

**Title**

Entropy Code or Memory Layout? A Negative Result in Lossless BF16 Weight Compression, and the 8-bit Block Index It Led To

**Authors**

Arnau Ferrerons Manich

**Abstract** (1,750 characters; limit is 1,920)

Entropy-coding the exponent field of BF16 weights, as in DFloat11, losslessly removes about a third of the bytes at the cost of a decompression kernel on every forward pass. This report tests whether a fixed-shape prefix code with a 10-byte table (a "ladder" code) decodes faster on a GPU than canonical Huffman with a 4 KiB lookup table, for under one point of compression. It does not: on Pascal, Ampere and Ada the ladder code loses 0.86 points and is never faster where it would be run. The test produced something that does work. Replacing the per-block uint32 offset index with 8-bit block lengths recovered by a warp prefix-sum cuts the index from 4 to 1.125 bytes per block (0.14 bits/weight, against about 0.21 for DFloat11's 5-bit per-thread gaps), adds 2.25 points of compression, and costs nothing measurable on any card. With the output coalesced through shared memory, the reference decoder becomes 2.0x faster and 2.25 points smaller, and decodes 64M symbols in 0.20 ms on an RTX 4090. Exact measurements on Qwen3-0.6B place canonical Huffman 0.033 bits/weight from the exponent-only floor and 0.100 from the true floor of the 16-bit symbol; the field split gives away 0.076 bits/weight. A 16x collapse at large block sizes on Pascal is an L2 effect that shrinks to 2x and 1.7x on cards with more L2 per thread, and the optimised decoder's time tracks SM clock, not bandwidth. Every number traces to a committed log published with the code.

**Comments**

10 pages, 11 tables. Code, raw logs and methodology: https://github.com/ArnauFerma/bf16-exponent-compression (archived at doi:10.5281/zenodo.22736348)

**ACM class** (optional)

E.4; C.1.4; I.2.6

**Report number:** leave blank. **Journal reference / DOI:** leave blank (the Zenodo DOI is for the code archive, not this paper; it goes in Comments).

## Step 3 — Process

arXiv compiles the source. Expected: pdflatex, two passes, 10 pages, no warnings. Check the generated PDF page 1 (title, ORCID link) and the tables.

## Endorsement

A first submission to cs.PF may require an endorser. If the form says so, the endorser must be someone who has submitted to cs.PF (or cs.LG for that cross-list) recently; arXiv shows a link to send them a request. Ask before submitting rather than after — a submission held for endorsement does not get a date until it clears.

## After acceptance

- Add the arXiv ID to `README.md` (Authorship section), `CITATION.cff` (`preferred-citation`), and the Zenodo record (edit → related identifiers).
- Claim the paper on Hugging Face Papers (huggingface.co/papers/<arXiv id>) — that links it to a profile.
- Link it from the ORCID record (Works → add → arXiv).
