# I tried to beat Huffman for BF16 weight compression. I measured why I couldn't — and found something better in the index.

*Arnau Ferrerons Manich · September 2026 · [code, logs and methodology](https://github.com/ArnauFerma/bf16-exponent-compression) · [report (PDF)](https://github.com/ArnauFerma/bf16-exponent-compression/blob/main/paper/report.pdf) · [doi:10.5281/zenodo.22736348](https://doi.org/10.5281/zenodo.22736348)*

---

This is a negative result. The idea I started with is wrong, and I have the numbers from three GPU architectures to show exactly how wrong. It's also the story of how testing a wrong idea properly produced a right one that I would never have looked for on purpose.

If you only want the takeaways, they're at the end. The rest is the order things actually happened in, because the order is the point.

## The idea

A bfloat16 weight is 16 bits: 1 sign, 8 exponent, 7 mantissa. In a trained model the sign is a coin flip and the mantissa is nearly uniform — together they're 8 bits of entropy in 8 bits. The exponent is different. Weights are roughly Gaussian, so almost all exponents land in a handful of values. On Qwen3-0.6B, three exponent values carry 72% of the weights, and the whole field has 2.65 bits of entropy in 8.

That's the observation behind [DFloat11](https://arxiv.org/abs/2504.11651) (NeurIPS 2025): Huffman-code the exponents, leave everything else raw, decompress on the GPU right before each matrix multiply. You get bit-exact outputs and a model about 30% smaller. You also get a decompression kernel running on every forward pass, which is why DFloat11 is 1.4–2x slower than plain BF16 at batch size 1.

So the decoder's speed matters. And when I looked at how a canonical Huffman decoder works on a GPU — a lookup table in shared memory, one table access per symbol — I thought I saw a shortcut.

A Huffman code is *adaptive*: the code lengths come from the data, and you need the table to invert them. But the exponent distribution has the same *shape* in every model: a few very frequent values, then a tail. What if the code had a fixed shape, and only the assignment of symbols to slots changed? Count the leading ones to pick a "rung", read one or two index bits, look up in a table of ten bytes. No LUT. A `clz`, a branch, a shift.

I called it a ladder code. On paper it costs about 0.8 points of compression against Huffman, and I hypothesised it would decode faster because it never touches memory.

I wrote that hypothesis down before I had a GPU to test it on, along with the reason it might fail: *"Canonical Huffman uses a LUT: one SRAM access. The ladder uses clz + branch + shift + mask. On instruction count it may lose."* That sentence turned out to be the whole story. But I didn't know that yet.

## Phase 1: the rate

First the cheap part, on CPU, on the real weights of Qwen3-0.6B — all 596 million unique ones (more on "unique" later).

| whole model | bits/exponent | reduction |
|---|---|---|
| entropy (the floor) | 2.645 | — |
| canonical Huffman | 2.678 | 32.48% |
| ladder | 2.815 | 31.62% |

Huffman is 0.033 bits from the theoretical floor. The ladder gives up 0.86 points. Within budget. Both roundtrip bit-exactly. The rate half of the hypothesis held.

## Phase 2: the GPU, and the question that decides everything

I wrote two CUDA kernels with an identical interface — same block index, same thread mapping, same output layout — differing only in the symbol decoder. And a third kernel that reads the same words and writes the same bytes but *doesn't decode*. A memory floor. If the real decoders sit near it, memory dominates and the entropy code cannot matter.

GTX 1050 Ti, 64 million exponents, blocks of 64 symbols per thread:

| | Huffman | ladder | floor |
|---|---|---|---|
| time | 9.70 ms | 9.75 ms | 9.57 ms |

Both decoders within 2% of the floor. Within 1% of each other. The ladder is marginally *slower*, because its stream is 4.8% bigger.

At larger block sizes the ladder did win — by up to 33% — which is the compute-bound regime I had predicted. Except those configurations were 3–16x slower in absolute terms. Nobody would run them.

So the hypothesis was already dead at the only operating point that matters. But "memory-bound" had a caveat I couldn't ignore: the best configuration was moving 9 GB/s on a card whose peak is 112. **Eight percent of peak.** That's not memory-bound. That's a terrible access pattern.

## Phase 2b: where the time actually went

Each thread decoded its own block, so each thread read its own region of the bitstream — a warp of 32 threads scattered into 32 streams. And each thread wrote 64 consecutive bytes of output, so a warp's 32 stores landed in 32 different sectors. Both problems have the same fix: the threads in a CUDA block own a contiguous range, so stage it through shared memory and move it as one coalesced transfer.

I compiled four variants — input staged, output staged, both, neither — so I could *attribute* the gain instead of guessing:

| variant | 1050 Ti |
|---|---|
| base | 1.00 |
| input staged | 1.01x |
| output staged | **1.92x** |
| both | 1.48x |

Output alone nearly doubled throughput. Input alone did nothing. And doing both was *worse* than output alone — the input staging spent shared memory and a barrier without buying anything, and paid in occupancy.

Then I redid the Huffman-vs-ladder comparison with the fast kernel, because moving the bottleneck can change the ranking. It didn't. Huffman 4.80 ms, ladder 5.07. Now the ladder lost on both axes, in exactly the regime where compute mattered more. One shared-memory lookup beats a dependent chain of clz, branch, shift, mask. The sentence I wrote before starting had been right.

## The thing I wasn't looking for

The block index was the bit of the design I'd thought about least. One 32-bit offset per block of 64 symbols. Four bytes per block. At block size 64 that's half a bit per weight — a real cost, and the reason everyone wants bigger blocks, which are slower because the serial chain inside each block gets longer.

I looked at the actual block lengths. At 64 symbols per block, a block is between 130 and 273 bits long. A range of 143. **That fits in eight bits.**

So: one 32-bit offset per *superblock* of 32 blocks — one warp — and one byte per block holding its length minus the global minimum. Each thread recovers its own offset with an exclusive prefix-sum across the warp: five `__shfl_up_sync` steps, no memory traffic. 1.125 bytes per block instead of 4.

| block size 64, output staged | uint32 index | 8-bit index |
|---|---|---|
| time | 4.81 ms | 4.82 ms |
| compression | 30.73% | **32.97%** |

Free. +2.25 points. Five shuffles are invisible next to 64 dependent loads.

Look at what that did to the design. Before it, the fast configuration (small blocks) and the compact configuration (big blocks) were 2.9 points apart. After it, 0.7 points. The entire reason to want large blocks — the one thing that made the ladder's compute-bound regime interesting — was gone.

So here's the score at that point. The entropy code question the whole project was built on: 0.86 points, in the wrong direction. Two changes that never touched the entropy code: 2x faster and 2.25 points smaller.

## Things I checked that didn't pan out, briefly

Because "I didn't find it" is only worth something if you say where you looked.

- **Correlation between neighbouring exponents**, which vector coding could exploit: mutual information at most 0.01 bits, measured in six windows across the model. Nothing there.
- **Correlation between the fields of one weight.** This one mattered, because a coder over the whole 16-bit symbol can never do worse than coding fields separately, and I needed to know how much my design was giving away. Answer, exactly, over all 596M weights: 0.040 bits between exponent and mantissa, zero between sign and anything. A full-alphabet Huffman would reach 10.602 bits/weight versus 10.678 for the field split. **0.076 bits/weight, 0.47 points.** Almost all of it in the two largest exponents, where the Gaussian tail decays *inside* the binade so small mantissas are more likely. That's the ceiling on any rate claim for this design, and now it's a measurement instead of a hope.
- **A duplicated tensor.** Qwen3-0.6B ties its embedding and output matrices, but the safetensors file stores both, bit-identical. My first numbers counted 155 million weights twice. The extractor now drops exact duplicates; every figure was recomputed; the difference was 0.08 points and the conclusions didn't move. It's in the methodology because it would be the first thing a careful reader found.

## Phase 2e: the part I couldn't do at home

Everything so far was one card: a 2016 Pascal GPU with 1 MiB of L2 cache, whose most dramatic behaviour was a 16x collapse when blocks got large. I'd attributed that to the working set overflowing L2 and I'd written down a prediction: on a card with more L2 per thread, the cliff should disappear, staging the input should stop mattering, and large blocks should become viable.

Two rented cards, about 1.7 USD in total, same code, same sample, unattended:

| block 1024 vs 64, base kernel | GTX 1050 Ti (85 B L2/thread) | A100 (190 B) | RTX 4090 (384 B) |
|---|---|---|---|
| slowdown | **16.2x** | 1.96x | 1.71x |

The cliff prediction held.

The second prediction didn't, and the way it failed was more interesting than a yes. Staging the input was supposed to stop mattering once the data fit in L2. Instead its benefit *decayed* — 4.4x, 2.0x, 1.2x across the three cards. Even when every byte is an L2 hit, 32 scattered streams per warp are served more slowly than one contiguous load. L2 per thread predicts the size of the effect, not whether it exists.

The third prediction was moot: large blocks are still 3–6x slower than small ones on the new cards, cliff or no cliff, because the serial chain is 16x longer. And the 8-bit index had already removed the reason to want them.

Two things I hadn't predicted at all. The Pascal lesson "don't stage both input and output" was Pascal-specific — on Ampere and Ada, both is the best variant. And the optimised decoder is 2.2x faster on the 4090 than on the A100, with *half* the memory bandwidth. Their SM clocks differ by 2.2x. The decoder, once you fix the access pattern, is bound by the latency of its 64-step chain of dependent loads. It runs at the speed of the clock, not the speed of memory.

Best times for 64 million exponents: 4.8 ms, 0.44 ms, 0.20 ms. Ladder-to-Huffman ratio: 1.05, ~1.0, 1.3. On no card does the ladder win where you'd run it.

## What the field did while I was doing this

In June, [Tan et al.](https://arxiv.org/abs/2606.15789) (ISCA 2026) reported up to 11x DFloat11's throughput. They use rANS instead of Huffman, code the whole 16-bit symbol, and — this is the part that matters — fuse decompression into the GEMM so decompressed tiles never touch global memory.

Their own attribution is that the gain comes from fusion, not from rANS. My measurements say the same thing from the other side: after every fix I found, the standalone decoder sits at 10–45% of peak bandwidth, its time tracks SM clock, and its output is the majority of its traffic. Those are precisely the costs fusion removes. The Shannon-gap argument for rANS is weak on this data — Huffman is at 98.8% efficiency on the exponent alphabet. If I continued, I'd keep Huffman and fuse.

One more correction I owe. In an early draft I wrote that my 8-bit index "applies unchanged" to DFloat11's index. Then I read their kernel instead of their paper. Their design is the other corner of the same trade-off: each thread gets a fixed 8 *bytes* of stream and a 5-bit gap to the first whole codeword (about 0.21 bits/weight of index), and because symbols per thread then vary, they decode twice with a prefix-sum in between for output positions. Mine fixes symbols and prefix-sums the input side: 0.14 bits/weight, one pass. A third smaller, not a drop-in, and I haven't run their kernel so I can't tell you which is faster. That's now what the report says.

## What I'd want you to take away

1. **The entropy code was never where the time was.** A 10-byte table versus a 4 KiB LUT is worth nothing when one thread's dependent-load chain is the bottleneck. Measure the floor before optimising the coder.
2. **Attribute, don't combine.** Four compiled variants told me output staging was worth 1.9x and input staging nothing — and that "both" was worse. Then a different architecture reversed the last part. Results from one card are results from one card.
3. **The index is a real place to look.** Fixed symbols per block, 8-bit lengths, a warp prefix-sum: free, +2.25 points, and it deletes the block-size dilemma. It works for any variable-length stream with bounded block-length spread.
4. **Write the prediction down first.** I got two of three right on the rented cards and the one I got wrong taught me the most. That only works if the prediction exists before the data does.
5. **The rate is done; the structure isn't.** Huffman is 0.033 bits from its floor, the field split is 0.076 bits from the true floor, and neighbouring exponents are independent. Everything left is in the access pattern, the index, and the serial chain — and the serial chain only goes away with fusion.

Everything above traces to a committed log in the repository, including the runs that didn't go as predicted. The methodology document lists what I couldn't measure and why.

---

*Claude Code (Anthropic) was used extensively in this project — for the codecs, kernels and harnesses, the CPU analysis, the literature review, driving the rented GPU runs, and drafting and revising the documentation and this article — under my direction. It was also used to audit the repository against its own logs before publication, which is how the duplicated tensor was caught, and to read DFloat11's kernel, which is how the misreading was. The hypothesis, the decisions, and the mistakes are mine.*
