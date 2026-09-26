# Graph Report - bf16-exponent-compression  (2026-09-26)

## Corpus Check
- 6 files · ~43,866 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 278 nodes · 505 edges · 15 communities (9 shown, 6 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 26 edges (avg confidence: 0.92)
- Token cost: 53,977 input · 0 output

## Community Hubs (Navigation)
- GPU Benchmark Harness
- Compression Concepts & Literature
- Entropy Field Analysis
- Ladder Codec & CPU Benchmark
- Report Findings & Citations
- L2 Cliff & Hardware Protocol
- GPU Codec Benchmark Results
- Access-Pattern Kernel Optimisation
- 8-bit Index Kernel
- Linux Run & Profile Scripts
- Cloud Setup Script
- Linux Setup Script

## God Nodes (most connected - your core abstractions)
1. `Entropy Code or Memory Layout? A Negative Result in Lossless BF16 Weight Compression (report)` - 27 edges
2. `Entropy Code or Memory Layout? (BF16 compression report)` - 22 edges
3. `canonical_huffman()` - 12 edges
4. `main()` - 11 edges
5. `Phase 2e: L2 prediction on A100 and RTX 4090` - 11 edges
6. `Lossless BF16 weight compression project` - 10 edges
7. `DFloat11 (arXiv 2504.11651, NeurIPS 2025)` - 10 edges
8. `roundtrip_check()` - 9 edges
9. `8-bit block index + warp prefix-sum` - 9 edges
10. `BitWriter` - 8 edges

## Surprising Connections (you probably didn't know these)
- `Correctness protocol: bit-exact before timing` --references--> `BitWriter`  [EXTRACTED]
  METHODOLOGY.md → ladder_codec.py
- `measure()` --implements--> `Timing protocol (CUDA events, warm-up, median of 11, shuffled order)`  [EXTRACTED]
  bench_gpu.py → METHODOLOGY.md
- `RTX 3060 prediction: cliff in the same place despite 3x L2` --semantically_similar_to--> `RTX 4070 prediction (Linux operator guide)`  [INFERRED] [semantically similar]
  OPERATOR_WINDOWS.md → OPERATOR_LINUX.md
- `Rented-GPU prediction: cliff flattens, input staging stops helping at BLOCK=256` --semantically_similar_to--> `RTX 4070 prediction (Linux operator guide)`  [INFERRED] [semantically similar]
  RENT_A_GPU.md → OPERATOR_LINUX.md
- `Entropy Code or Memory Layout? (BF16 compression report)` --shares_data_with--> `RTX 4090 base-kernel benchmark (Huffman vs ladder vs floor)`  [INFERRED]
  paper/report.pdf → results/rtx4090/1_bench_base.txt

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Four-stage GPU decode benchmark suite (base, smem opt, head-to-head, idx8) run on all three GPUs** — bench_gpu, bench_opt, bench_head2head, bench_idx8, results_a100sxm480gb_gpu_info_nvidia_a100_sxm4_80gb, results_gtx1050ti_gpu_info_nvidia_geforce_gtx_1050_ti, results_rtx4090_gpu_info_nvidia_geforce_rtx_4090 [EXTRACTED 1.00]
- **Ladder code family (fixed-table prefix codes with escape) compared against canonical Huffman on CPU** — results_cpu_bench_log_ladder_1_1_1_2, results_cpu_bench_log_ladder_1_1_2_2, results_cpu_bench_log_ladder_1_2_2_3, results_cpu_bench_log_ladder_escape_mechanism, results_cpu_bench_log_canonical_huffman_cpu_result [EXTRACTED 1.00]
- **Exact rate-floor decomposition of the field split** — handoff_field_split, results_phase_2d_cross_field_mi, results_neighbour_mutual_information, handoff_canonical_huffman, handoff_arxiv_2606_15789 [EXTRACTED 1.00]
- **Structural changes dominate the entropy code (access pattern, index, serial chain, fusion)** — handoff_output_coalescing, handoff_8bit_block_index, handoff_serial_decode_chain, handoff_gemm_fusion, handoff_ladder_code, handoff_canonical_huffman [EXTRACTED 1.00]
- **Three-architecture test of the L2-per-thread cliff prediction** — handoff_gtx_1050_ti, handoff_a100, handoff_rtx_4090, handoff_l2_per_resident_thread, handoff_block_cliff_l2_effect, rent_a_gpu_cliff_prediction, results_phase_2e_a100_4090 [EXTRACTED 1.00]
- **CPU data pipeline: extract deduped Qwen3-0.6B BF16 weights, analyse field entropies, benchmark codecs** — extract_real_weights, results_cpu_extract_log_lm_head_embed_tokens_duplication, analysis_fields, results_cpu_bench_log_qwen3_0_6b_real_dataset, results_cpu_analysis_fields_bf16_field_entropy [INFERRED 0.85]
- **Structural decoder fixes that beat entropy-code choice** — paper_report_output_smem_coalescing, paper_report_index8_prefix_sum, paper_report_sm_clock_bound, paper_report_fused_decompression [EXTRACTED 1.00]
- **RTX 4090 benchmark logs backing the report** — results_rtx4090_1_bench_base_bench_base, results_rtx4090_2_bench_opt_bench_opt, results_rtx4090_4_bench_idx8_bench_idx8, paper_report_report [INFERRED 0.95]
- **Random-access index designs for variable-length streams** — paper_report_uint32_block_index, paper_report_index8_prefix_sum, paper_report_dfloat11_gap_index [EXTRACTED 1.00]

## Communities (15 total, 6 thin omitted)

### Community 0 - "GPU Benchmark Harness"
Cohesion: 0.06
Nodes (54): Can vector (pairwise) coding gain anything on the exponent? Two very different…, argparse, main(), measure(), Phase 2: timing of the two decode kernels. Central question: MEMORY-bound or…, The format spec says uint32 index. With ~600M weights the stream is ~1.6e9…, sustained warm-up + repetitions; returns (median_ms, iqr_ms, mhz), sm_clock_mhz() (+46 more)

### Community 1 - "Compression Concepts & Literature"
Cohesion: 0.07
Nodes (55): 8-bit block index + warp prefix-sum, Approaching Shannon Bound with Lossless LLM Weight Compression (arXiv 2606.15789, ISCA 2026), Fixed symbols / fixed bits / fixed both trilemma, Canonical Huffman over BF16 exponents, Compressed file format (header, sign_mantissa, exponents, index), Deep Compression (Han et al., arXiv 1510.00149), DFloat11 (arXiv 2504.11651, NeurIPS 2025), EntroLLM (arXiv 2505.02380) (+47 more)

### Community 2 - "Entropy Field Analysis"
Cohesion: 0.07
Nodes (17): huff(), How much rate does coding the BF16 fields separately give away? The format…, (a) Mutual information ~0 must not be an artefact of looking only at the first…, BitReader, BitWriter, canonical_huffman(), decode(), encode() (+9 more)

### Community 3 - "Ladder Codec & CPU Benchmark"
Cohesion: 0.10
Nodes (21): huffman_analytic(), index_overhead_bytes(), Bit-exact roundtrip of Huffman and ladder on a large random sample, using the…, Test bench: canonical Huffman vs ladder, on synthetic weights and on real model…, report_file(), roundtrip_check(), ladder_code_arrays(), -> code_of[256] uint32, len_of[256] uint8, slots, escape (+13 more)

### Community 4 - "Report Findings & Citations"
Cohesion: 0.17
Nodes (26): Qwen3-0.6B (596,049,920 unique BF16 weights), outputs/gpu_bench.json, BF16 exponent entropy (~2.645 bits), Canonical Huffman exponent code (11-bit 4 KiB LUT), Schwartz & Kallick 1964, canonical prefix encoding, Deep Compression (Han et al., ICLR 2016), DFloat11 (Zhang et al., NeurIPS 2025), DFloat11 5-bit per-thread gap index (two-pass decode) (+18 more)

### Community 5 - "L2 Cliff & Hardware Protocol"
Cohesion: 0.14
Nodes (22): bench_gpu.mem_floor (memory-floor kernel), A100-SXM4-80GB (Ampere, 40 MiB L2, 190 B/thread), BLOCK cliff as an L2 effect, GTX 1050 Ti (Pascal, 1 MiB L2, 85 B/thread), L2 per resident thread, Nsight Compute unavailable (Pascal unsupported, counters blocked in containers), RTX 4090 (Ada, 72 MiB L2, 384 B/thread), Committed raw data under results/ (+14 more)

### Community 6 - "GPU Codec Benchmark Results"
Cohesion: 0.12
Nodes (18): BLOCK x threads sweep (BLOCK 64..1024, thr 32..256) benchmark grid, Canonical Huffman exponent codec (GPU kernel: 24 regs/thread, 4096 B shared, 2.5838 bits/symbol), Ladder exponent codec (GPU kernel: 18 regs/thread, 16 B shared, 2.7089 bits/symbol), Qwen3-0.6B real exponent stream, 64,000,000 symbols (GPU benchmark input), l/h ratio (ladder time / huffman time); ~1.0 at BLOCK=64, ladder slower at large BLOCK, uint32 block index (4.000 B/block; 30.73% compression at BLOCK=64), uint8 index infeasible at BLOCK=256 (block length range 409 > 255), uint8 + prefix-sum block index (1.125 B/block; +2.25 pts at BLOCK=64, +1.12 pts at BLOCK=128, ~same time) (+10 more)

### Community 7 - "Access-Pattern Kernel Optimisation"
Cohesion: 0.32
Nodes (7): kernel(), kernel_h(), launch(), launch_h(), max_words_per_cudablock(), EXACT bound (not the theoretical worst case) on the word span a CUDA block…, Access-pattern optimisation of the decode kernels. The base kernel reaches 8.2%…

### Community 8 - "8-bit Index Kernel"
Cohesion: 0.33
Nodes (3): build_index8(), -> len_codes uint8[n_blocks], sb_base uint32[...], minlen, n_blocks sb_base is…, 8-bit block index + warp prefix-sum. Instead of one absolute uint32 offset per…

## Knowledge Gaps
- **23 isolated node(s):** `setup_linux.sh script`, `Compressed file format (header, sign_mantissa, exponents, index)`, `Licensing: MIT for code, CC BY 4.0 for docs and data`, `MLX issue #3043 (rANS quantization request)`, `EntroLLM (arXiv 2505.02380)` (+18 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 91 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Entropy Code or Memory Layout? A Negative Result in Lossless BF16 Weight Compression (report)` connect `Compression Concepts & Literature` to `Report Findings & Citations`, `L2 Cliff & Hardware Protocol`?**
  _High betweenness centrality (0.428) - this node is a cross-community bridge._
- **Why does `measure()` connect `GPU Benchmark Harness` to `L2 Cliff & Hardware Protocol`?**
  _High betweenness centrality (0.303) - this node is a cross-community bridge._
- **Why does `Timing protocol (CUDA events, warm-up, median of 11, shuffled order)` connect `L2 Cliff & Hardware Protocol` to `GPU Benchmark Harness`, `Compression Concepts & Literature`?**
  _High betweenness centrality (0.302) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `Entropy Code or Memory Layout? (BF16 compression report)` (e.g. with `RTX 4090 base-kernel benchmark (Huffman vs ladder vs floor)` and `RTX 4090 shared-memory staging benchmark (base vs smem in/out)`) actually correct?**
  _`Entropy Code or Memory Layout? (BF16 compression report)` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `setup_linux.sh script`, `Compressed file format (header, sign_mantissa, exponents, index)`, `Licensing: MIT for code, CC BY 4.0 for docs and data` to the rest of the system?**
  _23 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `GPU Benchmark Harness` be split into smaller, more focused modules?**
  _Cohesion score 0.056535504296698326 - nodes in this community are weakly interconnected._
- **Should `Compression Concepts & Literature` be split into smaller, more focused modules?**
  _Cohesion score 0.06868686868686869 - nodes in this community are weakly interconnected._