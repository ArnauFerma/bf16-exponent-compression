# Renting a GPU by the hour for the missing measurements

Self-contained: this is enough to run the test without consulting anything
else. Total cost **~1.40 USD**, wall-clock time **under an hour**.

> **Done on 2026-09-13** on an A100-SXM4-80GB (1.59 USD/h, Secure Cloud —
> Community had no stock) and an RTX 4090 (0.74 USD/h), ~1.7 USD in total.
> Outcome against the prediction in section 7: RESULTS.md, Phase 2e. Kept as
> the procedure for reproducing it or adding a card.

---

## 1. What is missing, and why it needs a different card

Everything measured so far is a GTX 1050 Ti: Pascal, 6 SMs, **1 MiB of L2**.
Two limitations that cannot be worked around on that machine:

1. Performance collapses as the `BLOCK` parameter grows (9.7 ms at BLOCK=64,
   154 ms at BLOCK=1024). The proposed explanation is that the resident
   working set overflows the L2. **On a card with a large L2 that should
   disappear** — and large BLOCK is exactly where the format compresses best.
2. Nsight Compute **dropped Pascal support** in 2020.1. There is no way to
   profile that card with any current version.

## 2. Which card to rent

The figure that governs the effect is **not** L2 size or raw speed, but
**L2 per resident thread**:

| card | L2 | SMs | resident threads | **L2 / thread** |
|---|---|---|---|---|
| GTX 1050 Ti (the reference) | 1 MB | 6 | 12,288 | 85 B |
| RTX 3060 | 3 MB | 28 | 43,008 | 73 B |
| **A100 80GB** | 40 MB | 108 | 221,184 | **190 B** |
| H100 SXM | 50 MB | 132 | 270,336 | 194 B |
| RTX 4090 | 72 MB | 128 | 196,608 | 384 B |
| RTX 4070 | 36 MB | 46 | 70,656 | 534 B |

### Recommendation: **A100 80GB, ~1.39 USD/hour**

Two reasons:

1. **It fills the middle of the range.** The cards available for free sit at
   the extremes (85 B and 534 B). With the A100 there are four spread points:
   85 -> 190 -> 384 -> 534.
2. **It is DFloat11's target hardware.** Any claim becomes directly
   comparable with their published numbers, instead of a discussion about
   consumer cards.

### Cheap alternative: **RTX 4090, ~0.34 USD/hour**

Useful, but at 384 B/thread it sits close to a 4070, so it gives less
information per euro. Choose it only to spend cents.

### Do not rent

**RTX 3090 or RTX A6000**: only 6 MB of L2 and many resident threads, i.e.
~50 B/thread — **worse than the 1050 Ti**. Nothing older than Turing.

## 3. Where

| provider | RTX 4090 | A100 80GB | notes |
|---|---|---|---|
| **RunPod** Community | ~0.34 $/h | ~1.39 $/h | simple interface, per-second billing, no traffic charges |
| RunPod Secure | ~0.69 $/h | ~1.39 $/h | with SLA; unnecessary here |
| **Vast.ai** on-demand | ~0.29–0.59 $/h | varies | cheapest, but peer-to-peer and quality varies |
| Vast.ai spot | ~0.11–0.35 $/h | varies | interruptible; acceptable for 30 minutes |

For a one-off test: **RunPod Community**. Deploying takes less time than
comparing machines on Vast.

- RunPod: <https://www.runpod.io/pricing>
- Vast.ai: <https://vast.ai/pricing/gpu/RTX-4090>
- Comparison site: <https://getdeploying.com/gpus/nvidia-rtx-4090>

Prices as of September 2026; check before renting.

## 4. Important: profiling may not work (and it does not matter)

Inside a rented container `ncu` usually fails with `ERR_NVGPUCTRPERM`: the
counter permission is controlled by the **host**, not the container, and
cannot be fixed from inside.

**That is fine, and there is no need to hunt for an instance that allows
it.** The timing sweeps — which are what answer the L2 question — do not
need `ncu`, and `run_all.sh` detects it and skips it cleanly. Take the cheap
instance.

## 5. Step by step (RunPod)

1. Sign up at <https://runpod.io> and add credit (the minimum is 10 USD;
   ~1.40 will be spent).
2. **Deploy** -> **Community Cloud** tab -> choose **A100 80GB**
   (or RTX 4090).
3. Template: any PyTorch or CUDA image on Ubuntu.
   **Container disk: at least 20 GB** — the model plus the extracted weights
   take ~4 GB and the default is sometimes too small.
4. Deploy and wait for *Running*. Then **Connect** -> web terminal or
   **JupyterLab**.
5. Get the code onto the pod. Either:
   - clone it (the pod has internet):
     ```bash
     git clone https://github.com/ArnauFerma/bf16-exponent-compression.git ~/bf16
     ```
   - or upload a zip of the repository: in **JupyterLab**, drag it into the
     file browser on the left; or with **runpodctl** on the source PC,
     `runpodctl send repo.zip` and run the `receive` command it prints on
     the pod; then unzip into `~/bf16`.
6. In the pod's terminal:
   ```bash
   cd ~/bf16
   bash setup_cloud.sh
   ```
   `setup_cloud.sh` is the variant meant for containers: it works as root,
   without `sudo`, and with images that lack the `venv` module.

   **Check two things in its output:**

   - the line `L2 per resident thread:` — on an A100 it should read ~190 B.
     If it shows something near 50–85 B, the instance is not the card that
     was ordered: terminate it and choose again.
   - at the end, `RESULT: both kernels correct`. If it fails, the script
     exits with an error on purpose: **do not continue**, the timings would
     be worthless.

7. Run everything:
   ```bash
   bash run_all.sh
   ```
   ~20 minutes. Leaves `results.tar.gz` (and the same files unpacked under
   `results/<gpu-name>/`).

8. Retrieve the file: download it from JupyterLab (right-click ->
   Download) or `runpodctl send results.tar.gz` from the pod.

9. **TERMINATE the pod.** Not "Stop": **Terminate**. A stopped pod keeps
   billing for storage.

## 6. Cost

| | A100 80GB | RTX 4090 |
|---|---|---|
| setup + model download | ~10 min | ~10 min |
| measurements | ~15 min | ~15 min |
| **total (~40 min, rounded to 1 h)** | **~1.40 USD** | **~0.35 USD** |

## 7. What to look at in the result

The prediction, recorded before measuring, so the result can be read without
help:

> On a card with a large L2 the **BLOCK cliff should flatten**. On the
> 1050 Ti, going from BLOCK=64 to BLOCK=1024 costs ~14x in time. If the L2
> explanation is right, that ratio should shrink a lot as L2 per thread
> grows, and staging the **input** in shared should stop helping at
> BLOCK=256, because the data already fits in cache.

The files to look at inside `results.tar.gz`:

- `env_info.json` — versions and card, with nothing that identifies the
  machine or the account.
- `1_bench_base.txt` — the BLOCK sweep. **This answers the question.**
- `2_bench_opt.txt` — attribution: if `smem in=1` stops winning at
  BLOCK=256, the L2 explanation is confirmed.
- `3_head2head.txt` — Huffman vs ladder with the optimised kernel.
- `4_bench_idx8.txt` — 8-bit index.
- `ncu_out/` — only if profiling worked.

**If the cliff is still there, the L2 explanation is false** and section 2b
of RESULTS.md has to be rewritten before any of this leaves the repo.

## 8. If something fails

| What you see | What to do |
|---|---|
| `nvidia-smi` does not respond | the instance has no visible GPU; terminate and take another |
| `L2 per resident thread` does not match the card ordered | not the advertised card; terminate and take another |
| `FAIL` in the verification | stop; keep the full output |
| `ERR_NVGPUCTRPERM` | profiling blocked by the host; **ignore**, the rest is valid |
| `No space left on device` | container disk too small; redo with 20+ GB |
| `venv` missing | `setup_cloud.sh` already handles it |
| spot instance cut mid-run | relaunch `run_all.sh`; the downloaded model is reused |
