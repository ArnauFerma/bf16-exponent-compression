# Renting a GPU for this test — what to pick and how to run it

Everything here is for *you*, not for a third party. Total cost is a couple of
dollars and the run takes well under an hour.

---

## 1. Which card to rent

What governs the effect we're chasing is **L2 cache per resident thread**, not
raw L2 size and not raw speed:

| card | L2 | SMs | resident threads | **L2 / thread** |
|---|---|---|---|---|
| GTX 1050 Ti (yours) | 1 MB | 6 | 12,288 | 85 B |
| RTX 3060 (brother) | 3 MB | 28 | 43,008 | 73 B |
| **A100 80GB** | 40 MB | 108 | 221,184 | **190 B** |
| H100 SXM | 50 MB | 132 | 270,336 | 194 B |
| RTX 4090 | 72 MB | 128 | 196,608 | 384 B |
| RTX 4070 (Cristian) | 36 MB | 46 | 70,656 | 534 B |

Cristian's 4070 already covers the **high** end (534 B). Your own machine
covers the **low** end (85 B). What's missing is the **middle** — and that is
exactly where the A100 and H100 sit.

### Recommendation: **A100 80GB, ~$1.39/hr**

Two reasons, both good:

1. It fills the gap at 190 B/thread, so you end up with four points spanning
   85 → 190 → 384 → 534 instead of a cluster at each extreme.
2. **It is DFloat11's target hardware.** Any claim you make becomes directly
   comparable to their published numbers instead of an apples-to-oranges
   argument about consumer cards.

**Budget alternative: RTX 4090, ~$0.34/hr.** Cheapest genuinely useful option,
but at 384 B/thread it sits close to the 4070 you're already getting for free,
so it adds less. Pick this only if you want to spend cents rather than dollars.

**Don't bother with:** RTX 3090 or A6000 (Ampere consumer/pro — only 6 MB of
L2, *worse* per-thread than your 1050 Ti), or anything pre-Turing.

## 2. Where to rent

| provider | RTX 4090 | A100 80GB | notes |
|---|---|---|---|
| **RunPod** Community | ~$0.34/hr | ~$1.39/hr | easiest UI, per-second billing, no egress fees |
| **RunPod** Secure | ~$0.69/hr | ~$1.39/hr | SLA, not needed here |
| **Vast.ai** on-demand | ~$0.29–0.59/hr | varies | cheapest, but peer-to-peer: machine quality varies |
| Vast.ai spot | ~$0.11–0.35/hr | varies | can be interrupted — fine for a 30-min run |

I'd use **RunPod Community** for a one-off: the interface is simpler, billing is
per second, and you won't spend twenty minutes evaluating hosts.

Sources: [Vast.ai RTX 4090 pricing](https://vast.ai/pricing/gpu/RTX-4090),
[RunPod pricing](https://www.runpod.io/pricing),
[provider comparison](https://getdeploying.com/gpus/nvidia-rtx-4090)

## 3. One thing that may not work, and why it doesn't matter

**GPU profiling inside rented containers is often blocked.** `ncu` needs
driver-level counter access that the *host* controls, so you may hit
`ERR_NVGPUCTRPERM` with no way to fix it from inside. RunPod instances
sometimes allow it, sometimes don't.

This is fine. The timing benchmarks — the BLOCK sweep and the staging
attribution — are what actually answer the L2 question. `ncu` only adds
confirmation via L2 hit rate. `run_all.sh` detects `ncu` and skips it cleanly.

So: **don't pay extra or shop around for a profileable instance.** Take the
cheap one.

## 4. Step by step (RunPod)

1. Sign up at runpod.io, add credit (minimum is $10; you'll use ~$2).
2. **Deploy** → **Community Cloud** → pick **A100 80GB** (or RTX 4090).
3. Template: any PyTorch or CUDA Ubuntu image. **Container disk: 20 GB**
   minimum — the model plus the extracted weights need ~4 GB and the default
   is sometimes tight.
4. Deploy, then **Connect** → either the web terminal or **JupyterLab**.
5. Get the bundle onto the box. Easiest is JupyterLab: open it and
   **drag `portable_bundle.tar.gz` into the file browser**. Alternatives:
   ```bash
   # from your PC, if you installed runpodctl
   runpodctl send portable_bundle.tar.gz
   # then on the pod, run the receive command it prints
   ```
6. In the pod's terminal:
   ```bash
   mkdir -p ~/bf16 && cd ~/bf16
   tar xzf ~/portable_bundle.tar.gz     # adjust path if needed
   bash setup_cloud.sh
   ```
   `setup_cloud.sh` is the container-aware variant — it copes with being root,
   with no `sudo`, and with images that lack the `venv` module.

   Wait for:
   ```
   RESULTADO: ambos kernels correctos
   ```
   If that fails, stop — the script exits non-zero on purpose, because timings
   from an incorrect decoder are worthless.

7. Run everything:
   ```bash
   bash run_all.sh
   ```
   ~20 minutes. Produces `resultados.tar.gz`.

8. Get it back: download through JupyterLab (right-click → Download), or
   `runpodctl send resultados.tar.gz` from the pod.

9. **Terminate the pod.** Not "stop" — *terminate*, or you keep paying for
   storage.

## 5. What it costs

| | A100 | RTX 4090 |
|---|---|---|
| setup + download | ~10 min | ~10 min |
| benchmarks | ~15 min | ~15 min |
| **total (~40 min, rounded to 1 hr)** | **~$1.40** | **~$0.35** |

## 6. What to look for in the output

The prediction, so you can read the result yourself without waiting for me:

> On any card with a large L2, the **BLOCK cliff should flatten**. On the
> 1050 Ti, going from BLOCK=64 to BLOCK=1024 costs ~14× in time. If the L2
> explanation is right, that ratio should shrink dramatically as L2-per-thread
> rises, and `smem in=1` (input staging) should stop helping at BLOCK=256,
> because the data already fits in cache.

If the cliff is still there on an A100, my explanation is wrong and we rethink
before anyone writes anything up.
