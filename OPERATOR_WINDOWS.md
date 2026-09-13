# Running the measurements on a Windows machine with an NVIDIA GPU

Written for someone lending a machine. Prepared for an **RTX 4070**, but it
works on any NVIDIA card with compute capability >= 7.0 (Turing, Ampere, Ada
or newer). Nothing about the project needs to be understood: it is copy and
paste.

- **Time:** ~40 min, of which ~30 is waiting.
- **Disk:** ~5 GB (everything can be deleted at the end).
- **Everything stays in one folder**, apart from two programs you can
  uninstall afterwards.

---

## Why this card

The question is whether performance collapses when the data stops fitting in
the GPU's L2 cache. The figure that governs it is not L2 size but **L2 per
resident thread**:

| | L2 | SMs | resident threads | **L2 / thread** |
|---|---|---|---|---|
| GTX 1050 Ti (the reference card) | 1 MB | 6 | 12,288 | 85 B |
| RTX 3060 | 3 MB | 28 | 43,008 | 73 B |
| RTX 4090 | 72 MB | 128 | 196,608 | 384 B |
| **RTX 4070** | **36 MB** | 46 | 70,656 | **534 B** |

The 4070 has **more L2 per thread than a 4090**, because it has
proportionally more cache per SM. For this specific question it is the best
card in the list, 6.3x better than the reference. A 3060 is interesting for
the opposite reason: it has 3x the L2 but 3.5x the threads, so per thread it
is slightly *worse* than the 1050 Ti.

> ### Predictions, written before measuring
>
> **RTX 4070 (or any card with a large L2 per thread):** the performance
> collapse when raising the `BLOCK` parameter **should disappear** up to
> BLOCK=1024. At 1024 the resident working set is 24.5 MB and the L2 is
> 36 MB: **it fits**. On the reference card it does not fit even at
> BLOCK=256, which is where it collapses there.
>
> **RTX 3060 (or any card near 73 B/thread):** the `BLOCK` cliff should
> appear **in the same place** — between 128 and 256 — despite 3x the L2,
> 4.7x the SMs and 3.2x the bandwidth. Working set at BLOCK=256: 3.73 MB
> against 3 MB of L2. At BLOCK=128: 1.86 MB, fits.
>
> If the measurement matches, the L2 explanation is confirmed on different
> silicon. If it does not, the explanation is false — which is also useful
> information.

Nsight Compute also supports these cards (it does not support the 1050 Ti),
so the run additionally captures what could not be measured on the
reference: real occupancy, stall reasons and warp divergence.

---

## What gets installed (and how to remove it)

| What | Why | How to remove |
|---|---|---|
| Python 3.12 | runs the program | Settings > Apps |
| NVIDIA Nsight Compute | NVIDIA's official GPU measurement tool | Settings > Apps |
| The project folder | code + a 1.4 GB AI model that gets downloaded | delete the folder |

Nothing starts with Windows, nothing runs in the background, drivers and
games are untouched.

---

## Step 1 — Install Python and Nsight Compute

Open **PowerShell** (Windows key, type `powershell`, Enter) and paste:

```powershell
winget install Python.Python.3.12
winget install Nvidia.Nsight.Compute
```

Accept whatever it asks. The second one takes a few minutes.

**Close PowerShell when done.** Otherwise it will not find Python later.

## Step 2 — Allow reading the GPU counters

Windows blocks GPU performance counters by default. Without this the detailed
profiling part fails (everything else still works).

1. Right-click the desktop > **NVIDIA Control Panel**
2. Top menu: **Desktop**
3. Tick **"Enable GPU performance counters"**
   (or *Manage GPU Performance Counters* > **allow all users**)
4. **Reboot**

> If you cannot find the option, skip it. Later you can open PowerShell
> **as administrator** (right-click > *Run as administrator*) and it works
> the same.

## Step 3 — Get the code and prepare

1. Download the repository as a zip (GitHub: *Code* > *Download ZIP*) or
   `git clone` it, into a folder with **5 GB free**, e.g. `C:\test\`
2. Enter the folder
3. Right-click **inside** the folder, on an empty spot > **Open in Terminal**
4. Paste:

```powershell
powershell -ExecutionPolicy Bypass -File setup_windows.ps1
```

This installs what is needed, **downloads a 1.4 GB model** (the slow part)
and checks that everything works.

### How to know it went well

At the end it must print:

```
RESULT: both kernels correct
```

> **If it prints `FAIL` or a red error, STOP HERE** and send a screenshot.
> Without that check the timings are worthless: a fast decompressor that
> decompresses wrongly is useless.

## Step 4 — Run all the measurements

**First close everything that uses the GPU**: browsers (Chrome, Edge),
Discord, Steam, Spotify, YouTube, games, Blender, OBS. The cleaner, the more
reliable the numbers. Under Windows the GPU clocks cannot be locked, so this
is the only thing that helps.

```powershell
powershell -ExecutionPolicy Bypass -File run_all.ps1
```

It asks you to confirm everything is closed and then takes **~25 minutes**.
It prints 5 stages:

1. base sweep (the two coding schemes, original kernel)
2. memory access-pattern variants
3. the two schemes again, with the optimised version
4. 8-bit index
5. Nsight Compute profiling

Between stages it can go a while without printing anything. That is normal.
**Do not use the PC meanwhile**; it distorts the measurements.

When it finishes it leaves a file **`results.zip`** (and the same files
unpacked under `results\<gpu-name>\`).

## Step 5 — Send it back

Only that file:

```
results.zip
```

A few KB of text. **Do not send** the model or the `.venv\` folder: they are
gigabytes and useless.

### What is in it, for whoever reads the result

- `env_info.json` — versions and card, with nothing that identifies the
  machine or the account.
- `1_bench_base.txt` — the BLOCK sweep. **This answers the question.**
- `2_bench_opt.txt` — attribution: if `smem in=1` stops helping at
  BLOCK=256, the L2 explanation is confirmed.
- `3_head2head.txt` — Huffman vs ladder with the optimised kernel.
- `4_bench_idx8.txt` — 8-bit index.
- `ncu_out\*.csv` — only if profiling worked. The decisive column is
  **`lts__t_sector_hit_rate.pct`** (L2 hit rate): if the explanation is
  right it should collapse between BLOCK=128 and BLOCK=256 on a card that
  shows the cliff, and stay flat on one that does not.

---

## If something fails

| What you see | What to do |
|---|---|
| `python is not recognized...` | you did not close PowerShell after step 1 |
| `ERR_NVGPUCTRPERM` | counters: step 2, or PowerShell as administrator |
| `ncu.exe not found` | Nsight Compute did not install; the rest is still valid |
| `Failed to find CUDA headers` | `.\.venv\Scripts\python.exe -m pip install "cupy-cuda12x[ctk]"` |
| `Windows error 32` on a DLL | wait 30 s and repeat the command; a download was still open |
| `exceeds 48 KiB` on some rows | **normal and expected**, not an error |
| Looks hung | step 3 downloads 1.4 GB; step 4 takes ~25 min |
| Anything else | stop and send the text. Nothing can break. |

## To delete everything

1. Delete the folder (`C:\test\...`)
2. Settings > Apps > uninstall **Python 3.12** and **NVIDIA Nsight
   Compute**, if you do not want to keep them

Thank you. This card answers a question that is physically impossible to
answer on the reference one.
