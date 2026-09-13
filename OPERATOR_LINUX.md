# Running the measurements on a Linux machine with an NVIDIA GPU

Written for someone lending a machine. Prepared for an **RTX 4070** on
Ubuntu, but it works on any NVIDIA card with compute capability >= 7.0.
Nothing about the project needs to be understood: it is copy and paste.

It is an experiment in **lossless compression of AI model weights**. There
are two ways to encode the data and the question is which decodes faster on
a GPU. The reference card is a GTX 1050 Ti (Pascal), too old for NVIDIA's
measurement tools and with only 1 MB of L2.

- **Time:** ~30 min, of which ~20 is waiting.
- **Disk:** ~5 GB (everything can be deleted at the end).
- **Everything stays inside the folder**, apart from two system packages
  (`python3-venv` and, optionally, `nsight-compute`).

---

## Why this card

What is being tested is whether performance collapses when the data stops
fitting in the GPU's L2 cache. The figure that governs it is not L2 size but
**L2 per resident thread**:

| | L2 | SMs | resident threads | **L2 / thread** |
|---|---|---|---|---|
| GTX 1050 Ti (the reference card) | 1 MB | 6 | 12,288 | 85 B |
| RTX 3060 | 3 MB | 28 | 43,008 | 73 B |
| RTX 4090 | 72 MB | 128 | 196,608 | 384 B |
| **RTX 4070** | **36 MB** | 46 | 70,656 | **534 B** |

The 4070 has more L2 per thread than a 4090, because it has proportionally
more cache per SM. For this specific question it is **the best card in the
list**, 6.3x better than the reference.

> ### Prediction, written before measuring
> On the 4070 the performance collapse when raising the `BLOCK` parameter
> **should disappear** up to BLOCK=1024. At 1024 the resident data is
> 24.5 MB and the L2 is 36 MB: **it fits**. On the reference card it does
> not fit even at BLOCK=256, which is where it collapses.
>
> If it does not collapse here either, the explanation is confirmed. If it
> collapses the same way, the explanation is false — also useful.

---

## Step 1 — Check the GPU responds

```bash
nvidia-smi
```

A table with the card's name must appear. If it says `command not found`
or errors out, the drivers are missing; stop there and say so.

Also check whether the processes column is empty. If the monitors are driven
by an integrated GPU and this card is not painting the desktop, the
measurements come out much cleaner.

## Step 2 — Install Nsight Compute (optional but very useful)

NVIDIA's official tool for measuring from the inside. Without it the
experiment still works, but half the information is lost.

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y nsight-compute
```

Check with `ncu --version`. If the binary is not on the PATH:

```bash
export PATH=$PATH:/opt/nvidia/nsight-compute/$(ls /opt/nvidia/nsight-compute | tail -1)
```

### Counter permissions

By default the driver only lets root read the counters. Two options:

- **Easy:** do nothing. The script detects the missing permission and uses
  `sudo` automatically for that part.
- **Permanent:**
  ```bash
  echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
  sudo update-initramfs -u
  ```
  and reboot.

## Step 3 — Get the code and prepare

```bash
git clone https://github.com/FixemBCN/bf16-exponent-compression.git ~/bf16
cd ~/bf16
bash setup_linux.sh
```

(Or unpack a zip of the repository into `~/bf16` if you were sent one.)

This creates an isolated Python environment (`.venv/`, it does not touch the
system Python), installs CuPy, **downloads a 1.4 GB model** (the slow part)
and checks that everything works.

### How to know it went well

At the end it must print:

```
RESULT: both kernels correct
```

> **If it prints `FAIL` or any error, STOP HERE** and send the output.
> Without that check the timings are worthless: a fast kernel that
> decompresses wrongly is useless.

## Step 4 — Run all the measurements

**First close whatever uses the GPU** (browsers playing video, games,
Blender, anything with CUDA). If the monitors run off an integrated GPU,
that is enough.

```bash
bash run_all.sh
```

It takes ~20 minutes and prints 5 stages:

1. base sweep (Huffman vs ladder, original kernel)
2. access-pattern variants
3. Huffman vs ladder with the optimisation
4. 8-bit index
5. Nsight Compute profiling

The script also tries to **lock the GPU clocks** with `nvidia-smi -lgc`
(it will ask for sudo). That removes the noise from variable frequencies;
it cannot be done under Windows, which is why the reference measurements
are noisier than these will be. If it fails, nothing is lost; it continues.

When it finishes it leaves a file **`results.tar.gz`** (and the same files
unpacked under `results/<gpu-name>/`).

## Step 5 — Send it back

Only the file:

```
~/bf16/results.tar.gz
```

A few KB of text. **Do not send** the downloaded model or the `.venv/`
folder: they are gigabytes and useless.

---

## If something fails

| What you see | What to do |
|---|---|
| `nvidia-smi: command not found` | drivers missing; say so |
| `ensurepip is not available` | `sudo apt install python3-venv python3-dev` |
| `Failed to find CUDA headers` | `./.venv/bin/python -m pip install "cupy-cuda12x[ctk]"` |
| `ERR_NVGPUCTRPERM` | counter permissions, step 2 |
| `ncu: command not found` | not installed or not on the PATH, step 2 |
| `cudaErrorIllegalAddress` | send the whole output; it is a bug in the code |
| Looks hung | step 3 downloads 1.4 GB and step 4 takes ~20 min without saying much |

None of this can touch the system outside the folder.

## To delete everything

```bash
rm -rf ~/bf16
sudo apt remove nsight-compute      # only if you do not want to keep it
```

Thank you. This answers a question that is physically impossible to answer
on the reference card.
