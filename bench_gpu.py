#!/usr/bin/env python3
"""
Phase 2: timing of the two decode kernels.

Central question: MEMORY-bound or COMPUTE-bound? A "memory floor" kernel
that moves the same traffic without decoding is measured alongside. Close to
the floor => memory dominates and the entropy code is irrelevant. Far above
it => compute dominates and the ladder has room to win.

Method (this GPU cannot lock clocks: a consumer card under WDDM that also
drives the desktop, so the clock swings between 607 and 1923 MHz):
  - sustained warm-up per configuration to bring the clock to boost
  - SM clock sampled around each measurement
  - randomised configuration order, to decorrelate drift
  - median and IQR reported, plus SM cycles per symbol (clock-invariant)
"""
import json, random, subprocess, sys, time
import numpy as np, cupy as cp
import bitpack as bp, gpu_kernels as gk

# isdigit(): bench_gpu is imported from profile_run.py, which has its own
# dashed arguments. Without the guard the import blows up.
N        = int(sys.argv[1]) if (len(sys.argv) > 1 and sys.argv[1].isdigit()) else 64_000_000
REPEATS  = 11
WARMUP_S = 0.6

_FLOOR_SRC = gk._COMMON + r'''
extern "C" __global__ void mem_floor(
    const u32* __restrict__ words, const u32* __restrict__ block_bitpos,
    u8* __restrict__ out, int n_blocks, int block_size, i64 n_syms, i64 total_bits)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    i64 p0 = (i64)block_bitpos[b];
    i64 p1 = (b + 1 < n_blocks) ? (i64)block_bitpos[b + 1] : total_bits;
    u32 acc = 0;
    for (i64 k = (p0 >> 5); k <= (p1 >> 5) + 1; ++k) acc ^= words[k];
    i64 start = (i64)b * block_size, end = min(start + (i64)block_size, n_syms);
    for (i64 i = start; i < end; ++i) out[i] = (u8)(acc >> ((i & 3) * 8));
}
'''
_floor_k = cp.RawKernel(_FLOOR_SRC, "mem_floor", backend="nvrtc")


def to_u32_index(offs, total_bits):
    """The format spec says uint32 index. With ~600M weights the stream is
    ~1.6e9 bits, which fits in uint32 (max 4.29e9); beyond that it would need
    superblock-relative offsets."""
    if total_bits >= 2**32:
        raise ValueError(f"stream of {total_bits} bits does not fit a uint32 index")
    return np.ascontiguousarray(offs).astype(np.uint32)


def sm_clock_mhz():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        return float(o.stdout.strip().split("\n")[0])
    except Exception:
        return float("nan")


def measure(fn):
    """sustained warm-up + repetitions; returns (median_ms, iqr_ms, mhz)"""
    t_end = time.time() + WARMUP_S
    while time.time() < t_end:
        fn()
        cp.cuda.Stream.null.synchronize()   # otherwise thousands of launches queue up
    mhz = [sm_clock_mhz()]
    ts = []
    for i in range(REPEATS):
        s, e = cp.cuda.Event(), cp.cuda.Event()
        s.record(); fn(); e.record(); e.synchronize()
        ts.append(cp.cuda.get_elapsed_time(s, e))
    mhz.append(sm_clock_mhz())
    q1, med, q3 = np.percentile(ts, [25, 50, 75])
    return float(med), float(q3 - q1), float(np.nanmedian(mhz))


def main():
    props = cp.cuda.runtime.getDeviceProperties(0)
    n_sm = props["multiProcessorCount"]
    peak_bw = 2 * props["memoryClockRate"] * 1e3 * (props["memoryBusWidth"] / 8) / 1e9
    print(f"GPU: {props['name'].decode()}  SMs={n_sm}  peak={peak_bw:.1f} GB/s  "
          f"(SM clock sampled per configuration; run_all tries to lock it)\n")

    mm = np.memmap("outputs/real_weights_bf16.bin", dtype=np.uint16, mode="r")
    counts = np.load("outputs/real_exp_counts.npy")
    expo = ((np.asarray(mm[:N]) >> 7) & 0xFF).astype(np.uint8)
    print(f"symbols: {expo.size:,} (real Qwen3-0.6B)")

    co_h, lo_h, lengths, codes = bp.huffman_code_arrays(counts)
    th = gk.build_huffman_gpu_tables(lengths, codes)
    th.update({f"d_{k}": cp.asarray(th[k]) for k in
               ("lut", "first_code", "first_index", "cnt", "sorted_syms")})
    co_l, lo_l, slots, escape = bp.ladder_code_arrays(counts)
    d_slots = cp.asarray(gk.ladder_slots_flat(slots))

    for kn, kern in (("huffman", gk.huffman_kernel()), ("ladder", gk.ladder_kernel())):
        a = kern.attributes
        print(f"  {kn:8} regs/thread={a['num_regs']:3d}  shared={a['shared_size_bytes']:5d} B")

    streams = {}
    for cname, (code_of, len_of) in (("huffman", (co_h, lo_h)), ("ladder", (co_l, lo_l))):
        lens = len_of[expo].astype(np.int64)
        starts = np.cumsum(lens) - lens
        total_bits = int(starts[-1] + lens[-1])
        packed, _, _ = bp.encode_stream(expo, code_of, len_of, N)
        streams[cname] = (cp.asarray(bp.to_words(packed, total_bits)), starts, total_bits)
        print(f"  {cname:8} {total_bits/8/2**20:7.2f} MiB  ({total_bits/expo.size:.4f} bits/symbol)")
    print()

    d_out = cp.zeros(expo.size, dtype=cp.uint8)
    configs = [(b, t) for b in (64, 128, 256, 512, 1024) for t in (64, 128, 256)]
    random.Random(0).shuffle(configs)                 # decorrelate drift

    out = {}
    for block, threads in configs:
        for cname in ("huffman", "ladder", "floor"):
            src = "ladder" if cname == "floor" else cname
            d_words, starts, total_bits = streams[src]
            offs = to_u32_index(starts[::block], total_bits)
            d_offs = cp.asarray(offs)
            nb = offs.size
            grid = (nb + threads - 1) // threads
            if cname == "huffman":
                k, args = gk.huffman_kernel(), (
                    d_words, d_offs, d_out, th["d_lut"], th["d_first_code"],
                    th["d_first_index"], th["d_cnt"], th["d_sorted_syms"],
                    np.int32(th["maxlen"]), np.int32(nb), np.int32(block), np.int64(expo.size))
            elif cname == "ladder":
                k, args = gk.ladder_kernel(), (
                    d_words, d_offs, d_out, d_slots,
                    np.int32(nb), np.int32(block), np.int64(expo.size))
            else:
                k, args = _floor_k, (
                    d_words, d_offs, d_out, np.int32(nb), np.int32(block),
                    np.int64(expo.size), np.int64(total_bits))
            ms, iqr, mhz = measure(lambda: k((grid,), (threads,), args))
            moved = total_bits / 8 + expo.size + offs.nbytes
            out[(block, threads, cname)] = dict(
                ms=ms, iqr=iqr, mhz=mhz, gbs=moved / (ms * 1e-3) / 1e9,
                cyc_per_sym=ms * 1e-3 * mhz * 1e6 * n_sm / expo.size)

    hdr = (f"{'BLOCK':>6}{'thr':>5} | {'huff ms':>9}{'ladd ms':>9}{'floor':>8} | "
           f"{'cycles/symbol':>16} | {'h/floor':>8}{'l/floor':>8}{'l/h':>7}")
    print(hdr); print("-" * len(hdr))
    for block in (64, 128, 256, 512, 1024):
        for threads in (64, 128, 256):
            h, l, f = (out[(block, threads, c)] for c in ("huffman", "ladder", "floor"))
            print(f"{block:6d}{threads:5d} | {h['ms']:7.2f}±{h['iqr']:3.0f}"
                  f"{l['ms']:7.2f}±{l['iqr']:3.0f}{f['ms']:7.2f} | "
                  f"h={h['cyc_per_sym']:5.1f} l={l['cyc_per_sym']:5.1f} | "
                  f"{h['ms']/f['ms']:8.2f}{l['ms']/f['ms']:8.2f}{l['ms']/h['ms']:7.3f}")

    json.dump({f"{b}_{t}_{c}": v for (b, t, c), v in out.items()},
              open("outputs/gpu_bench.json", "w"), indent=2)
    print("\nsaved to outputs/gpu_bench.json   (ms = median of 11, ± IQR)")


if __name__ == "__main__":
    main()
