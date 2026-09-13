#!/usr/bin/env python3
"""
8-bit block index + warp prefix-sum.

Instead of one absolute uint32 offset per block (4 B), store:
  - one uint32 per SUPERBLOCK of 32 blocks (= one warp)   -> 0.125 B/block
  - one uint8 per block with its LENGTH in bits, minus a global minimum
                                                          -> 1.0   B/block
Total 1.125 B/block instead of 4. Each block's offset is recovered with an
exclusive prefix-sum within the warp (5 steps of __shfl_up_sync).

It fits in 8 bits because block length has little spread: at BLOCK=64,
measured on Qwen3-0.6B, it runs from 130 to 273 bits (range 143 < 256). The
builder checks that condition and fails if it does not hold.

Combined with staging the OUTPUT in shared, which was the winning variant.
"""
import numpy as np, cupy as cp
import gpu_kernels as gk

_PRE = gk._COMMON + r'''
// block offset = superblock base + exclusive sum of the lengths of the
// preceding lanes in the warp
__device__ __forceinline__ i64 block_offset(
    const u32* __restrict__ sb_base, const u8* __restrict__ len_codes,
    int b, int lane, int minlen, int n_blocks)
{
    int val = (b < n_blocks) ? ((int)len_codes[b] + minlen) : 0;
    int sum = val;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        int n = __shfl_up_sync(0xffffffffu, sum, off);
        if (lane >= off) sum += n;
    }
    return (i64)sb_base[b >> 5] + (i64)(sum - val);
}
'''

_LADDER = _PRE + r'''
extern "C" __global__ void decode_ladder_idx8(
    const u32* __restrict__ words,
    const u32* __restrict__ sb_base,
    const u8*  __restrict__ len_codes,
    u8*        __restrict__ out,
    const u8*  __restrict__ slots_flat,
    int minlen, int n_blocks, int block_size, i64 n_syms)
{
    extern __shared__ u8 s_out[];
    const int T = blockDim.x, tid = threadIdx.x;
    const int b = blockIdx.x * T + tid, lane = tid & 31;

    __shared__ u8 s_slots[16];
    if (tid < 16) s_slots[tid] = slots_flat[tid];
    __syncthreads();

    i64 p = block_offset(sb_base, len_codes, b, lane, minlen, n_blocks);

    if (b < n_blocks) {
        i64 start = (i64)b * block_size;
        i64 end   = min(start + (i64)block_size, n_syms);
        for (i64 i = start; i < end; ++i) {
            u32 w = peek32(words, p);
            int n = __clz(~w);
            u8 sym; int len;
            if      (n == 0) { sym = s_slots[0 + ((w >> 30) & 1u)]; len = 2;  }
            else if (n == 1) { sym = s_slots[2 + ((w >> 29) & 1u)]; len = 3;  }
            else if (n == 2) { sym = s_slots[4 + ((w >> 28) & 1u)]; len = 4;  }
            else if (n == 3) { sym = s_slots[6 + ((w >> 26) & 3u)]; len = 6;  }
            else             { sym = (u8)((w >> 20) & 0xFFu);       len = 12; }
            s_out[(i64)tid * block_size + (i - start)] = sym;
            p += len;
        }
    }
    __syncthreads();
    i64 outBase = (i64)blockIdx.x * T * block_size;
    i64 tot = min((i64)T * block_size, n_syms - outBase);
    for (i64 k = tid; k < tot; k += T) out[outBase + k] = s_out[k];
}
'''

_HUFF = _PRE + r'''
#define LUT_BITS %d
#define LUT_SIZE (1 << LUT_BITS)
extern "C" __global__ void decode_huffman_idx8(
    const u32* __restrict__ words,
    const u32* __restrict__ sb_base,
    const u8*  __restrict__ len_codes,
    u8*        __restrict__ out,
    const u16* __restrict__ lut,
    const u32* __restrict__ first_code,
    const int* __restrict__ first_index,
    const int* __restrict__ cnt,
    const u8*  __restrict__ sorted_syms,
    int maxlen, int minlen, int n_blocks, int block_size, i64 n_syms)
{
    extern __shared__ u8 s_out[];
    __shared__ u16 s_lut[LUT_SIZE];
    const int T = blockDim.x, tid = threadIdx.x;
    const int b = blockIdx.x * T + tid, lane = tid & 31;
    for (int k = tid; k < LUT_SIZE; k += T) s_lut[k] = lut[k];
    __syncthreads();

    i64 p = block_offset(sb_base, len_codes, b, lane, minlen, n_blocks);

    if (b < n_blocks) {
        i64 start = (i64)b * block_size;
        i64 end   = min(start + (i64)block_size, n_syms);
        for (i64 i = start; i < end; ++i) {
            u32 w = peek32(words, p);
            u16 e = s_lut[w >> (32 - LUT_BITS)];
            int len = e >> 8; u8 sym;
            if (len) { sym = (u8)(e & 0xFF); }
            else {
                sym = 0;
                for (int L = LUT_BITS + 1; L <= maxlen; ++L) {
                    u32 code = w >> (32 - L);
                    int d = (int)(code - first_code[L]);
                    if (cnt[L] > 0 && d >= 0 && d < cnt[L]) {
                        sym = sorted_syms[first_index[L] + d]; len = L; break;
                    }
                }
            }
            s_out[(i64)tid * block_size + (i - start)] = sym;
            p += len;
        }
    }
    __syncthreads();
    i64 outBase = (i64)blockIdx.x * T * block_size;
    i64 tot = min((i64)T * block_size, n_syms - outBase);
    for (i64 k = tid; k < tot; k += T) out[outBase + k] = s_out[k];
}
'''

_k = {}
def ladder_idx8():
    if "l" not in _k:
        _k["l"] = cp.RawKernel(_LADDER, "decode_ladder_idx8", backend="nvrtc")
    return _k["l"]
def huffman_idx8():
    if "h" not in _k:
        _k["h"] = cp.RawKernel(_HUFF % gk.LUT_BITS, "decode_huffman_idx8", backend="nvrtc")
    return _k["h"]


def build_index8(starts, block, total_bits, threads):
    """-> len_codes uint8[n_blocks], sb_base uint32[...], minlen, n_blocks

    sb_base is padded to cover the whole grid: the spare threads of the last
    CUDA block take part in the prefix-sum (it needs the full warp) and read
    sb_base[b>>5] before being discarded."""
    offs = np.ascontiguousarray(starts[::block]).astype(np.int64)
    n_blocks = offs.size
    ends = np.empty_like(offs)
    ends[:-1] = offs[1:]
    ends[-1] = total_bits
    blens = ends - offs
    minlen, maxlen = int(blens.min()), int(blens.max())
    if maxlen - minlen > 255:
        raise ValueError(f"block length range {maxlen-minlen} > 255; "
                         f"does not fit in 8 bits at BLOCK={block}")
    len_codes = (blens - minlen).astype(np.uint8)

    grid = (n_blocks + threads - 1) // threads
    n_sb_pad = (grid * threads + 31) // 32
    sb = np.zeros(n_sb_pad, dtype=np.uint32)
    real = offs[::32]
    sb[:real.size] = real.astype(np.uint32)
    sb[real.size:] = real[-1] if real.size else 0
    # len_codes is padded too, so nothing reads out of bounds
    lc = np.zeros(grid * threads, dtype=np.uint8)
    lc[:n_blocks] = len_codes
    return lc, sb, minlen, n_blocks, 1.0 + 4.0 / 32
