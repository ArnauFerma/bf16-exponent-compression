#!/usr/bin/env python3
"""
Access-pattern optimisation of the decode kernels.

The base kernel reaches 8.2% of peak bandwidth. Two separable causes:

  INPUT: each thread walks its own bit region, so a warp scatters into 32
    independent streams. But the regions of all threads in a CUDA block are
    CONTIGUOUS -> they can be loaded cooperatively (coalesced) into shared
    memory and decoded from there.

  OUTPUT: 71% of the traffic, and it behaves worse. Each thread writes
    block_size consecutive bytes, so a warp writes 32 bytes spaced
    block_size apart: 32 distinct sectors for 32 bytes. Same solution: the
    whole CUDA block's output is one contiguous span, so accumulate it in
    shared and flush it coalesced.

4 variants (in x out) are compiled so the improvement can be ATTRIBUTED
rather than guessed.
"""
import numpy as np, cupy as cp
import gpu_kernels as gk

_SRC = gk._COMMON + r'''
#define SMEM_IN  %d
#define SMEM_OUT %d

extern "C" __global__ void decode_ladder_opt(
    const u32* __restrict__ words,
    const u32* __restrict__ block_bitpos,
    u8*        __restrict__ out,
    const u8*  __restrict__ slots_flat,
    int n_blocks, int block_size, i64 n_syms, i64 total_bits,
    int smem_words)
{
    extern __shared__ u32 smem[];
    const int T   = blockDim.x;
    const int tid = threadIdx.x;
    const int b0  = blockIdx.x * T;
    const int b   = b0 + tid;

    __shared__ u8 s_slots[16];
    if (tid < 16) s_slots[tid] = slots_flat[tid];

    const u32* src = words;
    i64 pbase = 0;
#if SMEM_IN
    // Word range covering the WHOLE CUDA block: contiguous, which is why
    // this load is coalesced.
    i64 w0 = (i64)block_bitpos[b0] >> 5;
    int bEnd = min(b0 + T, n_blocks);
    i64 pEnd = (bEnd < n_blocks) ? (i64)block_bitpos[bEnd] : total_bits;
    int nw = (int)((pEnd >> 5) - w0) + 2;
    for (int k = tid; k < nw; k += T) smem[k] = words[w0 + k];
    src = smem;
    pbase = w0 << 5;
#endif
    __syncthreads();   // unconditional: s_slots is read by every thread

#if SMEM_OUT
    // The offset only applies if the input region exists. Without this
    // guard, with SMEM_IN=0 the reserved shared is just threads*block bytes
    // and s_out would point outside it -> cudaErrorIllegalAddress.
#if SMEM_IN
    u8* s_out = (u8*)(smem + smem_words);
#else
    u8* s_out = (u8*)smem;
#endif
#endif

    if (b < n_blocks) {
        i64 p     = (i64)block_bitpos[b] - pbase;
        i64 start = (i64)b * block_size;
        i64 end   = min(start + (i64)block_size, n_syms);
        for (i64 i = start; i < end; ++i) {
            u32 w = peek32(src, p);
            int n = __clz(~w);
            u8 sym; int len;
            if      (n == 0) { sym = s_slots[0 + ((w >> 30) & 1u)]; len = 2;  }
            else if (n == 1) { sym = s_slots[2 + ((w >> 29) & 1u)]; len = 3;  }
            else if (n == 2) { sym = s_slots[4 + ((w >> 28) & 1u)]; len = 4;  }
            else if (n == 3) { sym = s_slots[6 + ((w >> 26) & 3u)]; len = 6;  }
            else             { sym = (u8)((w >> 20) & 0xFFu);       len = 12; }
#if SMEM_OUT
            s_out[(i64)tid * block_size + (i - start)] = sym;
#else
            out[i] = sym;
#endif
            p += len;
        }
    }

#if SMEM_OUT
    __syncthreads();
    // The whole CUDA block's output is contiguous: coalesced flush.
    i64 outBase = (i64)b0 * block_size;
    i64 tot = min((i64)T * block_size, n_syms - outBase);
    for (i64 k = tid; k < tot; k += T) out[outBase + k] = s_out[k];
#endif
}
'''

_cache = {}

def kernel(smem_in, smem_out):
    key = (smem_in, smem_out)
    if key not in _cache:
        _cache[key] = cp.RawKernel(_SRC % (int(smem_in), int(smem_out)),
                                   "decode_ladder_opt", backend="nvrtc")
    return _cache[key]


def max_words_per_cudablock(offs, threads, total_bits):
    """EXACT bound (not the theoretical worst case) on the word span a CUDA
    block touches. Computed on the host from the real offsets; using the
    theoretical bound (12 bits/symbol) would oversize shared ~4x and sink
    occupancy."""
    o = offs.astype(np.int64)
    starts = o[::threads]
    ends = np.empty_like(starts)
    ends[:-1] = o[threads::threads][:len(starts) - 1]
    ends[-1] = total_bits
    return int((((ends >> 5) - (starts >> 5)) + 2).max())


def launch(d_words, d_offs, d_out, d_slots, n_blocks, block, n_syms,
           total_bits, threads, smem_in, smem_out, smem_words):
    grid = (n_blocks + threads - 1) // threads
    nbytes = (smem_words * 4 if smem_in else 0) + (threads * block if smem_out else 0)
    k = kernel(smem_in, smem_out)
    k((grid,), (threads,),
      (d_words, d_offs, d_out, d_slots, np.int32(n_blocks), np.int32(block),
       np.int64(n_syms), np.int64(total_bits), np.int32(smem_words)),
      shared_mem=nbytes)
    return nbytes


# ---------------------------------------------------------------- huffman
_SRC_H = gk._COMMON + r'''
#define SMEM_IN  %d
#define SMEM_OUT %d
#define LUT_BITS %d
#define LUT_SIZE (1 << LUT_BITS)

extern "C" __global__ void decode_huffman_opt(
    const u32* __restrict__ words,
    const u32* __restrict__ block_bitpos,
    u8*        __restrict__ out,
    const u16* __restrict__ lut,
    const u32* __restrict__ first_code,
    const int* __restrict__ first_index,
    const int* __restrict__ cnt,
    const u8*  __restrict__ sorted_syms,
    int maxlen, int n_blocks, int block_size, i64 n_syms, i64 total_bits,
    int smem_words)
{
    extern __shared__ u32 smem[];
    __shared__ u16 s_lut[LUT_SIZE];
    const int T   = blockDim.x;
    const int tid = threadIdx.x;
    const int b0  = blockIdx.x * T;
    const int b   = b0 + tid;

    for (int k = tid; k < LUT_SIZE; k += T) s_lut[k] = lut[k];

    const u32* src = words;
    i64 pbase = 0;
#if SMEM_IN
    i64 w0 = (i64)block_bitpos[b0] >> 5;
    int bEnd = min(b0 + T, n_blocks);
    i64 pEnd = (bEnd < n_blocks) ? (i64)block_bitpos[bEnd] : total_bits;
    int nw = (int)((pEnd >> 5) - w0) + 2;
    for (int k = tid; k < nw; k += T) smem[k] = words[w0 + k];
    src = smem;
    pbase = w0 << 5;
#endif
    __syncthreads();

#if SMEM_OUT
#if SMEM_IN
    u8* s_out = (u8*)(smem + smem_words);
#else
    u8* s_out = (u8*)smem;
#endif
#endif

    if (b < n_blocks) {
        i64 p     = (i64)block_bitpos[b] - pbase;
        i64 start = (i64)b * block_size;
        i64 end   = min(start + (i64)block_size, n_syms);
        for (i64 i = start; i < end; ++i) {
            u32 w = peek32(src, p);
            u16 e = s_lut[w >> (32 - LUT_BITS)];
            int len = e >> 8;
            u8 sym;
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
#if SMEM_OUT
            s_out[(i64)tid * block_size + (i - start)] = sym;
#else
            out[i] = sym;
#endif
            p += len;
        }
    }

#if SMEM_OUT
    __syncthreads();
    i64 outBase = (i64)b0 * block_size;
    i64 tot = min((i64)T * block_size, n_syms - outBase);
    for (i64 k = tid; k < tot; k += T) out[outBase + k] = s_out[k];
#endif
}
'''

_cache_h = {}

def kernel_h(smem_in, smem_out):
    key = (smem_in, smem_out)
    if key not in _cache_h:
        _cache_h[key] = cp.RawKernel(
            _SRC_H % (int(smem_in), int(smem_out), gk.LUT_BITS),
            "decode_huffman_opt", backend="nvrtc")
    return _cache_h[key]


def launch_h(d_words, d_offs, d_out, t, n_blocks, block, n_syms, total_bits,
             threads, smem_in, smem_out, smem_words):
    grid = (n_blocks + threads - 1) // threads
    nbytes = (smem_words * 4 if smem_in else 0) + (threads * block if smem_out else 0)
    kernel_h(smem_in, smem_out)(
        (grid,), (threads,),
        (d_words, d_offs, d_out, t["d_lut"], t["d_first_code"], t["d_first_index"],
         t["d_cnt"], t["d_sorted_syms"], np.int32(t["maxlen"]), np.int32(n_blocks),
         np.int32(block), np.int64(n_syms), np.int64(total_bits), np.int32(smem_words)),
        shared_mem=nbytes)
    return nbytes
