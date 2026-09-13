#!/usr/bin/env python3
"""
Phase 2: two CUDA kernels with an IDENTICAL interface for decoding the BF16
exponent field.

  decode_huffman  -- canonical Huffman with a hierarchical LUT in shared
                     memory (primary LUT of 2^LUT_BITS entries = 4 KB, plus
                     canonical search for the long codes). DFloat11's scheme.
  decode_ladder   -- ladder code: clz + switch + shift, 10-byte table. The
                     handoff's proposal.

Both share:
  - the same coarse index (one bit offset per block of BLOCK symbols)
  - the same one-thread-per-block mapping
  - the same 32-bit window reader (peek32)
  - the same output layout (one exponent byte per weight)

The only thing that changes is the symbol decoder. That is the comparison.
"""
import numpy as np
import cupy as cp

LUT_BITS = 11          # primary LUT: 2048 entries x 2 B = 4 KB in SRAM

_COMMON = r'''
typedef unsigned int   u32;
typedef unsigned char  u8;
typedef unsigned short u16;
typedef long long      i64;

// MSB-aligned 32-bit window starting at bit p of the stream.
// words[] is big-endian: bit p lives in words[p>>5], bit 31-(p&31).
__device__ __forceinline__ u32 peek32(const u32* __restrict__ w, i64 p)
{
    i64 wi = p >> 5;
    int s = (int)(p & 31);
    u32 hi = w[wi];
    if (s == 0) return hi;
    u32 lo = w[wi + 1];
    return (hi << s) | (lo >> (32 - s));
}
'''

_LADDER_SRC = _COMMON + r'''
// Ladder (1,1,1,2): prefix = n ones + a zero, then rung_bits[n] index bits.
// n>=4 is the escape: 1111 + 8 raw bits.
// slots_flat: 10 bytes, the symbols of each rung in index order.
extern "C" __global__ void decode_ladder(
    const u32* __restrict__ words,
    const u32* __restrict__ block_bitpos,
    u8*        __restrict__ out,
    const u8*  __restrict__ slots_flat,
    int n_blocks, int block_size, i64 n_syms)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ u8 s_slots[16];
    if (threadIdx.x < 16) s_slots[threadIdx.x] = slots_flat[threadIdx.x];
    __syncthreads();          // ALL threads must reach the barrier:
    if (b >= n_blocks) return;   // the return goes after it, never before.

    i64 p     = (i64)block_bitpos[b];
    i64 start = (i64)b * block_size;
    i64 end   = min(start + (i64)block_size, n_syms);

    for (i64 i = start; i < end; ++i) {
        u32 w = peek32(words, p);
        int n = __clz(~w);                 // number of leading ones
        u8 sym; int len;
        if      (n == 0) { sym = s_slots[0 + ((w >> 30) & 1u)]; len = 2;  }
        else if (n == 1) { sym = s_slots[2 + ((w >> 29) & 1u)]; len = 3;  }
        else if (n == 2) { sym = s_slots[4 + ((w >> 28) & 1u)]; len = 4;  }
        else if (n == 3) { sym = s_slots[6 + ((w >> 26) & 3u)]; len = 6;  }
        else             { sym = (u8)((w >> 20) & 0xFFu);       len = 12; }
        out[i] = sym;
        p += len;
    }
}
'''

_HUFF_SRC = _COMMON + r'''
#define LUT_BITS %d
#define LUT_SIZE (1 << LUT_BITS)

// Canonical Huffman. Primary LUT indexed by the next LUT_BITS bits:
// entry = (length << 8) | symbol, length 0 => long code.
// Long path: canonical search with first_code/first_index/cnt per length.
extern "C" __global__ void decode_huffman(
    const u32* __restrict__ words,
    const u32* __restrict__ block_bitpos,
    u8*        __restrict__ out,
    const u16* __restrict__ lut,
    const u32* __restrict__ first_code,
    const int* __restrict__ first_index,
    const int* __restrict__ cnt,
    const u8*  __restrict__ sorted_syms,
    int maxlen, int n_blocks, int block_size, i64 n_syms)
{
    __shared__ u16 s_lut[LUT_SIZE];
    for (int k = threadIdx.x; k < LUT_SIZE; k += blockDim.x)
        s_lut[k] = lut[k];
    __syncthreads();

    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;

    i64 p     = (i64)block_bitpos[b];
    i64 start = (i64)b * block_size;
    i64 end   = min(start + (i64)block_size, n_syms);

    for (i64 i = start; i < end; ++i) {
        u32 w = peek32(words, p);
        u16 e = s_lut[w >> (32 - LUT_BITS)];
        int len = e >> 8;
        u8 sym;
        if (len) {
            sym = (u8)(e & 0xFF);
        } else {                            // long code: canonical search
            sym = 0;
            for (int L = LUT_BITS + 1; L <= maxlen; ++L) {
                u32 code = w >> (32 - L);
                int d = (int)(code - first_code[L]);
                if (cnt[L] > 0 && d >= 0 && d < cnt[L]) {
                    sym = sorted_syms[first_index[L] + d];
                    len = L;
                    break;
                }
            }
        }
        out[i] = sym;
        p += len;
    }
}
''' % LUT_BITS


# ------------------------------------------------------------------ modules
_ladder_k = None
_huff_k = None

def ladder_kernel():
    global _ladder_k
    if _ladder_k is None:
        _ladder_k = cp.RawKernel(_LADDER_SRC, "decode_ladder", backend="nvrtc")
    return _ladder_k

def huffman_kernel():
    global _huff_k
    if _huff_k is None:
        _huff_k = cp.RawKernel(_HUFF_SRC, "decode_huffman", backend="nvrtc")
    return _huff_k


# ---------------------------------------------------------- host-side tables
def build_huffman_gpu_tables(lengths, codes, lut_bits=LUT_BITS):
    maxlen = max(lengths.values())
    order = sorted(lengths, key=lambda s: (lengths[s], s))   # canonical order
    sorted_syms = np.array(order, dtype=np.uint8)

    cnt = np.zeros(33, dtype=np.int32)
    for s in order:
        cnt[lengths[s]] += 1

    first_code = np.zeros(33, dtype=np.uint32)
    first_index = np.zeros(33, dtype=np.int32)
    idx = 0
    for L in range(1, 33):
        if cnt[L]:
            first_index[L] = idx
            first_code[L] = codes[order[idx]]
            idx += int(cnt[L])

    lut = np.zeros(1 << lut_bits, dtype=np.uint16)
    for s, L in lengths.items():
        if L <= lut_bits:
            base = int(codes[s]) << (lut_bits - L)
            lut[base:base + (1 << (lut_bits - L))] = (L << 8) | s

    return dict(lut=lut, first_code=first_code, first_index=first_index,
                cnt=cnt, sorted_syms=sorted_syms, maxlen=maxlen)


def ladder_slots_flat(slots):
    """10 bytes: the symbols of each rung, in index order."""
    flat = np.zeros(16, dtype=np.uint8)
    i = 0
    for chunk in slots:
        for sym in chunk:
            flat[i] = sym
            i += 1
    return flat


# ------------------------------------------------------------- launchers
def run_ladder(d_words, d_offs, d_out, d_slots, n_blocks, block, n_syms, threads=128):
    grid = (n_blocks + threads - 1) // threads
    ladder_kernel()((grid,), (threads,),
                    (d_words, d_offs, d_out, d_slots,
                     np.int32(n_blocks), np.int32(block), np.int64(n_syms)))


def run_huffman(d_words, d_offs, d_out, t, n_blocks, block, n_syms, threads=128):
    grid = (n_blocks + threads - 1) // threads
    huffman_kernel()((grid,), (threads,),
                     (d_words, d_offs, d_out, t["d_lut"], t["d_first_code"],
                      t["d_first_index"], t["d_cnt"], t["d_sorted_syms"],
                      np.int32(t["maxlen"]), np.int32(n_blocks),
                      np.int32(block), np.int64(n_syms)))
