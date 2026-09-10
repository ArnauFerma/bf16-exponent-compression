#!/usr/bin/env python3
"""
Fase 2: dos kernels CUDA con interfaz IDENTICA para decodificar el campo
exponente de BF16.

  decode_huffman  -- Huffman canonico con LUT jerarquica en shared memory
                     (LUT primaria de 2^LUT_BITS entradas = 4 KB, mas
                     busqueda canonica para los codigos largos). Es el
                     esquema de DFloat11.
  decode_ladder   -- codigo por escalones: clz + switch + shift, tabla de
                     10 bytes. La propuesta del handoff.

Ambos comparten:
  - el mismo indice grueso (un offset en bits por bloque de BLOCK simbolos)
  - el mismo mapeo un-hilo-por-bloque
  - el mismo lector de ventana de 32 bits (peek32)
  - el mismo layout de salida (un byte de exponente por peso)

Lo unico que cambia es el decodificador de simbolo. Esa es la comparacion.
"""
import numpy as np
import cupy as cp

LUT_BITS = 11          # LUT primaria: 2048 entradas x 2 B = 4 KB en SRAM

_COMMON = r'''
typedef unsigned int   u32;
typedef unsigned char  u8;
typedef unsigned short u16;
typedef long long      i64;

// Ventana de 32 bits alineada a MSB empezando en el bit p del stream.
// words[] esta en big-endian: el bit p vive en words[p>>5], bit 31-(p&31).
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
// Escalera (1,1,1,2): prefijo = n unos + un cero, luego rung_bits[n] bits
// de indice. n>=4 es el escape: 1111 + 8 bits crudos.
// slots_flat: 10 bytes, los simbolos de cada escalon en orden de indice.
extern "C" __global__ void decode_ladder(
    const u32* __restrict__ words,
    const u32* __restrict__ block_bitpos,
    u8*        __restrict__ out,
    const u8*  __restrict__ slots_flat,
    int n_blocks, int block_size, i64 n_syms)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;

    __shared__ u8 s_slots[16];
    if (threadIdx.x < 16) s_slots[threadIdx.x] = slots_flat[threadIdx.x];
    __syncthreads();

    i64 p     = (i64)block_bitpos[b];
    i64 start = (i64)b * block_size;
    i64 end   = min(start + (i64)block_size, n_syms);

    for (i64 i = start; i < end; ++i) {
        u32 w = peek32(words, p);
        int n = __clz(~w);                 // numero de unos iniciales
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

// Huffman canonico. LUT primaria indexada por los LUT_BITS siguientes bits:
// entrada = (longitud << 8) | simbolo, longitud 0 => codigo largo.
// Camino largo: busqueda canonica con first_code/first_index/cnt por longitud.
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
        } else {                            // codigo largo: busqueda canonica
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


# ------------------------------------------------------------------ modulos
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


# ------------------------------------------------------- tablas para el host
def build_huffman_gpu_tables(lengths, codes, lut_bits=LUT_BITS):
    maxlen = max(lengths.values())
    order = sorted(lengths, key=lambda s: (lengths[s], s))   # orden canonico
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
    """10 bytes: los simbolos de cada escalon, en orden de indice."""
    flat = np.zeros(16, dtype=np.uint8)
    i = 0
    for chunk in slots:
        for sym in chunk:
            flat[i] = sym
            i += 1
    return flat


# ------------------------------------------------------------- lanzadores
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
