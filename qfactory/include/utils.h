#pragma once

#include <cuda_fp16.h>

static inline __device__ uint32_t __pack_half2(const half x, const half y) {
    uint32_t v0 = *((uint16_t*)&x);
    uint32_t v1 = *((uint16_t*)&y);
    return (v1 << 16) | v0;
}

template <int NBITS, int N>
__device__ void decode_u32_to_f16(uint32_t &ints, uint32_t *halfs) {
    static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint32_t MASK = (1 << NBITS) - 1;
    static constexpr uint32_t BOTTOM_MASK = MASK | (MASK << 16);
    static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64006400;
    static constexpr uint32_t MEDIAN_NUM = 0x64006400;
#pragma unroll
    for (int i = 0; i < (N / 2); i++) {
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                     : "=r"(halfs[i])
                     : "r"(ints >> (NBITS * i)), "n"(BOTTOM_MASK), "n"(FP16_TOP_MAGIC_NUM), "n"(immLut));
        asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(halfs[i]) : "r"(halfs[i]), "r"(MEDIAN_NUM));
    }
}

template <int NBITS, int N>
__device__ void decode_f16(int8_t *_src, half *_dst) {
    if (NBITS == 8) {
        uint8_t *src = (uint8_t*)_src;
        half *dst = (half*)_dst;
        for (int i = 0; i < N; i++) {
            dst[i] = __float2half((float)src[i]);
        }
    } else {
        uint32_t *src = (uint32_t*)_src;
        uint32_t *dst = (uint32_t*)_dst;
        for (int i = 0; i < NBITS * N / 32; i++) {
            decode_u32_to_f16<NBITS, N>(src[i], &dst[i * (16 / NBITS)]);
        }
    }
    static_assert(NBITS == 8 || NBITS == 4 || NBITS == 2, "NBITS must be 8 or 4 or 2");
    static_assert(NBITS * N % 32 == 0, "NBITS * N must be a multiple of 32");
}

template <int NBITS, int N>
__device__ void decode_u32_to_f16_scale_zeros_quantized(uint32_t &ints, uint32_t *halfs, const half scale, const int8_t zero) {
    static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint32_t MASK = (1 << NBITS) - 1;
    static constexpr uint32_t BOTTOM_MASK = MASK | (MASK << 16);
    static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64006400;
    uint32_t const packed_scales = __pack_half2(scale, scale);
    uint32_t median_num = ((0xe400 | zero) << 16) | (0xe400 | zero);
#pragma unroll
    for (int i = 0; i < (N / 2); i++) {
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                     : "=r"(halfs[i])
                     : "r"(ints >> (NBITS * i)), "n"(BOTTOM_MASK), "n"(FP16_TOP_MAGIC_NUM), "n"(immLut));
        asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(halfs[i]) : "r"(halfs[i]), "r"(median_num));
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(halfs[i]) : "r"(halfs[i]), "r"(packed_scales), "r"(0));
    }
}

template <int NBITS, int N>
__device__ void decode_f16_quantized(int8_t *_src, half *_dst, const half scale, const int8_t zero) {
    if (NBITS == 8) {
        uint8_t *src = (uint8_t*)_src;
        half *dst = (half*)_dst;
        for (int i = 0; i < N; i++) {
            dst[i] = __float2half((float)(src[i] - *(uint8_t*)(&zero))) * scale;
        }
    } else {
        uint32_t *src = (uint32_t*)_src;
        uint32_t *dst = (uint32_t*)_dst;
        for (int i = 0; i < NBITS * N / 32; i++) {
            decode_u32_to_f16_scale_zeros_quantized<NBITS, N>(src[i], &dst[i * (16 / NBITS)], scale, zero);
        }
    }
    static_assert(NBITS == 8 || NBITS == 4 || NBITS == 2, "NBITS must be 8 or 4 or 2");
    static_assert(NBITS * N % 32 == 0, "NBITS * N must be a multiple of 32");
}

template <int BITS, bool EVICT>
__device__ void ld_global_reg_aligned(void* reg_ptr, const void* global_ptr) {
    if (BITS % (4 * 32) == 0) {
        if (!EVICT) {
            #pragma unroll
            for (int i = 0; i < BITS / (4 * 32); i++) {
                ((int4*)reg_ptr)[i] = ((int4*)global_ptr)[i];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < BITS / (4 * 32); i++) {
                int4 &x = ((int4*)reg_ptr)[i];
                int4 *gptr = &((int4*)global_ptr)[i];
                asm volatile(
                    "ld.global.cs.nc.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(x.x), "=r"(x.y), "=r"(x.z), "=r"(x.w) : "l"(gptr)
                );
            }
        }
    } else if (BITS == 2 * 32) {
        if (!EVICT) {
            *((int2*)reg_ptr) = *((int2*)global_ptr);
        } else {
            int2 &x = *((int2*)reg_ptr);
            asm volatile(
                "ld.global.cs.nc.v2.u32 {%0, %1}, [%2];\n"
                : "=r"(x.x), "=r"(x.y) : "l"(global_ptr)
            );
        }
    } else if (BITS == 32) {
        if (!EVICT) {
            *((int*)reg_ptr) = *((int*)global_ptr);
        } else {
            int &x = *((int*)reg_ptr);
            asm volatile(
                "ld.global.cs.nc.u32 %0, [%1];\n"
                : "=r"(x) : "l"(global_ptr)
            );
        }
    } else if (BITS == 16) {
        if (!EVICT) {
            *((int16_t*)reg_ptr) = *((int16_t*)global_ptr);
        } else {
            int16_t &x = *((int16_t*)reg_ptr);
            asm volatile(
                "ld.global.cs.nc.u16 %0, [%1];\n"
                : "=h"(x) : "l"(global_ptr)
            );
        }
    } else {
        static_assert(BITS % (4 * 32) == 0 || BITS == 2 * 32 || BITS == 32 || BITS == 16, "BITS must be 128, 64, 32, or 16");
    }
}

template <int NUM_ELM, int BITS, bool EVICT, bool SPLIT>
__device__ void ld_global_reg(void* reg_ptr, const void* global_ptr, bool last_iter, int num_remain) {
    if (!SPLIT) {
        ld_global_reg_aligned<NUM_ELM * BITS, EVICT>(reg_ptr, global_ptr);
    } else if (!last_iter) {
        ld_global_reg_aligned<NUM_ELM * BITS, EVICT>(reg_ptr, global_ptr);
    } else if (num_remain >= NUM_ELM) {
        ld_global_reg_aligned<NUM_ELM * BITS, EVICT>(reg_ptr, global_ptr);
    } else if (num_remain > 0) {
        if ((num_remain * BITS) % 16 == 0) {
            int16_t *reg_ptr_16 = (int16_t*)reg_ptr;
            int16_t *global_ptr_16 = (int16_t*)global_ptr;
            for (int i = 0; i < (num_remain * BITS) / 16; i++) {
                reg_ptr_16[i] = global_ptr_16[i];
            }
        } else {
            int8_t *reg_ptr_8 = (int8_t*)reg_ptr;
            int8_t *global_ptr_8 = (int8_t*)global_ptr;
            for (int i = 0; i < (num_remain * BITS) / 8; i++) {
                reg_ptr_8[i] = global_ptr_8[i];
            }
        }
    }
}

template <int BLOCK_SIZE>
__device__ void ld_global_shared(void* shared_ptr, const void* global_ptr, int nbytes, int tid) {
    int stride = BLOCK_SIZE * 4;
    int8_t *shared_ptr_byte = (int8_t*)shared_ptr;
    int8_t *global_ptr_byte = (int8_t*)global_ptr;
    bool aligned_gptr = ((uintptr_t)global_ptr_byte % 4 == 0);
    for (int i = tid * 4; i < nbytes; i += stride) {
        if (i + 3 < nbytes && aligned_gptr) {
            *(int*)(&shared_ptr_byte[i]) = *(int*)(&global_ptr_byte[i]);
        } else {
            for (int j = 0; j < 4 && i + j < nbytes; j++) {
                shared_ptr_byte[i + j] = global_ptr_byte[i + j];
            }
        }
    }
}

template <int NUM_ELM, bool SPLIT>
__device__ void reduce_k(half *A, half *B, half2 &sum2, bool last_iter, int num_remain) {
    if (!SPLIT || !last_iter || num_remain >= NUM_ELM) {
        #pragma unroll
        for (int k = 0; k < NUM_ELM; k += 2) {
            sum2 = __hfma2(__halves2half2(A[k], A[k + 1]), __halves2half2(B[k], B[k + 1]), sum2);
        }
    } else if (num_remain > 0) {
        for (int k = 0; k < num_remain; k += 2) {
            sum2 = __hfma2(__halves2half2(A[k], A[k + 1]), __halves2half2(B[k], B[k + 1]), sum2);
        }
    }
}

template <int NUM_ELM, bool SPLIT>
__device__ void reduce_k_fused(half scale, half zero, half *A, half *B, half2 &sum2, bool last_iter, int num_remain) {
    half2 scale2 = __halves2half2(scale, scale), zero2 = __halves2half2(zero, zero);
    half2 product_sum2 = __float2half2_rn(0.0), A_sum2 = __float2half2_rn(0.0);
    if (!SPLIT || !last_iter || num_remain >= NUM_ELM) {
        #pragma unroll
        for (int k = 0; k < NUM_ELM; k += 2) {
            product_sum2 = __hfma2(__halves2half2(A[k], A[k + 1]), __halves2half2(B[k], B[k + 1]), product_sum2);
            A_sum2 = __hadd2(A_sum2, __halves2half2(A[k], A[k + 1]));
        }
        half A_sum = A_sum2.x + A_sum2.y;
        sum2 += (scale2 * (product_sum2 - zero2 * A_sum2));
    } else if (num_remain > 0) {
        for (int k = 0; k < num_remain; k += 2) {
            product_sum2 = __hfma2(__halves2half2(A[k], A[k + 1]), __halves2half2(B[k], B[k + 1]), product_sum2);
            A_sum2 = __hadd2(A_sum2, __halves2half2(A[k], A[k + 1]));
        }
        half A_sum = A_sum2.x + A_sum2.y;
        sum2 += (scale2 * (product_sum2 - zero2 * A_sum2));
    }
}
