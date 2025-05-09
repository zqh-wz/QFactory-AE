#include <iostream>
#include <cuda_fp16.h>
#include <cub/cub.cuh>

#include "utils.h"

template <typename BlockReduce, int shared_size, int PACK_FACTOR>
struct SharedStorageStruct {
    BlockReduce::TempStorage temp_storage;
    alignas(16) half shared_scales[shared_size];
    alignas(16) int8_t shared_zeros[shared_size / PACK_FACTOR];
};

template <
    int M, int N, int K, int NBITS, int GS, bool SYM,
    int BLOCK_N, int BLOCK_SIZE, int NUM_ELM,
    bool EVICT_A, bool EVICT_B, bool SMEM_QUANT,
    bool DEFER_DECODE, bool COL_PARALLEL,
    typename SharedStorage, typename BlockReduce
>
__global__ void kernel_gptq(
    half *__restrict__ A,
    int8_t *__restrict__ B,
    half *__restrict__ C,
    half *__restrict__ Scale,
    int8_t *__restrict__ Zero
) {
    constexpr int PACK_FACTOR = 8 / NBITS;

    A += blockIdx.x * K;
    B += blockIdx.y * BLOCK_N * (K / PACK_FACTOR);
    C += blockIdx.x * N;

    half A_local[M][NUM_ELM];
    int8_t B_local[NUM_ELM * NBITS / 8];
    half B_decode_local[NUM_ELM];

    constexpr int num_elements_per_iter = NUM_ELM * BLOCK_SIZE;
    constexpr int num_iters_per_row = (K + num_elements_per_iter - 1) / num_elements_per_iter;
    constexpr bool split_flag = true;
    int last_num_elements = K % num_elements_per_iter - threadIdx.x * NUM_ELM;

    constexpr int num_groups_per_iter = num_elements_per_iter / GS;
    constexpr int MASK = (1 << NBITS) - 1;

    extern __shared__ char shared_memory[];
    SharedStorage &smem = *reinterpret_cast<SharedStorage*>(shared_memory);
    typename BlockReduce::TempStorage &temp_storage = smem.temp_storage;
    half *shared_scales = smem.shared_scales;
    int8_t *shared_zeros = smem.shared_zeros;

    Scale += blockIdx.y * BLOCK_N * (K / GS);
    Zero += blockIdx.y * BLOCK_N * (K / GS / PACK_FACTOR);

    if constexpr (SMEM_QUANT) {
        constexpr int total_num_groups = BLOCK_N * K / GS;
        ld_global_shared<BLOCK_SIZE>(shared_scales, Scale, total_num_groups * sizeof(half), threadIdx.x);
        ld_global_shared<BLOCK_SIZE>(shared_zeros, Zero, total_num_groups / PACK_FACTOR, threadIdx.x);
        __syncthreads();
    }

    half scale;
    int8_t zero;

    auto fetch_scale_zero = [&] (int i, int j) {
        int group_idx = i * (K / GS) + j * num_groups_per_iter + threadIdx.x * NUM_ELM / GS;
        if constexpr (SMEM_QUANT) {
            scale = shared_scales[group_idx];
        } else {
            scale = Scale[group_idx];
        }
        if constexpr (SYM) {
            zero = 1 << (NBITS - 1);
        } else {
            if constexpr (SMEM_QUANT) {
                zero = shared_zeros[group_idx / PACK_FACTOR];
            } else {
                zero = Zero[group_idx / PACK_FACTOR];
            }
            zero = (zero >> ((group_idx % PACK_FACTOR) * NBITS)) & MASK;
        }
    };

    auto compute_local = [&] (half2 &in_thread_C_local, half A_local[NUM_ELM], bool last_iter) {
        if constexpr (DEFER_DECODE) {
            decode_f16<NBITS, NUM_ELM>(B_local, B_decode_local);
            reduce_k_fused<NUM_ELM, split_flag>(scale, (half)zero, A_local, B_decode_local, in_thread_C_local, last_iter, last_num_elements);
        } else {
            decode_f16_quantized<NBITS, NUM_ELM>(B_local, B_decode_local, scale, zero);
            reduce_k<NUM_ELM, split_flag>(A_local, B_decode_local, in_thread_C_local, last_iter, last_num_elements);
        }
    };

    if constexpr (COL_PARALLEL) {
        half2 in_thread_C_local[M][BLOCK_N];
        #pragma unroll
        for (int b = 0; b < M; b++) {
            for (int i = 0; i < BLOCK_N; i++) {
                in_thread_C_local[b][i] = __float2half2_rn(0.0);
            }
        }
        #pragma unroll
        for (int j = 0; j < num_iters_per_row; j++) {
            bool last_iter = (j == num_iters_per_row - 1);
            for (int b = 0; b < M; b++)
                ld_global_reg<NUM_ELM, 16, EVICT_A, split_flag>(A_local[b], A + b * K + j * num_elements_per_iter + threadIdx.x * NUM_ELM, last_iter, last_num_elements);
            for (int i = 0; i < BLOCK_N; i++) {
                int8_t *B_cur = B + i * (K / PACK_FACTOR);
                ld_global_reg<NUM_ELM, NBITS, EVICT_B, split_flag>(B_local, B_cur + j * (num_elements_per_iter / PACK_FACTOR) + threadIdx.x * NUM_ELM / PACK_FACTOR, last_iter, last_num_elements);
                fetch_scale_zero(i, j);
                for (int b = 0; b < M; b++)
                    compute_local(in_thread_C_local[b][i], A_local[b], last_iter);
            }
        }
        for (int b = 0; b < M; b++) {
            for (int i = 0; i < BLOCK_N; i++) {
                float reduction_result = BlockReduce(temp_storage).Sum(__half2float(in_thread_C_local[b][i].x) + __half2float(in_thread_C_local[b][i].y));
                __syncthreads();
                // write back
                if (threadIdx.x == 0) {
                    C[b * N + blockIdx.y * BLOCK_N + i] = __float2half(reduction_result);
                }
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < BLOCK_N; i++) {
            half2 in_thread_C_local[M];
            for (int b = 0; b < M; b++)
                in_thread_C_local[b] = __float2half2_rn(0.0);
            int8_t *B_cur = B + i * (K / PACK_FACTOR);
            #pragma unroll
            for (int j = 0; j < num_iters_per_row; j++) {
                bool last_iter = (j == num_iters_per_row - 1);
                for (int b = 0; b < M; b++)
                    ld_global_reg<NUM_ELM, 16, EVICT_A, split_flag>(A_local[b], A + b * K + j * num_elements_per_iter + threadIdx.x * NUM_ELM, last_iter, last_num_elements);
                ld_global_reg<NUM_ELM, NBITS, EVICT_B, split_flag>(B_local, B_cur + j * (num_elements_per_iter / PACK_FACTOR) + threadIdx.x * NUM_ELM / PACK_FACTOR, last_iter, last_num_elements);
                fetch_scale_zero(i, j);
                for (int b = 0; b < M; b++)
                    compute_local(in_thread_C_local[b], A_local[b], last_iter);
            }
    
            for (int b = 0; b < M; b++) {
                float reduction_result = BlockReduce(temp_storage).Sum(__half2float(in_thread_C_local[b].x) + __half2float(in_thread_C_local[b].y));
                __syncthreads();
            
                // write back
                if (threadIdx.x == 0) {
                    C[b * N + blockIdx.y * BLOCK_N + i] = __float2half(reduction_result);
                }
            }
        }
    }
}

template <
    int M, int N, int K, int NBITS, int GS, bool SYM,
    int BLOCK_N, int BLOCK_SIZE, int NUM_ELM,
    bool EVICT_A, bool EVICT_B, bool SMEM_QUANT,
    bool DEFER_DECODE, bool COL_PARALLEL
>
int gptq_gemv_impl(
    half* A, int8_t* B, half* C,
    half* Scale, int8_t* Zero,
    cudaStream_t stream
) {
    dim3 grid_size(1, N / BLOCK_N);

    if (N % BLOCK_N != 0) {
        std::cerr << "\033[31mError: N(" << N << ") must be a multiple of BLOCK_N(" << BLOCK_N << ")\033[0m" << std::endl;
        return 1;
    }

    if (NBITS * NUM_ELM % 32 != 0) {
        std::cerr << "\033[31mError: NBITS(" << NBITS << ") * NUM_ELM(" << NUM_ELM << ") must be a multiple of 32\033[0m" << std::endl;
        return 1;
    }

    if (NUM_ELM * BLOCK_SIZE % GS != 0) {
        std::cerr << "\033[31mError: GS(" << GS << ") must be a multiple of 8 * BLOCK_SIZE(" << 8 * BLOCK_SIZE << ")\033[0m" << std::endl;
        return 1;
    }

    constexpr int PACK_FACTOR = 8 / NBITS;
    constexpr int elm_per_iter = NUM_ELM * BLOCK_SIZE;
    constexpr int padded_K = (K / elm_per_iter * elm_per_iter) + (K % elm_per_iter > 0 ? elm_per_iter : 0);
    using blockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    using sharedStorage = SharedStorageStruct<blockReduce, SMEM_QUANT ? (BLOCK_N * padded_K / GS) : 0, PACK_FACTOR>;

    kernel_gptq <
        M, N, K, NBITS, GS, SYM,
        BLOCK_N, BLOCK_SIZE, NUM_ELM,
        EVICT_A, EVICT_B, SMEM_QUANT,
        DEFER_DECODE, COL_PARALLEL,
        sharedStorage, blockReduce
    > <<<grid_size, BLOCK_SIZE, sizeof(sharedStorage), stream>>> (
        A,
        B,
        C,
        Scale,
        Zero
    );

    return 0;
}
