#include <iostream>
#include <cuda_fp16.h>
#include <cub/cub.cuh>

#include "utils.h"

template <typename BlockReduce>
struct SharedStorageStruct {
    BlockReduce::TempStorage temp_storage;
};

template <
    int N, int K, int NBITS,
    int BLOCK_N, int BLOCK_SIZE, int NUM_ELM,
    bool EVICT_A, bool EVICT_B,
    bool COL_PARALLEL,
    typename SharedStorage, typename BlockReduce
>
__global__ void kernel_gemv(
    half *__restrict__ A,
    int8_t *__restrict__ B,
    half *__restrict__ C
) {
    constexpr int PACK_FACTOR = 8 / NBITS;

    A += blockIdx.x * K;
    B += blockIdx.y * BLOCK_N * (K / PACK_FACTOR);
    C += blockIdx.x * N;

    half A_local[NUM_ELM];
    int8_t B_local[NUM_ELM * NBITS / 8];
    half B_decode_local[NUM_ELM];

    constexpr int num_elements_per_iter = NUM_ELM * BLOCK_SIZE;
    constexpr int num_iters_per_row = (K + num_elements_per_iter - 1) / num_elements_per_iter;
    constexpr bool split_flag = (K % num_elements_per_iter == 0) ? false : true;
    int last_num_elements = K % num_elements_per_iter - threadIdx.x * NUM_ELM;

    extern __shared__ char shared_memory[];
    SharedStorage &smem = *reinterpret_cast<SharedStorage*>(shared_memory);
    typename BlockReduce::TempStorage &temp_storage = smem.temp_storage;

    auto compute_local = [&] (half2 &in_thread_C_local, bool last_iter) {
        decode_f16<NBITS, NUM_ELM>(B_local, B_decode_local);
        reduce_k<NUM_ELM, split_flag>(A_local, B_decode_local, in_thread_C_local, last_iter, last_num_elements);
    };

    if constexpr (COL_PARALLEL) {
        half2 in_thread_C_local[BLOCK_N];
        #pragma unroll
        for (int i = 0; i < BLOCK_N; i++) {
            in_thread_C_local[i] = __float2half2_rn(0.0);
        }
        #pragma unroll
        for (int j = 0; j < num_iters_per_row; j++) {
            bool last_iter = (j == num_iters_per_row - 1);
            ld_global_reg<NUM_ELM, 16, EVICT_A, split_flag>(A_local, A + j * num_elements_per_iter + threadIdx.x * NUM_ELM, last_iter, last_num_elements);
            for (int i = 0; i < BLOCK_N; i++) {
                int8_t *B_cur = B + i * (K / PACK_FACTOR);
                ld_global_reg<NUM_ELM, NBITS, EVICT_B, split_flag>(B_local, B_cur + j * (num_elements_per_iter / PACK_FACTOR) + threadIdx.x * NUM_ELM / PACK_FACTOR, last_iter, last_num_elements);
                compute_local(in_thread_C_local[i], last_iter);
            }
        }
        for (int i = 0; i < BLOCK_N; i++) {
            float reduction_result = BlockReduce(temp_storage).Sum(__half2float(in_thread_C_local[i].x) + __half2float(in_thread_C_local[i].y));
            __syncthreads();
            // write back
            if (threadIdx.x == 0) {
                C[blockIdx.y * BLOCK_N + i] = __float2half(reduction_result);
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < BLOCK_N; i++) {
            half2 in_thread_C_local = __float2half2_rn(0.0);
            int8_t *B_cur = B + i * (K / PACK_FACTOR);
            #pragma unroll
            for (int j = 0; j < num_iters_per_row; j++) {
                bool last_iter = (j == num_iters_per_row - 1);
                ld_global_reg<NUM_ELM, 16, EVICT_A, split_flag>(A_local, A + j * num_elements_per_iter + threadIdx.x * NUM_ELM, last_iter, last_num_elements);
                ld_global_reg<NUM_ELM, NBITS, EVICT_B, split_flag>(B_local, B_cur + j * (num_elements_per_iter / PACK_FACTOR) + threadIdx.x * NUM_ELM / PACK_FACTOR, last_iter, last_num_elements);
                compute_local(in_thread_C_local, last_iter);
            }
    
            float reduction_result = BlockReduce(temp_storage).Sum(__half2float(in_thread_C_local.x) + __half2float(in_thread_C_local.y));
            __syncthreads();
        
            // write back
            if (threadIdx.x == 0) {
                C[blockIdx.y * BLOCK_N + i] = __float2half(reduction_result);
            }
        }
    }
}

template <
    int N, int K, int NBITS,
    int BLOCK_N, int BLOCK_SIZE, int NUM_ELM,
    bool EVICT_A, bool EVICT_B,
    bool COL_PARALLEL
>
int gemv_impl(
    int M,
    half* A, int8_t* B, half* C,
    cudaStream_t stream
) {
    dim3 grid_size(M, N / BLOCK_N);

    if (N % BLOCK_N != 0) {
        std::cerr << "\033[31mError: N(" << N << ") must be a multiple of BLOCK_N(" << BLOCK_N << ")\033[0m" << std::endl;
        return 1;
    }

    if (NBITS * NUM_ELM % 32 != 0) {
        std::cerr << "\033[31mError: NBITS(" << NBITS << ") * NUM_ELM(" << NUM_ELM << ") must be a multiple of 32\033[0m" << std::endl;
        return 1;
    }

    using blockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    using sharedStorage = SharedStorageStruct<blockReduce>;

    kernel_gemv <
        N, K, NBITS,
        BLOCK_N, BLOCK_SIZE, NUM_ELM,
        EVICT_A, EVICT_B,
        COL_PARALLEL,
        sharedStorage, blockReduce
    > <<<grid_size, BLOCK_SIZE, sizeof(sharedStorage), stream>>> (
        A,
        B,
        C
    );

    return 0;
}
