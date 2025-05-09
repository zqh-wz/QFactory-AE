import torch
import logging
from triton.testing import do_bench

from .qlinear import QLinear
from .profile import benchmark

logger = logging.getLogger(__name__)

try:
    import bitblas
    BITBLAS_AVAILABLE = True
except:
    BITBLAS_AVAILABLE = False
    logger.warning("BitBLAS is not available.")
try:
    import marlin
    MARLIN_AVAILABLE = True
except:
    MARLIN_AVAILABLE = False
    logger.warning("Marlin is not available.")


class LatencyTest:
    def __init__(self, func):
        self.func = func

    def latency(self):
        return benchmark(self.func)

class QFactoryLatencyTest(LatencyTest):
    def __init__(self, M, N, K, NBITS, GS, sym, compile_flags={}, base_impl=False):
        activation = torch.empty(M, K, dtype=torch.float16, device='cuda')
        qlinear = QLinear.empty(M, K, N, NBITS, GS, sym, compile_flags=compile_flags, base_impl=base_impl)
        super().__init__(func=lambda: qlinear(activation))

class CuBLASLatencyTest(LatencyTest):
    def __init__(self, M, N, K):
        activation = torch.randn(M, K, dtype=torch.float16, device='cuda')
        weight = torch.randn(K, N, dtype=torch.float16, device='cuda')
        super().__init__(lambda: torch.matmul(activation, weight))

if BITBLAS_AVAILABLE:
    class BitBLASLatencyTest(LatencyTest):
        def _get_or_create_bitblas_operator(self, config, enable_tuning):
            from bitblas.cache import global_operator_cache, get_database_path
            BITBLAS_DATABASE_PATH = get_database_path()
            BITBLAS_TARGET = bitblas.auto_detect_nvidia_target()
            if global_operator_cache.size() == 0:
                global_operator_cache.load_from_database(BITBLAS_DATABASE_PATH, BITBLAS_TARGET)
                print(f"Loaded {global_operator_cache.size()} operators from database.")

            bitblas_matmul = global_operator_cache.get(config)
            if bitblas_matmul is None:
                # should disable tuning for the first time because we may require loading bitblas operator from database.
                bitblas_matmul = bitblas.Matmul(config, target=BITBLAS_TARGET, enable_tuning=False, backend="tir")
                if enable_tuning:
                    bitblas_matmul.hardware_aware_finetune(topk=20)
                    global_operator_cache.add(config, bitblas_matmul)
                    global_operator_cache.save_into_database(BITBLAS_DATABASE_PATH, BITBLAS_TARGET)
                    print("BitBLAS Tuning done, appended operator to global_operator_cache.")
                else:
                    print("BitBLAS Operator created.")
            else:
                print("BitBLAS Operator found in global_operator_cache.")
            return bitblas_matmul

        def _get_bitblas_matmul(self, M, N, K, B, GS, SYM, enable_tuning):
            BITBLAS_WType = {
                8: "uint8",
                4: "uint4",
                2: "uint2"
            }
            if GS == -1 or GS == K:
                matmul_config = bitblas.MatmulConfig(
                    M=M, N=N, K=K,
                    A_dtype="float16",
                    W_dtype=BITBLAS_WType[B],
                    accum_dtype="float16",
                    out_dtype="float16",
                    layout="nt",
                    with_bias=False
                )
                bitblas_matmul = self._get_or_create_bitblas_operator(matmul_config, enable_tuning)
                return bitblas_matmul
            else:
                storage_dtype = "int32" if B == 8 else "int8"
                W_dtype = BITBLAS_WType[B] if SYM == 0 else BITBLAS_WType[B][1:]
                with_zeros = True if SYM == 0 else False
                gptq_config = bitblas.MatmulConfig(
                    M=M, N=N, K=K,
                    A_dtype="float16",
                    W_dtype=W_dtype,
                    accum_dtype="float16",
                    out_dtype="float16",
                    layout="nt",
                    with_bias=False,
                    group_size=GS,
                    with_scaling=True,
                    with_zeros=with_zeros,
                    zeros_mode='quantized',
                    storage_dtype=storage_dtype,
                )
                bitblas_gptq = self._get_or_create_bitblas_operator(gptq_config, enable_tuning)
                return bitblas_gptq

        def __init__(self, M, N, K, B, GS=-1, SYM=0, enable_tuning=True):
            bitblas_matmul = self._get_bitblas_matmul(M, N, K, B, GS, SYM, enable_tuning)
            x = torch.randn(M, K, dtype=torch.float16, device='cuda')
            w = torch.randint(-128, 128, (K, N), dtype=torch.int8, device='cuda')
            # w = bitblas_matmul.transform_weight(w)
            if GS == -1:
                super().__init__(lambda: bitblas_matmul.forward(x, w))
            else:
                scale = torch.randn(K // GS, N, dtype=torch.float16, device='cuda')
                if SYM == 1:
                    super().__init__(lambda: bitblas_matmul.forward(x, w, scale))
                else:
                    zero = torch.randint(-128, 128, (K // GS, N // 2), dtype=torch.int8, device='cuda')
                    super().__init__(lambda: bitblas_matmul.forward(x, w, scale, zero))

if MARLIN_AVAILABLE:
    class MarlinLatencyTest(LatencyTest):
        def _marlin_get_problem(self, m, n, k, groupsize):
            if groupsize == -1:
                groupsize = k
            A = torch.randn((m, k), dtype=torch.half, device='cuda')
            B = torch.randint(low=-2**31, high=2**31, size=(k * n // 8,), device='cuda')
            C = torch.zeros((m, n), dtype=torch.half, device='cuda')
            s = torch.zeros((k // groupsize, n), dtype=torch.half, device='cuda')
            torch.cuda.synchronize()
            return A, B, C, s

        def __init__(self, M, N, K, GS=-1):
            assert GS == -1 or GS == 128
            A, B, C, s = self._marlin_get_problem(M, N, K, GS)
            workspace = torch.zeros(C.shape[1] // 128 * 16, device='cuda')
            super().__init__(lambda: marlin.mul(A, B, C, s, workspace))
