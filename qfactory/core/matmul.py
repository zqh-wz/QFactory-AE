import os
import json
import torch
import logging
from itertools import chain

from .qtile import QTile, SymQTile, AsymQTile
from .qfunc import QMatmulQFunc, MatmulQFunc
from ..jit import jit
from ..profile import benchmark

logger = logging.getLogger(__name__)

class QMatmul:
    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        nbits: int,
        sym: bool,
        dtype_a,
        dtype_b,
        group_size: int = -1,
        layout: str = "nt",
        base_impl: bool = False,
    ):
        self.m, self.n, self.k = m, n, k
        self.nbits, self.sym = nbits, sym
        self.dtype_a, self.dtype_b = dtype_a, dtype_b
        self.group_size = k if group_size == -1 else group_size
        self.layout = layout
        self.base_impl = base_impl
        assert self.k % self.group_size == 0, f"Group size {self.group_size} must divide k {self.k}"
        assert self.layout == "nt", f"Layout {layout} is not supported"
        assert dtype_a == torch.float16, f"Data type {dtype_a} is not supported"
        assert dtype_b == torch.int8, f"Data type {dtype_b} is not supported"
        
    def init_qfuncs(self):
        if self.layout == "nt":
            ClassTileB = SymQTile if self.sym else AsymQTile
            qfunc = QMatmulQFunc(
                QTile((self.m, self.k), self.dtype_a, 16),
                ClassTileB((self.n, self.k), self.dtype_b, self.nbits, (1, self.group_size)),
                "nt",
                self.base_impl,
            )
        else:
            raise NotImplementedError(f"Layout {self.layout} is not supported")
        self.qfuncs = [qfunc]
    
    def get_dummy_inputs(self):
        activation = torch.empty((self.m, self.k), dtype=self.dtype_a, device='cuda')
        weight = torch.empty((self.n, self.k // (8 // self.nbits)), dtype=self.dtype_b, device='cuda')
        output = torch.empty((self.m, self.n), dtype=self.dtype_a, device='cuda')
        scale = torch.empty((self.n, self.k // self.group_size), dtype=self.dtype_a, device='cuda')
        zero = torch.empty((self.n, self.k // (8 // self.nbits) // self.group_size), dtype=self.dtype_b, device='cuda')
        return activation, weight, output, scale, zero, torch.cuda.current_stream()

    def compile_profile(
        self,
        enable_transform: bool = False,
        enable_schedule: bool = False,
        enable_lower: bool = False,
    ):
        self.init_qfuncs()

        predefined_config_path = os.getenv("QFACTORY_CONFIG")
        config = None
        if predefined_config_path is not None and os.path.exists(predefined_config_path):
            config = json.load(open(predefined_config_path))
        shape = f"{self.nbits}-{self.m}-{self.n}-{self.k}-{self.group_size}"
        if config is not None and shape in config:
            logger.info(f"[Profile] Using predefined config for {shape}")
            qfunc = self.qfuncs[0]
            for key, value in config[shape].items():
                qfunc.__setattr__(key, value)
            self.qfuncs = [qfunc]
        else:
            if enable_transform:
                self.qfuncs = list(chain.from_iterable(
                    qfunc.transform() for qfunc in self.qfuncs
                ))
            if enable_schedule:
                self.qfuncs = list(chain.from_iterable(
                    qfunc.schedule() for qfunc in self.qfuncs
                ))
            if enable_lower:
                self.qfuncs = list(chain.from_iterable(
                    qfunc.lower() for qfunc in self.qfuncs
                ))
        
        from concurrent.futures import ThreadPoolExecutor, as_completed

        self.runtimes = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(jit.compile, qfunc) for qfunc in self.qfuncs]
            for future in as_completed(futures):
                self.runtimes.append(future.result())
        
        dummy_inputs = self.get_dummy_inputs()
        best_runtime, best_time = None, None
        if len(self.runtimes) == 1:
            best_runtime = self.runtimes[0]
            best_time = "[skip]"
        else:
            for runtime in self.runtimes:
                ret_code = runtime.run(*dummy_inputs)
                if ret_code != 0:
                    continue
                time = benchmark(lambda : runtime.run(*dummy_inputs))
                if best_time is None or time < best_time:
                    best_runtime, best_time = runtime, time
                logger.debug(f"[Profile] Kernel: {time * 1e6} us")
        assert best_runtime is not None, f"No valid runtime found"
        if len(self.runtimes) > 1:
            logger.info(f"[Profile] Best latency: {best_time * 1e6:.2f} us")

        def wrapper(runtime):
            pack_factor = 8 // self.nbits
            zero_empty = torch.empty((self.n, self.k // pack_factor // self.group_size), dtype=torch.int8, device='cuda')

            def inner(activation, weight, output, scale, zero):
                # tensor shape check
                assert activation.shape[1] == self.k
                assert weight.shape == (self.n, self.k // pack_factor)
                assert output.shape[1] == self.n
                assert activation.shape[0] == output.shape[0]
                assert scale.shape == (self.n, self.k // self.group_size)
                assert zero.shape == (self.n, self.k // pack_factor // self.group_size)
                
                # tensor dtype check
                assert activation.dtype == torch.float16
                assert weight.dtype == torch.int8
                assert output.dtype == torch.float16
                assert scale.dtype == torch.float16
                assert zero.dtype == torch.int8
                
                # contiguous check
                assert activation.is_contiguous()
                assert weight.is_contiguous()
                assert output.is_contiguous()
                assert scale.is_contiguous()
                assert zero.is_contiguous()

                return runtime.run(activation, weight, output, scale, zero, torch.cuda.current_stream())
            
            def inner_sym(activation, weight, output, scale):
                return inner(activation, weight, output, scale, zero_empty)
            
            return inner_sym if self.sym else inner

        return wrapper(best_runtime)

class Matmul:
    def __init__(
        self,
        n: int,
        k: int,
        nbits: int,
        dtype_a,
        dtype_b,
        layout: str = "nt",
    ):
        self.n, self.k = n, k
        self.nbits = nbits
        self.dtype_a, self.dtype_b = dtype_a, dtype_b
        self.layout = layout
        assert self.layout == "nt", f"Layout {layout} is not supported"
        assert dtype_a == torch.float16, f"Data type {dtype_a} is not supported"
        assert dtype_b == torch.int8, f"Data type {dtype_b} is not supported"
        
    def init_qfuncs(self):
        if self.layout == "nt":
            qfunc = MatmulQFunc(
                QTile((0, self.k), self.dtype_a, 16),
                QTile((self.n, self.k), self.dtype_b, self.nbits),
                "nt",
            )
        else:
            raise NotImplementedError(f"Layout {self.layout} is not supported")
        self.qfuncs = [qfunc]
    
    def get_dummy_inputs(self):
        m = 1
        activation = torch.empty((m, self.k), dtype=self.dtype_a, device='cuda')
        weight = torch.empty((self.n, self.k // (8 // self.nbits)), dtype=self.dtype_b, device='cuda')
        output = torch.empty((m, self.n), dtype=self.dtype_a, device='cuda')
        return m, activation, weight, output, torch.cuda.current_stream()

    def compile_profile(
        self,
        enable_transform: bool = False,
        enable_schedule: bool = False,
        enable_lower: bool = False,
    ):
        self.init_qfuncs()

        predefined_config_path = os.getenv("QFACTORY_CONFIG")
        config = None
        if predefined_config_path is not None and os.path.exists(predefined_config_path):
            config = json.load(open(predefined_config_path))
        shape = f"{self.nbits}-{1}-{self.n}-{self.k}-{-1}"
        if config is not None and shape in config:
            logger.info(f"[Profile] Using predefined config for {shape}")
            qfunc = self.qfuncs[0]
            for key, value in config[shape].items():
                qfunc.__setattr__(key, value)
            self.qfuncs = [qfunc]
        else:
            if enable_schedule:
                self.qfuncs = list(chain.from_iterable(
                    qfunc.schedule() for qfunc in self.qfuncs
                ))
            if enable_lower:
                self.qfuncs = list(chain.from_iterable(
                    qfunc.lower() for qfunc in self.qfuncs
                ))
        
        from concurrent.futures import ThreadPoolExecutor, as_completed

        self.runtimes = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(jit.compile, qfunc) for qfunc in self.qfuncs]
            for future in as_completed(futures):
                self.runtimes.append(future.result())
        
        dummy_inputs = self.get_dummy_inputs()
        best_runtime, best_time = None, None
        if len(self.runtimes) == 1:
            best_runtime = self.runtimes[0]
            best_time = "[skip]"
        else:
            for runtime in self.runtimes:
                ret_code = runtime.run(*dummy_inputs)
                if ret_code != 0:
                    continue
                time = benchmark(lambda : runtime.run(*dummy_inputs))
                if best_time is None or time < best_time:
                    best_runtime, best_time = runtime, time
                logger.debug(f"[Profile] Kernel: {time * 1e6} us")
        assert best_runtime is not None, f"No valid runtime found"
        if len(self.runtimes) > 1:
            logger.info(f"[Profile] Best latency: {best_time * 1e6:.2f} us")

        def wrapper(runtime):
            pack_factor = 8 // self.nbits

            def inner(activation, weight, output):
                # tensor shape check
                m = activation.shape[0]
                assert activation.shape == (m, self.k)
                assert weight.shape == (self.n, self.k // pack_factor)
                assert output.shape == (m, self.n)
                
                # tensor dtype check
                assert activation.dtype == torch.float16
                assert weight.dtype == torch.int8
                assert output.dtype == torch.float16
                
                # contiguous check
                assert activation.is_contiguous()
                assert weight.is_contiguous()
                assert output.is_contiguous()

                return runtime.run(m, activation, weight, output, torch.cuda.current_stream())
            
            return inner

        return wrapper(best_runtime)
