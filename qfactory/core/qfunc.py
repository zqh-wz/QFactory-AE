import copy
import torch
from typing import List

Python2CppTypes = { # (raw, actual)
    int: ('int', 'int'),
    torch.float16: ('void*', 'half*'),
    torch.int8: ('void*', 'int8_t*'),
    torch.cuda.streams.Stream: ('void*', 'cudaStream_t'),
}

class QFunc:
    def __init__(self):
        pass

class QMatmulQFunc(QFunc):
    def __init__(self, qtile_a, qtile_b, layout, base_impl=False):
        super().__init__()
        self.qtile_a = qtile_a
        self.qtile_b = qtile_b
        self.layout = layout
        assert qtile_a.quant_method == "none", "Quantization method not supported for A"
        assert qtile_b.quant_method == "sym" or qtile_b.quant_method == "asym", "Quantization method of B must be 'sym' or 'asym'"
        assert layout == "nt", f"Layout {layout} is not supported"
        assert qtile_a.shape[1] == qtile_b.shape[1], f"Shape mismatch: {qtile_a.shape[1]} != {qtile_b.shape[1]}"
        
        # Template-based Code Generation
        self.parallel_strategy = "row"
        self.tiling_configs = {
            "BLOCK_N": 1,
            "BLOCK_SIZE": 128,
            "NUM_ELM": 16,
        }
        
        # Computation Transformation
        self.decode_strategy = "eager"

        # Loadpath Scheduling
        self.l2cache_a = False # cache A in L2
        self.l2cache_b = False # cache B in L2
        self.smem_quant = False # cache quantization parameters in shared memory

        self.base_impl = base_impl # use base implementation

        self.template = """
// Problem Configs
constexpr int M = {M};
constexpr int N = {N};
constexpr int K = {K};
constexpr int NBITS = {NBITS};
constexpr int GS = {GS};
constexpr int SYM = {SYM};
// Tiling Configs
constexpr int BLOCK_N = {BLOCK_N};
constexpr int BLOCK_SIZE = {BLOCK_SIZE};
constexpr int NUM_ELM = {NUM_ELM};
// Scheduling
constexpr int EVICT_A = {EVICT_A};
constexpr int EVICT_B = {EVICT_B};
constexpr int SMEM_QUANT = {SMEM_QUANT};
// Decoding Strategy
constexpr int DEFER_DECODE = {DEFER_DECODE};
// Parallel Strategy
constexpr int COL_PARALLEL = {COL_PARALLEL};

__return_code = gptq_gemv_impl <
    M, N, K, NBITS, GS, SYM,
    BLOCK_N, BLOCK_SIZE, NUM_ELM,
    EVICT_A, EVICT_B, SMEM_QUANT,
    DEFER_DECODE, COL_PARALLEL
> (
    activation, weight, output,
    scale, zero,
    stream
);
"""
    
    def arg_name_type(self):
        return [
            ('activation', torch.float16), ('weight', torch.int8), ('output', torch.float16),
            ('scale', torch.float16), ('zero', torch.int8),
            ('stream', torch.cuda.streams.Stream)
        ]

    def transform(self) -> List[QFunc]:
        ret = []
        for decode_strategy in ["eager", "deferred"]:
            qfunc = copy.deepcopy(self)
            qfunc.decode_strategy = decode_strategy
            ret.append(qfunc)
        return ret
    
    def schedule(self) -> List[QFunc]:
        ret = []
        qfunc = copy.deepcopy(self)
        qfunc.l2cache_a = True
        qfunc.l2cache_b = False
        qfunc.smem_quant = True
        ret.append(qfunc)
        # for l2cache_a in [True, False]:
        #     for l2cache_b in [True, False]:
        #         for smem_quant in [True, False]:
        #             qfunc = copy.deepcopy(self)
        #             qfunc.l2cache_a = l2cache_a
        #             qfunc.l2cache_b = l2cache_b
        #             qfunc.smem_quant = smem_quant
        #             ret.append(qfunc)
        return ret

    def lower(self) -> List[QFunc]:
        ret = []
        BLOCK_SIZE_CANDIDATES = [64, 128]
        if self.qtile_b.nbits == 8:
            BLOCK_SIZE_CANDIDATES += [256]
        if self.qtile_b.nbits == 2:
            BLOCK_SIZE_CANDIDATES += [32]
        for parallel_strategy in ["row", "col"]:
            for BLOCK_N in [1, 2, 4, 8]:
                if parallel_strategy == "col" and BLOCK_N == 1:
                    continue
                for BLOCK_SIZE in BLOCK_SIZE_CANDIDATES:
                    for NUM_ELM in [8, 16, 32]:
                        if self.qtile_b.shape[0] % BLOCK_N != 0 or (self.qtile_b.nbits * NUM_ELM) % 32 != 0:
                            continue
                        qfunc = copy.deepcopy(self)
                        qfunc.parallel_strategy = parallel_strategy
                        qfunc.tiling_configs["BLOCK_N"] = BLOCK_N
                        qfunc.tiling_configs["BLOCK_SIZE"] = BLOCK_SIZE
                        qfunc.tiling_configs["NUM_ELM"] = NUM_ELM
                        ret.append(qfunc)
        return ret

    def emit(self) -> str:
        code = '// QFactory Auto-generated Code\n\n'
        code += '#include "gptq_gemv.h"\n\n' if not self.base_impl else '#include "gptq_gemv_base.h"\n\n'

        args_name_type = self.arg_name_type()

        # launch function signature
        code += f'extern "C" void launch (\n'
        for name, t in args_name_type:
            raw_name = f'_{name}' if Python2CppTypes[t][0] != Python2CppTypes[t][1] else name
            code += f'\t{Python2CppTypes[t][0]} {raw_name},\n'
        code += '\tint& __return_code\n'
        code += ') {\n'

        # cast arguments
        for name, t in args_name_type:
            if Python2CppTypes[t][0] != Python2CppTypes[t][1]:
                raw_name = f'_{name}'
                code += f'\t{Python2CppTypes[t][1]} {name} = reinterpret_cast<{Python2CppTypes[t][1]}>({raw_name});\n'
        code += "\n"

        def cpp_parse(template: str, keys: dict) -> str:
            new_template = copy.deepcopy(template)
            for key, value in keys.items():
                new_template = new_template.replace(f'{{{key}}}', f'{value}')
            return new_template

        # function body
        body = cpp_parse(self.template, {
            "M": self.qtile_a.shape[0],
            "N": self.qtile_b.shape[0],
            "K": self.qtile_b.shape[1],
            "NBITS": self.qtile_b.nbits,
            "GS": self.qtile_b.quant_granularity[1],
            "SYM": 1 if self.qtile_b.quant_method == "sym" else 0,
            "BLOCK_N": self.tiling_configs["BLOCK_N"],
            "BLOCK_SIZE": self.tiling_configs["BLOCK_SIZE"],
            "NUM_ELM": self.tiling_configs["NUM_ELM"],
            "EVICT_A": 0 if self.l2cache_a else 1,
            "EVICT_B": 0 if self.l2cache_b else 1,
            "SMEM_QUANT": 1 if self.smem_quant else 0,
            "DEFER_DECODE": 1 if self.decode_strategy == "deferred" else 0,
            "COL_PARALLEL": 1 if self.parallel_strategy == "col" else 0,
        })
        code += '\n'.join([('\t' if line else '') + line for line in body.split('\n')])

        code += '}\n'
        
        return code

class MatmulQFunc(QFunc):
    def __init__(self, qtile_a, qtile_b, layout):
        super().__init__()
        self.qtile_a = qtile_a
        self.qtile_b = qtile_b
        self.layout = layout
        assert qtile_a.quant_method == "none", "Quantization method not supported for A"
        assert qtile_b.quant_method == "none", "Quantization method not supported for B"
        assert layout == "nt", f"Layout {layout} is not supported"
        assert qtile_a.shape[1] == qtile_b.shape[1], f"Shape mismatch: {qtile_a.shape[1]} != {qtile_b.shape[1]}"
        
        # Template-based Code Generation
        self.parallel_strategy = "row"
        self.tiling_configs = {
            "BLOCK_N": 1,
            "BLOCK_SIZE": 128,
            "NUM_ELM": 16,
        }

        # Loadpath Scheduling
        self.l2cache_a = True # cache A in L2
        self.l2cache_b = False # cache B in L2

        self.template = """
// Problem Configs
constexpr int N = {N};
constexpr int K = {K};
constexpr int NBITS = {NBITS};
// Tiling Configs
constexpr int BLOCK_N = {BLOCK_N};
constexpr int BLOCK_SIZE = {BLOCK_SIZE};
constexpr int NUM_ELM = {NUM_ELM};
// Scheduling
constexpr int EVICT_A = {EVICT_A};
constexpr int EVICT_B = {EVICT_B};
// Parallel Strategy
constexpr int COL_PARALLEL = {COL_PARALLEL};

__return_code = gemv_impl <
    N, K, NBITS,
    BLOCK_N, BLOCK_SIZE, NUM_ELM,
    EVICT_A, EVICT_B,
    COL_PARALLEL
> (
    m,
    activation, weight, output,
    stream
);
"""
    
    def arg_name_type(self):
        return [
            ('m', int),
            ('activation', torch.float16), ('weight', torch.int8), ('output', torch.float16),
            ('stream', torch.cuda.streams.Stream)
        ]

    def schedule(self) -> List[QFunc]:
        ret = []
        for l2cache_a in [True, False]:
            for l2cache_b in [True, False]:
                qfunc = copy.deepcopy(self)
                qfunc.l2cache_a = l2cache_a
                qfunc.l2cache_b = l2cache_b
                ret.append(qfunc)
        return ret

    def lower(self) -> List[QFunc]:
        ret = []
        BLOCK_SIZE_CANDIDATES = [64, 128]
        if self.qtile_b.nbits == 8:
            BLOCK_SIZE_CANDIDATES += [256]
        if self.qtile_b.nbits == 2:
            BLOCK_SIZE_CANDIDATES += [32]
        for parallel_strategy in ["row", "col"]:
            for BLOCK_N in [1, 2, 4, 8]:
                if parallel_strategy == "col" and BLOCK_N == 1:
                    continue
                for BLOCK_SIZE in BLOCK_SIZE_CANDIDATES:
                    for NUM_ELM in [8, 16, 32]:
                        if self.qtile_b.shape[0] % BLOCK_N != 0 or (self.qtile_b.nbits * NUM_ELM) % 32 != 0:
                            continue
                        qfunc = copy.deepcopy(self)
                        qfunc.parallel_strategy = parallel_strategy
                        qfunc.tiling_configs["BLOCK_N"] = BLOCK_N
                        qfunc.tiling_configs["BLOCK_SIZE"] = BLOCK_SIZE
                        qfunc.tiling_configs["NUM_ELM"] = NUM_ELM
                        ret.append(qfunc)
        return ret

    def emit(self) -> str:
        code = '// QFactory Auto-generated Code\n\n'
        code += '#include "gemv.h"\n\n'

        args_name_type = self.arg_name_type()

        # launch function signature
        code += f'extern "C" void launch (\n'
        for name, t in args_name_type:
            raw_name = f'_{name}' if Python2CppTypes[t][0] != Python2CppTypes[t][1] else name
            code += f'\t{Python2CppTypes[t][0]} {raw_name},\n'
        code += '\tint& __return_code\n'
        code += ') {\n'

        # cast arguments
        for name, t in args_name_type:
            if Python2CppTypes[t][0] != Python2CppTypes[t][1]:
                raw_name = f'_{name}'
                code += f'\t{Python2CppTypes[t][1]} {name} = reinterpret_cast<{Python2CppTypes[t][1]}>({raw_name});\n'
        code += "\n"

        def cpp_parse(template: str, keys: dict) -> str:
            new_template = copy.deepcopy(template)
            for key, value in keys.items():
                new_template = new_template.replace(f'{{{key}}}', f'{value}')
            return new_template

        # function body
        body = cpp_parse(self.template, {
            "N": self.qtile_b.shape[0],
            "K": self.qtile_b.shape[1],
            "NBITS": self.qtile_b.nbits,
            "BLOCK_N": self.tiling_configs["BLOCK_N"],
            "BLOCK_SIZE": self.tiling_configs["BLOCK_SIZE"],
            "NUM_ELM": self.tiling_configs["NUM_ELM"],
            "EVICT_A": 0 if self.l2cache_a else 1,
            "EVICT_B": 0 if self.l2cache_b else 1,
            "COL_PARALLEL": 1 if self.parallel_strategy == "col" else 0,
        })
        code += '\n'.join([('\t' if line else '') + line for line in body.split('\n')])

        code += '}\n'
        
        return code
