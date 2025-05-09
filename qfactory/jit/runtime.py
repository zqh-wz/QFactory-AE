import os
import torch
import pickle
import ctypes


Python2CTypes = {
    torch.float16: ctypes.c_void_p,
    torch.int8: ctypes.c_void_p,
    torch.cuda.streams.Stream: ctypes.c_void_p,
    int: ctypes.c_int,
}

def map_ctype(arg, dtype):
    ctype = Python2CTypes[dtype]
    if isinstance(arg, torch.Tensor):
        return ctype(arg.data_ptr())
    if isinstance(arg, torch.cuda.Stream):
        return ctype(arg.cuda_stream)
    return ctype(arg)

class Runtime:
    def __init__(self, path: str):
        self.lib, self.args = None, None
        assert self.is_path_valid(path), f'Invalid Runtime Path {path}'
        
        self.lib = ctypes.CDLL(os.path.join(path, 'kernel.so'))
        self.args = pickle.load(open(os.path.join(path, 'kernel.args'), 'rb'))
    
    @staticmethod
    def is_path_valid(path: str) -> bool:
        if not os.path.exists(path) or not os.path.isdir(path):
            return False
        files = ['kernel.cu', 'kernel.args', 'kernel.so']
        return all(os.path.exists(os.path.join(path, file)) for file in files)

    def run(self, *args):
        assert len(args) == len(self.args), f'Expected {len(self.args)} arguments, got {len(args)}'
        cargs = []
        for arg, (name, dtype) in zip(args, self.args):
            if isinstance(arg, torch.Tensor):
                assert arg.dtype == dtype, f'Expected tensor dtype {dtype} for {name}, got {arg.dtype}'
            else:
                assert isinstance(arg, dtype), f'Expected built-in type {dtype} for {name}, got {type(arg)}'
            cargs.append(map_ctype(arg, dtype))

        return_code = ctypes.c_int(0)
        self.lib.launch(*cargs, ctypes.byref(return_code))
        return return_code.value

class RuntimeCache:
    def __init__(self):
        self.cache = {}

    def __getitem__(self, path: str):
        if path in self.cache:
            return self.cache[path]

        if os.path.exists(path) and Runtime.is_path_valid(path):
            runtime = Runtime(path)
            self.cache[path] = runtime
            return runtime
        return None

    def __setitem__(self, path, runtime):
        self.cache[path] = runtime
