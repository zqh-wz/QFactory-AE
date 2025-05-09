import time
import torch
from triton.testing import do_bench

def benchmark(func):
    do_bench(func) # warmup
    t = do_bench(func, warmup=50) / 1e3 # second
    torch.cuda.synchronize()
    time.sleep(1.0) # cool down
    return t
