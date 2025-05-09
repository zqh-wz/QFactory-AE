import torch
import json

from qfactory import QLinear

OK_str = "\033[92mOK\033[0m"
Err_str = "\033[91mErr\033[0m"

def acc_test(M, N, K, nbits):
    print(f"=== M={M}, N={N}, K={K}, nbits={nbits} ===")

    activation = torch.randn(M, K, dtype=torch.float16, device='cuda')
    weight = torch.randint(0, 2 ** nbits, (K, N), dtype=torch.int32, device='cuda')
    output = activation @ weight.to(dtype=torch.float16)

    qlinear = QLinear.init(M, K, N, nbits, weight, compile_flags={"enable_schedule": False, "enable_lower": False})
    qoutput = qlinear(activation)
    q_err = round(((output - qoutput) ** 2).mean().item(), 4)
    print(f"QFactory Error: {q_err} {OK_str if q_err < 0.1 else Err_str}")


if __name__ == '__main__':
    N_K_G = json.load(open("data/nkg_shapes.json"))
    for model, shapes in N_K_G.items():
        for N, K, G in shapes:
            for M in [1, 16]:
                acc_test(M=M, N=N, K=K, nbits=8)
                acc_test(M=M, N=N, K=K, nbits=4)
                acc_test(M=M, N=N, K=K, nbits=2)
