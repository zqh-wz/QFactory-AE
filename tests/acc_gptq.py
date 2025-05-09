import torch
import json

from qfactory import QLinear

OK_str = "\033[92mOK\033[0m"
Warn_str = "\033[93mWarn\033[0m"
Err_str = "\033[91mErr\033[0m"

def asymmetric_quant(weight, GS, nbits):
    K, N = weight.shape
    assert K % GS == 0

    qweight = torch.empty(K, N, dtype=torch.int32, device=weight.device)
    scales = torch.empty(K // GS, N, dtype=torch.float16, device=weight.device)
    zeros = torch.zeros(K // GS, N, dtype=torch.int32, device=weight.device)

    quant_range = 2 ** nbits - 1

    for k in range(K // GS):
        weight_max = weight[k * GS : (k + 1) * GS].max(axis=0).values
        weight_min = weight[k * GS : (k + 1) * GS].min(axis=0).values
        scale = (weight_max - weight_min) / quant_range
        zero = (-weight_min / (weight_max - weight_min) * quant_range - 1).clamp(min=0).to(torch.int32)
        scales[k] = scale
        zeros[k] = zero
        qweight[k * GS : (k + 1) * GS] = (weight[k * GS : (k + 1) * GS] - weight_min) / (weight_max - weight_min) * quant_range
    
    return qweight, scales, zeros

def dequant(qweight, scales, qzeros):
    K = qweight.shape[0]
    GS = K // scales.shape[0]

    k_indices = torch.arange(K, device=qweight.device)
    dequant_w = (qweight - (qzeros[k_indices // GS] + 1)) * scales[k_indices // GS]

    return dequant_w

def acc_test(M, N, K, nbits, GS):
    print(f"=== M={M}, N={N}, K={K}, nbits={nbits}, GS={GS} ===")
    if K % (GS * 8 // nbits) != 0:
        print(f"Skipping test for K={K}, GS={GS}, nbits={nbits} as K % (GS * 8 // nbits) != 0")
        return

    activation = torch.randn(M, K, dtype=torch.float16, device='cuda')
    weight = torch.randn(K, N, dtype=torch.float16, device='cuda')
    output = activation @ weight

    qweight, scales, zeros = asymmetric_quant(weight, GS, nbits)
    dequant_w = dequant(qweight, scales, zeros)
    quant_output = activation @ dequant_w

    print("Quantization Error:")
    weight_err = ((weight - dequant_w) ** 2).mean().item()
    print(f"weight MSE Error: {weight_err} {OK_str if weight_err < 0.1 else Warn_str}")
    print(f"output MSE Error: {((output - quant_output) ** 2).mean().item()}")

    qlinear = QLinear.init(M, K, N, nbits, qweight, GS, scales, zeros, compile_flags={"enable_transform": False, "enable_schedule": False, "enable_lower": False})
    qoutput = qlinear(activation)
    q_err = round(((quant_output - qoutput) ** 2).mean().item(), 4)
    print(f"QFactory Error: {q_err} {OK_str if q_err < 0.1 else Err_str}")


if __name__ == '__main__':
    N_K_G = json.load(open("data/nkg_shapes.json"))
    for model, shapes in N_K_G.items():
        for N, K, G in shapes:
            for M in [1, 16]:
                acc_test(M=M, N=N, K=K, nbits=8, GS=G)
                acc_test(M=M, N=N, K=K, nbits=4, GS=G)
                acc_test(M=M, N=N, K=K, nbits=2, GS=G)
