import torch

def _check_weight(K, N, weight, nbits):
    assert weight.shape == (K, N)
    assert weight.dtype == torch.int32
    assert weight.min() >= 0
    assert weight.max() < 2 ** nbits

def _check_scale(K, N, GS, scale):
    assert scale.shape == (K // GS, N)
    assert scale.dtype == torch.float16

def _check_zero(K, N, GS, zero, nbits):
    assert zero.shape == (K // GS, N)
    assert zero.dtype == torch.int32
    assert zero.min() >= 0
    assert zero.max() < 2 ** nbits

def _check_gptq_gemv(args, params):
    if len(params) != 3:
        raise ValueError("gptq_gemv requires 3 parameters")
    weight, scale, zero = params

    K = args["K"]
    N = args["N"]
    NBITS = args["NBITS"]
    GS = args["GS"]

    _check_weight(K, N, weight, NBITS)
    _check_scale(K, N, GS, scale)
    if zero is not None:
        _check_zero(K, N, GS, zero, NBITS)

    return K, N, NBITS, GS, weight, scale, zero

def _check_gemv(args, params):
    if len(params) != 1:
        raise ValueError("gemv requires 1 parameters")
    weight = params[0]

    K = args["K"]
    N = args["N"]
    NBITS = args["NBITS"]

    _check_weight(K, N, weight, NBITS)

    return K, N, NBITS, weight

PERMUTE_ORDER = {
    8: [0, 1, 2, 3],
    4: [0, 2, 4, 6, 1, 3, 5, 7],
    2: [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15],
}

def _transform_weight_basic(K, N, weight, nbits):
    pack_factor = 32 // nbits
    qweight_permuted = torch.zeros(K // pack_factor, N, dtype=torch.int32, device=weight.device)
    for i in range(K // pack_factor):
        qws = [weight[i * pack_factor + j] for j in range(pack_factor)]
        qweight_permuted[i] = 0
        for j in range(pack_factor):
            qweight_permuted[i] |= qws[PERMUTE_ORDER[nbits][j]] << (j * nbits)
    
    w = torch.empty(K // (8 // nbits), N, dtype=torch.int8, device=weight.device)
    for i in range(w.shape[0]):
        w[i] = (qweight_permuted[i // 4] >> (i % 4 * 8)) & 0xff
    w = w.T.contiguous()
    return w

def _transform_scale(K, N, GS, scale):
    s = scale.T.contiguous()
    return s

def _transform_zero(K, N, GS, zero, nbits):
    pack_factor = 8 // nbits
    zero = (zero + 1).clamp(max=(2 ** nbits - 1))
    z = torch.zeros(zero.shape[0] // pack_factor, zero.shape[1], dtype=torch.int8, device=zero.device)
    for i in range(zero.shape[0]):
        z[i // pack_factor] |= zero[i] << ((i % pack_factor) * nbits)
    z = z.T.contiguous()
    return z


def convert_params(args, params):
    if "GS" not in args:
        K, N, NBITS, weight = _check_gemv(args, params)
        w = _transform_weight_basic(K, N, weight, NBITS)
        return w
    else:
        K, N, NBITS, GS, weight, scale, zero = _check_gptq_gemv(args, params)
        w = _transform_weight_basic(K, N, weight, NBITS)
        s = _transform_scale(K, N, GS, scale)
        if zero is not None:
            z = _transform_zero(K, N, GS, zero, NBITS)
        else:
            z = None
        return w, s, z
