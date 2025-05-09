import torch
from .convert import convert_params
from .core import QMatmul, Matmul

class GPTQLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        n_bits,
        weight,
        group_size,
        scale,
        zero,
        bias,
        compile_flags,
        skip_convert,
        batch_size,
        base_impl=False,
    ):
        super().__init__()

        if bias:
            raise NotImplementedError("Bias is not supported")
        
        if skip_convert:
            w, s, z = weight, scale, zero
        else:
            w, s, z = convert_params({
                "K": in_features,
                "N": out_features,
                "NBITS": n_bits,
                "GS": group_size,
            }, (weight, scale, zero))
        
        assert in_features % (group_size * 2) == 0
        assert w.dtype == torch.int8
        assert s.dtype == torch.float16
        assert len(w.shape) == 2
        assert w.shape[0] * w.shape[1] == in_features * out_features // (8 // n_bits)
        assert s.shape == (out_features, in_features // group_size)

        if z is not None:
            assert z.dtype == torch.int8
            assert z.shape == (out_features, in_features // group_size // (8 // n_bits))
        
        self.w, self.s, self.z = w, s, z

        matmul = QMatmul(batch_size, out_features, in_features, n_bits, z is None, torch.float16, torch.int8, group_size, base_impl=base_impl)
        self.func = matmul.compile_profile(**compile_flags)

    def forward(self, x: torch.Tensor):
        output = torch.empty(x.shape[0], self.w.shape[0], dtype=torch.float16, device=x.device)
        if self.z is not None:
            self.func(x, self.w, output, self.s, self.z)
        else:
            self.func(x, self.w, output, self.s)
        return output

class GEMVLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        n_bits,
        weight,
        bias,
        compile_flags,
        skip_convert,
    ):
        super().__init__()

        if bias:
            raise NotImplementedError("Bias is not supported")
        
        if skip_convert:
            w = weight
        else:
            w = convert_params({
                "K": in_features,
                "N": out_features,
                "NBITS": n_bits,
            }, (weight, ))
        
        assert w.dtype == torch.int8
        assert len(w.shape) == 2
        assert w.shape[0] * w.shape[1] == in_features * out_features // (8 // n_bits)
        
        self.w = w

        matmul = Matmul(out_features, in_features, n_bits, torch.float16, torch.int8)
        self.func = matmul.compile_profile(**compile_flags)

    def forward(self, x: torch.Tensor):
        output = torch.empty(x.shape[0], self.w.shape[0], dtype=torch.float16, device=x.device)
        self.func(x, self.w, output)
        return output

class QLinear():
    @staticmethod
    def init(
        batch_size,
        in_features,
        out_features,
        n_bits,
        weight,
        group_size=-1,
        scale=None,
        zero=None,
        bias=False,
        compile_flags={}
    ):
        if group_size == -1:
            return GEMVLinear(
                in_features,
                out_features,
                n_bits,
                weight,
                bias,
                compile_flags,
                skip_convert=False
            )
        else:
            return GPTQLinear(
                in_features,
                out_features,
                n_bits,
                weight,
                group_size,
                scale,
                zero,
                bias,
                compile_flags,
                skip_convert=False,
                batch_size=batch_size
            )

    @staticmethod
    def empty(M, K, N, NBITS, GS, sym, compile_flags, base_impl=False):
        w = torch.empty(N, K // (8 // NBITS), dtype=torch.int8, device='cuda')
        if GS == -1:
            return GEMVLinear(K, N, NBITS, w, False, compile_flags, skip_convert=True)
        s = torch.empty(N, K // GS, dtype=torch.float16, device='cuda')
        z = torch.empty(N, K // GS // (8 // NBITS), dtype=torch.int8, device='cuda')
        return GPTQLinear(K, N, NBITS, w, GS, s, None if sym else z, False, compile_flags, skip_convert=True, batch_size=M, base_impl=base_impl)
