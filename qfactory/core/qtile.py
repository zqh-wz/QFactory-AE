class QTile:
    def __init__(
        self,
        shape,
        dtype,
        nbits,
        quant_method = "none",
        quant_granularity = None,
    ):
        self.shape, self.dtype, self.nbits = shape, dtype, nbits
        self.quant_method = quant_method
        self.quant_granularity = quant_granularity
        if quant_method != "none":
            assert len(shape) == len(quant_granularity), "Quantization granularity must match the number of dimensions"
            assert quant_granularity[0] == 1
            for s, g in zip(shape, quant_granularity):
                assert s % g == 0, f"Shape {s} must be divisible by granularity {g}"

class SymQTile(QTile):
    def __init__(self, shape, dtype, nbits, quant_granularity):
        super().__init__(shape, dtype, nbits, "sym", quant_granularity)

class AsymQTile(QTile):
    def __init__(self, shape, dtype, nbits, quant_granularity):
        super().__init__(shape, dtype, nbits, "asym", quant_granularity)
