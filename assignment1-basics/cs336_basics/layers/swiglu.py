import torch
from torch import nn

from cs336_basics.layers.linear import Linear
from cs336_basics.utils.silu import silu


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # W2 (SiLU(W1x) * W3x)
        X1 = silu(self.w1(x))
        X3 = self.w3(x)
        X13 = X1 * X3
        return self.w2(X13)
