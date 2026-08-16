import torch
from torch import nn
from einops import rearrange, einsum


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
        self.device = device
        self.dtype = dtype

        self.W1 = nn.Parameter(
            torch.empty(
                d_ff,
                d_model,
                device=device,
                dtype=dtype,
            )
        )
        self.W3 = nn.Parameter(
            torch.empty(
                d_ff,
                d_model,
                device=device,
                dtype=dtype,
            )
        )

        self.W2 = nn.Parameter(
            torch.empty(
                d_model,
                d_ff,
                device=device,
                dtype=dtype,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # W2 (SiLU(W1x) * W3x)
        def SiLU(x: torch.Tensor) -> torch.Tensor:
            return x * torch.sigmoid(x)

        X1 = SiLU(einsum(x, self.W1, "... I, FF I -> ... FF"))
        X3 = einsum(x, self.W3, "... I, FF I -> ... FF")
        X13 = X1 * X3
        return einsum(X13, self.W2, "... FF, O FF -> ... O")