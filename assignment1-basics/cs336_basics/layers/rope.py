import torch
import torch.nn as nn
from einops import einsum


class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device=None,
    ):
        super().__init__()

        assert d_k % 2 == 0

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # ang = i / theta**((2k-2)/d)
        pos = torch.arange(max_seq_len, device=device)
        ifreq = theta ** (-torch.arange(0, d_k, 2, device=device) / d_k)
        ang = einsum(pos, ifreq, "i, k -> i k")
        self.register_buffer("sin", ang.sin(), persistent=False)
        self.register_buffer("cos", ang.cos(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        sin = self.sin[token_positions]
        cos = self.cos[token_positions]

        X = x[..., 0::2]
        Y = x[..., 1::2]
        res = torch.zeros_like(x)
        res[..., 0::2] = cos * X - sin * Y
        res[..., 1::2] = sin * X + cos * Y
        return res
