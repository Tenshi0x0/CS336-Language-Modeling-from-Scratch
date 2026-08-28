import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from cs336_basics.layers.rmsnorm import RMSNorm
from cs336_basics.layers.rope import RoPE
from cs336_basics.layers.multihead_self_attention import MultiHeadSelfAttention
from cs336_basics.layers.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RoPE | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope)
        self.ln1 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        self.ln2 = RMSNorm(d_model)

    def forward(
        self,
        x: Float[Tensor, " ... sequence_length d_model"],
    ) -> torch.Tensor:
        x_first = x + self.attn(self.ln1(x))
        x_second = x_first + self.ffn(self.ln2(x_first))
        return x_second
