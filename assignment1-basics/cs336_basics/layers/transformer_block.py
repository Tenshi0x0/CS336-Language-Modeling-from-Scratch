import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from einops import rearrange, einsum

from cs336_basics.utils.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.layers.rope import RoPE


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RoPE | None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope = rope

    def forward(
        self,
        in_features: Float[Tensor, " batch sequence_length d_model"],
    ) -> torch.Tensor:
        pass
