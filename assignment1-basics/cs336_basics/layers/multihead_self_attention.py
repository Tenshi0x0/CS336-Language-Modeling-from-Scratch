import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from einops import rearrange

from cs336_basics.utils.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.layers.linear import Linear
from cs336_basics.layers.rope import RoPE


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RoPE | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

        self.rope = rope

    def forward(
        self,
        x: Float[Tensor, " ... sequence_length d_model"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> torch.Tensor:
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = rearrange(Q, "... seq (h d) -> ... h seq d", h=self.num_heads)
        K = rearrange(K, "... seq (h d) -> ... h seq d", h=self.num_heads)
        V = rearrange(V, "... seq (h d) -> ... h seq d", h=self.num_heads)

        if self.rope is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        seq_len = x.shape[-2]
        mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
        )

        attention = scaled_dot_product_attention(Q, K, V, mask)
        attention = rearrange(attention, "... h seq d -> ... seq (h d)")
        return self.output_proj(attention)
