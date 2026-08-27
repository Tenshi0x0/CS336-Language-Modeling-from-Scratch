import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from einops import rearrange, einsum

from cs336_basics.utils.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.layers.rope import RoPE


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RoPE | None,
        q_proj_weight: Float[Tensor, " d_model d_model"],
        k_proj_weight: Float[Tensor, " d_model d_model"],
        v_proj_weight: Float[Tensor, " d_model d_model"],
        o_proj_weight: Float[Tensor, " d_model d_model"],
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # self.max_seq_len = max_seq_len
        # self.theta = theta
        self.W_Q = nn.Parameter(q_proj_weight)
        self.W_K = nn.Parameter(k_proj_weight)
        self.W_V = nn.Parameter(v_proj_weight)
        self.W_O = nn.Parameter(o_proj_weight)

        self.rope = rope
        # d_k = d_model // num_heads
        # self.rope = None if theta is None else RoPE(
        #     theta, d_k, max_seq_len
        # )
        # self.rope = None
        # if theta is not None:
        #     assert max_seq_len is not None
        #     self.rope =  RoPE(
        #         theta, d_k, max_seq_len
        #     )

    def forward(
        self,
        x: Float[Tensor, " ... sequence_length d_model"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> torch.Tensor:
        Q = einsum(self.W_Q, x, "d_out d_in, ... d_in -> ... d_out")
        K = einsum(self.W_K, x, "d_out d_in, ... d_in -> ... d_out")
        V = einsum(self.W_V, x, "d_out d_in, ... d_in -> ... d_out")

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
        return einsum(attention, self.W_O, "... hd_v, d_model hd_v -> ... d_model")
