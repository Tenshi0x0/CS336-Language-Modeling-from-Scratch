import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from cs336_basics.layers.embedding import Embedding
from cs336_basics.layers.linear import Linear
from cs336_basics.layers.rmsnorm import RMSNorm
from cs336_basics.layers.rope import RoPE
from cs336_basics.layers.transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff

        rope = None if rope_theta is None else RoPE(rope_theta, d_model // num_heads, context_length)

        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.Sequential(
            *[TransformerBlock(d_model, num_heads, d_ff, rope) for _ in range(num_layers)]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(
        self,
        token_ids: Int[Tensor, " ... sequence_length"],
    ) -> torch.Tensor:
        x = self.token_embeddings(token_ids)
        x = self.layers(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x
