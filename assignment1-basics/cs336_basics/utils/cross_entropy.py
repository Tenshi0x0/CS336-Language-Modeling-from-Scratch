import torch
from jaxtyping import Float, Int
from torch import Tensor
from cs336_basics.utils.softmax import softmax


def cross_entropy(
    o: Float[Tensor, " batch vocab_size"], targets: Int[Tensor, " batch"]
) -> Float[Tensor, ""]:
    # p = softmax(inputs, dim=-1)
    target_logits = torch.take_along_dim(o, targets.unsqueeze(-1), dim=-1).squeeze(-1)
    return (torch.logsumexp(o, dim=-1) - target_logits).mean()