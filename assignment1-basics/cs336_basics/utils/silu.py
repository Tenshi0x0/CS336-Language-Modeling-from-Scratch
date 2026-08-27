import torch
from jaxtyping import Float
from torch import Tensor


def silu(x: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    return x * torch.sigmoid(x)
