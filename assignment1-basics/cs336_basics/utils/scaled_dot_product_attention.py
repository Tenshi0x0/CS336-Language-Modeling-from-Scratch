import torch
from jaxtyping import Float, Bool
from torch import Tensor
from cs336_basics.utils.softmax import softmax
from einops import rearrange, einsum


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """

    # n: queries; m: keys
    d_k = Q.shape[-1]
    QK = einsum(Q, K, "... n d_k, ... m d_k -> ... n m") / (d_k**0.5)
    if mask is not None:
        QK = torch.where(mask, QK, float("-inf"))
    QK = softmax(QK, dim=-1)
    return einsum(QK, V, "... n m, ... m d_v -> ... n d_v")
