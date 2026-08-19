from jaxtyping import Float
from torch import Tensor


def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    x = in_features - in_features.max(dim=dim, keepdim=True).values
    exp = x.exp()
    sum = exp.sum(dim=dim, keepdim=True)
    return exp / sum


# print(softmax(Tensor([1, 1]), dim=0))
