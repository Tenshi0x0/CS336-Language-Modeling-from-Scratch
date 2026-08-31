import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = len(dataset)
    cl = context_length
    starts = np.random.randint(0, seq_len - cl, batch_size)
    x = np.stack([dataset[i : i + cl] for i in starts])
    y = np.stack([dataset[i + 1 : i + cl + 1] for i in starts])
    return (torch.tensor(x, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device))
