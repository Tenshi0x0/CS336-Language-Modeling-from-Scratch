from collections.abc import Callable

import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        super().__init__(
            params,
            {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay},
        )

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(grad))
                v = state.get("v", torch.zeros_like(grad))

                with torch.no_grad():
                    a_t = lr * math.sqrt(1 - b2**t) / (1 - b1**t)
                    p -= lr * weight_decay * p
                    m = b1 * m + (1 - b1) * grad
                    v = b2 * v + (1 - b2) * (grad**2)
                    p -= a_t * (m / (v.sqrt() + eps))

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        lr = it / warmup_iters * max_learning_rate
    elif it <= cosine_cycle_iters:
        cur = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        lr = min_learning_rate + 0.5 * (1 + math.cos(cur * math.pi)) * (max_learning_rate - min_learning_rate)
    else:
        lr = min_learning_rate
    return lr
