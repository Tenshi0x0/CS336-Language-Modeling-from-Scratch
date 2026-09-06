"""Train a Transformer LM end to end (assignment section 5.3).

All paths are relative to the repository root, which is where this script is
meant to be launched from:

    uv run python scripts/train.py --run-name ts-baseline

Defaults follow the hyperparameters prescribed in handout section 7.2.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.data import get_batch
from cs336_basics.layers.transformer_lm import TransformerLM
from cs336_basics.optim import AdamW, get_lr_cosine_schedule, gradient_clipping
from cs336_basics.serialization import load_checkpoint, save_checkpoint
from cs336_basics.utils.cross_entropy import cross_entropy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    d = p.add_argument_group("data")
    d.add_argument("--train-data", default="data/tokens/ts_train.npy")
    d.add_argument("--valid-data", default="data/tokens/ts_valid.npy")

    m = p.add_argument_group("model (handout 7.2 defaults)")
    m.add_argument("--vocab-size", type=int, default=10000)
    m.add_argument("--context-length", type=int, default=256)
    m.add_argument("--d-model", type=int, default=512)
    m.add_argument("--d-ff", type=int, default=1344)
    m.add_argument("--num-layers", type=int, default=4)
    m.add_argument("--num-heads", type=int, default=16)
    m.add_argument("--rope-theta", type=float, default=10000.0)
    m.add_argument("--no-rope", action="store_true", help="NoPE ablation: build the model without RoPE")

    o = p.add_argument_group("optimizer")
    o.add_argument("--lr", type=float, default=3e-4, help="alpha_max of the cosine schedule")
    o.add_argument("--min-lr", type=float, default=3e-5, help="alpha_min of the cosine schedule")
    o.add_argument("--weight-decay", type=float, default=0.01)
    o.add_argument("--beta1", type=float, default=0.9)
    o.add_argument("--beta2", type=float, default=0.95)
    o.add_argument("--eps", type=float, default=1e-8)
    o.add_argument("--max-grad-norm", type=float, default=1.0)

    t = p.add_argument_group("training")
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--max-iters", type=int, default=20000, help="64 x 20000 x 256 ~= the 327,680,000 token budget")
    t.add_argument("--warmup-iters", type=int, default=400)
    t.add_argument(
        "--cosine-cycle-iters",
        type=int,
        default=None,
        help="step at which the cosine decay reaches min-lr; defaults to --max-iters",
    )
    t.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument(
        "--matmul-precision",
        default="high",
        choices=["highest", "high", "medium"],
        help="'high' turns on TF32 matmuls on CUDA; use 'highest' to disable",
    )

    r = p.add_argument_group("run")
    r.add_argument("--run-name", default=None, help="directory under --runs-dir; defaults to a timestamp")
    r.add_argument("--runs-dir", default="runs")
    r.add_argument("--log-interval", type=int, default=50)
    r.add_argument("--eval-interval", type=int, default=500)
    r.add_argument("--eval-batches", type=int, default=20)
    r.add_argument("--ckpt-interval", type=int, default=2000)
    r.add_argument("--resume", default=None, help="checkpoint to resume from")
    r.add_argument("--wandb-project", default=None, help="enable Weights & Biases logging")

    args = p.parse_args()
    if args.warmup_iters < 1:
        p.error("--warmup-iters must be >= 1 (the schedule divides by it)")
    if args.cosine_cycle_iters is None:
        # The handout suggests the decay should bottom out exactly at the last step.
        args.cosine_cycle_iters = args.max_iters
    if args.run_name is None:
        args.run_name = time.strftime("run-%Y%m%d-%H%M%S")
    return args


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def build_model(args: argparse.Namespace) -> TransformerLM:
    return TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=None if args.no_rope else args.rope_theta,
    )


@torch.no_grad()
def evaluate(model: TransformerLM, data: np.ndarray, args: argparse.Namespace) -> float:
    """Average cross-entropy over a few batches of held-out data."""
    model.eval()
    total = 0.0
    for _ in range(args.eval_batches):
        x, y = get_batch(data, args.batch_size, args.context_length, args.device)
        total += cross_entropy(model(x), y).item()
    model.train()
    return total / args.eval_batches


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision(args.matmul_precision)

    run_dir = Path(args.runs_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps({"argv": sys.argv, "args": vars(args), "git_commit": git_commit()}, indent=2)
    )

    # Memory-map once, up front: these files are far too large to read eagerly.
    train_data = np.load(args.train_data, mmap_mode="r")
    valid_data = np.load(args.valid_data, mmap_mode="r")

    model = build_model(args).to(args.device)
    opt = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_iter = 0
    if args.resume:
        start_iter = load_checkpoint(args.resume, model, opt) + 1
        print(f"resumed from {args.resume}, continuing at iteration {start_iter}", flush=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"run_dir={run_dir}  device={args.device}  params={n_params:,}\n"
        f"train tokens={len(train_data):,}  valid tokens={len(valid_data):,}  "
        f"budget={args.batch_size * args.max_iters * args.context_length:,} tokens",
        flush=True,
    )

    run = None
    if args.wandb_project:
        import wandb

        run = wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    log_path = run_dir / "train.log"
    log_file = log_path.open("a")
    t0 = time.perf_counter()

    def record(**fields) -> None:
        fields["wall_clock_s"] = round(time.perf_counter() - t0, 3)
        log_file.write(json.dumps(fields) + "\n")
        log_file.flush()
        if run is not None:
            run.log(fields, step=fields["step"])

    model.train()
    for it in range(start_iter, args.max_iters):
        lr = get_lr_cosine_schedule(it, args.lr, args.min_lr, args.warmup_iters, args.cosine_cycle_iters)
        for group in opt.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        loss = cross_entropy(model(x), y)
        loss.backward()
        # Clip after backward and before step: this is the only window in which
        # .grad holds this step's gradients.
        gradient_clipping(model.parameters(), args.max_grad_norm)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if it % args.log_interval == 0:
            elapsed = time.perf_counter() - t0
            train_loss = loss.item()
            record(step=it, train_loss=train_loss, lr=lr)
            print(
                f"it {it:>6}  loss {train_loss:7.4f}  ppl {math.exp(min(train_loss, 20)):9.2f}  "
                f"lr {lr:.3e}  {elapsed / 60:6.1f}m",
                flush=True,
            )

        if it > start_iter and it % args.eval_interval == 0:
            val_loss = evaluate(model, valid_data, args)
            record(step=it, valid_loss=val_loss)
            print(f"it {it:>6}  VALID loss {val_loss:7.4f}  ppl {math.exp(min(val_loss, 20)):9.2f}", flush=True)

        if it > start_iter and it % args.ckpt_interval == 0:
            save_checkpoint(model, opt, it, run_dir / f"step_{it}.pt")

    val_loss = evaluate(model, valid_data, args)
    record(step=args.max_iters, valid_loss=val_loss)
    save_checkpoint(model, opt, args.max_iters, run_dir / "final.pt")
    print(
        f"done. final valid loss {val_loss:.4f}  ppl {math.exp(min(val_loss, 20)):.2f}  "
        f"total {(time.perf_counter() - t0) / 60:.1f}m",
        flush=True,
    )
    log_file.close()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
