from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch

from cs336_basics.layers.transformer_lm import TransformerLM
from cs336_basics.utils.tokenizer import Tokenizer
from cs336_basics.utils.softmax import softmax



@torch.no_grad()
def generate(
    model: TransformerLM,
    prompt_ids: list[int],
    max_new_tokens: int,
    eos_id: int | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> list[int]:
    model.eval()
    ids = list(prompt_ids)
    context_length = model.context_length

    for _ in range(max_new_tokens):
        window = ids[-context_length:]
        x = torch.tensor(window, dtype=torch.long, device=model.lm_head.weight.device).unsqueeze(0)
        logits = model(x)
        logits = logits[0, -1, :]
        probs = softmax(logits / temperature, dim=-1)

        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = sorted_probs.cumsum(-1)
        keep = (cumsum - sorted_probs) < top_p
        sorted_probs = sorted_probs * keep
        sorted_probs = sorted_probs / sorted_probs.sum()

        pos = torch.multinomial(sorted_probs, num_samples=1)
        cur_id = sorted_idx[pos].item()

        ids.append(cur_id)
        if cur_id == eos_id:
            break

    return ids


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default=None, help="defaults to config.json next to the checkpoint")
    p.add_argument("--vocab", default="data/tokenizer/tinystories_vocab.pkl")
    p.add_argument("--merges", default="data/tokenizer/tinystories_merges.pkl")
    p.add_argument(
        "--prompt",
        default="Once upon a time",
        help='pass "" for unconditional generation (seeds with <|endoftext|>)',
    )
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # The checkpoint stores weights only, so the architecture comes from the
    # config.json that the training run wrote next to it.
    cfg_path = Path(args.config) if args.config else Path(args.checkpoint).parent / "config.json"
    cfg = json.loads(cfg_path.read_text())["args"]

    model = TransformerLM(
        vocab_size=cfg["vocab_size"],
        context_length=cfg["context_length"],
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        rope_theta=None if cfg.get("no_rope") else cfg["rope_theta"],
    ).to(args.device)
    # Inference only: take the weights directly instead of load_checkpoint(),
    # which insists on an optimizer we do not have here.
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device)["model"])

    # Once Tokenizer.from_files is implemented these three lines become one call.
    with open(args.vocab, "rb") as f:
        vocab = pickle.load(f)
    with open(args.merges, "rb") as f:
        merges = pickle.load(f)
    tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])
    eos_id = next(i for i, b in vocab.items() if b == b"<|endoftext|>")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    prompt_ids = tokenizer.encode(args.prompt)
    unconditional = not prompt_ids
    if unconditional:
        # A transformer cannot be run on an empty sequence, and the corpus is a
        # stream of documents separated by <|endoftext|>. Seeding with that token
        # is what "start a fresh document" means to this model.
        prompt_ids = [eos_id]

    shown = "unconditional" if unconditional else repr(args.prompt)
    print(f"model={args.checkpoint}  device={args.device}  prompt={shown}")
    print(f"temperature={args.temperature}  top_p={args.top_p}  max_new_tokens={args.max_new_tokens}\n")

    for k in range(args.num_samples):
        out = generate(
            model,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            eos_id=eos_id,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        new_ids = out[len(prompt_ids) :]
        hit_eos = bool(new_ids) and new_ids[-1] == eos_id
        # Drop the terminator so it does not render literally in the story, and
        # never echo the seed token of an unconditional sample.
        body = new_ids[:-1] if hit_eos else new_ids
        reason = "hit <|endoftext|>" if hit_eos else "hit max_new_tokens"
        prompt_note = "" if unconditional else f"prompt {len(prompt_ids)} tok + "
        print(f"--- sample {k + 1}/{args.num_samples}  ({prompt_note}{len(body)} new tok, {reason}) ---")
        if not unconditional:
            print(args.prompt, end="")
        print(tokenizer.decode(body))
        print()


if __name__ == "__main__":
    main()
