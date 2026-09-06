"""Encode a raw text corpus into a flat uint16 token array.

The corpus is split at `<|endoftext|>` boundaries so no document (and hence no
token) is ever cut in half, then the pieces are encoded in parallel and
concatenated back in order.

    uv run python scripts/encode_corpus.py \
        --input data/raw/TinyStoriesV2-GPT4-valid.txt \
        --vocab data/tokenizer/tinystories_vocab.pkl \
        --merges data/tokenizer/tinystories_merges.pkl \
        --output data/tokens/ts_valid.npy
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from cs336_basics.utils.pretokenization import find_chunk_boundaries
from cs336_basics.utils.tokenizer import Tokenizer

_TOKENIZER: Tokenizer | None = None


def _init_worker(vocab_path: str, merges_path: str, special_tokens: list[str]) -> None:
    """Build one Tokenizer per worker process and reuse it across chunks."""
    global _TOKENIZER
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    with open(merges_path, "rb") as f:
        merges = pickle.load(f)
    _TOKENIZER = Tokenizer(vocab, merges, list(special_tokens))


def _encode_chunk(job: tuple[str, int, int]) -> np.ndarray:
    path, start, end = job
    with open(path, "rb") as f:
        f.seek(start)
        raw = f.read(end - start)
    assert _TOKENIZER is not None, "worker was not initialized"
    # Boundaries sit on the ASCII '<' of the special token, so a multi-byte
    # UTF-8 sequence can never straddle a chunk edge -- decode strictly.
    return np.asarray(_TOKENIZER.encode(raw.decode("utf-8")), dtype=np.uint16)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="raw .txt corpus")
    p.add_argument("--vocab", required=True, help="pickled dict[int, bytes]")
    p.add_argument("--merges", required=True, help="pickled list[tuple[bytes, bytes]]")
    p.add_argument("--output", required=True, help="destination .npy (uint16)")
    p.add_argument("--special-token", default="<|endoftext|>")
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or (os.cpu_count() or 1))
    p.add_argument(
        "--chunks-per-worker",
        type=int,
        default=16,
        help="more, smaller chunks keep peak memory down and smooth out stragglers",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.vocab, "rb") as f:
        vocab_size = len(pickle.load(f))

    with open(args.input, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, args.workers * args.chunks_per_worker, args.special_token.encode("utf-8")
        )
    jobs = [(args.input, s, e) for s, e in zip(boundaries[:-1], boundaries[1:])]
    total_bytes = boundaries[-1] - boundaries[0]
    print(
        f"input={args.input}  {total_bytes / 2**20:.1f} MiB  vocab_size={vocab_size}\n"
        f"workers={args.workers}  chunks={len(jobs)}",
        flush=True,
    )

    parts: list[np.ndarray] = []
    done_bytes = 0
    t0 = time.perf_counter()
    with Pool(args.workers, initializer=_init_worker, initargs=(args.vocab, args.merges, [args.special_token])) as pool:
        # imap keeps the results in submission order, which is what makes the
        # concatenation below a faithful reconstruction of the corpus.
        for i, part in enumerate(pool.imap(_encode_chunk, jobs)):
            parts.append(part)
            done_bytes += jobs[i][2] - jobs[i][1]
            elapsed = time.perf_counter() - t0
            frac = done_bytes / total_bytes
            print(
                f"  [{i + 1}/{len(jobs)}] {frac:6.1%}  {done_bytes / 2**20 / elapsed:6.2f} MiB/s  "
                f"elapsed {elapsed / 60:5.1f}m  eta {elapsed / max(frac, 1e-9) * (1 - frac) / 60:5.1f}m",
                flush=True,
            )

    ids = np.concatenate(parts)
    # The handout asks for an explicit check that the memory-mapped data is sane.
    assert ids.max() < vocab_size, f"token id {ids.max()} >= vocab_size {vocab_size}"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, ids)
    print(
        f"saved {args.output}  {len(ids):,} tokens  {ids.nbytes / 2**20:.1f} MiB  "
        f"dtype={ids.dtype}  max_id={ids.max()}  total {(time.perf_counter() - t0) / 60:.1f}m",
        flush=True,
    )


if __name__ == "__main__":
    main()
