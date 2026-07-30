import os
from cs336_basics.utils import pretokenization
import regex as re


def train(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # init
    vocab: dict[int, bytes] = {}
    vocab_id = 0
    merges: list[tuple[bytes, bytes]] = []
    freq: dict[tuple[bytes, ...], int] = {}
    # can vocab_size be too small to restore these vocabs??
    for e in special_tokens:
        vocab[vocab_id] = e.encode("utf-8")
        vocab_id += 1
    for i in range(256):
        vocab[vocab_id] = bytes([i])
        vocab_id += 1

    # is eot always in special_tokens?
    pattern = "|".join(map(re.escape, special_tokens))
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = pretokenization.find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")

            if not pattern:
                strs = list(chunk)
            else:
                strs = re.split(pattern, chunk)
            for cur_string in strs:
                words = re.findall(PAT, cur_string)
                for word in words:
                    b_word = word.encode("utf-8")
                    tup: tuple[bytes, ...] = tuple(bytes([b]) for b in b_word)
                    freq[tup] = freq.get(tup, 0) + 1

    # merging:
    while True:
        if vocab_id >= vocab_size:
            break
        freq_bp = {}
        for b_tup, cnt in freq.items():
            for i in range(0, len(b_tup) - 1):
                byte_pair = b_tup[i : i + 2]
                freq_bp[byte_pair] = freq_bp.get(byte_pair, 0) + cnt
        if not freq_bp:  # no more word to merge
            break
        best_pair = max(freq_bp, key=lambda k: (freq_bp[k], k))
        merges.append(best_pair)
        vocab[vocab_id] = b"".join(best_pair)
        vocab_id += 1
        new_freq = {}
        for b_tup, cnt in freq.items():
            new_b_list = []
            i = 0
            while i < len(b_tup):
                if i + 1 < len(b_tup) and b_tup[i : i + 2] == best_pair:
                    new_b_list.append(b"".join(b_tup[i : i + 2]))
                    i += 2
                else:
                    new_b_list.append(b_tup[i])
                    i += 1
            new_b_tuple = tuple(new_b_list)
            new_freq[tuple(new_b_tuple)] = new_freq.get(tuple(new_b_tuple), 0) + cnt
        freq = new_freq
    return (vocab, merges)
