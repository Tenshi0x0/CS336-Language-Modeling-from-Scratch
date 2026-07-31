import os
from cs336_basics.utils import pretokenization
import regex as re
import heapq
import typing
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from functools import partial


@dataclass(slots=True, order=False)
class HeapNode:
    val: int
    name: tuple[bytes, ...]

    def __lt__(self, o: typing.Self) -> bool:
        if self.val != o.val:
            return self.val > o.val
        return self.name > o.name


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
    special_pattern = "|".join(map(re.escape, special_tokens))

    with open(input_path, "rb") as f:
        num_cpu = len(os.sched_getaffinity(0))
        num_processes = num_cpu * 3
        boundaries = pretokenization.find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start)
            chunks.append(chunk)

        with ProcessPoolExecutor(max_workers=num_cpu) as ex:
            for sub_freq in ex.map(partial(pretokenization.sub_upd_freq, special_pattern=special_pattern), chunks):
                for x, y in sub_freq.items():
                    freq[x] = freq.get(x, 0) + y

    pair_to_tupset: dict[tuple[bytes, ...], set[tuple[bytes, ...]]] = {}
    freq_bp = {}
    bp_heap = []
    for b_tup, cnt in freq.items():
        for i in range(0, len(b_tup) - 1):
            byte_pair = b_tup[i : i + 2]
            freq_val = freq_bp.get(byte_pair, 0) + cnt
            freq_bp[byte_pair] = freq_val
            heapq.heappush(bp_heap, HeapNode(freq_val, byte_pair))
            pair_to_tupset.setdefault(byte_pair, set()).add(b_tup)

    # merging:
    while True:
        if vocab_id >= vocab_size:
            break
        if not freq_bp:  # no more word to merge
            break
        # best_pair = max(freq_bp, key=lambda k: (freq_bp[k], k))
        best_pair = None
        while bp_heap:
            node = heapq.heappop(bp_heap)
            if freq_bp.get(node.name, 0) == node.val:
                best_pair = node.name
                break
        assert best_pair, "cannot find valid pair"
        merges.append(best_pair)
        vocab[vocab_id] = b"".join(best_pair)
        vocab_id += 1

        tuplist = pair_to_tupset[best_pair]
        upd_freq_event = []
        for b_tup in tuplist:
            new_b_list = []
            i = 0
            b_tup_size = len(b_tup)
            while i < b_tup_size:
                if i + 1 < b_tup_size and b_tup[i : i + 2] == best_pair:
                    new_b_list.append(b"".join(b_tup[i : i + 2]))
                    i += 2
                else:
                    new_b_list.append(b_tup[i])
                    i += 1
            new_b_tuple = tuple(new_b_list)
            upd_freq_event.append((b_tup, new_b_tuple, freq.get(b_tup, 0)))

        for pre_tup, nxt_tup, val in upd_freq_event:
            freq.pop(pre_tup, None)
            freq[nxt_tup] = freq.get(nxt_tup, 0) + val
            for i in range(0, len(pre_tup) - 1):
                byte_pair = pre_tup[i : i + 2]
                freq_val = freq_bp.get(byte_pair, 0) - val
                freq_bp[byte_pair] = freq_val
                heapq.heappush(bp_heap, HeapNode(freq_val, byte_pair))
                if freq_bp[byte_pair] == 0:
                    freq_bp.pop(byte_pair, None)
                pair_to_tupset.setdefault(byte_pair, set()).discard(pre_tup)

            for i in range(0, len(nxt_tup) - 1):
                byte_pair = nxt_tup[i : i + 2]
                freq_val = freq_bp.get(byte_pair, 0) + val
                freq_bp[byte_pair] = freq_val
                heapq.heappush(bp_heap, HeapNode(freq_val, byte_pair))
                pair_to_tupset.setdefault(byte_pair, set()).add(nxt_tup)

    return (vocab, merges)
