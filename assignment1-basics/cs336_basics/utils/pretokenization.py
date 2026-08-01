import os
from typing import BinaryIO
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def sub_upd_freq(bchunk: bytes, special_pattern: str) -> dict[tuple[bytes, ...], int]:
    chunk = bchunk.decode("utf-8", errors="ignore")
    if not special_pattern:
        strs = [chunk]
    else:
        strs = re.split(special_pattern, chunk)
    freq = {}
    for cur_string in strs:
        words = re.findall(PAT, cur_string)
        for word in words:
            b_word = word.encode("utf-8")
            tup: tuple[bytes, ...] = tuple(bytes([b]) for b in b_word)
            freq[tup] = freq.get(tup, 0) + 1
    return freq


def pretokenization(chunk: str, special_tokens: list[str]) -> list[str]:
    special_pattern = "|".join(map(re.escape, special_tokens))
    if not special_pattern:
        strs = [chunk]
    else:
        # strs = re.split(special_pattern, chunk)
        strs = re.split(f"({special_pattern})", chunk)
    word_list = []
    for cur_string in strs:
        if cur_string in special_tokens:
            word_list.append(cur_string)
        else:
            words = re.findall(PAT, cur_string)
            word_list.extend(words)
    return word_list
