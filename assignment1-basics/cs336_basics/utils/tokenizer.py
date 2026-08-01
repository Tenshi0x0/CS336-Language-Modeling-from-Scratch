from collections.abc import Iterable, Iterator
from typing import Self
import heapq
from cs336_basics.utils import pretokenization


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab: dict[int, bytes] = vocab
        self.merges: list[tuple[bytes, bytes]] = merges
        self.special_tokens: list[str] = special_tokens if special_tokens is not None else []
        self.special_tokens.sort(key=lambda x: -len(x))
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {}
        self.word_ids: dict[bytes, int] = {}
        self.LOWESTRANK: int = len(merges)
        for idx, merge in enumerate(merges):
            self.merge_ranks[merge] = idx
        for x, y in vocab.items():
            self.word_ids[y] = x

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Self:
        raise NotImplementedError

    def encode_bytes(self, blist: list[bytes]) -> list[int]:
        """
        I ignore that text is from pretokenization, thus brute force should be totally enough...
        """
        blist_size = len(blist)

        # init: each byte belong to its own group: [0] [1] [2] [3] [4] ...
        # after merging, it can be like [00] [22] [4]

        # belong = [x for x in range(blist_size)]
        seg = [[x, x] for x in range(blist_size)]
        nxt = [x + 1 for x in range(blist_size)]
        pre = [x - 1 for x in range(blist_size)]
        valid_group: dict[tuple[int, int], int] = {}  # valid group just guarantee adjacency
        candidate_heap = []  # candidate heap for merging
        for i in range(blist_size - 1):
            tup: tuple[bytes, bytes] = (blist[i], blist[i + 1])
            valid_group[(i, i + 1)] = self.merge_ranks.get(tup, self.LOWESTRANK)
            if tup in self.merge_ranks:
                heapq.heappush(candidate_heap, (self.merge_ranks[tup], (i, i + 1)))

        def group_to_bytes(group_id: int) -> bytes:
            return b"".join(blist[seg[group_id][0] : seg[group_id][1] + 1])

        def valid_merge(group_pair: tuple[int, int]) -> bool:
            x, y = group_pair
            x_bytes = group_to_bytes(x)
            y_bytes = group_to_bytes(y)
            return (x_bytes, y_bytes) in self.merge_ranks

        while candidate_heap:
            top = heapq.heappop(candidate_heap)
            group_pair = top[1]
            if top[0] != valid_group.get(group_pair, self.LOWESTRANK) or not valid_merge(group_pair):
                continue
            x, y = group_pair  # [xx] [yy] -> [xxxx]
            seg[x][1] = seg[y][1]
            valid_group.pop((x, y))

            nxt[x] = nxt[y]
            if nxt[x] != blist_size:
                valid_group.pop((y, nxt[y]))
                valid_group[(x, nxt[x])] = self.merge_ranks.get(
                    (group_to_bytes(x), group_to_bytes(nxt[x])), self.LOWESTRANK
                )
                pre[nxt[x]] = x
            if pre[x] != -1:
                valid_group[(pre[x], x)] = self.merge_ranks.get(
                    (group_to_bytes(pre[x]), group_to_bytes(x)), self.LOWESTRANK
                )
            if nxt[x] != blist_size and valid_merge((x, nxt[x])):
                heapq.heappush(
                    candidate_heap, (self.merge_ranks[(group_to_bytes(x), group_to_bytes(nxt[x]))], (x, nxt[x]))
                )
            if pre[x] != -1 and valid_merge((pre[x], x)):
                heapq.heappush(
                    candidate_heap, (self.merge_ranks[(group_to_bytes(pre[x]), group_to_bytes(x))], (pre[x], x))
                )

        cur_g = 0
        encode_list: list[int] = []
        while cur_g != blist_size:
            word = group_to_bytes(cur_g)
            encode_list.append(self.word_ids[word])
            cur_g = nxt[cur_g]
        return encode_list

    def encode(self, text: str) -> list[int]:
        word_list = pretokenization.pretokenization(text, self.special_tokens)

        res: list[int] = []
        for word in word_list:
            if word in self.special_tokens:
                res.append(self.word_ids[word.encode("utf-8")])
            else:
                blist = []
                bword = word.encode("utf-8")
                for byte in bword:
                    blist.append(bytes([byte]))
                res.extend(self.encode_bytes(blist))
        return res

    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        bstr = b"".join(self.vocab[vid] for vid in ids)
        return bstr.decode("utf-8", errors="replace")
