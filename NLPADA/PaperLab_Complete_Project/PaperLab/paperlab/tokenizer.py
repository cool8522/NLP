"""A byte-level BPE tokenizer. Only the Python standard library is required."""


from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import re

# Each match preserves its original characters, including spaces and newlines.
PIECES = re.compile(r" ?\w+| ?[^\w\s]+|\s+", re.UNICODE)
PAD, BOS, EOS = 256, 257, 258
FIRST_MERGE = 259


def merge_pair(symbols, pair, new_id):
    result = []
    pos = 0
    while pos < len(symbols):
        if pos + 1 < len(symbols) and (symbols[pos], symbols[pos + 1]) == pair:
            result.append(new_id)
            pos += 2
        else:
            result.append(symbols[pos])
            pos += 1
    return tuple(result)


class ByteBPE:
    def __init__(self, merges=()):
        self.merges = tuple(tuple(pair) for pair in merges)
        self.ranks = {}
        self.pieces = {i: bytes([i]) for i in range(256)}
        for new_id, pair in enumerate(self.merges, FIRST_MERGE):
            if len(pair) != 2 or pair in self.ranks or any(i not in self.pieces for i in pair):
                raise ValueError(f"Invalid BPE merge at id {new_id}")
            self.ranks[pair] = new_id
            self.pieces[new_id] = self.pieces[pair[0]] + self.pieces[pair[1]]

    @property
    def vocab_size(self):
        return FIRST_MERGE + len(self.merges)

    @classmethod
    def train(cls, documents, vocab_size=448, min_frequency=2):
        if vocab_size < FIRST_MERGE or min_frequency < 1:
            raise ValueError(f"Use vocab_size >= {FIRST_MERGE} and min_frequency >= 1.")
        words = Counter(tuple(piece.encode("utf-8")) for doc in documents for piece in PIECES.findall(doc))
        if not words:
            raise ValueError("BPE training needs nonempty text.")
        merges = []
        while FIRST_MERGE + len(merges) < vocab_size:
            counts = Counter()
            for word, frequency in words.items():
                for pair in zip(word, word[1:]):
                    counts[pair] += frequency
            if not counts:
                break
            # Count ties are resolved by the lexicographic order of the two ids.
            pair, frequency = min(counts.items(), key=lambda item: (-item[1], item[0]))
            if frequency < min_frequency:
                break
            new_id = FIRST_MERGE + len(merges)
            updated = Counter()
            for word, weight in words.items():
                updated[merge_pair(word, pair, new_id)] += weight
            words = updated
            merges.append(pair)
        return cls(merges)

    @lru_cache(maxsize=16384)
    def _encode_piece(self, piece):
        symbols = tuple(piece.encode("utf-8"))
        while len(symbols) > 1:
            candidates = (self.ranks[p] for p in zip(symbols, symbols[1:]) if p in self.ranks)
            new_id = min(candidates, default=None)
            if new_id is None:
                break
            symbols = merge_pair(symbols, self.merges[new_id - FIRST_MERGE], new_id)
        return symbols

    def encode(self, text, *, boundaries=False):
        result = [i for match in PIECES.findall(text) for i in self._encode_piece(match)]
        return [BOS, *result, EOS] if boundaries else result

    def decode(self, ids, *, errors="strict"):
        pieces = []
        for idx in ids:
            if idx in (PAD, BOS, EOS):
                continue
            if idx not in self.pieces:
                raise ValueError(f"Unknown token id {idx}")
            pieces.append(self.pieces[idx])
        return b"".join(pieces).decode("utf-8", errors=errors)

    def save(self, path):
        Path(path).write_text(json.dumps({"format": "byte-bpe-v1", "merges": self.merges}), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != "byte-bpe-v1":
            raise ValueError("Unsupported tokenizer format.")
        return cls(data["merges"])
