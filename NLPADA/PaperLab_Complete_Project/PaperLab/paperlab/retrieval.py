"""Page-local PDF extraction and a sparse BM25 passage index."""


from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
import re
import unicodedata

from pypdf import PdfReader


STOP = set("""a an the of for from at to on in as by with and or but is are was were be been being
it its this that these those what which who whom whose how did does do has have had
there their they them his her we our you your can could would should many much used""".split())


def terms(text):
    words = re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold())
    return [w for w in words if w not in STOP]


@dataclass(frozen=True)
class Passage:
    file: str
    page: int       # Physical PDF page, counted from 1.
    chunk: int      # Chunk within this page, counted from 1.
    text: str

    @property
    def citation(self):
        return f"{self.file}, p. {self.page}"


@dataclass(frozen=True)
class Hit:
    passage: Passage
    score: float
    coverage: float


def split_page(text, size=130, overlap=28):
    if size < 1 or not (0 <= overlap < size):
        raise ValueError("Use a positive chunk size and an overlap smaller than the size.")
    words = unicodedata.normalize("NFKC", text).split()
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        yield " ".join(words[start:end])
        if end == len(words):
            break
        start = end - overlap


def read_papers(folder, *, size=130, overlap=28):
    folder = Path(folder)
    paths = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")
    if not paths:
        raise ValueError(f"No PDFs in {folder}. Add at least one searchable PDF.")
    passages, issues = [], []
    for path in paths:
        name = path.relative_to(folder).as_posix()
        try:
            reader = PdfReader(str(path))
            if reader.is_encrypted and not reader.decrypt(""):
                issues.append(f"{name}: password needed.")
                continue
            pages = len(reader.pages)
        except Exception as exc:
            issues.append(f"{name}: cannot read PDF ({type(exc).__name__}).")
            continue
        for index in range(pages):
            try:
                text = reader.pages[index].extract_text() or ""
            except Exception as exc:
                issues.append(f"{name}, p. {index+1}: extraction failed ({type(exc).__name__}).")
                continue
            if not text.strip():
                issues.append(f"{name}, p. {index+1}: no selectable text. Use OCR for an image-only page.")
                continue
            for chunk, part in enumerate(split_page(text, size, overlap), 1):
                passages.append(Passage(name, index + 1, chunk, part))
    if not passages:
        raise ValueError("No searchable text was found. " + " ".join(issues[:5]))
    return passages, issues


class BM25:
    def __init__(self, passages, *, k1=1.5, b=0.75):
        self.passages = list(passages)
        if not self.passages or k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("Provide passages, k1 > 0 and b between 0 and 1.")
        self.k1, self.b = k1, b
        self.lengths = []
        self.postings = defaultdict(list)
        for idx, p in enumerate(self.passages):
            words = terms(p.text)
            self.lengths.append(len(words))
            for word, count in Counter(words).items():
                self.postings[word].append((idx, count))
        self.average = sum(self.lengths) / len(self.lengths) or 1

    def _idf(self, df):
        return math.log1p((len(self.passages) - df + 0.5) / (df + 0.5))

    def search(self, question, k=3, *, distinct_pages=False):
        if not question.strip() or k < 1:
            raise ValueError("Enter a nonempty question and a positive result count.")
        scores, matched = defaultdict(float), defaultdict(float)
        total_idf = 0.0
        for word in sorted(set(terms(question))):
            postings = self.postings.get(word, ())
            idf = self._idf(len(postings))
            total_idf += idf
            for idx, count in postings:
                adjustment = 1 - self.b + self.b * self.lengths[idx] / self.average
                scores[idx] += idf * count * (self.k1 + 1) / (count + self.k1 * adjustment)
                matched[idx] += idf
        hits, pages = [], set()
        for idx in sorted(scores, key=lambda i: (-scores[i], i)):
            passage = self.passages[idx]
            key = (passage.file, passage.page)
            if distinct_pages and key in pages:
                continue
            pages.add(key)
            hits.append(Hit(passage, scores[idx], matched[idx] / total_idf))
            if len(hits) == k:
                break
        return hits
