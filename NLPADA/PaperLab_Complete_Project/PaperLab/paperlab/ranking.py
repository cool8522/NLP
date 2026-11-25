"""Lexical retrieval ablations. TF-IDF is sparse; no neural embeddings are used."""
from collections import Counter, defaultdict
import math
from .retrieval import BM25, Hit, terms


class RetrievalIndex(BM25):
    def __init__(self, passages, *, mode="bm25", k1=1.5, b=0.75):
        super().__init__(passages, k1=k1, b=b)
        if mode not in {"bm25", "tfidf", "hybrid"}:
            raise ValueError("Mode must be bm25, tfidf or hybrid.")
        self.mode = mode
        self.tf_postings = defaultdict(list)
        norms = defaultdict(float)
        for word, posts in self.postings.items():
            idf = math.log((1 + len(self.passages)) / (1 + len(posts))) + 1
            for i, count in posts:
                w = (1 + math.log(count)) * idf
                self.tf_postings[word].append((i, w))
                norms[i] += w * w
        self.tf_norms = {i: math.sqrt(n) for i, n in norms.items()}

    def search(self, question, k=3, *, distinct_pages=False):
        if not question.strip() or k < 1:
            raise ValueError("Enter a nonempty question and a positive result count.")
        if self.mode == "bm25":
            return super().search(question, k, distinct_pages=distinct_pages)
        bm = super().search(question, len(self.passages))
        coverage = {h.passage: h.coverage for h in bm}
        scores = defaultdict(float)
        qnorm = 0.0
        for word, count in Counter(terms(question)).items():
            idf = math.log((1 + len(self.passages)) / (1 + len(self.postings.get(word, ())))) + 1
            qw = (1 + math.log(count)) * idf
            qnorm += qw * qw
            for i, weight in self.tf_postings.get(word, ()):
                scores[i] += qw * weight
        for i in scores:
            scores[i] /= math.sqrt(qnorm) * self.tf_norms[i]
        tf = sorted(scores, key=lambda i: (-scores[i], i))
        if self.mode == "hybrid":
            lookup = {p: i for i, p in enumerate(self.passages)}
            fused = defaultdict(float)
            for rank, h in enumerate(bm, 1):
                fused[lookup[h.passage]] += 1 / (60 + rank)
            for rank, i in enumerate(tf, 1):
                fused[i] += 1 / (60 + rank)
            scores = fused
            tf = sorted(scores, key=lambda i: (-scores[i], i))
        hits, pages = [], set()
        for i in tf:
            p = self.passages[i]
            page = (p.file, p.page)
            if distinct_pages and page in pages:
                continue
            pages.add(page)
            hits.append(Hit(p, scores[i], coverage.get(p, 0)))
            if len(hits) >= k:
                break
        return hits

