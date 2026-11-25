"""Page retrieval and optional end-to-end answer checks on labeled questions."""

from .rag import answer

import json
from pathlib import Path
import re
import unicodedata
import time



def normalize_answer(text):
    # Keep numbers, units, signs and decimal separators intact for exact match.
    value = unicodedata.normalize("NFKC", text).casefold().strip().rstrip(".!?")
    value = re.sub(r"^(the|a|an)\s+", "", value)
    return " ".join(value.split())


def evaluate(index, labels, *, k=3, generator=None):
    rows = json.loads(Path(labels).read_text(encoding="utf-8"))
    if not rows or k < 1:
        raise ValueError("Provide at least one evaluation question and a positive cutoff.")
    cases = []
    for row in rows:
        q = row["question"]
        expected = (row["file"], row["page"]) if row["file"] else None
        rank = None
        # Ranking metrics count distinct physical pages.
        hits = index.search(q, k, distinct_pages=True)
        for pos, hit in enumerate(hits, 1):
            if (hit.passage.file, hit.passage.page) == expected:
                rank = pos
                break
        case = {"question": q, "answerable": expected is not None, "expected": expected, "rank": rank,
                "retrieved": [h.passage.citation for h in hits]}
        if generator is not None:
            started = time.perf_counter()
            reply = answer(index, generator, q, k=k)
            case.update(status=reply.status, answer=reply.text, citation=reply.citation, quote=reply.quote,
                        latency_ms=1000 * (time.perf_counter() - started))
            if expected:
                case["exact_match"] = reply.status == "answered" and normalize_answer(reply.text) in {
                    normalize_answer(a) for a in row["answers"]}
                case["correct_page"] = reply.citation == f"{expected[0]}, p. {expected[1]}"
                from .rag import locate_span
                case["reference_in_response"] = reply.status == "answered" and any(
                    locate_span(a, reply.text) is not None for a in row["answers"])
        cases.append(case)
    known = [c for c in cases if c["answerable"]]
    unknown = [c for c in cases if not c["answerable"]]
    metrics = {"k": k, "answerable": len(known), "unanswerable": len(unknown)}
    if known:
        metrics[f"page_hit@{k}"] = sum(c["rank"] is not None for c in known) / len(known)
        metrics[f"page_mrr@{k}"] = sum(1 / c["rank"] for c in known if c["rank"]) / len(known)
    if generator is not None:
        if known:
            metrics["answer_exact_match"] = sum(c["exact_match"] for c in known) / len(known)
            metrics["answer_and_page_correct"] = sum(c["exact_match"] and c["correct_page"] for c in known) / len(known)
            metrics["reference_containment_and_page"] = sum(c["reference_in_response"] and c["correct_page"] for c in known) / len(known)
            metrics["accepted_answer_rate"] = sum(c["status"] == "answered" for c in known) / len(known)
        if unknown:
            metrics["unanswerable_abstention_rate"] = sum(c["status"] != "answered" for c in unknown) / len(unknown)
    return {"metrics": metrics, "cases": cases}
