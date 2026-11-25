"""Generate a short factual answer, then map its quoted span back to a PDF."""

from .retrieval import Hit

from dataclasses import dataclass
from pathlib import Path
import re



MODEL_ID = "google/flan-t5-small"
REVISION = "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"


@dataclass(frozen=True)
class Candidate:
    text: str
    context: str  # Exact portion of the retrieved passage visible to the generator.


class FlanGenerator:
    def __init__(self, model_id=MODEL_ID, *, device=None):
        self.model_id = str(model_id)
        self.device = device
        self.model = None

    def _load(self):
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install requirements-llm.txt to enable FLAN-T5, or choose offline evidence.") from exc
        import os
        threads = int(os.getenv("PAPERLAB_CPU_THREADS", "2"))
        if not 1 <= threads <= 32:
            raise ValueError("PAPERLAB_CPU_THREADS must be between 1 and 32.")
        torch.set_num_threads(threads)
        opts = {"revision": REVISION} if self.model_id == MODEL_ID and not Path(self.model_id).exists() else {}
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True, **opts)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id, use_safetensors=True, **opts)
        except OSError as exc:
            raise RuntimeError("Cannot load the generator. Connect for the first download or pass a local FLAN-T5 model directory.") from exc
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer

    @staticmethod
    def _prompt(question, context):
        return ("Answer with a short phrase copied from the passage. If the passage does "
                "not answer the question, write unanswerable.\n"
                f"Passage: {context}\nQuestion: {question}\nAnswer:")

    def _fit(self, question, passage, budget=512):
        # Fit the actual model tokenizer. Keep the whole question and never silently cut it off.
        def fits(text):
            return len(self.tokenizer.encode(self._prompt(question, text), verbose=False)) <= budget
        if not fits(""):
            raise ValueError("The question exceeds the generator input budget. Shorten it.")
        if fits(passage):
            return passage
        words = passage.split()
        lo, hi = 0, len(words)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if fits(" ".join(words[:mid])):
                lo = mid
            else:
                hi = mid - 1
        return " ".join(words[:lo])

    def propose(self, question, passage):
        self._load()
        import torch
        visible = self._fit(question, passage)
        if not visible:
            return Candidate("", "")
        inputs = self.tokenizer(self._prompt(question, visible), return_tensors="pt").to(self.device)
        with torch.inference_mode():
            tokens = self.model.generate(**inputs, do_sample=False, num_beams=3, max_new_tokens=44)
        return Candidate(self.tokenizer.decode(tokens[0], skip_special_tokens=True).strip(), visible)


@dataclass
class Answer:
    status: str
    text: str
    citation: str | None
    quote: str | None
    hits: list[Hit]

    def display(self):
        if self.citation:
            return f"{self.text} [{self.citation}]\nEvidence: “{self.quote}”"
        return self.text


def locate_span(guess, context):
    """Find literal wording with word boundaries; avoid partial decimals, units and negatives."""
    raw = re.sub(r"^answer\s*:\s*", "", guess.strip(), flags=re.I)
    phrase = raw.strip(" \"'“”‘’").rstrip(".!?").strip().strip("\"'“”‘’").strip()
    if not phrase or not re.search(r"\w", phrase) or phrase.casefold() in {
        "unanswerable", "unknown", "not enough information", "not mentioned", "yes", "no"
    }:
        return None
    left = r"(?<![\w.,+/−–-])" if phrase[0].isdigit() else r"(?<!\w)"
    right = r"(?![\w%/]|[.,]\d|[-–]\d)" if phrase[-1].isdigit() else r"(?!\w)"
    literal = r"\s+".join(re.escape(word) for word in phrase.split())
    return re.search(left + literal + right, context, re.I)


def evidence_excerpt(text, match, margin=165):
    # A bounded, verbatim substring of the normalized PDF passage.
    left, right = match.span()
    start = max(0, left - margin)
    end = min(len(text), right + margin)
    previous = list(re.finditer(r"[.!?]\s+", text[start:left]))
    if previous:
        start += previous[-1].end()
    elif start > 0:
        space = text.find(" ", start, left)
        if space != -1:
            start = space + 1
    following = re.search(r"[.!?](?=\s|$)", text[right:end])
    if following:
        end = right + following.end()
    elif end < len(text):
        space = text.rfind(" ", right, end)
        if space != -1:
            end = space
    return text[start:end].strip()


def answer(index, generator, question, *, k=3, min_coverage=0.23):
    if not 0 <= min_coverage <= 1:
        raise ValueError("min_coverage must be between 0 and 1.")
    hits = index.search(question, k)
    candidates = [hit for hit in hits if hit.coverage >= min_coverage]
    if not candidates:
        return Answer("low_retrieval", "I could not find enough relevant text in these papers.", None, None, hits)
    for hit in candidates:
        proposed = generator.propose(question.strip(), hit.passage.text)
        if not proposed.context or proposed.context not in hit.passage.text:
            continue
        match = locate_span(proposed.text, proposed.context)
        if match:
            return Answer("answered", match.group(), hit.passage.citation,
                          evidence_excerpt(proposed.context, match), hits)
    return Answer("no_supported_span", "I could not verify an answer in the retrieved passages.", None, None, hits)
