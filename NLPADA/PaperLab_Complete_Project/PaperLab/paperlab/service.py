"""Filesystem-backed single-user collection and a reusable application service."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import io
import os
import re
import shutil
import tempfile
import threading
import time
from pypdf import PdfReader
from .retrieval import read_papers
from .ranking import RetrievalIndex
from .rag import answer
from .generators import make_generator

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "data" / "starter"


class PaperService:
    def __init__(self, workspace=None):
        self.workspace = Path(workspace or os.getenv("PAPERLAB_WORKSPACE", ROOT / "workspace"))
        self.folder = self.workspace / "papers"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        # Seed only a genuinely new collection. Removing all PDFs later does not
        # silently restore starter files.
        marker = self.workspace / "initialized"
        if not marker.exists():
            for p in (STARTER / "papers").glob("*.pdf"):
                if not (self.folder / p.name).exists():
                    shutil.copy2(p, self.folder / p.name)
            marker.write_text("1\n")
        self.generators = {}
        self.rebuild()

    def rebuild(self):
        with self.lock:
            self.passages, self.issues = read_papers(self.folder)
            self.indexes = {name: RetrievalIndex(self.passages, mode=name)
                            for name in ("bm25", "tfidf", "hybrid")}

    def status(self):
        with self.lock:
            files = []
            for p in sorted(self.folder.glob("*.pdf")):
                pages = {x.page for x in self.passages if x.file == p.name}
                files.append({"name": p.name, "searchable_pages": len(pages),
                              "bytes": p.stat().st_size,
                              "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
            return {"documents": files, "chunks": len(self.passages),
                    "searchable_pages": len({(p.file, p.page) for p in self.passages}),
                    "issues": self.issues, "scope": "One local shared collection"}

    def upload(self, name, content):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_. -]{0,115}\.pdf", name, re.I):
            raise ValueError("Use a PDF filename with letters, numbers, spaces, dots, underscores or hyphens; no folders.")
        if not content.startswith(b"%PDF-"):
            raise ValueError("This file is not a PDF.")
        if len(content) > 20 * 1024 * 1024:
            raise ValueError("PDFs must be 20 MiB or smaller.")
        try:
            reader = PdfReader(io.BytesIO(content))
            if reader.is_encrypted:
                raise ValueError("Upload a PDF without a password.")
            if not 1 <= len(reader.pages) <= 200:
                raise ValueError("PDFs must contain 1 to 200 pages.")
            if not any((p.extract_text() or "").strip() for p in reader.pages):
                raise ValueError("No selectable text found. Run OCR before uploading.")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("The PDF could not be read.") from exc
        with self.lock:
            target = self.folder / name
            if target.exists():
                if target.read_bytes() == content:
                    return {"name": name, "status": "already_present"}
                raise FileExistsError("A different PDF already has this name. Rename your upload.")
            with tempfile.NamedTemporaryFile(dir=self.folder, suffix=".part", delete=False) as f:
                temp = Path(f.name)
                f.write(content)
            try:
                temp.replace(target)
                self.rebuild()
            except Exception:
                target.unlink(missing_ok=True)
                self.rebuild()
                raise
            finally:
                temp.unlink(missing_ok=True)
            return {"name": name, "status": "added"}

    def ask(self, question, *, mode="bm25", generator="extractive", k=3, min_coverage=0.23):
        started = time.perf_counter()
        with self.lock:
            if mode not in self.indexes:
                raise ValueError("Unknown retrieval mode.")
            if generator not in self.generators:
                self.generators[generator] = make_generator(generator)
            reply = answer(self.indexes[mode], self.generators[generator], question,
                           k=k, min_coverage=min_coverage)
        result = asdict(reply)
        result.update(question=question, retriever=mode, generator=generator,
                      answer_kind="evidence_sentence" if generator == "extractive" else "verified_literal_span",
                      latency_ms=1000 * (time.perf_counter() - started),
                      verification="Literal source match; semantic correctness still needs review.")
        return result

