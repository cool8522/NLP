"""Optional real papers for retrieval and official QASPER abstracts for language modeling."""


import json
import os
from pathlib import Path
import tarfile
import tempfile
from urllib.request import Request, urlopen
import shutil

from pypdf import PdfReader


PAPERS = {
    "Sennrich_2016_BPE.pdf": "https://aclanthology.org/P16-1162.pdf",
    "Devlin_2019_BERT.pdf": "https://aclanthology.org/N19-1423.pdf",
    "Dasigi_2021_QASPER.pdf": "https://aclanthology.org/2021.naacl-main.365.pdf",
}
QASPER_ARCHIVE = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"


def _download(url, destination):
    request = Request(url, headers={"User-Agent": "paperlab-academic-project/1.0"})
    with urlopen(request, timeout=60) as response, open(destination, "wb") as out:
        shutil.copyfileobj(response, out)


def download_papers(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, url in PAPERS.items():
        target = folder / name
        if target.is_file():
            if len(PdfReader(str(target)).pages) > 0:
                continue
        with tempfile.NamedTemporaryFile(dir=folder, suffix=".part", delete=False) as f:
            part = Path(f.name)
        try:
            _download(url, part)
            if len(PdfReader(str(part)).pages) < 1:
                raise ValueError(f"No pages in {url}")
            part.replace(target)
        finally:
            part.unlink(missing_ok=True)
    return [folder / name for name in PAPERS]


def import_qasper(archive, output, *, train_limit=200, validation_limit=50):
    """Use titles/abstracts only, preserving official paper-level train and dev splits."""
    if min(train_limit, validation_limit) < 1:
        raise ValueError("Both QASPER split limits must be positive.")
    rows, seen_text, seen_ids = [], set(), set()
    with tarfile.open(archive, "r:gz") as tar:
        members = {Path(item.name).name: item for item in tar if item.isfile()}
        for name, split, limit in [("train", "train", train_limit), ("dev", "validation", validation_limit)]:
            member = members[f"qasper-{name}-v0.3.json"]
            with tar.extractfile(member) as f:
                papers = json.load(f)
            count = 0
            for paper_id, record in sorted(papers.items()):
                abstract = (record.get("abstract") or "").strip()
                if not abstract:
                    continue
                text = (record["title"].strip() + "\n" + abstract).strip()
                normalized = " ".join(text.casefold().split())
                if normalized in seen_text or paper_id in seen_ids:
                    continue
                seen_text.add(normalized)
                seen_ids.add(paper_id)
                rows.append({"id": paper_id, "split": split, "text": text})
                count += 1
                if count == limit:
                    break
            if not count:
                raise ValueError(f"The QASPER {name} split contains no usable abstracts.")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, delete=False) as f:
        temp = Path(f.name)
        try:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    try:
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)
    return {s: sum(r["split"] == s for r in rows) for s in ("train", "validation")}


def download_qasper(output, *, train_limit=200, validation_limit=50):
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "qasper.tgz"
        _download(QASPER_ARCHIVE, archive)
        return import_qasper(archive, output, train_limit=train_limit, validation_limit=validation_limit)
