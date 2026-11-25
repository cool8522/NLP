"""Document-level train/validation splits, AdamW, checkpointing and sampling."""

from .tokenizer import ByteBPE, PAD, BOS, EOS
from .model import MiniGPT, Config, cross_entropy, softmax

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np
from threadpoolctl import threadpool_limits



def load_corpus(path):
    groups = {"train": [], "validation": []}
    seen_ids, seen_text = set(), set()
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        split = row["split"]
        text = row["text"].strip()
        fingerprint = " ".join(text.casefold().split())
        if split not in groups or not row["id"] or not text:
            raise ValueError(f"Invalid document at line {line_no}.")
        if row["id"] in seen_ids or fingerprint in seen_text:
            raise ValueError(f"Duplicate document at line {line_no}.")
        seen_ids.add(row["id"])
        seen_text.add(fingerprint)
        groups[split].append(text)
    if not all(groups.values()):
        raise ValueError("The corpus needs at least one train and one validation document.")
    return groups


def make_batch(items, context):
    """Each item is (document token array, start offset). Windows never cross papers."""
    x = np.full((len(items), context), PAD, dtype=np.int64)
    y = np.full_like(x, -1)
    for row, (doc, start) in enumerate(items):
        seq = doc[start:start + context + 1]
        n = len(seq) - 1
        x[row, :n], y[row, :n] = seq[:-1], seq[1:]
    return x, y


def evaluate_lm(model, docs, batch_size=12):
    context = model.config.context
    windows = [(doc, pos) for doc in docs for pos in range(0, len(doc) - 1, context)]
    nll, n = 0.0, 0
    for start in range(0, len(windows), batch_size):
        x, y = make_batch(windows[start:start + batch_size], context)
        count = int((y >= 0).sum())
        nll += count * cross_entropy(model.forward(x), y)
        n += count
    return nll / n


class AdamW:
    def __init__(self, weights, lr):
        self.lr = lr
        self.m = {key: np.zeros_like(w) for key, w in weights.items()}
        self.v = {key: np.zeros_like(w) for key, w in weights.items()}
        self.step_number = 0

    def step(self, weights, grads):
        self.step_number += 1
        t = self.step_number
        norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
        if not math.isfinite(norm):
            raise FloatingPointError("Non-finite gradient. Try a smaller learning rate.")
        clip = 1 / max(1.0, norm)
        for name, weight in weights.items():
            g = grads[name] * clip
            m, v = self.m[name], self.v[name]
            m *= 0.9
            m += 0.1 * g
            v *= 0.999
            v += 0.001 * g * g
            if weight.ndim > 1:
                weight *= 1 - self.lr * 0.01
            weight -= self.lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)


def save_checkpoint(path, model, tokenizer, report):
    """Bundle token ids with the matching weights in a single atomic, pickle-free file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"format": 1, "config": asdict(model.config), "merges": tokenizer.merges, "report": report}
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temp = Path(handle.name)
    try:
        np.savez_compressed(temp, metadata=json.dumps(metadata), **model.weights)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["metadata"]))
        if meta["format"] != 1:
            raise ValueError("Unsupported checkpoint version.")
        tok = ByteBPE(meta["merges"])
        model = MiniGPT(Config(**meta["config"]))
        if tok.vocab_size != model.config.vocab:
            raise ValueError("Tokenizer and model vocabularies disagree.")
        for name, weight in model.weights.items():
            saved = archive[name]
            if weight.shape != saved.shape:
                raise ValueError(f"Wrong weight shape for {name}.")
            weight[:] = saved
    return model, tok, meta["report"]


def train(corpus, output, *, steps=400, batch_size=8, lr=0.003, vocab_size=448,
          context=48, width=48, heads=4, layers=2, seed=17, log=print):
    if steps < 1 or batch_size < 1 or not (0 < lr < 1):
        raise ValueError("Use positive steps and batch size and a learning rate between 0 and 1.")
    text = load_corpus(corpus)
    tokenizer = ByteBPE.train(text["train"], vocab_size)
    docs = {s: [np.asarray(tokenizer.encode(d, boundaries=True)) for d in items] for s, items in text.items()}
    model = MiniGPT(Config(tokenizer.vocab_size, context, width, heads, layers), seed)
    optim = AdamW(model.weights, lr)
    rng = np.random.default_rng(seed)
    num_starts = np.asarray([max(1, len(doc) - context) for doc in docs["train"]])
    ends = np.cumsum(num_starts)
    interval = max(1, steps // 8)
    history, best, best_step, snapshot = [], float("inf"), 0, None
    with threadpool_limits(limits=1, user_api="blas"):
        for step in range(steps + 1):
            if step % interval == 0 or step == steps:
                train_loss = evaluate_lm(model, docs["train"])
                val_loss = evaluate_lm(model, docs["validation"])
                history.append({"step": step, "train_loss": train_loss, "validation_loss": val_loss})
                if val_loss < best:
                    best, best_step = val_loss, step
                    snapshot = {k: w.copy() for k, w in model.weights.items()}
                if log:
                    log(f"step {step:>4} | train {train_loss:.3f} | validation {val_loss:.3f}")
            if step == steps:
                break
            starts = rng.integers(0, ends[-1], size=batch_size)
            doc_ids = np.searchsorted(ends, starts, side="right")
            offsets = starts - (ends[doc_ids] - num_starts[doc_ids])
            x, y = make_batch([(docs["train"][i], j) for i, j in zip(doc_ids, offsets)], context)
            logits, cache = model.forward(x, with_cache=True)
            _, dlogits = cross_entropy(logits, y, gradient=True)
            optim.step(model.weights, model.backward(dlogits, cache))
    model.weights = snapshot
    report = {
        "seed": seed, "steps": steps, "batch_size": batch_size, "learning_rate": lr,
        "corpus_sha256": hashlib.sha256(Path(corpus).read_bytes()).hexdigest(),
        "documents": {s: len(rows) for s, rows in docs.items()},
        "next_token_targets": {s: sum(len(doc) - 1 for doc in rows) for s, rows in docs.items()},
        "parameters": model.parameter_count, "vocabulary": tokenizer.vocab_size,
        "best_step": best_step, "best_validation_loss": best, "validation_perplexity": math.exp(best),
        "history": history
    }
    save_checkpoint(output, model, tokenizer, report)
    return model, tokenizer, report


def generate(model, tokenizer, prompt="", *, max_new_tokens=64, temperature=0.7, top_k=20, seed=17):
    if temperature <= 0 or top_k < 1 or max_new_tokens < 0:
        raise ValueError("Use positive temperature and top_k and a nonnegative length.")
    rng = np.random.default_rng(seed)
    ids = [BOS, *tokenizer.encode(prompt)]
    with threadpool_limits(limits=1, user_api="blas"):
        for _ in range(max_new_tokens):
            x = np.asarray([ids[-model.config.context:]])
            logits = model.forward(x)[0, -1].copy()
            logits[[PAD, BOS]] = -np.inf
            k = min(top_k, len(logits) - 2)
            choices = np.argsort(-logits, kind="stable")[:k]
            probs = softmax(logits[choices].astype(np.float64) / temperature)
            idx = int(rng.choice(choices, p=probs))
            if idx == EOS:
                break
            ids.append(idx)
    # Randomly sampled bytes can form incomplete UTF-8 sequences.
    return tokenizer.decode(ids, errors="replace")
