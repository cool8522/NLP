"""Reproducible command-line workflows for PaperLab."""
import argparse
import json
from pathlib import Path
import time
from .service import ROOT, STARTER


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("train", help="Train the custom BPE + NumPy Mini-GPT")
    p.add_argument("--corpus", type=Path, default=STARTER / "corpus.jsonl")
    p.add_argument("--output", type=Path, default=ROOT / "runs" / "mini_gpt.npz")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=17)
    p = commands.add_parser("ask", help="Ask about a PDF folder")
    p.add_argument("question")
    p.add_argument("--papers", type=Path, default=STARTER / "papers")
    p.add_argument("--generator", choices=["extractive", "flan", "ollama"], default="extractive")
    p.add_argument("--retriever", choices=["bm25", "tfidf", "hybrid"], default="bm25")
    p.add_argument("--k", type=int, default=3)
    p = commands.add_parser("sample", help="Sample the educational Mini-GPT")
    p.add_argument("--prompt", default="Research questions")
    p.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "mini_gpt.npz")
    p = commands.add_parser("evaluate", help="Run labeled retrieval and answer evaluation")
    p.add_argument("--papers", type=Path, default=STARTER / "papers")
    p.add_argument("--labels", type=Path, default=STARTER / "qa.json")
    p.add_argument("--generator", choices=["extractive", "flan", "ollama"], default="extractive")
    p.add_argument("--output", type=Path, default=ROOT / "runs" / "evaluation.json")
    p = commands.add_parser("download", help="Optionally fetch published PDFs or QASPER abstracts")
    p.add_argument("what", choices=["papers", "qasper"])
    args = parser.parse_args()
    try:
        if args.command == "train":
            from .training import train
            started = time.perf_counter()
            model, tok, report = train(args.corpus, args.output, steps=args.steps, seed=args.seed)
            report["wall_seconds"] = time.perf_counter() - started
            args.output.parent.mkdir(parents=True, exist_ok=True)
            (args.output.parent / "training.json").write_text(json.dumps(report, indent=2))
            tok.save(args.output.parent / "tokenizer.json")
            print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))
        elif args.command == "sample":
            from .training import load_checkpoint, generate
            m, t, _ = load_checkpoint(args.checkpoint)
            print(generate(m, t, args.prompt))
        elif args.command in {"ask", "evaluate"}:
            from .retrieval import read_papers
            from .ranking import RetrievalIndex
            from .generators import make_generator
            passages, issues = read_papers(args.papers)
            index = RetrievalIndex(passages, mode=getattr(args, "retriever", "bm25"))
            generator = make_generator(args.generator)
            if args.command == "ask":
                from .rag import answer
                print(answer(index, generator, args.question, k=args.k).display())
                print("\nMode:", args.generator, "| Literal evidence is not a guarantee of semantic correctness.")
            else:
                from .evaluation import evaluate
                report = evaluate(index, args.labels, generator=generator)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2))
                print(json.dumps(report["metrics"], indent=2))
            if issues:
                print("PDF notices:", issues)
        elif args.command == "download":
            from .datasets import download_papers, download_qasper
            if args.what == "papers":
                print(download_papers(ROOT / "data" / "published"))
            else:
                print(download_qasper(ROOT / "data" / "qasper_abstracts.jsonl"))
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"PaperLab: {exc}\n")


if __name__ == "__main__":
    main()
