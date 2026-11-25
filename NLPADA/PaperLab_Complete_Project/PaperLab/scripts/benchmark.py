"""Measure the bundled fixtures; all metrics are descriptive, not held-out claims."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
try:
    import resource
except ImportError:
    resource = None
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paperlab.service import STARTER, ROOT
from paperlab.retrieval import read_papers
from paperlab.ranking import RetrievalIndex
from paperlab.evaluation import evaluate
from paperlab.generators import make_generator


def summary(values):
    values = sorted(values)
    return {"median": statistics.median(values), "p95": values[min(len(values)-1, int(0.95*len(values)))],
            "min": min(values), "max": max(values), "count": len(values)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flan", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT/"runs/benchmark.json")
    args=parser.parse_args()
    if args.flan:
        import torch
        torch.set_num_threads(2)
    labels=json.loads((STARTER/"qa.json").read_text())
    started=time.perf_counter();passages,issues=read_papers(STARTER/"papers")
    extraction_ms=1000*(time.perf_counter()-started)
    result={"scope":"3 teaching PDFs, 6 pages, 10 answerable questions and 3 unanswerable questions; these fixtures were visible during development.",
            "limitations":"Tiny development set. No held-out QA test, no human relevance study, and no claim of research novelty or general accuracy. Literal-span checks do not establish entailment.",
            "environment":{"python":platform.python_version(),"platform":platform.platform(),"cpu_count":os.cpu_count(),"torch_threads":2 if args.flan else None},
            "pdf_extraction_ms":extraction_ms,"chunks":len(passages),"issues":issues,"retrieval":{},"answers":{}}
    result["environment"]["versions"]={}
    for package in ("numpy","pypdf","fastapi","uvicorn","threadpoolctl","torch","transformers","sentencepiece","safetensors"):
        try:result["environment"]["versions"][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:pass
    for mode in ("bm25","tfidf","hybrid"):
        started=time.perf_counter();index=RetrievalIndex(passages,mode=mode)
        build_ms=1000*(time.perf_counter()-started)
        timings=[]
        for _ in range(30):
            for q in labels:
                started=time.perf_counter();index.search(q["question"],3)
                timings.append(1000*(time.perf_counter()-started))
        result["retrieval"][mode]={**evaluate(index,STARTER/"qa.json"),"build_ms":build_ms,"latency_ms":summary(timings)}
        print(mode,result["retrieval"][mode]["metrics"],flush=True)
    index=RetrievalIndex(passages)
    for name in (["extractive","flan"] if args.flan else ["extractive"]):
        generator=make_generator(name)
        load_ms=0
        if name=="flan":
            started=time.perf_counter();generator._load();load_ms=1000*(time.perf_counter()-started)
        report=evaluate(index,STARTER/"qa.json",generator=generator)
        result["answers"][name]={**report,"model_load_ms":load_ms,"latency_ms":summary([c["latency_ms"] for c in report["cases"]])}
        print(name,report["metrics"],flush=True)
    # Chunk ablation measures retrieval only, keeping questions and ranking fixed.
    result["chunk_ablation"]=[]
    for size,overlap in [(65,0),(65,14),(130,0),(130,28),(200,40)]:
        chunks,_=read_papers(STARTER/"papers",size=size,overlap=overlap)
        result["chunk_ablation"].append({"size":size,"overlap":overlap,"chunks":len(chunks),
                    **evaluate(RetrievalIndex(chunks),STARTER/"qa.json")["metrics"]})
    result["process_peak_rss_mib_linux"]=(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if resource and sys.platform.startswith("linux") else None)
    result["memory_note"]="Whole process peak RSS, including model loading and evaluation; not isolated model memory. Linux units."
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print("Saved",args.output)


if __name__=="__main__":main()
