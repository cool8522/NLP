# PaperLab Research Paper Assistant

A complete local project built from the supplied `Basic.ipynb`: custom byte-level BPE, a NumPy Mini-GPT with manual gradients, and a PDF question-answering application with inspectable page evidence.

**Read `docs/PROJECT_GUIDE.md` or `docs/PaperLab_Project_Guide.pdf` for the full explanation.** Start with the distinction between the language-model learning lab and the practical PDF assistant. Mini-GPT is not used as the factual QA engine.

## Quick start on Fedora or Linux

Use Python 3.12, the version tested for this release. Python 3.11 is also a reasonable installation target, but was not tested here. You do not need Docker, Node.js, a database, an API key, or a GPU for the default mode.

```bash
unzip PaperLab_Complete_Project.zip
cd PaperLab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py
```

Open **http://127.0.0.1:8000**. Stop with **Ctrl+C**. If port 8000 is occupied, run `python run.py --port 8001` and open that port. If your `python3` is 3.15 and optional ML packages do not install, use a Python 3.12 interpreter to create the virtual environment.

Windows PowerShell: use `py -3.12 -m venv .venv`, then `.venv\Scripts\Activate.ps1`; the remaining Python commands are the same. Windows was not tested in this session.

The ZIP includes a trained Mini-GPT checkpoint, so Model lab works immediately. On its first run, the application copies the three teaching PDFs into `workspace/papers`. Upload your own searchable PDF with **Add a PDF**. Each upload is limited to 20 MiB and 200 pages. Questions are not saved to a server history; export an answer with the Markdown or JSON download button.

## Enable the pretrained answer model

```bash
python -m pip install -r requirements-llm.txt
python run.py
```

Choose **FLAN-T5** in Answer mode. The first question downloads `google/flan-t5-small` at a pinned revision. Later runs use the local cache. CPU works; GPU support depends on your PyTorch installation. Model files are downloaded separately, not included in the project ZIP.

For an explicitly CPU-only PyTorch installation, the version tested here was installed using:

```bash
python -m pip install torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-llm.txt
```

Offline evidence is always available with the core dependencies. It selects a relevant sentence and does not use an LLM. It can still choose the wrong sentence.

## Use an existing Ollama installation

```bash
ollama pull qwen3:4b
ollama serve
```

Run `python run.py` in another terminal and choose **Ollama**. If Ollama is already running as a service, do not start a second server. The adapter talks only to a local host. Qwen is optional; its actual inference was not run in the build environment.

Optional environment variables are read from the shell, not automatically from a `.env` file:

```bash
export PAPERLAB_OLLAMA_MODEL=qwen3:4b
export PAPERLAB_OLLAMA_URL=http://127.0.0.1:11434
export PAPERLAB_CPU_THREADS=2
```

`PAPERLAB_WORKSPACE` changes the application's local data directory. `PAPERLAB_FLAN_MODEL` can point to a locally saved FLAN-T5 directory.

## Reproduce the experiments

```bash
python -m paperlab train --steps 400
python -m paperlab sample --prompt "Research questions"
python -m paperlab ask "What was the alarm threshold in the distance experiment?"
python -m paperlab ask "What was the alarm threshold in the distance experiment?" --generator flan
python -m paperlab evaluate --generator flan
python scripts/benchmark.py --flan
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

Without FLAN installed, run `python scripts/benchmark.py` and `python -m paperlab evaluate`. A new benchmark run replaces `runs/benchmark.json`; the dashboard displays the last saved run. Training selects the best measured validation checkpoint, not the final training step.

## Optional published data

```bash
python -m paperlab download papers
python -m paperlab ask "What is byte pair encoding?" --papers data/published --generator flan
python -m paperlab download qasper
python -m paperlab train --corpus data/qasper_abstracts.jsonl --output runs/qasper/mini_gpt.npz
```

The published-paper command downloads BPE, BERT and QASPER papers from ACL Anthology. The QASPER command imports 200 training and 50 development titles/abstracts while preserving official split identities. This is an optional language-model corpus, **not a QASPER QA benchmark**. The CLI folder and dashboard collection are separate; upload a downloaded paper if you want to query it in the dashboard.

## What was verified

- 22 automated tests passed: gradient checks, causality, Unicode round trips, split isolation, checkpoint loading, retrieval, citation validation and API behavior.
- Mini-GPT: 79,584 parameters; best validation loss 3.839 at step 250; perplexity 46.49.
- Teaching collection: Page Hit@3 100%; MRR@3 0.95; FLAN exact answer plus correct page 7/10; abstention 3/3 on unanswerable questions.
- The real FLAN model was downloaded and evaluated on CPU. See per-question results in `runs/benchmark.json`.
- Visual browser verification was blocked by the execution environment. The dashboard routes and local HTTP flow were checked separately; no claim of a completed visual UI check is made.

These are small, visible development fixtures. They do not establish general research-paper accuracy. A copied phrase can answer the wrong question. BM25, TF-IDF and the combined ranker all tied on retrieval accuracy here; the combination is not demonstrated to be better.

## Files to explore

| Path | Contents |
|---|---|
| `paperlab/` | Tokenizer, model, training, retrieval, generators, API and application service |
| `paperlab/static/` | HTML, CSS and JavaScript dashboard |
| `data/starter/` | 36 educational text documents, 3 teaching PDFs and 13 QA labels |
| `runs/` | Checkpoint, tokenizer, training history and measured evaluation |
| `tests/` | Standard-library unittest suite; API tests also use httpx |
| `scripts/benchmark.py` | Retrieval, answer and chunking experiments |
| `notebooks/PaperLab_End_to_End.ipynb` | Annotated notebook; local or Google Colab |
| `docs/PROJECT_GUIDE.md` | Complete technical explanation and viva preparation |

This is a local, single-user academic prototype. Authentication, multi-user separation, OCR, table understanding, semantic embeddings and production deployment are not implemented.
