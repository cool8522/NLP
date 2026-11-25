# PaperLab Research Paper Assistant

## Complete implementation and explanation

PaperLab helps a reader find evidence in a collection of research PDFs and inspect the page supporting a proposed answer. It also provides a small language-model laboratory where the reader can study tokenization, Transformer computation, gradient calculation and training in code.

This guide explains the completed implementation derived from the supplied Basic.ipynb. It covers the problem, architecture, algorithms, design choices, measured performance, computational complexity, source files, practical benefits, limitations and viva questions. The intended reader is an M.Tech student who wants to run the project, understand its internals and explain it honestly in a demonstration.

The main result is a runnable local prototype with sample data, a trained 79,584-parameter model, a PDF dashboard, three retrieval methods, optional pretrained answer generation, automated checks and reproducible evaluation. Its small teaching benchmark establishes that the workflow runs; it does not establish broad scientific-paper understanding.

## 1 What problem does the project solve

Research papers distribute information across abstracts, methods, experimental settings and results. Finding one threshold, dataset, evaluation metric or experimental condition can require reading several pages. A conventional keyword search finds occurrences but leaves the reader to identify the useful passage and its context. A general chatbot can provide fluent answers without having the relevant paper available.

PaperLab addresses the evidence-location part of this problem. The user supplies searchable PDFs, asks a focused question and receives either a source-matched answer or a message that no answer could be verified. The interface exposes retrieved passages and physical PDF page numbers so that the user can check the result.

For example, a user asks what alarm threshold was used in the distance experiment. The assistant retrieves the relevant page, proposes the span 80 centimetres and verifies that those words occur in the model-visible passage. The citation identifies sensor_filter.pdf, page 1. The original PDF remains available for inspection.

The educational problem is different: many student projects call an LLM without explaining how tokenization, attention or training work. PaperLab includes a custom tokenizer and a manually differentiated Transformer so these operations can be inspected and tested.

## 2 Objectives and implemented scope

The practical objective is to make focused document questions easier to investigate. The engineering objectives are to keep the default system runnable on CPU, preserve document provenance, handle ordinary input errors and provide measurements that another student can reproduce. The learning objective is to connect NLP and deep-learning theory to a working implementation.

Implemented features include PDF ingestion, page-local chunking, lexical indexes, retrieval comparison, offline evidence selection, FLAN-T5 generation, a local Ollama adapter, answer-span validation, page links, export, a tokenizer laboratory, model sampling, checkpointing and evaluation. The application uses one collection on one computer.

Unimplemented features include OCR for scanned pages, reliable table or equation interpretation, comprehensive multi-paper synthesis, authenticated accounts, multi-tenant isolation, distributed indexing, a semantic embedding index and a trained entailment verifier. These are extension opportunities rather than capabilities to claim in a presentation.

## 3 The two parts of the architecture

PaperLab has two connected learning and application workflows. They share the repository and interface, but their models and tokenizers have different roles.

| Workflow | Input | Processing | Output |
|---|---|---|---|
| Language-model lab | Educational text corpus | Train-only byte BPE, token windows, NumPy Mini-GPT, cross-entropy, manual gradients, AdamW | Saved tokenizer, checkpoint, loss history and text samples |
| PDF assistant | Searchable PDFs and a question | Extract pages, split passages, lexical retrieval, answer proposal, literal-span verification | Evidence, page citation or abstention |

The custom Mini-GPT is trained from scratch to predict the next token. It is not instruction-tuned, has a context of only 48 BPE tokens and learns from a very small corpus. It is therefore not the factual question-answering engine. FLAN-T5 or Ollama supplies that optional engine; the offline mode selects a sentence without any generative model.

FLAN-T5 uses its own pretrained tokenizer. Feeding custom BPE IDs directly to FLAN would be incorrect because token IDs have meanings tied to each model's embedding matrix. Keeping the two tokenizers separate is required for correctness.

The PDF system follows the retrieve-then-generate pattern when FLAN or Ollama is selected. Offline evidence mode is retrieval plus extraction, not neural generative RAG. This implementation is also not a reproduction of the jointly trained retriever-generator system in the original RAG research paper.

## 4 Technologies and why they were chosen

| Technology | Role | Reason for use |
|---|---|---|
| Python | Application, algorithms, CLI and tests | One language covers the numerical and document workflows |
| NumPy | Mini-GPT tensors and manual backpropagation | Makes matrix operations and derivatives explicit |
| pypdf | Text extraction and physical page enumeration | Lightweight parsing of text-based PDFs |
| Python dictionaries and counters | BPE statistics and sparse postings | Inspectable algorithms without a separate database |
| threadpoolctl | BLAS thread control | Avoids excessive overhead for small training matrices |
| FastAPI and Pydantic | Local API and request validation | Typed request bodies and straightforward endpoints |
| Uvicorn | Local HTTP server | Runs the FastAPI application |
| HTML, CSS and JavaScript | Dashboard | No frontend build system or Node installation required |
| PyTorch and Transformers | Optional FLAN inference | Load and execute the pretrained T5 model |
| SentencePiece and safetensors | Optional model assets | Tokenizer support and non-pickle model weights |
| Ollama HTTP API | Optional local Qwen or other model | Reuses an existing local model service |
| unittest and httpx | Algorithm and API tests | Verify behavior without a separate test framework |
| JSON, JSONL and NPZ | Data, metrics and checkpoint files | Portable, inspectable storage for this project scale |

There is no MongoDB, SQL server, vector database, LangChain or paid API dependency. This is a deliberate size and teaching tradeoff. A production collection may need durable indexes, job processing and access controls that this local application does not provide.

## 5 Data and provenance

The starter corpus contains 36 educational documents: 30 training documents and 6 validation documents. Each JSONL row has id, split and text fields. These are short teaching notes rather than a large collection of scientific papers.

The retrieval collection contains three teaching PDFs: sensor_filter.pdf, window_overlap.pdf and retrieval_ranks.pdf. Each has two pages. Their controlled examples describe median filtering, overlapping passages and retrieval metrics. The QA file contains ten answerable questions with expected filename, physical page and acceptable answer strings, plus three unanswerable questions.

The embedded data from the original notebook was decoded from Base64 and its SHA-256 checked against the notebook's expected value. Base64 is a transport encoding of ZIP bytes; it is not training, encryption or a model. The delivered project exposes the files directly in data/starter, so the user can inspect them without decoding a bundle. provenance.json records source and data hashes.

The training code rejects duplicate IDs and duplicate normalized texts, including duplicates across the training and validation partitions. BPE learns merges from training documents only. This prevents those specific leakage paths. It does not prove absence of semantic overlap between related teaching notes, nor does it make the visible QA fixtures a blind test set.

An optional downloader obtains three published ACL papers for document experiments. Another imports 200 QASPER training and 50 development titles/abstracts while preserving official paper-level split assignments. It reads JSON members from the archive without extracting arbitrary archive paths. The importer was designed for language-model training: it does not load QASPER answer annotations or run the official QASPER QA evaluation. The optional corpus is separate from the starter results in this guide.

## 6 Custom byte-level BPE explained

Language models operate on integer token IDs rather than raw strings. Whole-word tokenization requires a very large vocabulary and struggles with unfamiliar words. Character or byte tokenization covers many inputs but produces long sequences. BPE learns frequent adjacent combinations to reduce sequence length while retaining a fallback representation.

The tokenizer starts with IDs 0 through 255 for UTF-8 bytes. ID 256 is PAD, 257 is BOS and 258 is EOS. Learned merges begin at ID 259. The requested vocabulary is 448, so at most 189 merges are learned. A vocabulary can stop earlier if no pair reaches the minimum frequency.

Training first splits text into regex pieces while preserving spaces and newlines. Each piece becomes a tuple of UTF-8 byte values. A Counter records how often each piece occurs. For every merge iteration, the algorithm counts adjacent pairs with those frequencies, chooses the most frequent pair, resolves ties deterministically by the pair IDs and replaces non-overlapping occurrences. It repeats until the vocabulary target or stopping rule is reached.

The learned tokenizer's first merge in this run combines byte 32, a space, with byte 116, the letter t. Another early merge combines h and e. As merges accumulate, common pieces become compact IDs. With the saved tokenizer, Research questions occupies 18 UTF-8 bytes and becomes six IDs: 82, 272, 101, 279, 331 and 444.

Encoding uses the learned merge ranks; it does not relearn merges on the user's question. Decoding concatenates the underlying bytes and decodes UTF-8. The encode/decode round trip therefore preserves unseen scripts and emoji without an unknown-token fallback. Randomly sampled model tokens can still form invalid or incomplete UTF-8; sampling displays replacement characters rather than crashing. Unicode coverage is not the same as multilingual understanding.

The tokenizer saves only its merge list and a format marker in JSON. _encode_piece is cached for up to 16,384 pieces, which makes repeated text cheaper to encode. A fresh process has a cold cache.

## 7 Mini-GPT architecture and dimensions

The custom model is a decoder-only, pre-normalized Transformer implemented in NumPy. Its configuration is vocabulary V=448, maximum context T=48, width d=48, heads H=4 and layers L=2. Each attention head therefore has width 12. The feed-forward hidden width is 4d=192. Training batch size is B=8.

For a full batch, input IDs have shape [8,48]. Token embeddings map these IDs to [8,48,48]. Learned position embeddings of shape [48,48] are added so that the model can distinguish token positions. Unlike attention without position information, this model has an explicit mechanism for order.

Each of the two Transformer blocks contains attention layer normalization, a combined QKV projection, causal multi-head attention, an output projection and residual addition, then feed-forward layer normalization, an expanding projection, GELU, a contracting projection and another residual addition. A final layer normalization precedes the vocabulary projection.

| Array | Shape for the default model |
|---|---|
| Token embeddings | [448,48] |
| Position embeddings | [48,48] |
| Combined QKV projection per block | [48,144] |
| Q, K and V after splitting heads | [8,4,48,12] each |
| Attention probabilities | [8,4,48,48] |
| Attention output projection | [48,48] |
| Feed-forward expansion | [48,192] |
| Feed-forward contraction | [192,48] |
| Final batch logits | [8,48,448] |

The input token embeddings and output vocabulary projection share weights. This is weight tying. The model omits linear projection biases and dropout; layer normalizations have learned scale and bias. These choices keep the implementation small and understandable.

The parameter count follows directly from the implemented matrices:

```text
P = V*d + T*d + L*(12*d*d + 4*d) + 2*d
  = 448*48 + 48*48 + 2*(12*48*48 + 4*48) + 2*48
  = 79,584 parameters
```

Float32 weights alone occupy 318,336 bytes, approximately 0.304 MiB. This excludes activations, gradients, optimizer state, Python overhead, model copies and library memory. Checkpoint compression changes file size, not the number of parameters.

## 8 Attention step by step

Let X be the current hidden representation. After layer normalization, a matrix multiplication produces Q, K and V. Q describes what each token is looking for; K describes what each token offers for matching; V contains the information to combine. These are learned linear projections, not manually assigned linguistic labels.

Within one head, attention scores are Q multiplied by the transpose of K, divided by the square root of the head width. The scaling helps keep dot-product magnitudes manageable. The causal mask sets scores for future positions to negative infinity before softmax. Softmax therefore gives those future positions zero probability.

```text
scores = (Q @ K_transpose) / sqrt(head_width)
A = softmax(scores + causal_mask)
head_output = A @ V
```

The four head outputs are joined, projected back to width 48 and added to the original hidden representation. The feed-forward sublayer then transforms each position separately. Residual paths provide direct routes for activations and gradients. Layer normalization controls activation statistics; GELU introduces nonlinearity.

The implementation uses learned absolute positional embeddings. It does not use rotary embeddings, a key/value cache, sparse attention or FlashAttention. Its attention matrix is explicitly dense. Those details matter when describing complexity and performance.

## 9 Training and manual backpropagation

The language-model task is next-token prediction. For a document token sequence [BOS, a, b, c, EOS], an input window [BOS,a,b,c] is paired with targets [a,b,c,EOS]. Each training row comes from one document; windows never concatenate the end of one paper with the start of another.

Short windows use PAD in the input and -1 for ignored targets. Because padding comes after meaningful tokens and attention is causal, earlier valid positions cannot attend to those later padding positions. The cross-entropy function excludes ignored targets from both loss and gradient.

Cross-entropy is the average negative log probability of the correct next token. Stable softmax subtracts the largest logit before exponentiation. For labeled positions, the logit gradient is predicted probability minus the one-hot target, divided by the number of valid targets. Ignored positions receive zero gradient.

The backward method manually propagates derivatives through the tied output projection, final normalization, feed-forward layers, GELU, residual paths, attention softmax, QK products and input embeddings. Shared embedding weights receive gradient contributions from both the input lookup and output projection. np.add.at correctly accumulates repeated token IDs; a simple overwrite would lose gradients.

AdamW maintains first and second moments of every parameter gradient. The update uses beta1=0.9, beta2=0.999 and epsilon=1e-8. Global gradient norm is clipped to at most 1. Matrix parameters receive decoupled weight decay of 0.01; one-dimensional normalization parameters do not. The learning rate is a constant 0.003 in the default run. There is no learning-rate scheduler.

There are 400 optimizer steps, not 400 full epochs. Random windows are sampled from available document starts. Evaluation traverses document windows at regular intervals and computes a target-count-weighted mean loss. The best evaluated validation checkpoint is restored after training. The validation set is used for model selection, so it is not an untouched final test set.

The checkpoint stores config, tokenizer merges, report and NumPy weights in a compressed NPZ archive. Loading uses allow_pickle=False and validates vocabulary and tensor shapes. It is an inference checkpoint: optimizer moments and RNG state are not saved for exact interrupted-training resumption.

## 10 Generation from Mini-GPT

Sampling begins with BOS and the encoded prompt. The model sees the most recent 48 IDs, produces next-token logits and suppresses PAD and BOS. Temperature rescales logits, top-k retains the highest-scoring choices and a seeded random sampler picks the next ID. Generation stops on EOS or the requested token limit.

The default temperature is 0.7 and top-k is 20. A lower temperature generally concentrates probability on likely continuations; it does not make factual answers trustworthy. The sampled text in this small run contains partly learned phrases and malformed words. That behavior is consistent with the model size and corpus; it is not hidden or labeled as successful document QA.

Every generation step recomputes the cropped context. There is no key/value cache, and positional indices restart within that context window. These choices simplify the educational implementation but limit efficiency and long-context coherence.

## 11 PDF extraction and chunking

read_papers recursively discovers PDFs in a chosen CLI folder, opens them with pypdf and enumerates physical pages starting from 1. The physical index can differ from a page number printed by the paper. Each extracted page is normalized and split into word windows.

The default chunk size is 130 whitespace-delimited words, with overlap 28. These are words for document chunking, not BPE tokens. Chunking never crosses a physical page, so each passage has an unambiguous filename and page citation. Each Passage stores file, page, chunk and normalized text.

Overlap repeats boundary words in neighboring chunks. This helps when an answer-bearing phrase would otherwise be split. It increases storage and retrieval work. A rough long-page redundancy factor is size divided by size minus overlap: 130/102 is approximately 1.27. Pages shorter than 130 words produce one chunk, so the six starter pages produce six chunks and the default overlap adds no redundancy in this fixture.

Unreadable PDFs, extraction failures, password requirements and blank pages produce notices. The dashboard rejects uploaded PDFs without selectable text. The system does not run OCR automatically and does not guarantee correct reading order in complex multi-column layouts. Charts, formula images and tables may lose structure even when some text is extracted.

## 12 BM25 retrieval

The retrieval tokenizer lowercases with Unicode case folding, normalizes NFKC, extracts alphanumeric terms and removes a fixed stopword list. It is independent of the custom BPE tokenizer. This lexical representation does not provide stemming or semantic embeddings.

An inverted index maps each term to passages containing it and the term's frequency there. BM25 combines inverse document frequency, term-frequency saturation and passage-length normalization. A rare query term contributes more than a term appearing in nearly every passage. Repeated appearances help, but their contribution saturates.

```text
idf(t) = log(1 + (N - df(t) + 0.5)/(df(t) + 0.5))
score(t,D) = idf(t) * tf(t,D)*(k1+1)
             / (tf(t,D) + k1*(1-b+b*len(D)/avg_len))
BM25(q,D) = sum of score(t,D) over unique query terms
```

Here N is the number of passages, df is passage document frequency, k1=1.5 and b=0.75. Search accumulates scores only for matching postings and then sorts matching candidates. The implementation uses a complete sort rather than a heap-based top-k selection.

Every hit also has IDF-weighted query-term coverage. The denominator includes query terms absent from the index, so a question dominated by unseen terms receives low coverage. Coverage is a heuristic, not a calibrated probability of answer correctness. A default threshold of 0.23 filters weakly matched candidates; this is a configurable development setting rather than a statistically certified threshold.

## 13 TF-IDF and combined ranking

The alternate TF-IDF retriever uses sublinear term frequency, 1+log(tf), and smoothed inverse document frequency, log((1+N)/(1+df))+1. Sparse term weights are normalized for cosine similarity. It uses the same lexical terms as BM25, so neither method understands synonyms by itself.

The combined retriever uses reciprocal rank fusion. A candidate receives 1/(60+rank) from each ranked list in which it appears, and these contributions are added. Using ranks avoids pretending that BM25 and cosine scores have the same numerical scale.

The implementation computes BM25 matches even in TF-IDF mode to reuse coverage values. Hybrid mode computes both lists and fuses them. Its asymptotic order is similar, but constant factors are larger. All three methods achieved the same aggregate retrieval scores on the tiny starter collection. The comparison demonstrates an experiment framework; it does not demonstrate that the combination improves quality.

## 14 Answer engines and citation validation

Offline evidence selects a sentence using query overlap, numeric-question checks and a small set of attribute checks. It is deterministic, cheap and fully local. It returns an evidence sentence instead of a concise model-generated answer. Lexical selection can still pick an irrelevant or misleading sentence, and the attribute heuristics are incomplete.

FLAN-T5-small is a separate pretrained encoder-decoder model. The adapter requests a short phrase copied from the passage or the word unanswerable. It uses a pinned model revision, safetensors, deterministic generation with three beams and a maximum of 44 new tokens. Only one retrieved passage is proposed to the model at a time.

FLAN's prompt is fitted against its own tokenizer with a 512-token budget. The full question is preserved. If necessary, a binary search finds the longest passage prefix that fits. If the question itself exceeds the budget, the user receives an explicit error. The source check then uses only the text actually visible to the generator, avoiding evidence attribution to a truncated portion the model never saw.

Ollama sends a bounded prompt to a local /api/chat endpoint. It requests deterministic generation and disables thinking for compatible models. Its default model name is qwen3:4b, but the user can configure another installed model. Unlike the FLAN adapter, it does not have the actual server tokenizer; character limits and a 4096-token requested context are practical bounds, not an exact tokenizer-level guarantee. Real Ollama inference was not executed during this build.

After proposal, answer checks that the supplied context is a literal substring of the retrieved passage. locate_span removes simple answer prefixes and looks for a word-boundary-respecting literal match, allowing whitespace differences and ignoring case. It rejects unsupported phrases and abstention tokens. Numeric boundary checks avoid matching 43 inside 143 or -43, and avoid matching 0.5 inside 0.50.

An accepted answer gets a citation from the stored Passage metadata, never from a model-invented filename. evidence_excerpt returns a bounded, contiguous substring of the normalized passage. This is source-wording verification. It is not semantic entailment, a complete numerical reasoning check or proof that the answer addresses the question.

The first candidate with a verified span is returned. If the highest-ranked page contains a plausible but wrong value, that value can pass the check. This happened in the measured run and is analyzed below. If retrieval coverage is too weak, status is low_retrieval. If passages exist but no proposed span is verified, status is no_supported_span.

## 15 The complete request lifecycle

First the user starts run.py, which launches Uvicorn on 127.0.0.1. PaperService initializes a local workspace and seeds a new collection with starter PDFs. It builds in-memory indexes. Indexes are rebuilt after a successful upload and on process restart; they are not serialized as a durable search database.

The browser loads static HTML, CSS and JavaScript, then fetches collection metadata. A question submission contains the text, retriever, generator and number of passages. FastAPI validates input lengths and limits. The service retrieves passages, obtains candidate wording, applies source checks and returns JSON with the answer, evidence, citations, hits and elapsed time.

The UI escapes document and model text before inserting it into HTML. It shows the answer mode, page link and retrieval details. The reader can inspect the source PDF and download a Markdown or JSON report. Exports are created in the browser; the application does not maintain a persistent question history.

The Model lab endpoint loads the saved Mini-GPT and matching tokenizer, returns IDs and the round trip, and samples a short continuation. The Results endpoint exposes saved experiment reports from runs. Those results are explicitly labeled as belonging to the starter collection, even after the user uploads new papers.

## 16 API and local storage

| Endpoint | Purpose |
|---|---|
| GET / | Serve the dashboard |
| GET /api/status | List documents, searchable pages, chunk count and notices |
| POST /api/upload | Validate a raw PDF upload, store it and rebuild indexes |
| POST /api/ask | Retrieve evidence and propose a source-matched answer |
| GET /api/papers/{name} | Open an existing collection PDF |
| POST /api/lab | Tokenize text and sample the saved Mini-GPT |
| GET /api/results | Read saved training and benchmark reports |
| GET /openapi.json | Machine-readable API schema |

POST requests need X-PaperLab: 1. If an Origin header is present, it must match the request host. This is a basic local cross-origin guard, not authentication. The app binds to loopback and is intended for one local user. Its Content Security Policy blocks external scripts and embedding. External-asset Swagger and ReDoc pages are disabled; the dashboard, CLI and JSON API schema are available.

Upload validation constrains filenames, file size, page count, PDF header, encryption and text availability. A changed file with an existing name is rejected rather than silently replacing it. Identical files with the same name are recognized. An index rebuild is protected by a process lock. Questions and model generation also share that lock, so requests are serialized in this prototype.

PDF parsing still runs in the application's process, and file/page limits are not a comprehensive defense against malicious files. A hosted service would need sandboxed parsing, authentication, quotas and multi-user isolation. Do not present this local prototype as ready for public exposure.

## 17 Time complexity of the actual implementation

There is no single meaningful Big-O expression for the whole project. Offline training, document indexing, retrieval and generation have different inputs and are performed at different times. The following symbols describe those costs.

| Symbol | Meaning |
|---|---|
| R | UTF-8 bytes in the tokenizer training corpus |
| S | Total byte length of unique pretokenized pieces, at most R |
| M | Number of learned BPE merges, at most 189 by default |
| n and m | Input byte length and length of an individual regex piece |
| W | Extracted word count before overlap |
| c and o | Chunk size and overlap in whitespace words |
| N and U | Number of chunks and total term occurrences after overlap |
| q and F | Unique query terms and sum of their posting-list lengths |
| r | Number of matching candidate chunks, at most N |
| B, T, d, L, V | Batch size, context length, width, layers and vocabulary |
| I and G | Optimizer steps and generated output tokens |
| P | Number of model parameters |

BPE training scans unique pieces to count pairs and apply each merge. A useful upper bound for this simple implementation is O(R + M*S), plus regex and hash-table costs. Worst-case S can be comparable to R, giving O(M*R). A more optimized tokenizer implementation could maintain pair updates incrementally, but this one does not.

BPE encoding repeatedly scans a piece for the best available merge and rebuilds its symbol tuple. A conservative bound per piece is O(m*min(M,m)), with O(m squared) as the unconstrained worst case. Across pieces, sum those costs. An LRU cache avoids recomputing repeated pieces while entries remain cached. Decoding is linear in the emitted byte length.

For a page containing w words, one chunk is produced if 0<w<=c. Otherwise the number is 1+ceil((w-c)/(c-o)). Total N is the sum over pages. Splitting and constructing all overlapping chunk strings costs O(U), with U roughly W*c/(c-o) on long pages. PDF parsing itself depends on compressed objects, layout and parser behavior; it is not accurately bounded by extracted character count alone.

Building the sparse index takes expected O(U) term processing. BM25 search tokenizes the query, traverses F matching postings and sorts r matching chunks: O(query_length + F + r*log(r)). A worst case with common query terms is O(q*N + N*log(N)). Distinct-page filtering scans ranked matches; it does not change this upper bound. A heap optimization could reduce the ranking part to O(r*log(k)) for some use cases, but is not implemented.

TF-IDF has similar sparse accumulation and sorting costs. The present implementation also computes BM25 coverage, so it has additional constant work. Hybrid performs both rankings, fuses ranks and sorts again; it retains the same broad worst-case order with larger constants.

For one Mini-GPT forward pass, projections and feed-forward layers cost O(L*B*T*d squared), dense attention costs O(L*B*T squared*d), and the full vocabulary projection costs O(B*T*d*V). Adding them gives:

```text
C_forward = O(L*B*(T*d^2 + T^2*d) + B*T*d*V)
C_training = O(I*(C_forward + P)) + evaluation cost + BPE cost
```

Backward propagation has the same asymptotic order as forward computation with a larger constant factor. AdamW adds O(P) per optimizer step. Evaluation has its own batch/window count and is not free; the training implementation evaluates the full train and validation partitions at configured intervals.

Doubling context length makes the attention term approximately four times larger, while the projection and vocabulary terms approximately double. Doubling width roughly quadruples the projection term. These statements concern individual terms; actual elapsed time also depends on BLAS, memory allocation and overhead.

Mini-GPT generation uses batch size one and no key/value cache. A conservative bound for G generated tokens at a full context is:

```text
O(G * (L*(T*d^2 + T^2*d) + T*d*V + V*log(V)))
```

The V*log(V) term comes from sorting logits for top-k. Early shorter contexts may cost less. This is the complexity of the delivered NumPy sampler, not of an optimized serving engine.

FLAN is an encoder-decoder model with a different architecture and library generation implementation. Its cost includes encoding each visible passage and autoregressive decoding with beam search. PaperLab may call it up to k times, stopping at the first accepted answer. A defensible application-level expression is retrieval_cost + sum of generator_cost(candidate_i) + span_checks. Do not apply the Mini-GPT formula directly to FLAN or Qwen. Model size, prompt length, output length, beams, caching and device affect their costs.

## 18 Space complexity and scaling

The document store retains PDF bytes. In memory, passages store normalized text and indexes store sparse postings. Their principal size is O(U), plus term dictionaries and string/object overhead. Building BM25 and TF-IDF representations increases constants. Every new upload currently triggers a full rebuild; this is convenient for a small collection but inefficient for thousands of papers.

Mini-GPT parameters use O(P) memory. Training additionally holds gradients, two Adam moments and a best-checkpoint copy. Cached hidden/MLP activations are roughly O(L*B*T*d), attention matrices O(L*B*H*T squared), and output logits/gradients O(B*T*V). NumPy temporaries and retained block tensors increase the actual peak beyond these simple component counts.

Generation avoids storing backward caches but still materializes dense attention and logits on each step. The code does not implement memory-efficient attention. The pretrained model is much larger than the custom educational model, so it dominates memory when FLAN is enabled.

To scale, first measure larger collections. Candidate improvements include incremental indexing, a top-k heap, persistent sparse indexes, semantic embeddings, a reranker and queued model inference. These have tradeoffs and should be evaluated rather than added solely to make the technology list longer.

## 19 Measured results and how to interpret them

The main build used Python 3.12.14 on Linux with NumPy 2.3.5. Mini-GPT training used one BLAS thread. FLAN used PyTorch 2.6.0+cpu, Transformers 4.56.2 and two CPU threads. Detailed versions and timing samples are recorded in runs/benchmark.json. Hardware and load differ from the user's laptop, so these times are measurements of this run, not performance guarantees.

| Training measure | Observed result |
|---|---|
| Training and validation documents | 30 and 6 |
| Next-token targets | 4,884 training and 897 validation |
| Parameters and vocabulary | 79,584 and 448 |
| Optimizer steps | 400 |
| Initial validation loss | 6.117 nats per BPE token |
| Best measured validation loss | 3.839 at step 250 |
| Best validation perplexity | 46.49 |
| Final-step training loss | 2.521 |
| Final-step validation loss | 4.179 |
| Training wall time including evaluation | Approximately 9.60 seconds |

Perplexity is exp(mean cross-entropy). It measures how uncertain the language model is about next tokens under this tokenizer and evaluation setup. It is not a percentage and is not QA accuracy. Comparisons across different tokenizers or corpora require care.

After step 250, training loss keeps falling while validation loss rises. The model increasingly fits the small training corpus. Selecting step 250 avoids delivering the later overfit snapshot. A larger corpus, carefully evaluated regularization and validation-based stopping are possible next experiments; simply increasing training steps is not automatically an improvement.

| Retriever | Page Hit at 3 | Page MRR at 3 | Median search time |
|---|---|---|---|
| BM25 | 10/10 | 0.950 | 0.0068 ms |
| TF-IDF | 10/10 | 0.950 | 0.0146 ms |
| Combined ranking | 10/10 | 0.950 | 0.0167 ms |

These microbenchmarks made 390 searches per method: 13 questions repeated 30 times. They measure in-process retrieval over six passages, without PDF parsing, model generation, HTTP transport or browser rendering. Such tiny latency numbers should not be extrapolated to large collections. Extracting the six pages took approximately 15.7 ms in this run.

Page Hit@3 asks whether the labeled page appears among the first three distinct retrieved pages. Page MRR@3 averages reciprocal rank, assigning zero to misses outside the cutoff. Here nine labeled pages ranked first and one ranked second: (9+0.5)/10=0.95. Retrieval metrics are computed on the ten answerable questions.

| Answer measure | Offline evidence | FLAN-T5 |
|---|---|---|
| Short-answer exact match | 0/10 | 7/10 |
| Exact answer and correct cited page | 0/10 | 7/10 |
| Reference phrase contained and correct page | 7/10 | 7/10 |
| Accepted outputs on answerable questions | 9/10 | 10/10 |
| Abstained on unanswerable questions | 3/3 | 3/3 |
| Median request computation | 0.215 ms | 334.7 ms |

Offline evidence returns full sentences, while the labels contain short answers. Exact-match scoring therefore penalizes its intentionally different output format. Reference containment checks whether an acceptable reference phrase appears in the response and the page is correct. It is still only a permissive automatic proxy, not human-assessed relevance or entailment.

The answer latency distribution includes all 13 questions, including cheap retrieval-level abstentions. FLAN model loading is measured separately and excluded from those per-question times. Its observed p95 was approximately 1.94 seconds. The benchmark process peaked at approximately 763 MiB RSS on Linux, including imports, model loading and evaluation; this is not isolated model-weight memory.

All questions and starter texts were visible during development. Results on this set can help debug behavior and explain metrics, but they are optimistic as estimates of unseen-paper performance. A serious claim needs a larger held-out QA set, a fixed tuning protocol, multiple seeds where relevant and human review of citation support.

## 20 Observed failures and what they teach

The FLAN run missed three answerable questions. For the question asking which statistic the distance filter used, it returned a longer passage about where filtered values were computed rather than the expected median. For the filtered distance at the spike, it returned 80 centimetres from page 1 instead of 43 centimetres from page 2. For the count of filtered alarms, it returned the raw count 1 instead of filtered count 0.

These are concrete examples of different failures. The first is answer selection and output-format failure. The second combines ranking and premature acceptance of a source-matched but wrong value. The third confuses two adjacent quantities in flattened PDF text. All can pass a literal source check. A citation should make a result inspectable, not hide the need for judgment.

The chunk ablation also illustrates how experiments should be interpreted. With 65-word chunks and 14-word overlap, MRR@3 increased to 1.0 on these fixtures; Hit@3 remained 1.0 for all tested chunk settings. This was an exploratory development result on ten answerable questions, not an independently validated improvement. The default remained 130/28, matching the supplied design, instead of silently tuning to the fixture.

Useful future work includes comparing every proposed answer against a question-aware reranker, adding an entailment check, preserving table structure and calibrating abstention on held-out data. None of these guarantees perfect factual correctness.

## 21 Testing and verification

The delivered unittest suite has 22 tests. It covers Unicode round trips, merge reproducibility, invalid tokenizer state, numerical gradients, causal masking, masked loss, token-range errors, checkpoint consistency, duplicate split detection, document-local batching, chunk boundaries, distinct-page retrieval, numeric span boundaries, invented evidence rejection, abstention, prompt budgeting, local-only Ollama configuration, upload behavior and API validation.

The finite-difference test perturbs 80 sampled coordinates across all model parameter tensors using float64. It compares the analytic gradient with the central-difference estimate (loss(theta+epsilon)-loss(theta-epsilon))/(2*epsilon). Maximum observed absolute discrepancy was approximately 5.7e-10. This provides strong local evidence for those derivatives; it is not an exhaustive mathematical proof for every possible input.

The causal test changes future token IDs and confirms earlier logits do not change. The checkpoint test confirms exact restored logits and matching tokenization. The data test intentionally places normalized duplicate text across splits and expects rejection. Citation tests reject fabricated spans and contexts, including partial numeric matches.

FastAPI TestClient exercises the dashboard response, PDF delivery, upload and indexing, answering, malformed requests and the local-origin guard. A real HTTP smoke test also started Uvicorn and verified requests with both offline evidence and FLAN. Real FLAN inference was executed during benchmark evaluation. All 13 default notebook code cells were executed sequentially in an IPython process; the optional FLAN cell was disabled, and the real FLAN benchmark was checked separately. A socket-based Jupyter kernel and browser access to the server were unavailable in this isolated build environment, so visual dashboard verification is explicitly uncompleted. Do a short browser check on your laptop before presenting: ask a question, open its page, upload a PDF, use Model lab and download an answer.

## 22 Source code map and function responsibilities

| File | Main responsibilities |
|---|---|
| tokenizer.py | merge_pair performs replacement; ByteBPE.train learns merges; encode and decode map text/IDs; save and load preserve vocabulary |
| model.py | Config validates dimensions; softmax, layer_norm and gelu implement operators; forward computes logits; backward computes gradients; cross_entropy supplies loss |
| training.py | load_corpus validates splits; make_batch forms windows; evaluate_lm computes weighted loss; AdamW updates parameters; train selects the best snapshot; generate samples |
| retrieval.py | terms normalizes retrieval words; split_page forms windows; read_papers preserves pages; BM25 builds postings and ranks passages |
| ranking.py | RetrievalIndex adds sparse TF-IDF and reciprocal rank fusion while preserving retrieval coverage |
| rag.py | FlanGenerator loads and budgets the pretrained model; locate_span checks wording; evidence_excerpt selects context; answer orchestrates retrieval and acceptance |
| generators.py | ExtractiveGenerator selects a sentence; OllamaGenerator calls a local model; make_generator selects an engine |
| datasets.py | download_papers fetches optional PDFs; import_qasper preserves official split IDs; download_qasper obtains the archive |
| evaluation.py | normalize_answer defines exact-match normalization; evaluate reports page ranking, answer and abstention measures with cases |
| service.py | PaperService owns the workspace, validates uploads, rebuilds indexes and serializes answer requests |
| api.py | Pydantic request models, FastAPI routes, host/origin checks and file delivery |
| __main__.py | CLI commands for train, sample, ask, evaluate and optional downloads |
| static/index.html | Dashboard structure and accessible labels |
| static/style.css | Desktop and narrow-screen layout rules |
| static/app.js | API calls, escaped rendering, navigation and local report downloads |
| run.py | Local Uvicorn startup and port selection |
| scripts/benchmark.py | Comparable retrieval timings, answer evaluation, chunk ablation and environment metadata |
| tests/ | Behavioral regression checks for algorithms and API |

Start reading tokenizer.py, then model.py and training.py. Next read retrieval.py, rag.py and evaluation.py. Finally read service.py, api.py and static/app.js to see how the algorithms become a user-facing application. This order separates mathematical understanding from request-handling details.

## 23 Why the project is useful and what distinguishes it

For students, the project supplies a controlled way to connect linear algebra, optimization, information retrieval and software engineering. A learner can see how matrix dimensions relate to attention, how gradients are checked and why a validation curve can worsen despite lower training loss. Those are stronger learning outcomes than merely showing a chatbot response.

For research readers, page-linked evidence can shorten the search for experimental details. It is particularly useful for narrow questions about named methods, settings or reported values in searchable PDFs. The system helps locate passages for review; it does not replace reading the paper or establish that its claims are scientifically sound.

For instructors and reviewers, the source map, small fixtures, tests and per-question reports make the demonstration inspectable. The offline mode lowers the cost of trying it, and the pretrained option illustrates the practical gap between training a tiny model and using a model with broad prior language knowledge.

The distinguishing implementation features are the combination of custom byte BPE, a genuinely trainable manually differentiated Transformer, verified numerical gradients, page-local evidence, literal-span checks, an offline path and explicit failure analysis in one project. These are engineering and educational characteristics. BPE, Transformers, BM25, TF-IDF, RRF and RAG are established methods; combining them does not by itself establish a novel research contribution.

To develop a research contribution, formulate a testable question. One candidate is whether a stronger evidence verifier reduces wrong cited answers at a tolerable abstention rate and latency. Another is whether overlap or reranking improves page retrieval on a fixed held-out corpus. Compare against controlled baselines, tune on development data and reserve a test set for final reporting. Report error bars or confidence intervals when sample size permits meaningful inference.

## 24 How to run and demonstrate it

The README provides installation commands. After extracting the ZIP, create a Python 3.12 virtual environment, install requirements.txt and run python run.py. The trained checkpoint is included. For the first demonstration, use the starter data so that the expected answers are known.

Ask the alarm-threshold question in offline mode and inspect the retrieved passage. If optional LLM dependencies are installed, select FLAN and ask it again to show a concise span. Click the citation and verify page 1. Ask who manufactured the distance sensor to demonstrate abstention. Then show a recorded failure from the benchmark and explain why a copied span can still be wrong.

In Model lab, enter Research questions and show the six BPE IDs and exact decode round trip. Generate a continuation and identify the model as educational. Open Measured results and explain why checkpoint step 250 was selected. Show one gradient test and the split-validation check in code.

A sensible five-minute presentation spends roughly one minute on the problem and architecture, two on the working PDF workflow, one on the custom model and one on evaluation and limitations. For a longer viva, derive the parameter count and attention complexity, then discuss the observed failures.

Use notebooks/PaperLab_End_to_End.ipynb to step through the experiment locally or in Google Colab. The notebook reads project files from the extracted ZIP. In Colab, upload the project ZIP when prompted; the notebook does not hide the code or data inside a Base64 payload. Optional pretrained inference is controlled by an explicit flag. The local dashboard is a separate launch workflow; the notebook does not automatically expose a public web server.

## 25 Common errors and practical fixes

| Symptom | Likely cause and fix |
|---|---|
| Address already in use | Run python run.py --port 8001 and open that port |
| Import error for core packages | Activate the same virtual environment used for installation |
| FLAN unavailable | Install requirements-llm.txt; allow the initial model download or use a local model directory |
| ML packages fail on a very new Python | Create the environment with the tested Python 3.12 version |
| No searchable PDF text | Use a text PDF or perform OCR before upload |
| File already exists | Same-name different bytes are rejected; rename the upload |
| Questions receive no verified answer | Inspect coverage and passages; simplify the question; the source may not contain the answer |
| Ollama connection error | Confirm the local service is running and the configured model is installed |
| API POST returns 403 | Send X-PaperLab: 1 and use the same local origin |
| Results did not change after upload | Saved benchmark results describe the starter collection, not the live collection |
| Model text looks poor | Tiny corpus, tiny model and no instruction tuning; inspect loss rather than treating it as a QA model |

## 26 Viva questions with direct answers

**What is the project in one sentence?** It is a local research-paper evidence assistant with an inspectable BPE and Mini-GPT learning laboratory.

**What did you implement yourself?** The project includes byte-pair training, a NumPy causal Transformer and its derivatives, AdamW training, sparse retrieval, evidence validation, evaluation and an application around these parts. The starting implementations came from the supplied notebook and were modularized and extended. FLAN and Qwen are pretrained external models, not models trained by this project.

**Why not use Mini-GPT to answer papers?** It has only 79,584 parameters, a 48-token context and a tiny next-token corpus. It is not trained to follow questions. Using its samples as factual answers would misrepresent its capability.

**Does the custom BPE tokenize FLAN input?** No. The custom vocabulary matches Mini-GPT's embedding table. FLAN must use its own tokenizer.

**Is the retrieval semantic?** The implemented BM25 and TF-IDF modes are lexical. They can miss paraphrases without shared terms. There is no neural embedding model in the default retrieval stack.

**Does a valid citation prove the answer?** No. The code confirms a literal span and source identity. It does not prove relevance or entailment, as the wrong filtered-distance answer demonstrates.

**Why use overlapping chunks?** They can preserve phrases crossing a chunk boundary, at the cost of duplicated storage and work. Page-local boundaries preserve citation identity.

**What is the overall time complexity?** There are separate costs. Sparse retrieval is O(F+r log r) after query normalization. Mini-GPT training is O(I[L B(T d squared + T squared d)+B T d V+P]) plus evaluation and BPE preprocessing. The guide defines each symbol and the implementation assumptions.

**What is the accuracy?** On ten visible answerable teaching questions, FLAN achieved seven exact answers with the correct page. Retrieval Hit@3 was ten out of ten. These are different metrics and do not imply general accuracy of 70 percent or 100 percent on unseen papers.

**What is unique?** The transparent combination and its evaluation are useful educational features. No globally new algorithm or publishable novelty has been established.

**How does it help others?** It provides a reproducible teaching project and a local way to locate evidence in papers. A reader still verifies the evidence and the underlying scientific claim.

**What would you improve first?** Evaluate on held-out real questions, analyze semantic citation failures, and compare a question-aware reranker or verifier before expanding the interface or model size.

## 27 References and further reading

These primary sources establish the relevant method families and pretrained components. The measurements in this guide come from the delivered code and run artifacts, not from performance numbers in these papers.

1. Sennrich, Haddow and Birch. Neural Machine Translation of Rare Words with Subword Units. ACL 2016. https://aclanthology.org/P16-1162/
2. Vaswani and colleagues. Attention Is All You Need. 2017. https://arxiv.org/abs/1706.03762
3. Lewis and colleagues. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. 2020. https://arxiv.org/abs/2005.11401
4. Dasigi and colleagues. A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers. NAACL 2021. https://aclanthology.org/2021.naacl-main.365/
5. Google. FLAN-T5-small model card and license. https://huggingface.co/google/flan-t5-small
6. Ollama. Generate a chat message API documentation. https://docs.ollama.com/api/chat

The byte-level tokenizer is an educational variant of BPE, not an exact reproduction of the original word-segmenting system. Mini-GPT is a decoder-only adaptation of Transformer ideas, not the original encoder-decoder translation architecture. Citation of these sources acknowledges the foundations without claiming that the project reproduces all of their experiments.
