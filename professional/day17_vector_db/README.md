# Vector Database with Chroma — Retrieval That Understands Meaning

Day 17 of the "UAV AI Application Development" series. Replaces Day 15's hand-rolled word-frequency vectors with a **real vector database (Chroma)** and a **real Chinese embedding model**, then measures exactly what that buys you.

## What's inside

`day17_vector_db.py` — a self-contained script that:

1. **loads the chunks produced by Day 16** (`chunks_parsed.json`, 8 heading-based chunks), searching its own folder, the sibling `day16_doc_parsing/` folder, and the parent folder;
2. **picks an embedding model with 3-tier graceful degradation** (see below);
3. **builds a persistent Chroma collection** — vectors + metadata written to `chroma_db/` on disk;
4. **runs semantic queries** and prints similarity scores;
5. **benchmarks against Day 15's word-frequency retrieval** on questions deliberately phrased *differently* from the manual ("电池能飞多久" vs the manual's "续航"), reporting both **Top-1** and **Top-3** recall;
6. **demonstrates metadata filtering** — search only inside chapter "2. 电压管理";
7. **verifies persistence** — reopens the database in a fresh client and confirms the vectors survived;
8. **closes the RAG loop** — Chroma retrieval → prompt → LLM answer with cited sources.

## Requirements

```bash
pip install chromadb requests            # minimum
pip install sentence-transformers        # optional, strongly recommended for Chinese
```

| Package | Needed for | Without it |
|---|---|---|
| `chromadb` | the vector database | script still runs, falls back to local hash vectors |
| `sentence-transformers` | the Chinese BGE model | falls back to an English ONNX model (poor on Chinese) |
| `requests` | the final generation step | everything except generation runs |
| LLM API key | the final RAG answer | everything except generation runs |

- **Python 3.12** (developed and tested on 3.12.6)

## Run

From this folder (`professional/day17_vector_db/`):

```bash
python day17_vector_db.py
```

It creates a `chroma_db/` directory next to itself (the persisted vector store, git-ignored).

---

## Design 1: three-tier graceful degradation

A teaching script must run on *anyone's* machine — with no key, no GPU, and possibly no network.

| Tier | Embedding | Dim | Chinese semantics | Needs |
|---|---|---|---|---|
| **A** | `BAAI/bge-small-zh-v1.5` | 512 | ✅ true semantics | `sentence-transformers` + ~95 MB download |
| **B** | Chroma's bundled ONNX `all-MiniLM-L6-v2` | 384 | ❌ English corpus only | ~90 MB download |
| **C** | local hash embedding | 256 | ❌ literal matching only | nothing |

The function returns `(fn, mode_name, is_semantic, is_chinese)` so the rest of the script knows whether the benchmark is meaningful — and says so honestly instead of overclaiming.

> **The single most important lesson in this file:** never use an English embedding model for a Chinese RAG system. Tier B is kept in the code *deliberately* so you can watch it fail — it ranks "7. 安全预警" above "5. 续航估算" for the question "电池能飞多久".

### Model choice (measured, not assumed)

I benchmarked two Chinese models on this exact manual:

| Model | Size | Top-1 | Top-3 |
|---|---|---|---|
| `bge-small-zh-v1.5` | 95 MB | 2/4 | 2/4 |
| `bge-base-zh-v1.5` | 400 MB | 2/4 | **3/4** |

`base` rescues one extra answer inside Top-3 at 4× the size. Small is the better default for learning and for most products; switch by changing one argument if your own evaluation set justifies it.

### BGE's official query instruction: tested and rejected

BGE recommends prefixing queries with `为这个句子生成表示以用于检索相关文章：`. On this corpus it was **worse**:

| | Top-1 |
|---|---|
| without prefix | **2/4** |
| with prefix | 1/4 |

The knob is kept in the code (`SentenceTransformerEmbedding.QUERY_INSTRUCTION`, default `""`) so you can retest on your own data. Best practices still need local validation.

---

## Design 2: probe the model host before downloading

Everyone hard-codes the China mirror:

```python
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
```

I hit a real failure with that: the mirror returned 502, the script silently fell back to the English model, and Chinese quality collapsed with no obvious cause. So this script **probes** instead:

```python
for url in ["https://huggingface.co", "https://hf-mirror.com"]:
    if requests.get(url, timeout=6).status_code < 500:
        os.environ.setdefault("HF_ENDPOINT", url); break
```

⚠️ It must run **before** importing `chromadb` / `sentence_transformers` — `huggingface_hub` reads the endpoint at import time. That is the real reason "I set the mirror but it had no effect" happens so often.

---

## Design 3: Chroma 1.x custom embedding functions

Chroma 1.x requires more than `__call__`. A plain Python class raises:

```
AttributeError: 'SentenceTransformerEmbedding' object has no attribute 'name'
AttributeError: 'SentenceTransformerEmbedding' object has no attribute 'embed_query'
```

Both custom classes here implement the full protocol:

| Method | Purpose |
|---|---|
| `__call__(input)` | embed documents on `add()` |
| `embed_query(input)` | embed the question on `query()` — separate so BGE/E5-style models can differ |
| `name()` | identity stored in the DB, so Chroma detects that you swapped models |
| `get_config()` | serializable config written to the DB |
| `build_from_config()` | rebuild the instance when reopening an old DB |

That `name()` check is Chroma protecting you from a subtle bug: **change the embedding model and every old vector becomes meaningless** unless you rebuild the collection.

---

## Measured results

Questions are phrased as a user would ask, not as the manual writes them. Expected sections were verified against the manual text — e.g. "电池胀起来了" belongs to *6. 充电与维护* ("电池鼓包、漏液应立即停止使用"), not to *7. 安全预警*. A mislabelled eval set will judge a good model wrong.

| Question | Word-frequency (Day 15) | Chroma + BGE-zh |
|---|---|---|
| 电池能飞多久 | 1. 电池规格 (0.434) ❌ | 4. 温度管理 (0.628) ❌ |
| 多少伏就必须降落了 | 2. 电压管理 (0.139) ✅ | 2. 电压管理 (0.574) ✅ |
| 冬天飞行掉电特别快是为什么 | 7. 安全预警 (0.143) ❌ | 7. 安全预警 (0.639) ❌ |
| 电池胀起来了还能继续用吗 | 1. 电池规格 (0.434) ❌ | 6. 充电与维护 (0.655) ✅ |
| **Top-1** | **1/4** | **2/4** |
| **Top-3** | **1/4** | **2/4** |

The raw count is not dramatic, and the script says so rather than overselling. The more important signal is the **score distribution**:

- word-frequency scores sit at **0.14–0.43** — essentially noise, no usable threshold;
- semantic scores cluster at **0.55–0.70** — spread exists, so a relevance cutoff becomes possible (that's Day 18).

Why the gap is narrow here: 8 chunks only, all about the same battery, so vectors crowd together; and some questions still share literal words with the target. This is what a small homogeneous corpus looks like — recognising it is part of doing RAG for real.

## Other patterns demonstrated

**Metadata turns a vector store into a searchable knowledge base:**

```python
collection.add(ids=ids, documents=documents, metadatas=metadatas)
collection.query(query_texts=[q], n_results=3, where={"section": "2. 电压管理"})
```

Filter-then-search is what production systems use for scoping and permission isolation (e.g. `where={"dept": "engineering"}`).

**Distance vs similarity** — the collection is created with `{"hnsw:space": "cosine"}`, so:

```python
similarity = 1.0 - distance
```

**Persistence** — `chromadb.PersistentClient(path=...)` writes to disk; reopening with a fresh client still returns all 8 vectors.

## Why it matters

| Problem (Day 15) | Solved by Chroma (Day 17) |
|---|---|
| literal matching only | semantic similarity (with a model that speaks Chinese) |
| brute-force scan of every chunk | HNSW index, millisecond lookup |
| vectors lost on exit | persisted to disk, reload instantly |
| no way to scope a search | metadata filtering |
| scores indistinguishable from noise | calibrated scores → thresholds become possible |

## About

Part of my learning series toward building a **UAV energy-consumption prediction AI System**. Full roadmap: see repo root `README.md`.

Next: **Day 18** — retrieval tuning (score thresholds, reranking, query rewriting).
