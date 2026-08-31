# RAG Fundamentals: Stop the Model from Making Things Up

Day 15 of the "UAV AI Application Development" series. Builds a minimal Retrieval-Augmented Generation (RAG) pipeline from scratch — no external libraries — to show *why* RAG exists and *how* it kills hallucination. A UAV battery/energy manual is chunked, turned into vectors, and used to answer a question, then compared against asking the model with no data at all.

## What's inside

- `day15_rag_basics.py` — a fully self-contained RAG demo that:
  - embeds a 7-paragraph UAV battery/energy manual (hard-coded, see Day16 for real files);
  - chunks the manual, builds a character-frequency vocabulary, and turns each chunk into a vector;
  - asks a question, converts it to a vector, and ranks chunks by **cosine similarity** (Top-3);
  - feeds the retrieved chunks + question to an LLM and compares the RAG answer vs a direct answer;
  - **degrades gracefully**: the chunk / vectorize / retrieve steps run fully offline; the generate step reuses `api_config.py` and is skipped automatically if no API key is set.
- `api_config.py` — a copied config center (PROVIDER switch + env-var keys) so the generate step can run in this folder without the rest of the repo.

## Requirements

- Python 3.12
- requests

```bash
pip install requests
```

## Setup (only needed for the generate step)

`api_config.py` reads its key from an environment variable. Set one (Windows):

```
设置 → 系统 → 关于 → 高级系统设置 → 环境变量 → 新建
名称: ZHIPU_API_KEY        (default backend = glm-4-flash, free tier)
值:   your-key
```

Reopen your terminal. If no key is set, the script still runs and demonstrates the retrieval half of RAG.

## Run

From this folder (`professional/day15_rag/`):

```bash
python day15_rag_basics.py
```

## Key pattern

**RAG in five steps** — `document → chunk → vectorize → retrieve → generate`:

```python
vocab   = build_vocab(chunks)                       # collect every character
vectors = [make_vector(c, vocab) for c in chunks]   # text -> char-frequency vector
hits    = search(question, chunks, vectors, vocab)  # cosine similarity -> Top-K
# then feed retrieved chunks + question to the model
```

**Cosine similarity** — why semantically close text scores high:

```python
def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)   # 1.0 = identical direction, 0.0 = unrelated
```

> Note: today's "vector" is a character-frequency count — chosen so the principle is transparent and runs with zero dependencies. Real systems use an embedding model (BGE / text2vec / OpenAI). That upgrade is Day17 (Chroma).

## Why it matters

Asking the model directly = it answers from training memory and may fabricate. RAG = "look up the manual first, then answer" → every answer is traceable to a source chunk. This is the single most common way enterprises ship LLMs in 2024+, and the foundation of the Week-4 capstone: a **UAV battery/energy manual Q&A system**.

## About

Part of my learning series toward building a **UAV energy-consumption prediction AI System**. Full roadmap: see repo root `README.md`.
