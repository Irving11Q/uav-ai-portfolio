# RAG with LangChain — What the Framework Actually Does for You

Day 20 of the "UAV AI Application Development" series. Days 15–18 built a full RAG pipeline **by hand** — chunking state machine, a 3-tier embedding fallback, Chroma retrieval, query rewriting, RRF fusion, thresholds, reranking, an eval set. That was ~1300 lines across Day 17 + Day 18. This day rewrites the *same* RAG with **LangChain 1.x** and answers the only question that matters for an interview:

> The framework saves effort on *plumbing*. It does **not** make retrieval accurate, prompts correct, or evaluations unnecessary. Know which half is yours.

## What's inside

`day20_langchain_rag.py` — one self-contained script that:

1. **loads the Day 16 chunks** (`chunks_parsed.json`, 8 heading-based chunks) from its own folder;
2. **adapts the Chinese BGE model into LangChain** via the `Embeddings` interface — `BgeEmbeddings` implements two methods and reuses the *locally cached* `BAAI/bge-small-zh-v1.5` (no network, no hardcoded path);
3. **builds the chain with LCEL** (`|`): `retriever → prompt → LLM → parser`, where the LLM is wrapped in `RunnableLambda` so we keep our own `requests`-based client instead of a LangChain chat model;
4. **runs a query end-to-end** and prints the retrieved chunks + a cited answer;
5. **compares the LangChain splitter against the Day 16 hand-written one** — same 8 chunks, proving the hand-written state machine was right, not lucky;
6. **prints a hand-written-vs-framework table** so you can say exactly what the framework bought and what it didn't.

## Requirements

```bash
pip install langchain-core langchain-text-splitters sentence-transformers
```

| Package | Needed for | Without it |
|---|---|---|
| `langchain-core` | the LCEL chain, `Embeddings`, `InMemoryVectorStore` | script does not run |
| `langchain-text-splitters` | the chunk-comparison demo | that one section is skipped |
| `sentence-transformers` | the Chinese BGE embedding | falls back to an English ONNX model (poor on Chinese) |
| `requests` + LLM key | the final generation | retrieval + comparison still run, generation is skipped |

- **Python 3.12**, **LangChain 1.3.18** (tested), **langgraph 1.2.11** co-installed in the same env — LangChain 1.x API differs greatly from 0.x tutorials you find online.
- Keys come from the `api_config.py` copy in this folder → environment variables (`ZHIPU_API_KEY` / `DEEPSEEK_API_KEY`). No key = offline mode, no crashes.

## Run

From this folder (`professional/day20_langchain_rag/`):

```bash
python day20_langchain_rag.py
```

It uses an **in-memory** vector store, so it creates **no `chroma_db` directory** — reruns start clean every time.

---

## Design 1: the adapter pattern — bring *your* model to the framework

LangChain does not care which embedding model you use; it only requires you to subclass `Embeddings` and implement two methods:

```python
class BgeEmbeddings(Embeddings):
    def embed_documents(self, texts): ...   # batch — called when building the index
    def embed_query(self, text):    ...      # single — called on each question
```

Inside, it lazily imports `SentenceTransformer` and loads the model from **`HF_HUB_CACHE`** (resolved via `huggingface_hub.constants.HF_HUB_CACHE`, never a hard-coded path). The `QUERY_INSTRUCTION` knob stays `""` — we measured on Day 17 that the BGE query prefix is *worse* on this corpus, so we don't turn it on.

> This is the single most reusable LangChain skill: **any** local or custom embedding (BGE, E5, a fine-tuned model) drops in through this same two-method interface.

## Design 2: LCEL — one pipe symbol instead of 50 lines

```python
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | RunnableLambda(call_llm)     # our own requests client, not a LangChain ChatModel
    | StrOutputParser()
)
```

`RunnableLambda` lets us keep the HTTP client from Day 10/17 instead of adopting LangChain's chat-model layer — useful when your backend is already wired to a specific provider. One gotcha worth knowing: LangChain's `HumanMessage.type` is `"human"` but OpenAI-compatible APIs want `"user"`, so the lambda maps roles explicitly (`system/user/assistant`).

## Design 3: `InMemoryVectorStore` does **not** persist

```python
vectorstore = InMemoryVectorStore(embedding)   # gone when the process exits
```

Day 17 used `chroma.PersistentClient` and wrote to disk. Here the store lives only in RAM. For production you swap `InMemoryVectorStore` → `langchain_chroma.Chroma` and **nothing else changes** — that swappability is the framework's real win, not the memory saving.

---

## Measured results (this corpus, bge-small-zh)

**Chunking — LangChain vs hand-written (Day 16):**

| | Blocks | Sections |
|---|---|---|
| `MarkdownHeaderTextSplitter` | 8 (1 title + 7 chapters) | identical to Day 16 |
| Day 16 hand-written state machine | 8 (1 title + 7 chapters) | identical |

✅ The two are **byte-for-section identical**. The hand-written version was correct, not a lucky guess — and writing it first is *why* we understood what the splitter does internally (it judges titles line-by-line to avoid false matches).

**Retrieval — "电池能飞多久":**

```
[4. 温度管理]  [1. 电池规格]  [7. 安全预警]      ← Top-3
⚠️  "5. 续航估算" is NOT in the Top-3
```

The user says "能飞多久", the manual says "续航" — the semantic model *understands* the wording (it ranks the right theme high), but the exact target chapter still misses the cut. **This is the whole point of the day:** the framework assembled the pipeline in ~5 lines, yet retrieval quality is *exactly* where Day 18 left it. Frameworks do not tune retrieval for you.

**Generation — honest caveat:** with a loose "use only the manual" instruction, the model produced a *plausible but fabricated* flight-time formula. The fix is a stricter prompt (cite sentence-by-sentence, forbid self-derivation) — frameworks don't write that for you either.

## Hand-written vs framework — what each side actually does

| Step | Hand-written (Day 15–18) | LangChain (Day 20) |
|---|---|---|
| chunking | ~60-line state machine | `MarkdownHeaderTextSplitter` — 2 lines |
| embedding | ~100-line 3-tier fallback | subclass `Embeddings` — 2 methods |
| retrieval | `search_chroma()` ~50 lines | `as_retriever()` — 1 line |
| prompt | f-strings scattered in functions | `ChatPromptTemplate` — centralised |
| generation | own `requests.post` | `RunnableLambda` into the chain |
| persistence | Chroma on disk, survives restart | `InMemoryVectorStore`, lost on exit |
| swap vector DB | edit call sites | change 1 line (interface-stable) |
| debugging | print every step, see everything | chain is a black box, trace via callbacks |

**My verdict:**
- Framework *earns* its place at: splitter, retriever abstraction, swappable backends (change DB/model in one line), centralised prompts.
- Framework does **not** help with: writing the prompt correctly, setting the threshold, building the eval set — those stay human work.
- Framework's cost: the chain is a black box (debug by `invoke(retriever)` first, then generation); fast version churn (1.x ≠ 0.x).

So the route is **hand-write to understand each layer, then use the framework to go faster** — not one-or-the-other.

## Why it matters

| Concern | Hand-written taught us | LangChain now handles |
|---|---|---|
| "what is a vector store?" | we built the calls ourselves | `InMemoryVectorStore` / `Chroma` |
| "how do I swap models?" | rewrite call sites | one `Embeddings` subclass |
| "is my chunking right?" | state machine, verified vs framework | both agree → confidence |
| "does the framework fix RAG?" | — | no — retrieval/tuning/eval still ours |

## About

Part of my learning series toward building a **UAV energy-consumption prediction AI System**. Full roadmap: see repo root `README.md`.

Next: **Day 21** — LangGraph, turning this retrieve-then-answer flow into an Agent that can plan and call tools. (LangChain/LangGraph is named in ~80% of the AI-application JD descriptions I'm targeting, so it is the next gap to close.)
