# RAG Retrieval Tuning — Five Techniques, Each Measured

Day 18 of the "UAV AI Application Development" series. Day 17 proved that a real vector database + a Chinese embedding model beats hand-rolled word frequency (Top-1 1/4 → 2/4), and that semantic scores finally land in a **usable range** (0.55–0.70) instead of noise (0.14–0.43). But it left three open problems:

1. **Top-1 is only 2/4** — if retrieval is wrong, no prompt can save the answer.
2. **Scores exist but aren't *used*** — how do you actually set a threshold?
3. **Wording gap** — users say "能飞多久", the manual says "续航".

This day answers "RAG retrieval is inaccurate, how do you tune it?" with **five concrete techniques, each measured on the same 9-question eval set** so you can say exactly how much each one helps and what it costs — not just name the tricks.

## What's inside

`day18_retrieval_tuning.py` — one self-contained script that:

1. **builds a baseline** — pure semantic retrieval, the floor we have to beat;
2. **rewrites the query** — an LLM bridges the wording gap ("能飞多久" → "续航时间"), with a literal term-map fallback when there is no key;
3. **hybrid retrieval + RRF fusion** — semantic and keyword召回 fused by Reciprocal Rank Fusion (`RRF_K = 60`);
4. **score threshold** — dares to answer "我不知道" when nothing is relevant (`SCORE_THRESHOLD = 0.50`);
5. **LLM reranking** — the model re-orders the candidates, plus a rule-based reranker that needs no key;
6. **MMR** — de-duplicates so results stay diverse;
7. **benchmarks all five** in one table — Top-1 / Top-3 recall + off-topic refusal, side by side;
8. **runs the best combination** end-to-end through a cited RAG answer.

It depends on `day17_vector_db.py` (copied into this folder), which provides the embedding function, the chunk loader, and the keyword retriever.

## The decision that mattered most: measure before tuning

Every technique is evaluated against the **same `EVAL_SET`** (6 on-topic questions with the expected section verified against the manual text, + 3 off-topic questions that should be refused). A tuning run with no eval set is just "I feel it got better." Industrial RAG starts with the eval set.

> ⚠️ One eval trap I hit myself: "电池胀起来了还能继续用吗" belongs to **6. 充电与维护** ("电池鼓包、漏液应立即停止使用"), *not* 7. 安全预警. A mislabelled expected section judges a correct model as wrong. Every expectation in this file was checked against the source manual.

## Requirements

```bash
pip install chromadb sentence-transformers requests
```

| Package | Needed for | Without it |
|---|---|---|
| `chromadb` | the vector database | falls back to local hash vectors, conclusions shift |
| `sentence-transformers` | the Chinese BGE model | falls back to English ONNX (poor on Chinese) |
| `requests` + LLM key | query rewriting / rerank / generation | those three steps skip, everything else runs |

- **Python 3.12** (developed and tested on 3.12.6)
- Keys come from the `api_config.py` copy in this folder → environment variables (`ZHIPU_API_KEY` / `DEEPSEEK_API_KEY`). No key = offline mode, no crashes.

## Run

From this folder (`professional/day18_retrieval_tuning/`):

```bash
python day18_retrieval_tuning.py
```

It creates a `chroma_db_day18/` directory next to itself (the persisted store, git-ignored).

---

## Three engineering calls worth explaining out loud

### 1. RRF is for ordering, NOT for a rejection threshold

Reciprocal Rank Fusion score depends **only on rank**, not on similarity:

```
RRF_score = 1 / (RRF_K + rank)        # RRF_K = 60
```

1st place is always `1/61 ≈ 0.016`; with 8 chunks the max is ~0.033. Comparing that to `0.50` would **never** fire — yet this is the single most common RAG mistake. The fix is a clean division of labour:

- **RRF** decides the *order* of fused results;
- **raw cosine similarity** (the Chroma distance) decides *whether* to answer at all.

### 2. BGE's official query instruction: tested and rejected

BGE recommends prefixing queries with `为这个句子生成表示以用于检索相关文章：`. On this corpus it was **worse**:

| | Top-1 |
|---|---|
| without prefix | **2/4** |
| with prefix | 1/4 |

The knob is kept in `day17_vector_db.py` (`QUERY_INSTRUCTION`, default `""`) so you can re-test on your own data. Even an official best practice needs local validation.

### 3. The score spread tells you where the fault is

Max − min of the retrieval scores is a cheap diagnostic:

| spread | meaning | fix |
|---|---|---|
| < 0.15 | scores bunched → chunks too large / weak embedding | re-chunk (Day 16), or stop using the hash fallback |
| 0.15–0.30 | usable | set the threshold in the upper-middle |
| > 0.30 | good separation | threshold is easy, retrieval is trustworthy |

On this 8-chunk manual the spread is **0.123** — the chunks are too big and the vectors average into mush, so the *correct* answer ("5. 续航估算") ranks 6th. Re-chunking would help more than any parameter tweak. Naming this honestly is more credible than pretending the system is great.

## Measured results (this corpus, bge-small-zh, threshold 0.50)

Every row is the same 6 on-topic questions + 3 off-topic questions, verified against the manual.

| 方案 | Top-1 | Top-3 | Off-topic refused |
|---|---|---|---|
| ① 基线（纯语义检索） | 4/6 | 4/6 | 3/3 |
| ② + 查询改写 | **5/6** | 5/6 | 3/3 |
| ③ + 混合检索（不改写提问） | 3/6 ⚠️ | 4/6 | 3/3 |
| ④ + 查询改写 + 混合检索 | **5/6** | **6/6** | 3/3 |
| ⑤ 再 + LLM 重排序 | 5/6 | 6/6 | 3/3 |

Two results are easy to get wrong if you only read the names:

- **③ 混合检索单独用，反而变差（3/6）.** Keyword (word-frequency) scores live at 0.14–0.43 while semantic scores sit at 0.55–0.70 — different scales, so fusing them by rank drags the good result down. Multi-retrieval is *not* "more is better"; a weak route dilutes the strong one. The fix is ordering: rewrite the query first (②), *then* the keyword route stops being noise and the two routes complement each other (④, Top-3 hits 6/6).
- **⑤ LLM reranking adds 0 at this scale.** With 8 homogeneous chunks the answer is already in the candidates, so reranking only shuffles the order — Top-1 unchanged — at the cost of one extra model call (~1 s + tokens). It earns its place at corpus scale (tens of thousands of chunks) or when accuracy budget is tight. Naming this honestly beats pretending every component helps.

The one remaining miss on ④/⑤ — "冬天飞行掉电特别快是为什么" — is a **ranking** failure, not a recall failure: the right section (4. 温度管理) *is* retrieved (0.553) but sits 3rd behind two higher-scoring but less on-target chunks. That is exactly the case reranking targets, and a fair thing to defend in an interview.

Off-topic refusal is 3/3 at every stage — the threshold does its job.

## Why it matters

| Problem (Day 17) | Addressed by Day 18 |
|---|---|
| Top-1 only 2/4 | query rewrite + hybrid lifts it to 5/6 |
| scores unused | a real threshold → "我不知道" when irrelevant |
| wording gap | LLM query rewrite bridges user vs manual |
| no idea if tuning worked | one eval set, one benchmark table |

## About

Part of my learning series toward building a **UAV energy-consumption prediction AI System**. Full roadmap: see repo root `README.md`.

Next: **Day 19** — wrap this into a desktop Q&A app with a visible retrieval pane and a bad-case eval harness.
