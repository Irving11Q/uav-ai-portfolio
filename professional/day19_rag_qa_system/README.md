# RAG Q&A System with Desktop UI — From Prototype to Measured Product

Day 19 of the *UAV AI Application Development* series, and the capstone of the RAG week.

Days 15–18 were all command-line: you ran a script and got a number back. This day turns the
whole pipeline into a **desktop Q&A app with a desktop UI** — and, more importantly, puts a
**measurable evaluation harness** around it, so the claim "it works" becomes "here is exactly
how well it works, and here is why the one failure happened."

---

## What's inside

Three files I wrote for this day:

| File | Lines | Role |
|---|---|---|
| `day19a_ui.py` | 350 | PySide6 UI skeleton. Runs on a **fake retriever** first — deliberately |
| `day19b_rag_backbone.py` | 218 | Swaps in the real retrieval + LLM chain. **UI code unchanged** |
| `day19c_eval_badcase.py` | 330 | Evaluation set + automatic bad-case attribution |

Copied in so this folder is self-contained (built on earlier days):

`day17a_embedding.py` · `day17b_chroma_store.py` · `day17c_rag_pipeline.py` ·
`day18a_baseline_eval.py` · `day18b_recall_tuning.py` · `chunks_parsed.json` ·
`uav_battery_manual.md` · `api_config.py`

---

## Why three files instead of one

Earlier days produced 1,300-line monoliths and they were miserable to debug. The rule I now
follow is **one file answers one question, cap it around 400 lines**:

- **19a — "What should this look like?"** Built the UI against a fake retriever so it could be
  run and seen immediately, with zero dependency on whether retrieval worked yet.
- **19b — "Now make it real."** Swapped the implementation, not the interface.
- **19c — "How good is it, actually?"** Evaluation is a separate concern and stays separate.

---

## The decision that mattered most: one interface, two implementations

Day 19-A defines a contract:

```python
retrieve(question, top_k) -> [(title, body, score), ...]
```

Day 19-B's `RealRetriever` honors exactly that contract. So going from fake to real changed
**one line** in `main()`:

```python
retriever = FakeRetriever()      # Day 19-A
retriever = RealRetriever()      # Day 19-B  ← the only change
```

All 350 lines of UI code stayed untouched. The payoff showed up immediately: on the question
"电池能飞多久" (how long can it fly), the fake retriever ranked `7. 安全预警` first (it matched
the character 电), while the real one correctly ranked `5. 续航估算` at 0.689 — and the model's
answer cited source [1].

The retrieval itself is also a single line, because it consumes all of Day 18's work:

```python
picked, refused = search_with_threshold(self.coll, question, threshold, top_k)
```

That one call is query rewriting → semantic + keyword recall → RRF fusion → similarity
thresholding. Roughly 2,000 lines of prior work, absorbed by one function.

---

## Measured results

Setup: `bge-small-zh-v1.5` (512-dim), rejection threshold 0.50, 8 heading-based chunks.

| Metric | Result | Meaning |
|---|---|---|
| Top-1 recall | **5/6** | the first chunk retrieved is the correct chapter |
| Top-3 recall | **6/6** | the correct chapter is somewhere in the top 3 |
| Off-topic rejection | **3/3** | questions the manual can't answer get "I don't know" |
| Wrongful rejection | **0** | no question the manual *can* answer was blocked |

Evaluation set: 6 on-topic questions (each expected chapter verified against the source text)
plus 3 off-topic questions that must be refused.

### The one failure — the part I would actually defend in an interview

> 「冬天飞行掉电特别快是为什么」(why does the battery drain so fast in winter)
> expects chapter `4. 温度管理`, but `5. 续航估算` ranked first.

`day19c_eval_badcase.py` attributed it as a **ranking failure**, not a recall failure:
temperature management *was* retrieved (0.553), just at position 3. That distinction decides
the fix — reranking, or smaller chunks — instead of blindly turning the threshold knob.

Being able to name your system's one failure and explain its cause is worth more than a
perfect score you can't account for.

---

## Three engineering calls worth explaining out loud

**1. RRF fusion scores cannot be used as a rejection threshold.**
RRF scores depend only on *rank* (1st place always gets 1/61 ≈ 0.016), not on how relevant
something actually is. With 8 chunks the maximum is ~0.033, so comparing it against a 0.50
threshold would never fire — the threshold would be dead code.
The fix is a **division of labor**: RRF score for *ordering*, raw cosine similarity for
*deciding*. This is the standard industry approach.

**2. I tested the BGE query instruction prefix and rejected it.**
BGE's official guidance is to prepend `为这个句子生成表示以用于检索相关文章：` to queries.
On my own manual, measured: **2/4 hits without it, 1/4 with it**. Worse.
Official best practices still need validating on your own data — so it ships disabled, with
the measurement recorded in the code.

**3. Measure before tuning.**
Retrieval is the most "mystical" part of RAG, and it stops being mystical once you can see it.
The UI renders retrieved chunks and their scores as bar charts on every single question.
A companion probe also reports the score **spread** (max − min): on this dataset the spread was
only **0.123**, which is the signature of chunks that are too large — their vectors average
themselves into mush. That single number told me to go fix chunking rather than tune parameters.

---

## Requirements

```bash
pip install chromadb sentence-transformers requests PySide6
```

| Package | Needed for | Without it |
|---|---|---|
| `chromadb` | the vector store | retrieval falls back to local hash vectors |
| `sentence-transformers` | the Chinese BGE model | falls back to an English model (poor on Chinese) |
| `PySide6` | the desktop UI | use `--cli` mode instead |
| `requests` | the generation step | everything except generation runs |
| LLM API key | the generation step | everything except generation runs |

- **Python 3.12** (developed on 3.12.6)

> **Environment note:** this project needs PySide6 *and* chromadb in the same interpreter.
> I used a venv created with `python -m venv --system-site-packages` so it can see the
> system-installed PySide6 while keeping the heavy ML packages isolated. Installing chromadb
> straight into a system Python previously broke a dozen unrelated packages for me.

### Enabling the generation step

The dependency modules look for `api_config.py` one level up in `w3_ai/`:

```bash
mkdir -p ../w3_ai && cp api_config.py ../w3_ai/
```

`api_config.py` reads its key from an environment variable (`ZHIPU_API_KEY` or
`DEEPSEEK_API_KEY` — never hardcoded). Without a key, retrieval, the UI, and the full
evaluation still run; only the LLM answer is skipped.

---

## Run

From this folder:

```bash
# 1. CLI self-test — fastest way to confirm the pipeline works (no UI)
python day19b_rag_backbone.py --cli "电池能飞多久"

# 2. Full desktop UI
python day19b_rag_backbone.py

# 3. Evaluation with automatic bad-case attribution
python day19c_eval_badcase.py

# 4. Same evaluation at a different threshold, to see the trade-off
python day19c_eval_badcase.py --threshold 0.45
```

`day19c` writes a Markdown report with per-question detail and attributed bad cases.

---

## Known limitations

Stated plainly, because the point of this project is knowing where the edges are:

- **8 chunks, single-machine Chroma** — this is a validated prototype, not production scale.
- **Chunks are still too large.** Measured score spread of 0.123 confirms it; re-chunking is
  the highest-value next step, ahead of any retrieval tuning.
- **9-question evaluation set** — enough to validate the approach, nowhere near enough to
  make production decisions. It needs to grow to hundreds.
- **No reranking in the shipped pipeline.** I measured it (Day 18-C) and found the cost/benefit
  not yet worth it; the rule-based rerank was the better cheap alternative.
- **No service layer yet** — no FastAPI, no Docker. Desktop app + CLI only.

---

## What this day added to the series

| Before (Days 15–18) | After (Day 19) |
|---|---|
| command-line only | desktop UI with visible retrieval |
| "it seems to work" | 5/6 Top-1, 3/3 rejection, measured |
| failures were confusing | failures auto-attributed to a cause |
| one-off scripts | the deliverable the week was aiming for |

The week's stated deliverable — *a UAV battery/energy manual Q&A system* — is complete here,
with numbers attached.
