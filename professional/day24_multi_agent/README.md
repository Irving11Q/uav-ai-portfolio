# Day 24 — Multi-Agent Collaboration: One Supervisor, Two Workers

**Week 5 (LangGraph Agent).** Day 21 built a single ReAct agent. Day 22/23 gave that
agent different tools (retrieval, then data analysis). This day puts **several agents
to work together**: a **supervisor** that only routes and delegates, plus two
**worker sub-graphs** that each own a specialty.

> An LLM is unreliable at a hard task done alone; split it into "a specialist" + "a
> dispatcher". The dispatcher decides *who* does *what*; the specialist carries one
> job end to end.

---

## What changed, in one table

| | Day 23 (single data agent) | Day 24 (multi-agent) |
|---|---|---|
| How many agents | 1 | 3 (1 supervisor + 2 workers) |
| Who decides the tool | the single agent | the **supervisor** (a router) |
| Who answers | the same agent | the **worker** it was routed to |
| Failure mode | wrong tool chosen | wrong worker routed — but easily observable |

---

## Files

### `day24_multi_agent.py` (≈470 lines) — a supervisor graph

Three pieces, all built from the same Day 21-B `build_agent` graph:

**1. `ManualIndex` — the manual worker's retriever (in-memory, no Chroma)**

Loads `uav_battery_manual.md`, splits it at `##` headings, embeds the 7 sections with
the local **bge-small-zh** model (offline, from HF cache — zero network), and answers
`search_manual(query)` by cosine similarity. No vector DB to build or persist, so the
directory stays self-contained and runnable when copied out.

**2. Two worker sub-graphs (reuse Day 23's tools)**

| Worker | Tools | Answers |
|---|---|---|
| `data_worker` | Day 23's `peek / describe / filter_rows / correlate` + `plot_series` | flight-data questions |
| `manual_worker` | `search_manual` (bge RAG over the battery manual) | manual / spec questions |

Each worker is the *exact same* `build_agent(model, tools=...)` graph from Day 21-B —
only the tool list and system prompt differ.

**3. The supervisor graph**

```
START → supervisor ─┬─ "data"   → call_data   → supervisor
                    ├─ "manual" → call_manual → supervisor
                    └─ "FINISH" → END
```

The supervisor node asks the model *which worker should handle this* (it returns
`{"next":"data"|"manual"|"FINISH"}`). A worker node calls its sub-graph with
`graph.invoke(...)` and **merges the worker's messages back into the shared state** —
that `.invoke` hand-off is the essence of LangGraph sub-graph collaboration. When a
worker's final answer is already in the messages, the supervisor returns `FINISH`
(plus a `step` cap as a safety net against loops).

Measured routing (model path, `glm-4-flash`):

| Question | Supervisor → | Worker tool call | Answer |
|---|---|---|---|
| 锂电池温度超过多少度必须降落散热？ | `manual` | `search_manual` | from 【1】安全预警 /【3】温度管理 |
| 这次飞行温度超过 55 度的有几次？ | `data` | `filter_rows(温度C,>,55)` | 共 **4** 次 |

---

## Three real lessons this day surfaced

### 1. Sub-graph collaboration = `dispatcher.invoke(worker)` + merge messages

The supervisor never answers. It calls a worker as a compiled sub-graph and appends
the returned `messages` to the shared state. That is the whole multi-agent trick —
nobody holds all the tools; the dispatcher just passes the conversation around.

### 2. How to stop the loop without an infinite hand-off

A naive supervisor re-routes forever. The fix here: the supervisor checks "is there
already a worker answer in the messages?" → if yes, `FINISH`. A `step` counter
(≥6) is a second safety net. **Always give a router a terminal state**, or it will
bounce between workers.

### 3. The retriever can be in-memory when the corpus is tiny

Day 22 used Chroma (a persisted vector DB). Here the manual is 7 chunks, so an
in-memory cosine search over bge embeddings is enough — and it keeps the directory
self-contained (no `chroma_db/` to gitignore or rebuild on copy). **Match the
retrieval backend to the corpus size.**

---

## Takeaway

> Single agent → add tools (Day 22/23) → multiple agents collaborating (Day 24).
> The graph from Day 21 was reused **four times** this week (21, 22, 23, and today's
> two workers). What scales is the *orchestration*, not the graph.
>
> Next step (not built here): let workers pass intermediate results to each other —
> e.g. the data worker's conclusion feeds the manual worker to cross-check a spec.

---

## Running

```bash
# chroma-env (needs chromadb + sentence-transformers + langgraph + pandas + matplotlib)
D:/Python-envs/chroma-env/Scripts/python.exe day24_multi_agent.py
```

Needs an API key for the full multi-agent conversation (supervisor + workers call the
model). Without a key it still runs: the supervisor uses keyword routing and each
worker demonstrates its own tools — so the *structure* of the collaboration is visible
offline. The first run loads bge from the local HF cache (~1 min, no network).

Set `ZHIPU_API_KEY` or `DEEPSEEK_API_KEY` (see `api_config.py`).

## Dependencies

Self-contained — bundled alongside: `day21b_agent_tools.py` (the `BaseChatModel`
adapter + ReAct graph), `day23a_data_agent.py` / `day23b_chart_agent.py` (the data
tools + charting tool), `api_config.py`, `uav_battery_manual.md` (source),
`w2_comms/flight_data_real.csv` (the Week-2 serial capture, 23 rows).

Environment: langgraph 1.2.11, langchain-core 1.6.1, chromadb 1.5.9,
sentence-transformers 6.0.1, pandas, matplotlib.
