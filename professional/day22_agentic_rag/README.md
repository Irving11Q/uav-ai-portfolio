# Day 22 — Agentic RAG: Retrieval Becomes a Tool

**Week 5 (LangGraph Agent).** Day 15–20 built a RAG pipeline where retrieval was **hard-coded**:
every question triggered a search, no matter what was asked. Day 21 built a ReAct agent.

This day connects them: the retriever is wrapped as a **tool**, and the model decides
when to use it, what query to search with, and whether to retry after a miss.

---

## What changed, in one table

| | Day 19 (fixed pipeline) | Day 22 (Agentic RAG) |
|---|---|---|
| Who decides *whether* to retrieve | `if/else` in your code | the model |
| What gets searched | the user's raw wording | a query the model rewrites |
| When retrieval misses | pipeline stops → "not in the manual" | model rewrites and tries again |
| Chit-chat | still runs a search (wasted) | skips the tool |

---

## Files

### `day22a_retrieval_as_tool.py` (554 lines) — the decision to search

Wraps the Day 18 retriever in a `@tool` and hands it to the Day 21 ReAct graph.
The graph is unchanged; only the tool list differs.

Four demos, all measured on this repo's data (bge-small-zh, threshold 0.50):

| Question | Tool calls | Outcome |
|---|---|---|
| 冬天低温飞行，电池要注意什么？ | 1 | hits `7. 安全预警` (0.558), answers with `【1】` citations |
| 你好，请用一句话介绍一下你自己。 | 0 | chit-chat — correctly skips retrieval |
| 这块电池的保修期是多久？ | **0** | ⚠️ model skips the tool and fabricates "usually 1 year" |
| same question + keyword guardrail | 1 (forced) | ✅ "手册里没有提到" — hallucination blocked |

### `day22b_retry_and_cite.py` (376 lines) — query rewriting and retry

The same question, two ways of searching it:

```
User asks: 能撑几个起落？          ("how many sorties can it last" — pilot slang)

Fixed pipeline: searches the raw words   → 0.404 → refused → "not in the manual"
Agent:          rewrites to 续航 时间     → 0.673 → hit     → "about 25 minutes" 【1】
```

**The rewrite was produced by the model, not by code.** Compare with Day 18-B: that
query rewriting was a hand-written synonym table, which dies on any phrase it does not
contain. The model's vocabulary *is* its rewrite rule.

A third demo confirms the agent stops after one attempt when the manual genuinely
cannot answer, instead of looping forever.

### `day22c_crag.py` (403 lines) — Corrective RAG: grade the retrieval

`day22a/22b` assume that if the tool returned something, the answer can be built from
it. That assumption is exactly what breaks in production. This file adds the node
Agentic RAG usually skips — a **grader** that judges the retrieval *before* the model
is allowed to answer:

```
agent → guard → tools → grade ─┬─ relevant     → agent  (answer, with citations)
                               └─ not relevant → rewrite (new query) → tools
                                                  (max 2 retries, then stop)
```

**Two different failures need two different mechanisms** — this is the whole point:

| Failure mode | Caught by | Mechanism |
|---|---|---|
| retrieved something, but it does not answer the question | `grade` node | rewrite the query and search again |
| never retrieved at all — the model skipped the tool | `guard` node | domain keyword → force one search in code |

A grader can only judge results that exist. It cannot notice that the model never
called the tool. That blind spot has to be closed in code, which is why `guard` is
the second half of CRAG rather than an optional extra.

Measured on 这块电池的保修期是多久？ (not in the manual):

| | behaviour |
|---|---|
| Day 22-A | 0 tool calls, fabricates "usually 1 year warranty" |
| Day 22-C | guard forces a search → "手册里没有提到这块电池的保修期" |

---

## Three real bugs this day surfaced

These are the parts worth talking about in an interview — both were found by running
the code, not by reading it.

### 1. `bind_tools` and `ToolNode` must receive the same tool list

```
🔧 model calls search_manual(...)
📄 Error: search_manual is not a valid tool, try one of [check_battery_health, estimate_endurance]
```

`day21b.build_agent()` had `ToolNode(TOOLS)` hard-coded to Day 21's battery tools.
The model asked for a tool the execution node did not know, got an error string back,
and quietly fell back to answering from its own (wrong) knowledge. Fixed by adding a
`tools=` parameter to `build_agent`.

### 2. A small model will skip the tool and hallucinate — and it is not deterministic

| Symptom | Cause | Fix |
|---|---|---|
| "Usually 1 year warranty…" with no tool call | glm-4-flash judged the manual irrelevant | keyword guardrail: force retrieval on domain terms |
| "What model do you mean?" instead of searching | small models ask for clarification | explicit "do not ask back" rule in the system prompt |
| Same code, same question: works once, skips the tool next run | `temperature=0.3` randomness | `temperature=0` for agent workloads |

After setting `temperature=0`, all demos behaved consistently across runs.

### 3. A dead variable that only crashes on the path you did not test

```python
# guard_node — the "model skipped the tool" branch
q = question          # NameError: question is never defined in this function
```

The first run passed because the model happened to call the tool itself, so this line
was never reached. A later run took the other branch and the graph died. Two lessons:

- Guard nodes are *fallback* code — by definition they run least often, so they are the
  least tested paths in the whole graph. Force them once on purpose.
- Leftover scaffolding (`q = question` was a placeholder from an earlier draft) survives
  review easily when the surrounding lines look right. Re-read fallback branches.

---

## Takeaway

> Real systems are not "fixed pipeline **vs** agent" — they are a mix.
> Use the deterministic pipeline as a floor for high-risk domains,
> and let the agent decide in open-ended ones.

The guardrail demo in `day22a` is exactly this: domain keyword hits → retrieval is
forced and the result is injected as context; the model is left to phrase the answer.

---

## Running

```bash
# chroma-env (needs chromadb + sentence-transformers + langgraph)
D:/Python-envs/chroma-env/Scripts/python.exe day22a_retrieval_as_tool.py
D:/Python-envs/chroma-env/Scripts/python.exe day22b_retry_and_cite.py
D:/Python-envs/chroma-env/Scripts/python.exe day22c_crag.py
```

`day22b` and `day22c` require an API key (they call the model to demonstrate retrying
and grading). `day22a` runs its retrieval demo without a key and skips the model part.

`day22c` output (measured, temperature=0):

```
演示一 手册里有的问题      → 1 次查中 (0.721)，带【1】【3】引用
演示二 口语「能撑几个起落」 → 改写为「电池续航」 (0.673) → 约 25 分钟
演示三 手册没有的「保修期」 → guard 强制检索 → 「手册里没有提到」← 不再编造
```

First run downloads nothing but loads the BGE model from the local HF cache
(~1–2 min). `HF_HUB_OFFLINE=1` is set inside the scripts to avoid a 5-minute
network timeout on huggingface.co.

Set `ZHIPU_API_KEY` or `DEEPSEEK_API_KEY` (see `api_config.py`).

## Dependencies

Self-contained — bundled alongside: `day18a/day18b` (retrieval, Day 18),
`day17a/17b/17c` (embedding + Chroma, Day 17), `day21b` (the `BaseChatModel`
adapter and ReAct graph, Day 21), `chunks_parsed.json` (8 manual chunks),
`uav_battery_manual.md` (source), `api_config.py`.

Environment: langgraph 1.2.11, langchain-core 1.6.1, chromadb 1.5.9,
sentence-transformers 6.0.1.
