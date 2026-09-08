# Day 23 — Data Analysis Agent: Let the Model *Compute*, Not *Recite*

**Week 5 (LangGraph Agent).** Day 21 built a ReAct agent; Day 22 wrapped retrieval
as a tool. This day wraps **data computation** (pandas) and **charting** (matplotlib)
as tools, on top of the real flight log you captured over serial in Week 2
(23 rows: time / voltage / current / temperature / altitude).

The core lesson:

> An LLM is unreliable at arithmetic (it miscounts, it fabricates), but it is
> excellent at *deciding what to compute*. So the division of labour is:
> **the model decides the question, the code produces the exact number.**

---

## What changed, in one table

| | Day 22 (Agentic RAG) | Day 23 (Data Agent) |
|---|---|---|
| Tool returns | a text chunk; model picks the answer out of it | a number; the number *is* the fact |
| Hallucination surface | medium (model rephrases retrieved text) | near zero (number is computed, not invented) |
| Model's real job | decide *whether / what* to retrieve | decide *which metric / filter / chart* |
| New failure mode | skipping the tool entirely | column-name mismatch, silent CSV-not-found |

---

## Files

### `day23a_data_agent.py` (390 lines) — compute tools + a controlled experiment

Four `@tool`s wrap pandas so the model never does arithmetic itself:

| Tool | What it computes |
|---|---|
| `peek_data` | show first rows / column names |
| `describe_column` | count, mean, std, min, quartiles, max |
| `filter_rows(col, op, value)` | "how many times was temp > 55" |
| `correlate(col1, col2)` | Pearson r, with a strength label |

The file runs a **side-by-side controlled experiment** on the same question:

```
Question: 电压低于 23.5 伏的记录有几条？
  Control (no tool, whole 23-row table pasted into the prompt):
      model counts → 3            ❌ wrong
  Treatment (pandas tool does the counting):
      filter_rows(电压V, <, 23.5) → 5   ✅ correct
  Truth is 5 — computed by pandas, not by the model.
```

Three agent questions, all natural language, all answered via tools:

| Question | Tool call | Answer |
|---|---|---|
| 这次飞行电压最低掉到多少？ | `describe_column(电压)` | 最小 **23.00** |
| 温度超过 55 度的有多少次？都什么时候？ | `filter_rows(温度C, >, 55)` | **4 条** (16:50:14–17) |
| 电流和电压是正相关吗？ | `correlate(电流A, 电压V)` | r = **0.161**（几乎没有相关） |

The graph is **identical** to Day 21-B / Day 22 — only the tool list changed
(`build_agent(model, tools=TOOLS)`). That one line is the whole point of the
week: same graph, swap tools → swap the entire application.

### `day23b_chart_agent.py` (249 lines) — a charting tool, and a hard boundary

Adds one tool on top of Day 23-A's four:

```python
@tool
def plot_series(cols, title) -> str:
    # draws a line chart to _out_day23b/<cols>.png
    # returns a TEXT summary of the numbers — NOT the image
```

The single most important sentence of the day:

> **The model cannot see the chart.** It calls `plot_series`; the tool draws the
> picture and saves it; what comes back to the model is a *text summary*. So when
> the model "talks about the chart", it is actually talking about numbers.

A contrast demo proves it: asked "in that voltage chart, when was the lowest point?",
the model does **not** look at the image — it calls `filter_rows` / `describe_column`
and answers from the numbers. Making the model truly *see* a chart means feeding the
PNG into a vision (multimodal) model — a different topic.

`matplotlib` Chinese-font gotcha is handled in `_setup_chinese_font()` (otherwise
every Chinese label renders as `□□□`, and negative signs break too).

---

## Three real bugs / lessons this day surfaced

These are the parts worth talking about in an interview — all found by running the
code, not by reading it.

### 1. `api_config` exposes `MODEL_NAME`, not `MODEL` — a silent-degrade trap

```python
from api_config import MODEL          # ❌ ImportError: no such name
# the whole try block dies → HAS_KEY stays False
# result: you HAVE a key, but the code behaves as if you don't
```

Fixed by importing the module and reading the attribute:
`import api_config; MODEL = api_config.MODEL_NAME`. One wrong name in a `from … import`
silently downgrades a working setup to the no-key branch — invisible until you
actually watch a demo that should have called the model but didn't.

### 2. Column-name fuzziness — the model says "电压", your CSV says "电压V"

```python
def _resolve_col(name):
    # "电压" must match "电压V", or the tool raises KeyError
    # and the model decides the column doesn't exist → starts hallucinating
```

`_resolve_col()` fuzzy-matches the model's phrasing to the real header and, on a
miss, returns the *list of available columns* so the model can self-correct. This is
a small but very common production gap: the model speaks natural language, your schema
speaks `snake_case` / suffixed units.

### 3. CSV-not-found must exit non-zero, not `return`

```python
if not os.path.exists(CSV_PATH):
    print("找不到数据文件…")
    sys.exit(1)          # ✅ was: return  → exit code 0 → "looked successful"
```

The first draft `return`ed, so a broken setup still exited 0 and *looked* like a
pass. Silent success is worse than a crash. Also added a **dual-layout** CSV search
(`<here>/w2_comms` for the self-contained package, `<parent>/w2_comms` for the local
repo) so the directory runs both in-place and when copied out.

---

## Takeaway

> A data-analysis agent is not "the model reads the spreadsheet". It is:
> the model decides *what* to compute, tools compute it exactly, and the
> agent only paraphrases verified numbers.
>
> The moment a tool returns a number instead of prose, the model's
> hallucination surface collapses — because the number was never its to invent.

Day 22 → 23 shows one graph serving three apps (manual Q&A, data analysis, charting).
Next, Week 5 Day 5: **multi-agent collaboration** (a supervisor that delegates to
several worker agents).

---

## Running

```bash
# chroma-env (needs chromadb + sentence-transformers + langgraph + pandas + matplotlib)
D:/Python-envs/chroma-env/Scripts/python.exe day23a_data_agent.py
D:/Python-envs/chroma-env/Scripts/python.exe day23b_chart_agent.py
```

`day23a`/`day23b` need an API key (they call the model to demonstrate tool use).
Without a key they still run the tools themselves and show what the model *would*
have received. `day23b` writes charts to `_out_day23b/`.

Set `ZHIPU_API_KEY` or `DEEPSEEK_API_KEY` (see `api_config.py`).

## Dependencies

Self-contained — bundled alongside: `day21b_agent_tools.py` (the `BaseChatModel`
adapter and ReAct graph, Day 21), `api_config.py`, `w2_comms/flight_data_real.csv`
(the Week-2 serial capture, 23 rows).

Environment: langgraph 1.2.11, langchain-core 1.6.1, chromadb 1.5.9,
sentence-transformers 6.0.1, pandas, matplotlib.
