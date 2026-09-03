# Day 21 — LangGraph Agent: when the model decides what happens next

Day 13 of this repo already had a tool-calling loop, written by hand.
This day rebuilds it with **LangGraph** — and the point is not "use a framework",
it is fixing a concrete defect in the hand-written version.

## What's inside

| File | Lines | Needs API key | What it teaches |
|---|---|---|---|
| `day21a_state_graph.py` | 387 | No | State / node / conditional edge / loop |
| `day21b_agent_tools.py` | 562 | Yes | Real tool-calling agent + a custom model adapter |
| `day21c_agent_memory.py` | 389 | Yes | Checkpointer + `thread_id` |

Read them in order. 21a runs with zero configuration on purpose — the shape of a
graph has nothing to do with the model, so you should see it working before any
network call is involved.

## The defect this day fixes

Here is Day 13's loop, reduced to its skeleton:

```python
msg = ask_model(question, tools=TOOLS)
if msg.get("tool_calls"):
    for call in msg["tool_calls"]:
        result = FUNCTIONS[call["function"]["name"]](**call["function"]["arguments"])
        messages.append({"role": "tool", ...})
    return ask_model(messages)      # <- returns. Two rounds, hard stop.
return msg["content"]
```

The model calls tools **once**. If the result makes it think "I need one more
lookup before I can answer", there is no path back. Supporting that means writing
`while True` yourself and then owning every question it raises: how many
iterations before giving up, what happens on a bad tool call, where intermediate
state lives.

That is the entire reason agent frameworks exist.

## Measured results

### 21a — the loop is real (no model involved)

A pre-flight battery self-check. `check` runs once per checklist item, so a
4-item checklist means the **same node executes 4 times**:

| Scenario | Node visits | Path |
|---|---|---|
| Healthy battery | 5 | check ×4, all pass → report |
| Aged battery (0.21V cell gap, 2°C) | 6 | check ×4 → 2 issues → consult manual → report |
| Low charge (18%) | 6 | check ×4 → 1 issue → consult manual → report |

The branch (skip the manual lookup when nothing is wrong) is a conditional edge,
not an `if` inside a node.

### 21b — the model chains two dependent tools

The two tools are deliberately built so the second **cannot** be answered without
the first:

```text
check_battery_health(cell_voltages, nominal_capacity_ah)  ->  usable capacity in Ah
estimate_endurance(usable_capacity_ah, avg_current_a)     ->  flight minutes
```

Question: *3S pack, cells at 3.92 / 3.68 / 3.88, nominal 5.0 Ah, 20 A draw — how
long can it fly?*

```text
step 2  ->  check_battery_health([3.92, 3.68, 3.88], 5.0)
            {"gap_V": 0.24, "health": "severely aged", "usable_capacity_Ah": 3.5}
step 4  ->  estimate_endurance(3.5, 20)          <- 3.5, not the nominal 5.0
            {"flight_minutes": 8.4}
step 6  ->  "severely aged, 8.4 minutes, replace the pack"
```

The model used the capacity returned by the tool instead of the number in the
question. Answering from the nominal 5.0 Ah would have produced 12 minutes — a
**43% overestimate** on a battery that is about to be retired.

Two tool-calling rounds in one turn. Day 13's loop cannot do this.

### 21c — the real cost of having no memory

Turn 1 gives the full battery spec and gets 8.4 minutes.
Turn 2 asks only *"what about 15 A instead?"* — every parameter omitted, because
a human would remember. Correct capacity is **3.5 Ah**.

| Setup | Capacity the model passed | Verdict |
|---|---|---|
| No checkpointer | 4.25 | ❌ fabricated (+21.4%) |
| `MemorySaver`, same `thread_id` | 3.5 | ✅ correct |
| `MemorySaver`, different `thread_id` | 4.76 | ❌ fabricated (+36.0%) |

All three rows call the tool exactly **once**. The difference is not how many
calls — it is whether the argument is right. With no memory the model does not
say "I don't have that"; it invents a plausible-looking number, and the invented
number changes every run (4.25, 4.76, ...), which is how you tell it apart from a
recall.

> Memory is not about asking the user one fewer question. It is about the model
> fabricating one fewer number.

## The custom model adapter

`langchain_openai` is not installed in this environment, so `day21b` ships its own
chat model instead of adding a dependency — `SimpleChatModel(BaseChatModel)`.
Subclassing `BaseChatModel` only requires:

1. `_generate(messages, ...) -> ChatResult` — send HTTP, parse the response
2. `_llm_type` — a name for the model
3. `bind_tools(tools)` — convert tools to OpenAI tool specs, return a copy via
   `self.model_copy(update={"tool_specs": specs})`

Roughly 120 lines, no new packages. This is the part worth explaining out loud in
an interview: anyone can call `ChatOpenAI(...)`, but being able to say what that
object actually does underneath is different.

## Requirements

```bash
pip install langgraph langchain_core requests
```

The environment used for the numbers above: `langgraph 1.2.11`,
`langchain_core 1.6.1`, model `glm-4-flash`.

An OpenAI-compatible endpoint is enough — the adapter only needs `base_url`,
`api_key`, and a model id. Set them in `api_config.py` (a copy sits in this
folder; `day21b` looks for `../w3_ai/api_config.py` first and falls back to it).

Without a key, `day21a` still runs fully and `day21b`/`day21c` skip only the
model-backed demos and explain why.

## Run

```bash
# from this folder
python day21a_state_graph.py     # no key needed
python day21b_agent_tools.py
python day21c_agent_memory.py    # ~60s, makes 6 model calls
```

## Known limitations

- **Memory is in-process.** `MemorySaver` keeps everything in RAM, so it dies with
  the process. Production needs `SqliteSaver` / `PostgresSaver`
  (`pip install langgraph-checkpoint-sqlite`, not installed here).
- **Messages grow without bound.** Long conversations get expensive. Real systems
  need trimming or summarising; this day does neither.
- **The fabricated-number finding is from one run.** It reproduced across runs
  (4.25, 4.76) but this is not a benchmark — it is a demonstration.
- **No error handling for tool failures.** A tool raising an exception propagates
  and stops the graph.
- **GLM does not support OpenAI strict mode**, so `convert_to_openai_tool(..., strict=False)`
  is required. On a model that does support it, turning it on constrains arguments
  to the schema.

## What this day added

| Before (Day 13) | After (Day 21) |
|---|---|
| Fixed two-round loop | Loop exits when the model stops asking for tools |
| Hand-written `while` state juggling | `State` carries messages between nodes |
| No memory between turns | `checkpointer` + `thread_id` |
| One tool call per turn | Chained dependent tool calls |
