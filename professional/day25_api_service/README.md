# Day 25 — Turn the Agent Into an HTTP Service (FastAPI)

**Week 6 (system integration), day 1.** Everything up to Day 24 was a *script*: you run
it, it prints, it exits. Real projects are different — the AI capability has to be
**called by somebody else**: a PySide6 desktop app, a WeChat mini-program, a web page.
None of those can `import` your `.py`. They send HTTP requests.

This day wraps Day 23's data-analysis ability and Day 24's manual retrieval into
**five HTTP endpoints**.

> Knowing how to call an LLM API = you can run a demo.
> Knowing how to wrap it as a service = you can hand it to somebody else.

---

## What changed, in one table

| | Day 24 (script) | Day 25 (service) |
|---|---|---|
| Who calls it | `main()` in the same process | anybody, over HTTP |
| When is the model loaded | at the top of `main()` | **once, at startup** (`lifespan`) |
| Bad input | crashes the script | Pydantic rejects it → **422** with the field name |
| Missing API key | whole run degrades | only `/ask` returns **503**; the rest still works |
| How you verify it | run it and read the console | `TestClient` self-test, no port needed |

---

## Files

### `day25_api_service.py` (≈560 lines) — the service

**Endpoints**

| Method | Path | What it does | Needs a key? |
|---|---|---|---|
| GET | `/health` | alive? key configured? index built? how many rows/chunks | no |
| POST | `/manual/search` | semantic search over the battery manual (bge) | no |
| POST | `/data/filter` | `filter_rows(col, op, value)` — real pandas math | no |
| POST | `/data/summary` | `describe_column(col)` — count/mean/min/max/quantiles | no |
| POST | `/ask` | full agent: natural language → tool call → answer | **yes** (else 503) |

**The four ideas that matter**

1. **`lifespan` pre-loads the heavy resource.** Loading bge takes ~30–60 s. If you load
   it *inside* an endpoint, every single request pays that cost. So it is built once at
   startup — and wrapped in `await run_in_threadpool(...)`, because a synchronous 30 s
   load inside an `async` lifespan would block the event loop and even `/health` would
   time out (in a container that means "unhealthy → restart loop").
2. **`def`, not `async def`.** FastAPI runs plain `def` endpoints in a thread pool, so
   blocking work (pandas, bge inference) does not stall the event loop. Writing
   `async def` for CPU-bound sync code is the classic FastAPI foot-gun — one request
   blocks the whole service.
3. **Pydantic is the gate.** Types, required fields, `top_k` bounds, and illegal `op`
   values are rejected with **422** before they reach any business logic. `/docs` and
   the request examples are generated from these models for free.
4. **Degrade, don't die.** No API key → `/ask` returns 503 explaining why, while
   `/health`, `/manual/search` and `/data/*` keep serving. One missing dependency must
   not take the service down.

**Self-test without starting a server**

`self_test()` drives the app through `fastapi.testclient.TestClient` inside a `with`
block — which also fires the `lifespan`. It exercises routing, validation,
serialisation and the startup hook **without occupying a port**, so it runs offline and
can go straight into CI.

Measured self-test output (local run, `glm-4-flash`):

```
✅ GET /health                      200  data_rows=23  manual_chunks=8  index_backend=bge
✅ POST /manual/search              200  Top1【4. 温度管理】score=0.631
✅ POST /data/filter                200  col=温度C  → 共 4 条   (59.1/60.0/57.1/56.3)
✅ POST /data/summary               200  电压V: 23 个, 均值 24.03, 最小 23.00, 最大 25.00
✅ POST /data/filter (bad op)       422  detail=不支持的比较符：大于
✅ POST /data/filter (bad column)   404  detail=找不到列… 可用列: ['时间','电压V','电流A','温度C','高度m']
✅ POST /ask                        200  answer=温度超过 55 度的有 4 次
```

---

## Three real lessons this day surfaced

### 1. `bind_tools` must happen *before* you build the graph

`build_agent(model, tools=TOOLS)` (Day 21-B) expects a model that **already has tools
bound**; it never binds for you. My first version passed a bare `SimpleChatModel`, so
the model had no idea tools existed — and confidently answered **"0 条"** where the
truth is **4 条**. That is Day 23's lesson seen from the other side: *if you don't hand
the model a calculator, it will do arithmetic wrong, politely.* After
`.bind_tools(TOOLS)`, the same question routes to `filter_rows(温度C,>,55)` → **4**.

### 2. Forgetting `SystemMessage` silently disables tool use

The first `/ask` implementation sent only `HumanMessage`, and the model replied
"insufficient information, please provide more data". The system prompt is *where the
model is told it has tools*. Same failure shape as #1, different cause — worth
remembering because both fail **silently** (HTTP 200, plausible text, wrong answer).

### 3. Fuzzy column matching belongs on the server side

Callers (and models) say 温度 / 电压, while the CSV headers are `温度C` / `电压V`.
`_match_col()` resolves exact → substring → 404-with-the-available-columns. Never push
"remember the exact header" onto the caller.

---

## Takeaway

> Days 21–24 changed *what the agent can do*. Day 25 changes *who can use it*.
> The graph is unchanged; the front door is new.
>
> Next step (not built here): point the PySide6 upper-computer (Week 2) at these
> endpoints — the desktop app posts flight data and renders the AI conclusion. That is
> the skeleton of the Week 6–7 integrated system.

---

## Running

```bash
# A. self-test (recommended first — no port, no server)
D:/Python-envs/chroma-env/Scripts/python.exe day25_api_service.py

# B. real server, then open http://127.0.0.1:8000/docs
D:/Python-envs/chroma-env/Scripts/python.exe -m uvicorn day25_api_service:app --reload

# C. skip the bge warm-up (instant start; /manual/search degrades to keyword matching)
DAY25_PREHEAT=0 D:/Python-envs/chroma-env/Scripts/python.exe day25_api_service.py
```

Without a key: everything works except `/ask` (503). Set `ZHIPU_API_KEY` or
`DEEPSEEK_API_KEY` (see `api_config.py`). First run loads bge from the local HF cache
(~30–60 s, no network).

## Dependencies

Runtime needs `fastapi`, `uvicorn`, `httpx` (for `TestClient`) on top of the earlier
stack. Self-contained — bundled alongside: `day23a_data_agent.py` (pandas tools),
`day21b_agent_tools.py` (the `BaseChatModel` adapter + ReAct graph), `api_config.py`,
`uav_battery_manual.md` (retrieval source), `w2_comms/flight_data_real.csv` (Week-2
serial capture, 23 rows).

Environment: fastapi 0.141.1, uvicorn 0.52.4, langgraph 1.2.11, langchain-core 1.6.1,
sentence-transformers 6.0.1, pandas.
