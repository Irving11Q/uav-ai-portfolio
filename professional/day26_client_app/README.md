# Day 26 — Desktop Client for the AI Service (PySide6 → HTTP)

**Week 6 (system integration), day 2.** Day 25 built the service. A service nobody calls
is just a process, so today it gets a **caller**: the Week-2 style PySide6 upper-computer,
rewired so that every button is an HTTP request instead of a local function call.

The client also **starts the server itself** (spawns `uvicorn` as a subprocess and polls
`/health` until ready) — you do not have to open a terminal first.

> Day 25 answers *"how do I expose this ability?"*
> Day 26 answers *"how does somebody actually use it?"*

---

## What changed, in one table

| | Day 25 (server) | Day 26 (client) |
|---|---|---|
| Runs where | its own process, headless | desktop window, one user |
| Entry point | HTTP request | a button click |
| Who waits for bge | the server, once at startup | the client, polling `/health` |
| Failure looks like | a JSON `detail` | a coloured message the user can act on |
| Verification | `TestClient` self-test | the same endpoints, driven from the client side |

**Buttons → endpoints**

| Button | Request | Notes |
|---|---|---|
| 刷新状态 | `GET /health` | lights the status lamp green when ready |
| 检索手册 | `POST /manual/search` | bge retrieval, shows title + score + snippet |
| 过滤数据 | `POST /data/filter` | real pandas filtering, not model counting |
| 统计摘要 | `POST /data/summary` | count / mean / min / max / quantiles |
| 问 AI | `POST /ask` | full agent; 503 if no API key |

---

## Files

### `day26_client_app.py` (≈556 lines) — the client

**Structure (7 blocks)**

1. environment probe — no PySide6 ⇒ fall back to CLI self-test instead of crashing
2. `ServiceManager` — spawn/stop `uvicorn`, poll `/health` until ready
3. `call_api()` — one exit point, translates HTTP into an **action level**
4. two `QThread`s — booting the server and sending requests
5. `MainWindow` — three tabs (manual / data / AI) plus a log pane
6. `self_test()` — no window, no Qt dependency, exercises everything
7. `main()` — GUI or self-test

---

## The three ideas that matter

### 1. Nothing that waits may run on the GUI thread

The main thread owns the window. Block it and Windows labels the app *Not Responding*.
That includes not just requests but **waiting for the 25-second bge warm-up** — so there
are two workers, not one:

- `BootWorker(QThread)` — spawns the server and polls `/health`
- `HttpWorker(QThread)` — performs one request, returns `(level, text)` via a signal

The worker object is kept on `self` (`self.worker`), otherwise Python garbage-collects a
running `QThread` and the app dies with it — a classic PySide6 foot-gun.

### 2. Errors have three layers — never collapse them into "request failed"

`call_api()` returns a *level*, and the level decides what the user is told:

| Level | HTTP | Meaning | What the UI says |
|---|---|---|---|
| `business` | 422 / 404 | *your input* is wrong | show the server's `detail` verbatim |
| `config` | 503 | a capability is missing (no API key) | point at what to configure |
| `offline` | — | the server is not up | "click 启动服务" |

These three need completely different user actions. Merging them into one message means
the user has no idea whether to retype a column name, set an env var, or start a process.

### 3. A deliverable client starts its own server

Expecting the user to run `uvicorn` in another terminal first is not delivery. So the
client `Popen`s the server with `cwd` set to the script directory (otherwise uvicorn
cannot import `day25_api_service`, which in turn imports `day23a` / `day21b`), then polls
`/health` every second with a progress line every 10 s. That loop is a **readiness
probe** — the same concept Kubernetes health checks use.

---

## Self-test output (local run, `glm-4-flash`, 31 s)

```
▶ 启动服务：-m uvicorn day25_api_service:app --host 127.0.0.1 --port 8123
   … 仍在加载（11 秒），首次要加载中文向量模型
✅ 服务就绪（耗时 24.4 秒，bge 预热占大头）

✅ 服务体检 → ok      data_rows=23  manual_chunks=8  backend=bge
✅ 手册检索 → ok      Top1【4. 温度管理】score=0.631
✅ 数据过滤 → ok      温度C > 55 → 4 条 (59.1 / 60.0 / 57.1 / 56.3)
✅ 列统计   → ok      电压V: 23 个, 均值 24.03, 最小 23.00, 最大 25.00
✅ 非法比较符 → business  【422】不支持的比较符：大于
✅ 乱写列名   → business  【404】找不到列「不存在的列」，可用列：[…]
✅ AI 提问   → ok      温度超过 55 度的有 4 次
✅ 服务不可达 → offline   (提示去启动服务)

自测完成：8 / 8 项通过
```

The last case points `BASE_URL` at a dead port to prove the `offline` branch fires —
error handling is *tested*, not just written.

---

## Three real lessons this day surfaced

### 1. "Wait for the server" is also blocking work

My first instinct was to call `mgr.start()` directly in the button handler. That freezes
the window for 25 seconds. The fix is not a `time.sleep()` tweak — it is a second thread.
Anything measured in seconds rather than milliseconds belongs off the GUI thread.

### 2. `QThread` needs an owner

`worker = HttpWorker(...); worker.start()` looks fine and crashes randomly: the local
variable goes out of scope, gets collected mid-run, and the process aborts. Keep a
reference on `self`. Same for `BootWorker`.

### 3. Format the response where the caller understands it

`/data/filter` returns `result` as a *string* containing a printed table, not JSON rows.
Choosing `QTextEdit` with a monospace font over `QTableWidget` was deliberate — it keeps
the client honest about what the server actually returns, instead of inventing a schema
the endpoint does not have.

---

## Running

```bash
# A. window (click 启动服务, wait ~25 s for the first warm-up)
D:/Python-envs/chroma-env/Scripts/python.exe day26_client_app.py

# B. CLI self-test, no window (used for release verification / CI)
D:/Python-envs/chroma-env/Scripts/python.exe day26_client_app.py --selftest

# C. point at an already-running server
DAY26_BASE=http://127.0.0.1:8000 D:/Python-envs/chroma-env/Scripts/python.exe day26_client_app.py
```

Env: `DAY26_PORT` (default `8123`), `DAY26_BASE` (overrides host+port).

Without a key: everything works except 问 AI (503, shown as a config-level message).
Without PySide6: the script detects it and runs the CLI self-test instead of crashing.

## Dependencies

`PySide6` (GUI only — optional), `requests`, plus the Day 25 stack (`fastapi`,
`uvicorn`, `httpx`, `langgraph`, `pandas`, `sentence-transformers`).

Self-contained — bundled alongside: `day25_api_service.py` (the server it launches),
`day23a_data_agent.py`, `day21b_agent_tools.py`, `api_config.py`,
`uav_battery_manual.md`, `w2_comms/flight_data_real.csv` (Week-2 serial capture, 23 rows).

---

## Takeaway

> Days 21–24 changed *what the agent can do*. Day 25 changed *who can call it*.
> Day 26 closes the loop: the person at the desktop never sees Python at all —
> they click a button and read an answer.
>
> Next step (not built here): push real serial data up instead of reading a static CSV —
> the Week-2 `RealSerial` reader posts each frame to the service and renders the AI
> conclusion live. That is the Week 6–7 integrated system.
