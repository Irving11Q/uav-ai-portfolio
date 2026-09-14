# Day 28 — Turning the Inspector Into a Service (background collector + `/live/*`)

**Week 6 (system integration), day 4.** Day 27's inspector was a *script*: it read the
serial port itself, printed to its own console, and wrote its own report file. That means
one inspector, visible only on the machine running it.

Today splits it in two:

- **collection moves into a background thread inside the service process** — it keeps
  running whether or not anyone is asking
- **anybody can read the current state** with `GET /live/status`

The inspection logic itself is not rewritten — `day28_live_api.py` imports Day 27 and adds
only the service layer. That reuse is the point: Day 27 wrote the domain logic, Day 28
publishes it.

---

## What changed, in one table

| | Day 27 (script) | Day 28 (service) |
|---|---|---|
| Who runs collection | the script's own loop | a background thread, always on |
| Who can see the state | whoever is looking at that console | any client over HTTP |
| Lifetime | starts and ends with the run | starts/stops on request, survives between reads |
| Output | console + a report file | live JSON snapshot + alerts + Markdown |
| Concurrency worry | none (single thread) | **collector writes while requests read** |

**Endpoints**

| Method | Path | What it does |
|---|---|---|
| POST | `/live/start` | begin collecting `{source: replay\|synth\|real, speed: N}` |
| POST | `/live/stop` | stop collecting |
| GET | `/live/status` | ★ current snapshot: latest frame, window stats, prediction, recent alerts |
| GET | `/live/alerts` | full alert timeline (`?limit=N`) |
| GET | `/live/report` | current inspection report as Markdown text |
| GET | `/health` | service + collection status |

---

## The four ideas that matter

### 1. The state lives in the service process, not in the client

There is exactly one "current flight state" — the aircraft's. Ten clients must all see the
same one. If collection lived in the client, every viewer would be looking at data it made
up itself. This is why the collector is a thread *inside* the server and clients can only
read.

### 2. A background thread plus a lock — because FastAPI endpoints run in a thread pool

FastAPI executes plain `def` endpoints in worker threads (Day 25), so a request thread
reads the window at the same moment the collector thread is appending to it. Same object,
two threads, no lock — that is not a theoretical race, it is a crash waiting for load.

The rule used here:

- writes to the window/alerts are done **while holding the lock**
- reads return a **snapshot** (`status()` copies into a plain dict under the lock)

Copying on read keeps the critical section microscopic and means callers can never hold a
reference to a structure that is still being mutated.

One deliberate exception: the AI inspection call is made **outside** the lock. It takes
seconds; holding the lock across a network call would block every endpoint for that whole
time.

### 3. Polling first — and being able to say why

`/live/status` is polled about once a second. For a human-watched dashboard this is
perfect: stateless, curl-able, self-healing after a dropped connection, no server-side
session to leak.

WebSocket / SSE is the right upgrade when the update rate climbs into the tens of Hz or
when many clients must be pushed at once. Choosing not to use it on day one — and being
able to explain the threshold — is worth more in an interview than reaching for it
reflexively.

### 4. Thread lifetime has to be owned by somebody

Three failure modes, three defences:

| Failure | Defence |
|---|---|
| `/live/start` clicked twice → two collectors, double the data | `start()` is **idempotent**, returns `409` if already running |
| `stop` leaves the thread mid-frame | `stop()` sets an event and **`join(timeout=10)`** |
| uvicorn exits with the collector alive → port stays occupied | `lifespan` shutdown calls `stop()`; thread is also `daemon=True` as a backstop |

---

## Self-test output (12/12, no port, no model calls)

```
✅ GET /health                    初始应为未采集
✅ POST /live/start（replay）
✅ GET /live/status               frames=10 running=True
✅ 告警链路（真实数据温度>55℃）      alarm=3 total=5
✅ 重复 start 被拒（幂等）          已经在采集了（先 /live/stop）
✅ 非法 source → 422
✅ 重新 start（synth）
✅ 预测触发（合成轨迹）              R²=1.000 → 约 64 秒后跌破 23.0V
✅ GET /live/alerts / GET /live/report / POST /live/stop
✅ 停止后 running=False
自测完成：12 / 12 项通过
```

Two rounds are run on purpose: **`replay`** proves the alert path end-to-end (real capture,
temperature > 55 °C → alarms recorded and readable through the endpoint), and **`synth`**
proves the prediction path (R² = 1.000 clears the confidence gate and the extrapolation
fires). A single source would have left one of the two links untested.

---

## Lessons this day surfaced

### A list attribute and a method cannot share a name

`Collector` held `self.alerts = []` **and** defined `def alerts(self, limit)`. The instance
attribute wins, so the endpoint hit `TypeError: 'list' object is not callable` at request
time — the module imported fine and the app started fine. Renamed the attribute to
`alert_log`. Worth remembering because it fails only when the endpoint is actually called.

### Snapshot semantics need to be stated, not assumed

`/live/status` returns `alerts_recent` (last 5) while `/live/alerts` returns the timeline.
That split is intentional: a dashboard wants the tail, an audit wants everything. Don't
make one endpoint serve both.

### Restarting a source resets the state

`/live/start` clears the window and the alert log — a new flight is a new flight. That is
why the second self-test round reports `count=0` on `/live/alerts`: it is a fresh `synth`
run with no alarms yet, not a bug in the endpoint.

---

## Files

### `day28_live_api.py` (425 lines)

| Block | What it holds |
|---|---|
| 1 | reuse layer — imports Day 27's sources / rules / window / prediction |
| 2 | `Collector`: background thread, lock, snapshot, start/stop |
| 3 | `app` + `lifespan` + Pydantic request model |
| 4 | the five endpoints |
| 5 | `self_test()` driving the real app through `TestClient` |

Also bundled: `day27_live_monitor.py` (the logic being published), `day21b_agent_tools.py`,
`api_config.py`, `w2_comms/flight_data_real.csv`.

---

## Running

```bash
# self-test (no port, no model cost)
D:/Python-envs/chroma-env/Scripts/python.exe day28_live_api.py

# real server, then open http://127.0.0.1:8000/docs
D:/Python-envs/chroma-env/Scripts/python.exe -m uvicorn day28_live_api:app --port 8000

# drive it once the server is up
curl -X POST localhost:8000/live/start -H "Content-Type: application/json" \
     -d '{"source":"synth","speed":5}'
curl localhost:8000/live/status
curl localhost:8000/live/report
curl -X POST localhost:8000/live/stop
```

`speed` is a playback multiplier (`0` = as fast as possible, for tests); with the real
board it should stay at `1`. Output files land in `_out_day28/` (gitignored).

Without an API key everything works except the AI inspection text inside
`inspections_recent`, which reports `origin: "local"` (Day 27's fallback).

## Dependencies

`fastapi`, `uvicorn`, `httpx` (TestClient), plus the Day 27 stack.

---

## Takeaway

> Day 25 exposed an ability as HTTP. Day 27 made that ability run continuously.
> Day 28 is where it becomes a **system**: state that outlives any single request,
> guarded by a lock because the framework itself is multi-threaded, and readable by
> any client that shows up.
>
> Next step (not built here): the Day 26 desktop client polls `/live/status` on a timer and
> renders the flight as it happens — that closes the Week 6–7 loop, board to screen.
