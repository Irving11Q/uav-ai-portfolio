# Day 29 — The Dashboard (a PySide6 client that polls `/live/*`)

**Week 6 (system integration), day 5 — and the day the loop closes.**

Day 28 turned the inspector into a service, but a service is just JSON — nobody can see it.
Today it gets a dashboard. Open the window, and once a second it asks `/live/status` and
renders the flight as it happens: voltage, current, temperature, altitude, window statistics,
endurance prediction and the alert timeline.

| | |
|---|---|
| Day 27 | rules + prediction (the domain logic) |
| Day 28 | the same logic, running continuously inside a service |
| **Day 29** | **the screen — board to screen, closed** |

---

## What is new compared with Day 26

Day 26's client fired one request per button press. A dashboard is a different animal:

| | Day 26 | Day 29 |
|---|---|---|
| What drives the request | the user clicking | a `QTimer` |
| How many requests | one | ~1/second, forever |
| History | not needed | the client must accumulate it itself |
| New failure mode | — | **overlapping requests** |

Three things had to be added, and each of them is the interesting part of the day.

### 1. A timer, and an in-flight guard

The timer fires every second. If a request takes two seconds to come back, a naive
implementation fires another one anyway — and then another. The queue grows, the UI shows
stale data, and the eventual burst of replies makes it thrash.

```python
def _tick(self):
    if self.poll_worker is not None:      # ★ still in flight → skip this beat
        return
    self.poll_worker = PollWorker()
    self.poll_worker.got.connect(self.on_poll)
    self.poll_worker.start()
```

Three lines. In production this is called *in-flight guarding* and it is the number one
timer bug people write.

### 2. The server gives you *now*; history is the client's problem

`/live/status` answers "what is the value right now". It does not answer "what did the last
minute look like". To draw a curve, the client has to copy the current value into its own
ring buffer on every poll. `SampleBuffer` does exactly that with
`deque(maxlen=CURVE_POINTS)` — bounded memory, no leak after a long flight.

This is a **structural cost of polling**, not an implementation detail. Moving to SSE or
WebSocket removes the *missing frames*, not the buffering.

### 3. Polling loses frames — so the UI shows the loss

The collector runs at `speed` frames per second; the client asks once per second. At
`speed=5` the client sees one frame in five. Rather than hide that, the panel displays both
numbers side by side:

```
服务端帧数   10
客户端采样点 6
采样比       1.67x
```

That ratio is the honest measurement of what a polling dashboard can and cannot show.
For a human watching gauges it is fine. For anything that must not miss a frame, it is the
signal to switch protocols — and being able to point at the number is worth more in an
interview than reaching for WebSocket on day one.

---

## The four ideas worth explaining

### 1. Logic that can be tested must not live inside a window class

Testing GUIs is hard for a boring reason: the window will not open — no display, no
windowing system in CI, no Qt in a headless image. So the split here is deliberate:

| File | Contents | Needs Qt? |
|---|---|---|
| `day29a_live_client.py` (471 lines) | service manager, HTTP, all formatting, sample buffer, **the self-test** | **no** |
| `day29b_dashboard.py` (473 lines) | widgets, threads, timer, wiring | yes |

The payoff is not prettiness, it is **permission to change things**: after any edit to the
client's logic, one command tells you whether you broke it. That command is the self-test,
and it never opens a window.

### 2. De-duplicate alerts with a monotonic counter, not with content

The same message can be emitted twice; `alert_log` length cannot go down. So the client
tracks `alerts_total` from the server and appends only what it has not seen.

The subtlety: `/live/status` returns only the last 5 alerts. If one polling interval
produces more than five, the client physically cannot show them all. It says so instead of
pretending:

```
… 另有 4 条未显示（一拍里涨太多，见 /live/alerts）
```

A dashboard that silently drops data is worse than one that admits it.

### 3. Keep the business judgement out of the GUI

There is not a single `if voltage < 23` in `day29b_dashboard.py`. Every decision —
is the lamp green, is this prediction trustworthy, why not — comes from `fmt_*` functions
in `day29a`. A rough way to check whether a GUI is properly layered: *grep the window class
for business numbers*. If they are there, the layering leaked.

### 4. Reuse the curve object; never re-create it

```python
self.curve.setData(xs, ys)      # ★ correct
# self.plot.plot(xs, ys, clear=True)   # ✗ wipes the threshold line and the marker too
```

`clear=True` removes *every* item in the plot, including the safety-threshold line and the
prediction marker that were added once at start-up. The symptom is "the curve is there but
the red line vanished after the first update" — easy to miss, easy to write.

---

## Self-test output (19/19, no window, no model calls)

```
✅ fmt_frame 四列
✅ fmt_prediction 拒绝噪声外推            R²=0.030
✅ fmt_stats 含均值
✅ fmt_prediction 可信时给秒数
✅ fmt_lamp 运行中给绿灯
✅ SampleBuffer 受 maxlen 约束
✅ new_alerts 增量取 3 条
✅ new_alerts 超 5 条如实报遗漏
✅ GET /health
✅ POST /live/start(replay)
✅ 轮询 6 拍都拿到快照
✅ 服务端帧数在增长                       3 → 10
✅ 告警被增量读到（真实数据>55℃）           5 条
✅ 客户端采样点 < 服务端帧数（漏帧证据）       采样 6 / 服务端 10
✅ fmt_counters 给出采样比
✅ 续航预测给出明确结论                    拒绝外推：R²=0.450 低于门槛 0.50
✅ 预测触发（synth，R² 过门槛）             R²=1.000 → 71 秒
✅ 停止后 running=False
✅ 服务不可达 → offline
自测完成：19 / 19 项通过
```

Two kinds of checks in one run:

- **pure-function checks** — hand-written snapshots fed straight into `fmt_*`. No service
  needed, so they also pin down the *refusal* cases (a noisy window must produce
  "拒绝外推", not a number).
- **contract checks** — a real uvicorn child process, real HTTP, real collection. This is
  what proves client and server still agree.

The GUI path is verified separately, offscreen:

```
QT_QPA_PLATFORM=offscreen python -c "... win._apply(fake_snapshot) ..."
→ label text, prediction text, alert lines, curve points and marker position all checked
```

---

## Lessons this day surfaced

### A single file hit 799 lines

The first draft was one file. The project rule is ≤600 lines per teaching file, and it was
already 199 over. Splitting it was not mechanical — the natural seam turned out to be
*"everything that does not need Qt"*, which is exactly the seam that makes the day testable.
The line limit forced a better design.

### `str(data).__contains__` is not a state machine

Deciding "did the collection start?" by testing `"start" in str(response)` works by
accident and breaks the moment a field name changes. The window now records
`self._ctrl_is_start` at request time. Signal/slot APIs carry fixed arguments, so the
intent has to be stored somewhere explicit.

### Deterministic test setups produce deterministic numbers

The synthetic trajectory is a perfect straight line, so its `R²` is exactly 1.000 and the
prediction fires every time. The real replay data is noisy — in this run its window
scored `R² = 0.450`, just under the 0.50 gate, and the panel correctly refused to
extrapolate. Both behaviours are the *desired* ones; a test that only ever exercises the
straight line would never catch a broken confidence gate.

---

## Files

### `day29a_live_client.py` (471 lines) — the core, no Qt

| Block | What it holds |
|---|---|
| 1 | environment probe |
| 2 | `ServiceManager` — spawn uvicorn, poll `/health` until ready, stop |
| 3 | `call_api()` — one exit point, errors classified into four levels |
| 4 | `fmt_lamp` / `fmt_frame` / `fmt_stats` / `fmt_prediction` / `fmt_counters` / `new_alerts` |
| 5 | `SampleBuffer` — the client-side ring buffer |
| 6 | `self_test()` — 19 checks |

### `day29b_dashboard.py` (473 lines) — the window

| Block | What it holds |
|---|---|
| 1 | layout: service bar, run bar, top row (numbers │ curve), bottom row (alerts │ log) |
| 2 | `BootWorker`, `PollWorker`, `CtrlWorker` — every blocking call leaves the main thread |
| 3 | `_tick()` — the timer callback with the in-flight guard |
| 4 | `_apply()` — spread one snapshot across the widgets |

Bundled for self-containment: `day28_live_api.py`, `day27_live_monitor.py`,
`day21b_agent_tools.py`, `api_config.py`, `w2_comms/flight_data_real.csv`.

---

## Running

```bash
# self-test, no window (this is what CI would run)
D:/Python-envs/chroma-env/Scripts/python.exe day29a_live_client.py
D:/Python-envs/chroma-env/Scripts/python.exe day29b_dashboard.py --selftest

# open the dashboard: click 启动服务 → pick a source → 开始巡检
D:/Python-envs/chroma-env/Scripts/python.exe day29b_dashboard.py

# just the service, then drive it by hand
D:/Python-envs/chroma-env/Scripts/python.exe day29a_live_client.py --serve
curl localhost:8124/live/status
```

The panel starts the Day 28 service itself on port 8124 (Day 26 uses 8123) so all three
programs can run at once. `DAY29_BASE` points the client at an already-running server.

| Source | Frames | Try it with | What it demonstrates |
|---|---|---|---|
| `replay` | 23, real capture | speed 1 | the alert path (temperature > 55 ℃) and a *refused* prediction |
| `synth` | 60, synthetic straight line | speed 5 | the prediction path (R² = 1.000, threshold crossed in ~71 s) |
| `real` | the STM32 board | speed 1 | the same code on live hardware at 115200 baud |

## Dependencies

`PySide6`, `pyqtgraph`, `requests`, plus the Day 28 stack (`fastapi`, `uvicorn`, `httpx`).

---

## Takeaway

> Day 26 proved a desktop client could *ask* a service a question.
> Day 29 is a client that *watches* it — driven by a clock instead of a click,
> measuring what its own polling cannot see, and structured so that everything
> except the pixels can be tested without opening a window.
>
> Week 6 is closed: a real serial stream goes in one end, and a live dashboard comes out
> the other, with the domain logic written once (Day 27), published once (Day 28),
> and observed here.
