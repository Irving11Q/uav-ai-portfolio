# Day 27 — Live Flight-Data Monitoring (serial → rules → prediction → AI)

**Week 6 (system integration), day 3.** Days 23–26 all worked on a *static* CSV:
Day 23 read the whole table, Day 25/26 answered when a human clicked a button.
Real flight data is not a table — it arrives **one frame at a time**, and the operator
needs an answer *now*: is the temperature over the line? how many seconds of flight are
left?

This day turns the Week-2 serial reader and the Week-5 agent into a **live inspector**:

```
frame → rule engine → sliding window → endurance prediction → (throttled) AI summary → report
        ↑ real-time, free   ↑ fixed length    ↑ linear extrapolation   ↑ every 10 frames
```

---

## What changed, in one table

| | Day 23 (batch) | Day 27 (streaming) |
|---|---|---|
| Input | whole CSV at once | one frame per second, forever |
| Latency requirement | "eventually" | "now" — safety line crossed = alarm |
| Who detects a problem | you ask, agent queries | rules fire on the frame that crosses |
| Model calls | one per question | **throttled to 1 per 10 frames** |
| Memory | grows with table | **fixed** (window is a bounded `deque`) |
| Output | printed answer | per-frame `.jsonl` + Markdown report |

**Three data sources, one interface** (`readline()`)

| Source | Flag | Use |
|---|---|---|
| `ReplaySerial` | `--source=replay` | replays the Week-2 real capture (23 frames) — no hardware |
| `SynthSerial` | `--source=synth` | generates a falling-voltage track, to exercise the prediction |
| `RealSerial` | `--source=real` | actual board (Yangtao-1 STM32, 115200), falls back to replay if absent |

---

## The three ideas that matter

### 1. Two time scales — rules for real time, the model for insight

The interview question this day answers is: *"your sensor runs at 1 Hz — do you call the
LLM once per second?"* No:

- **Rules** decide alarms. Comparison against a threshold is arithmetic — microseconds,
  free, works offline, and never hallucinates. **Safety alarms must not depend on a
  network round-trip.**
- **The LLM** writes the human-readable summary. Useful, but slow and metered — so it is
  **throttled to once every 10 frames**, and it is fed *features* (min/mean/max/slope),
  not the raw 30 frames. That keeps both latency and token cost down.

### 2. Extrapolation without a confidence measure is fortune-telling

The endurance predictor fits a least-squares line to the recent voltage and solves
`(current − threshold) / drop_rate` for "seconds until the safety line".

The first version reported **"about 84 seconds until 23 V"** on the *real* capture — a
false alarm. The measurement says why:

| Data | Slope | R² |
|---|---|---|
| Real capture (23 frames) | −0.0166 V/s | **0.030** ← pure noise |
| Synthetic falling track | −0.0300 V/s | **1.000** ← a real trend |

So the predictor now requires `R² ≥ 0.50` before it will extrapolate at all. On the real
data the report says *"no significant downward trend (R²=0.030), endurance alarm **not**
triggered"*, and on the synthetic track it says *"about 14 s until 23.0 V"*. A negative
result stated honestly beats a confident wrong one.

### 3. Single-point differencing turns noise into alarms

The voltage-drop rule was first written as `(previous − current) / dt`. On real data a
0.6 V jitter between consecutive frames was instantly judged a "0.6 V/s plunge" —
**10 false alarms in 23 frames**.

Switching the rule to the **window slope** (the same fitted trend the predictor uses)
fixed it: the real data's trend is a normal ≈ −0.013 V/s discharge, so nothing fires,
and the 23-frame run reports exactly the **4 real alarms** (temperature > 55 °C), matching
ground truth. Fitting looks at the *trend*; differencing amplifies the *noise*.

---

## Self-test output (no model calls, ~1 s, 2/2)

```
【自测 1/2】replay the Week-2 real capture (23 frames) — rule engine
  采集 23 帧 ｜ 告警 4 条     ← 温度 > 55℃ 真值就是 4 条
【自测 2/2】synthetic falling-voltage track (60 frames) — prediction
  巡检报告 → 结论：按当前速率约 **14 秒**后跌破 23.0V，建议提前返航
  自测结果：✅ 规则引擎  ✅ 续航预测   通过 2 / 2
```

Measured with the model enabled (`--source=synth --frames=22`, `glm-4-flash`):

```
[ 20] 电压 24.63V  电流 1.88A  温度 43.4C  高度 105m  ✅
── 🔍 第 20 帧巡检（ai）──
   整体状态：注意。电压即将跌破安全值，约54秒后跌破23.0V。建议检查电源系统，确保电压稳定。
```

Note the summary above needed one prompt fix: the first version returned only
*"状态：注意。电压略低，建议检查电源连接"* — it silently dropped the 54-second
prediction, which is the single most actionable number on the screen. If a metric is not
named in the prompt, the model will not mention it.

---

## Two more things this day surfaced

### Logical time vs wall-clock time

Replaying 60 frames takes ~0.05 s of wall clock, so a slope computed from real
timestamps came out as **−120 V/s** — meaningless. Frame time is therefore
`frame_index × sampling_period` (1 Hz). Any rate-of-change or extrapolation work needs a
correct `dt` first; getting it from the clock only works for genuinely real-time input.

### Reports use the *final* window, so read them as a snapshot

The Markdown report summarises the last 30 frames, not the whole run — the alarm
timeline covers the run, the statistics cover the window. Keeping those two scopes
distinct is what makes a long-running report readable.

---

## Files

### `day27_live_monitor.py` (600 lines) — the inspector

| Block | What it holds |
|---|---|
| 1 | thresholds (sourced from the Day 22 manual + Week-2 data distribution) and `evaluate()` |
| 2 | `ReplaySerial` / `SynthSerial` / `RealSerial` — one `readline()` interface |
| 3 | `Window`: bounded `deque`, stats, `volt_slope()`, `fit_r2()`, `predict_endurance()`, `energy_used_ah()` |
| 4 | throttled AI inspection + local fallback when no key is configured |
| 5 | per-frame `.jsonl` + Markdown report writer |
| 6 | frame loop, argument parsing, self-test |

---

## Running

```bash
# replay the real capture (no hardware, no cost)
python day27_live_monitor.py --source=replay

# synthetic falling-voltage track — watch the prediction fire
python day27_live_monitor.py --source=synth

# the physical board (STM32, 115200), falls back to replay if unavailable
python day27_live_monitor.py --source=real

# self-test, no model calls (release / CI)
python day27_live_monitor.py --selftest
```

Options: `--frames=N` to stop after N frames, `--no-ai` to skip model calls entirely.
Output goes to `_out_day27/` (gitignored): `frames_<src>_<ts>.jsonl` + `巡检报告_<src>_<ts>.md`.

Without an API key the rule engine, prediction, and report all still run; only the AI
summary degrades to a locally assembled paragraph (labelled `local` instead of `ai`).

## Dependencies

`pyserial` (real-serial mode only), `langchain-core` + `langgraph` for the
`SimpleChatModel` adapter. Bundled for self-containment: `day21b_agent_tools.py`
(the OpenAI-compatible `BaseChatModel`), `api_config.py`, and
`w2_comms/flight_data_real.csv` (the Week-2 capture, 23 frames).

---

## Takeaway

> Day 23 analysed data *after* the flight. Day 27 watches it *during* the flight.
> The interesting part is not the model — it is deciding what must **not** go through
> the model: alarms are arithmetic, and any number you extrapolate has to come with a
> reason to believe it.
>
> Next step (not built here): publish the report as a `/report` endpoint on the Day 25
> service so the Day 26 desktop client can poll it while a flight is running — the last
> seam in the Week 6–7 integrated system.
