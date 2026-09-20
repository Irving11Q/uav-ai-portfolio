# Day 30 — Deploy & Evaluate the W6 Stack (Capstone)

Bring W6 from "it runs on my machine" to "someone else can ship it and prove it works".
Two things separate a demo from a deliverable — and both were gaps I flagged in my own
job-hunt self-review:

1. **Deployability** — a stranger clones the repo and gets the whole service stack up with one command.
2. **Evaluability** — before it goes live, there is a test set that prints a real pass/total score proving "it actually answers correctly".

This day closes W6 by adding both, on top of the services already built:

| Layer | Built in | What it does |
|---|---|---|
| `day25_api_service.py` | Day 25 | data analysis + manual RAG + agent, HTTP service (port 8000) |
| `day28_live_api.py` | Day 28 | real-time inspection, background collector, `/live/*` snapshot (port 8001) |
| `day27_live_monitor.py` | Day 27 | rule engine + endurance prediction (the domain logic, reused verbatim) |
| `day30_eval.py` | **Day 30** | the evaluation harness (this file's companion) |

## What is in this folder

```
day30_deploy/
├── Dockerfile            # one image, two services (compose picks which via command)
├── docker-compose.yml    # `docker compose up` → agent :8000 + live :8001
├── requirements.txt      # pinned to versions verified in the learning env
├── eval_questions.json   # the human-reviewed retrieval golden set
├── day30_eval.py         # evaluation harness (pure-logic / TestClient / remote)
├── day25_api_service.py  # ↓ copied from their own publish dirs, kept self-contained
├── day23a_data_agent.py  #   Day25's data-analysis dependency
├── day27_live_monitor.py  #   Day28's domain logic (evaluated directly, not re-implemented)
├── day28_live_api.py
├── day21b_agent_tools.py #   shared tool definitions
├── api_config.py         #   reads keys from env (no secrets in code)
├── uav_battery_manual.md #   the manual that /manual/search retrieves from
└── w2_comms/flight_data_real.csv
```

Everything is self-contained: copy this folder anywhere with Python + the deps in
`requirements.txt` and it runs. No parent-project imports, no hardcoded paths.

## The four evaluation dimensions

The harness (`day30_eval.py`) covers four dimensions. Crucially, **the prediction and
rule checks call Day27's own `Window.predict_endurance` / `evaluate` — the exact same
code the live service runs**. If I had re-written "expected answers" in the test, the
test and the code could drift apart and the score would mean nothing.

| # | Dimension | How | Source of truth |
|---|---|---|---|
| 1 | Endurance R² gate | clean line → extrapolates with seconds & r²≈1; noisy → `None` (refuses) | `day27.Window.predict_endurance` |
| 2 | Rule alarm | temp 60°C > 55°C → must raise `alarm`; all-normal frame → 0 alerts | `day27.evaluate` |
| 3 | Retrieval recall | ask the manual, assert the correct keyword is in the top-3 hits | `day25 /manual/search` |
| 4 | Service contract smoke | both `/health`=200, `/data/summary` works, live synth predicts `trustworthy` | both services' endpoints |

## How to run

```bash
# A. Full evaluation — spins both services up *in-process* via TestClient (no port needed)
python day30_eval.py

# B. Pure-logic self-test only — zero network, zero model load, for CI
python day30_eval.py --selftest

# C. Against a real deployment (after `docker compose up`)
python day30_eval.py --base-url http://127.0.0.1:8000 http://127.0.0.1:8001

# D. Docker — one command brings up the whole stack
docker compose up --build
docker compose ps            # watch the healthchecks go green
docker compose down
```

## Real results (run in this folder, no Docker needed)

```
Day30 评测（完整 / TestClient 进程内起服务）
  ┌─ 纯逻辑：预测门槛 + 规则告警（4/4 通过）
  │  ✓ 直线下降 seconds=19.3 slope=-0.0517 r²=1.000
  │  ✓ 噪声抖动 seconds=None r²=0.037（拒绝外推）
  │  ✓ 温度60℃ 触发等级=['alarm']
  │  ✓ 正常帧 告警数=0
  ┌─ 检索召回（Day25 /manual/search）（4/4 通过）
  │  ✓ 电池鼓包 → 命中「鼓包」
  │  ✓ 温度超过多少度降落 → 命中「55」
  │  ✓ 低电压报警阈值 → 命中「10.2」
  │  ✓ 多大电流充电 → 命中「1C」「5.2A」
  ┌─ 服务契约冒烟（4/4 通过）
  │  ✓ agent /health → 200
  │  ✓ live /health → 200
  │  ✓ agent /data/summary 电压V → 200
  │  ✓ live synth 预测 trustworthy=True seconds=64.3
  合计：12/12 通过
```

`--selftest` (pure logic) is **4/4**; the full run (A) is **12/12**.

## Honest note on Docker

The build machine in this project has **no Docker installed**, so `docker compose up`
was **not actually executed** here. What was verified:
- `docker-compose.yml` parses as valid YAML (`yaml.safe_load`);
- `Dockerfile` uses the standard `python:3.11-slim` base, `COPY . /app`, and a `CMD` that
  matches what `compose` overrides — syntactically sound, and identical in shape to the
  patterns used by the `shopmind` thesis project in this same repo.

The evaluation **does not depend on Docker** — it uses FastAPI's `TestClient` to bring
both services up in-process, so the 12/12 score above is a real number from this machine,
not a Docker demo. On any host with Docker, `docker compose up --build` is the one-liner
that turns this folder into a running stack.

## Why this day matters (interview talking points)

This is the line between "I can call an API" and "I can ship an AI system":
- **Deployability** — services need a healthcheck, a pinned port, declared deps, and a
  one-command start. That is what `docker-compose.yml` encodes.
- **Evaluability** — RAG recall, agent tool-hit rate, rule false-alarm rate: if you can't
  put a number on it, you can't tell if a change helped. The harness prints that number.
- **Reuse the real logic** — evaluation targets the production code path, never a
  parallel "expected" copy. Drift between test and code is the silent killer of evals.
