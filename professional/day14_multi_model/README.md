# Multi-Model Comparison: DeepSeek vs Qwen vs GLM

Day 14 of the "UAV AI Application Development" series. Asks the same question to three different LLM providers, compares their answers, latency and token usage, and prints a selection table.

The point of this day: **switching model providers is a 3-line change** — `base_url` + `api_key` + `model` — because every mainstream provider speaks the same OpenAI-compatible format.

## What's inside

- `day14_multi_model.py` — a provider-agnostic benchmark that:
  - declares three model configs (DeepSeek, Qwen, GLM) as plain dicts;
  - runs one shared `gen_response(cfg, question)` function against each of them;
  - measures wall-clock latency and `total_tokens` per call;
  - skips providers whose API key is missing instead of crashing;
  - prints a comparison table plus a 4-dimension selection guide.

Self-contained — no dependency on `api_config` or any other day.

## Requirements

- Python 3.12
- requests

```bash
pip install requests
```

## Setup

Each provider reads its **own** environment variable. Configure whichever you have; the rest are skipped.

| Provider | Environment variable | Where to get it |
|---|---|---|
| DeepSeek | `DEEPSEEK_API_KEY` | platform.deepseek.com → API Keys |
| Qwen (通义千问) | `DASHSCOPE_API_KEY` | bailian.aliyun.com → API-KEY |
| GLM (智谱) | `ZHIPU_API_KEY` | open.bigmodel.cn → API 密钥 |

On Windows:

```
设置 → 系统 → 关于 → 高级系统设置 → 环境变量 → 新建
名称: DEEPSEEK_API_KEY
值:   sk-your-key
```

Then reopen your terminal.

## Run

From this folder (`professional/day14_multi_model/`):

```bash
python day14_multi_model.py
```

## Key pattern

**One config dict per provider** — this is all that separates them:

```python
MODELS = [
    {"name": "DeepSeek",
     "base_url": "https://api.deepseek.com/v1",
     "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
     "model": "deepseek-chat"},
    {"name": "通义千问 (Qwen)",
     "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
     "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),
     "model": "qwen-turbo"},
    {"name": "智谱 GLM",
     "base_url": "https://open.bigmodel.cn/api/paas/v4",
     "api_key": os.environ.get("ZHIPU_API_KEY", ""),
     "model": "glm-4-flash"},
]
```

**One shared call function** — the request body is identical for all three:

```python
url = f"{cfg['base_url']}/chat/completions"
payload = {
    "model": cfg["model"],
    "messages": [{"role": "user", "content": question}],
    "temperature": 0.3,
    "max_tokens": 256,
}
resp = requests.post(url, headers=headers, json=payload, timeout=60)
```

> Note: this file deliberately does **not** reuse a single global config, so that each row in the comparison table really is the provider it claims to be.

## How to choose a model

| Scenario | Pick |
|---|---|
| Learning / side project | cheapest one — `deepseek-chat`, `glm-4-flash`, `qwen-turbo` |
| Conversational product | lowest first-token latency |
| Hard reasoning / code | strongest one — `deepseek-reasoner`, `qwen-plus`, `glm-4-air` |
| Always check | context window, function-calling support (see day13) |

## About

Part of my learning series toward building a **UAV energy-consumption prediction AI System**. Full roadmap: see repo root `README.md`.
