# DeepSeek API Client Wrapper

A reusable Python client for the DeepSeek chat API. Wraps request sending, streaming output, and automatic retry into a single class so callers get a one-line interface.

## What's inside

- `deepseek_client.py` — the `DeepSeekClient` class with:
  - `chat(messages)` — normal request, returns the full answer;
  - `chat_stream(messages)` — streaming output, yields text piece by piece (typewriter effect);
  - automatic retry with exponential backoff on rate-limit (429) and server errors (5xx);
  - API key read from the `DEEPSEEK_API_KEY` environment variable (never hardcoded).

## Requirements

- Python 3.12
- requests

```bash
pip install requests
```

## Setup

Set the API key as an environment variable (Windows):

```
设置 → 系统 → 关于 → 高级系统设置 → 环境变量 → 新建
名称: DEEPSEEK_API_KEY
值:   sk-your-key
```

Then reopen your terminal.

## Run

```bash
python deepseek_client.py
```

Demonstrates both normal chat and streaming chat.

## Usage

```python
from deepseek_client import DeepSeekClient

client = DeepSeekClient()
messages = [
    {"role": "system", "content": "你是无人机技术专家，回答要简短。"},
    {"role": "user", "content": "无人机电池电压降到多少算危险？"},
]

# normal
answer = client.chat(messages)

# streaming
for piece in client.chat_stream(messages):
    print(piece, end="", flush=True)
```

## Key design

- Streaming uses SSE (`data:`-prefixed lines, terminated by `[DONE]`).
- Retry distinguishes transient server errors (429/5xx) from permanent ones (e.g. 401), and applies exponential backoff (`1s, 2s, 4s`).

## About

Part of my learning series toward building a **UAV energy-consumption prediction AI system**. Full roadmap: see repo root `README.md`.
