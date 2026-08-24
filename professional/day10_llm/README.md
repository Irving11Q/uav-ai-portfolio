# DeepSeek LLM API Command-Line Chat

First LLM API demo: a command-line chat with the DeepSeek model. Type a question, get a model response, and see the token usage for each call. Shows the core pattern of calling a large-language-model API from Python.

## What's inside

- `first_llm_call.py` — a command-line Q&A loop that:
  - sends a request to DeepSeek's `/chat/completions` endpoint via `requests.post`;
  - keeps the conversation history so the model remembers previous turns;
  - prints token usage after each answer;
  - reads the API key from the environment variable `DEEPSEEK_API_KEY` (never hardcoded).

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
python first_llm_call.py
```

Type a question and press Enter; type `exit` to quit.

## Key pattern

```python
resp = requests.post(f"{BASE_URL}/chat/completions",
                     headers={"Authorization": f"Bearer {API_KEY}", ...},
                     json={"model": ..., "messages": [...], ...})
reply = resp.json()["choices"][0]["message"]["content"]
```

This is the standard OpenAI-compatible chat API shape, shared by DeepSeek, Qwen (Tongyi), GLM and others.

## About

Part of my learning series toward building a **UAV energy-consumption prediction AI system**. Full roadmap: see repo root `README.md`.
