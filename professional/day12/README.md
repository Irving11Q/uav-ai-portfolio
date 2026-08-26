# Prompt Engineering: System, Few-Shot, JSON, Temperature

Four hands-on experiments showing how to control LLM output through prompt design. Covers the core techniques used in real-world AI applications: setting a system persona, giving examples (few-shot), requesting structured JSON output, and tuning temperature.

## What's inside

- `day12_prompt_engineering.py` — four side-by-side experiments:
  - **System prompt**: compare "no persona" vs "expert persona" to see how the output structure changes
  - **Few-shot**: give the model 2 examples so it follows a consistent format
  - **JSON output**: request structured JSON that your code can parse directly
  - **Temperature**: compare `0` (stable/deterministic) vs `1.5` (divergent/creative)

## Requirements

- Python 3.12
- requests

```bash
pip install requests
```

## Setup

Set your API key as an environment variable (default uses GLM, can switch to DeepSeek):

```
设置 → 系统 → 关于 → 高级系统设置 → 环境变量 → 新建
名称: ZHIPU_API_KEY  (or DEEPSEEK_API_KEY)
值:   your-key-here
```

Edit `day11_api_wrapper.py` to switch between providers if needed.

## Run

```bash
python day12_prompt_engineering.py
```

The program runs all four experiments sequentially, printing both the prompt and the model response for comparison.

## Key techniques

**System prompt** — Set the model's persona and response rules:

```python
messages = [
    {"role": "system", "content": "You are a UAV data analyst with 10 years experience. Answer concisely: 1) conclusion, 2) reasoning, 3) suggestion."},
    {"role": "user", "content": "Analyze: 24.6,1.2,38.5,120"},
]
```

**Few-shot** — Show 2-3 examples, the model imitates the format:

```python
messages = [
    {"role": "user", "content": "Data: 23.1,1.5,42.0,95. Normal or abnormal?"},
    {"role": "assistant", "content": "Normal. Voltage 23.1V above 22V threshold."},
    {"role": "user", "content": question},
]
```

**JSON output** — Request structured data for programmatic use:

```python
messages = [
    {"role": "system", "content": "Output only JSON: {\"voltage\": 24.6, \"risk\": \"low\", \"suggestion\": \"...\"}"},
    {"role": "user", "content": "Data: 24.6,1.2,38.5,120"},
]
data = json.loads(client.chat(messages))
```

**Temperature** — Control randomness:

| Value | Use case |
|---|---|
| 0.0-0.3 | Facts, code, structured output (want stable) |
| 0.5-0.8 | Daily conversation, Q&A (balanced) |
| 1.0-1.5 | Creative writing, naming (want variety) |

## About

Part of my learning series toward building a **UAV energy-consumption prediction AI system**. Full roadmap: see repo root `README.md`.