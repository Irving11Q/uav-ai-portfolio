# Function Calling: Let the Model Call Your Tools

Day 13 of the "UAV AI Application Development" series. Shows how to make a large language model **drive your own code** through function calling — the foundation of AI Agents. The model decides *when* to call a tool, *which* tool, and *what arguments* to pass; your code executes the function and feeds the result back.

## What's inside

- `day13_function_calling.py` — a UAV flight-decision assistant that:
  - defines two real Python tools: `get_weather(city)` and `estimate_flight_time(battery_capacity_ah, avg_current_a)`;
  - sends a `tools` spec so the model knows what it can call;
  - parses the model's `tool_calls`, executes the matching function, and returns the result with `role="tool"` + `tool_call_id`;
  - runs a 2-round loop (ask → execute tool → ask again for the final answer);
  - reuses `DeepSeekClient` from `day11_api_client/deepseek_client.py` (same pattern as day12).

## Requirements

- Python 3.12
- requests

```bash
pip install requests
```

## Setup

Set your API key as an environment variable (Windows):

```
设置 → 系统 → 关于 → 高级系统设置 → 环境变量 → 新建
名称: DEEPSEEK_API_KEY
值:   sk-your-key
```

Then reopen your terminal.

## Run

From this folder (`professional/day13_function_calling/`):

```bash
python day13_function_calling.py
```

The program asks three questions covering: calling the weather tool, calling the battery tool, and calling both together.

## Key pattern

**Tools spec** — describe each tool so the model can decide to call it:

```python
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询城市天气并判断是否适合飞行",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "城市名"}},
            "required": ["city"],
        },
    },
}]
```

**Two-round loop** — the core of function calling:

```python
# Round 1: send question + tools, read tool_calls from the raw JSON
data = client._request_with_retry({
    "model": client.model, "messages": messages, "tools": TOOLS,
}).json()
msg = data["choices"][0]["message"]
messages.append(msg)

if msg.get("tool_calls"):
    for tc in msg["tool_calls"]:
        result = FUNCTIONS[tc["function"]["name"]](
            **json.loads(tc["function"]["arguments"]))
        messages.append({"role": "tool", "tool_call_id": tc["id"],
                         "content": json.dumps(result, ensure_ascii=False)})

# Round 2: send the tool results back, get the final answer
final = client._request_with_retry({
    "model": client.model, "messages": messages,
}).json()
```

> Note: we call `client._request_with_retry(...).json()` instead of `chat()` because `chat()` only returns the text and would drop the `tool_calls` field we need to see the model's decision.

## About

Part of my learning series toward building a **UAV energy-consumption prediction AI System**. Full roadmap: see repo root `README.md`.
