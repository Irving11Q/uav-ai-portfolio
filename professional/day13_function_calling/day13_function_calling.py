"""
Day 13：Function Calling —— 让模型「会调用工具」

演示如何让大模型指挥你的代码去干活：
- 定义两个真实工具函数：查天气 get_weather、估算续航 estimate_flight_time
- 把工具「说明书」通过 tools 参数告诉模型
- 模型决定调哪个工具、传什么参数 → 你的代码执行 → 结果回传 → 模型综合回答
场景：无人机飞行前决策助手

依赖：需要配置 DEEPSEEK_API_KEY 环境变量，或复制 day11_api_client/deepseek_client.py 到同目录。
"""

import json
import sys
from pathlib import Path

# 动态导入 deepseek_client（支持从上级目录或同目录导入），和 day12 保持一致
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from day11_api_client.deepseek_client import DeepSeekClient
except ImportError:
    try:
        from deepseek_client import DeepSeekClient
    except ImportError:
        print("错误：找不到 deepseek_client.py")
        print("请将 day11_api_client/deepseek_client.py 复制到本目录")
        sys.exit(1)


# ── 工具函数：这些是真正会执行的 Python 函数 ──
# 真实项目里可换成查数据库、发 HTTP 请求、控制硬件等。

def get_weather(city):
    """查询城市天气（模拟）。返回是否适合无人机飞行。"""
    weather_data = {
        "北京": {"天气": "晴", "风力": 3, "温度": 26, "适合飞行": True},
        "上海": {"天气": "小雨", "风力": 5, "温度": 24, "适合飞行": False},
        "深圳": {"天气": "多云", "风力": 4, "温度": 28, "适合飞行": True},
    }
    return weather_data.get(city, {"天气": "未知", "风力": 0, "温度": 0, "适合飞行": False})


def estimate_flight_time(battery_capacity_ah, avg_current_a):
    """根据电池容量(Ah)和平均电流(A)，估算续航时间(分钟)。"""
    SAFETY_FACTOR = 0.8  # 留 20% 余量，电池不能放空
    minutes = (battery_capacity_ah / avg_current_a) * 60 * SAFETY_FACTOR
    return {"续航时间分钟": round(minutes, 1), "建议": "留足返航余量，不要飞满"}


# TOOLS：工具的「说明书」，告诉模型有什么工具可用。
# 模型看完这份说明，才能正确地说「我要调 get_weather」。
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的实时天气、风力，判断是否适合无人机飞行",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，如：北京"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_flight_time",
            "description": "根据电池容量和平均电流，估算无人机续航时间（分钟）",
            "parameters": {
                "type": "object",
                "properties": {
                    "battery_capacity_ah": {
                        "type": "number",
                        "description": "电池容量，单位安时(Ah)。例如 5000mAh 要填 5.0",
                    },
                    "avg_current_a": {
                        "type": "number",
                        "description": "平均工作电流，单位安培(A)",
                    },
                },
                "required": ["battery_capacity_ah", "avg_current_a"],
            },
        },
    },
]

# 工具名 → 真实函数的映射，方便按模型返回的名字执行
FUNCTIONS = {
    "get_weather": get_weather,
    "estimate_flight_time": estimate_flight_time,
}


def ask_with_tools(user_question):
    """核心函数：发问题 → 模型要调工具就执行 → 回传 → 拿最终回答。"""
    client = DeepSeekClient()

    messages = [
        {"role": "system", "content": "你是无人机飞行助手。需要实时信息时，调用提供的工具获取。"},
        {"role": "user", "content": user_question},
    ]

    # 第 1 轮：把问题 + 工具说明书一起发给模型。
    # 注意用 _request_with_retry + .json() 而非 chat()，因为 chat() 只返回文字，
    # 会丢掉模型「想调工具」的 tool_calls 信息。
    body = {
        "model": client.model,
        "messages": messages,
        "tools": TOOLS,
        "temperature": 0.3,
    }
    data = client._request_with_retry(body).json()
    msg = data["choices"][0]["message"]
    messages.append(msg)  # 把模型这条消息（含其「想法」）存入对话历史

    # 模型没要调工具：直接返回文字回答（例如纯聊天问题）
    if not msg.get("tool_calls"):
        return msg.get("content", "（模型没有回答）")

    # 模型要调工具！逐个执行真实函数
    print("  🔧 模型决定调用工具：")
    for tool_call in msg["tool_calls"]:
        func_name = tool_call["function"]["name"]
        args = json.loads(tool_call["function"]["arguments"])
        print(f"    → {func_name}({args})")

        result = FUNCTIONS[func_name](**args)  # 模型指挥你的代码干活
        print(f"      工具返回：{result}")

        # 把工具执行结果以 role="tool" 回传，tool_call_id 用于配对
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": json.dumps(result, ensure_ascii=False),
        })

    # 第 2 轮：把工具结果发回模型，让它综合回答
    print("  🤖 模型综合工具结果，给出最终回答：\n")
    final_data = client._request_with_retry({
        "model": client.model,
        "messages": messages,
        "temperature": 0.3,
    }).json()
    return final_data["choices"][0]["message"]["content"]


def main():
    print("=" * 50)
    print("Day 13：Function Calling —— 无人机飞行决策助手")
    print("=" * 50)

    questions = [
        "今天北京天气适合飞无人机吗？",
        "我的电池是 5000mAh，平均电流 20A，能飞多久？",
        "我明天想去深圳飞无人机，电池 8000mAh 平均电流 16A，帮我看看行不行？",
    ]

    for i, question in enumerate(questions, start=1):
        print(f"\n{'─' * 50}")
        print(f"【问题 {i}】{question}")
        print(f"{'─' * 50}")
        answer = ask_with_tools(question)
        print(f"最终回答：{answer}\n")

    print("✅ Day 13 完成！模型会自己决定「什么时候调工具、调哪个」了。")


if __name__ == "__main__":
    main()
