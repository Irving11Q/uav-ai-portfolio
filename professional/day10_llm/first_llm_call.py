"""
DeepSeek 大模型 API 命令行问答。

向 DeepSeek 的对话接口发送请求，实现命令行交互问答：
输入问题，模型回答，并显示本次调用的 Token 用量。

API Key 从环境变量 DEEPSEEK_API_KEY 读取（不写死在代码里）。

运行前设置环境变量：
    Windows:  设置 → 系统 → 关于 → 高级系统设置 → 环境变量
              → 新建 DEEPSEEK_API_KEY = sk-你的key
    然后重新打开终端。

模型选择：deepseek-chat（通用对话）/ deepseek-reasoner（推理）。
"""

import json
import os

import requests

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
BASE_URL = "https://api.deepseek.com/v1"
MODEL_NAME = "deepseek-chat"


def call_deepseek(user_text: str, history: list | None = None) -> tuple:
    """发送一次对话请求，返回 (回答文字, 用量字典)；失败返回 (None, None)。"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    if history is None:
        history = []
    messages = history + [{"role": "user", "content": user_text}]

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1024,
    }

    resp = requests.post(
        f"{BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    data = resp.json()

    if resp.status_code != 200:
        print(f"请求失败，状态码 {resp.status_code}")
        print(f"错误信息：{data}")
        return None, None

    reply = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return reply, usage


def main() -> None:
    if not API_KEY:
        print("未找到环境变量 DEEPSEEK_API_KEY，请先配置后重试。")
        return

    print("=" * 50)
    print("DeepSeek 命令行问答")
    print("=" * 50)
    print("输入你的问题，按回车；输入 exit 退出。\n")

    history: list = []

    while True:
        try:
            user_text = input("你：").strip()
        except EOFError:
            print("\n再见。")
            break

        if user_text.lower() in ("exit", "quit"):
            print("再见。")
            break
        if not user_text:
            continue

        print("AI 思考中...")
        reply, usage = call_deepseek(user_text, history)

        if reply is None:
            print("请求失败，请检查上方错误信息后重试。\n")
            continue

        print(f"AI：{reply}\n")

        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": reply})

        if usage:
            print(f"（Token：提示 {usage.get('prompt_tokens', '?')} + "
                  f"回答 {usage.get('completion_tokens', '?')} = "
                  f"总计 {usage.get('total_tokens', '?')}）\n")


if __name__ == "__main__":
    main()
