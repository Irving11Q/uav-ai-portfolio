"""
DeepSeek API 客户端封装。

将 DeepSeek 对话接口封装为可复用的类，支持：
- chat(): 普通问答，一次请求返回完整回答
- chat_stream(): 流式问答，逐段返回生成内容
- 自动重试：遇到限流(429)或服务器错误(5xx)时指数退避重试

API Key 从环境变量 DEEPSEEK_API_KEY 读取。
"""

import json
import os
import time

import requests

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
BASE_URL = "https://api.deepseek.com/v1"
MODEL_NAME = "deepseek-chat"

_RETRYABLE_STATUS = (429, 500, 502, 503, 504)


class DeepSeekClient:
    """DeepSeek 对话接口的轻量封装。"""

    def __init__(self, api_key: str | None = None,
                 base_url: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or API_KEY
        self.base_url = base_url or BASE_URL
        self.model = model or MODEL_NAME

    def chat(self, messages: list, temperature: float = 0.7,
             max_tokens: int = 2048, **extra) -> str:
        """发送一次对话，返回完整回答文字。"""
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **extra,
        }
        data = self._request_with_retry(body).json()
        return data["choices"][0]["message"]["content"]

    def chat_stream(self, messages: list, temperature: float = 0.7, **extra):
        """流式对话：逐段返回生成内容。"""
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            **extra,
        }
        resp = self._request_with_retry(body, stream=True)

        for line in resp.iter_lines():
            if not line:
                continue
            text = line.decode("utf-8")
            if not text.startswith("data:"):
                continue
            data_str = text[len("data:"):].strip()
            if data_str == "[DONE]":
                break
            chunk = json.loads(data_str)
            delta = chunk["choices"][0]["delta"].get("content")
            if delta:
                yield delta

    def _request_with_retry(self, body: dict, stream: bool = False,
                            max_retries: int = 3) -> requests.Response:
        """发送请求，失败时指数退避重试。"""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(max_retries):
            try:
                resp = requests.post(url, headers=headers, json=body,
                                     stream=stream, timeout=120)
                if resp.status_code in _RETRYABLE_STATUS:
                    wait = 2 ** attempt
                    print(f"状态码 {resp.status_code}，{wait} 秒后重试"
                          f"（第 {attempt + 1}/{max_retries} 次）")
                    time.sleep(wait)
                    continue
                return resp
            except requests.exceptions.RequestException as e:
                print(f"网络异常：{e}（第 {attempt + 1}/{max_retries} 次）")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        raise RuntimeError("请求多次重试仍然失败，请检查网络或稍后再试")


def main() -> None:
    if not API_KEY:
        print("未找到环境变量 DEEPSEEK_API_KEY，请先配置后重试。")
        return

    client = DeepSeekClient()
    messages = [
        {"role": "system", "content": "你是无人机技术专家，回答要简短。"},
        {"role": "user", "content": "无人机电池电压降到多少算危险？"},
    ]

    print("=" * 50)
    print("演示 1：普通问答")
    print("=" * 50)
    answer = client.chat(messages)
    print(f"AI：{answer}\n")

    print("=" * 50)
    print("演示 2：流式问答（打字效果）")
    print("=" * 50)
    print("AI：", end="", flush=True)
    for piece in client.chat_stream(messages):
        print(piece, end="", flush=True)
    print("\n\n完成。")


if __name__ == "__main__":
    main()
