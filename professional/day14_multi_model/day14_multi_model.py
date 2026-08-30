"""
Day 14: 多模型对比 —— DeepSeek / 通义千问 / 智谱 GLM 怎么选

同一道题问三个模型，对比回答、耗时和 Token 消耗，最后输出选型对比表。

核心知识点：所有主流大模型 API 都是 OpenAI 兼容格式，
换模型只需要改三个东西：base_url（接口地址）+ api_key（密钥）+ model（模型名）。

运行：python day14_multi_model.py
配置：哪个模型配了 key 就测哪个，没配的自动跳过（并提示怎么申请）
"""

import os
import time

import requests

# 环境变量：读系统里配的 key，没配就是空字符串
# 注意：这里故意不复用统一配置，三个模型各自读自己的 key，
# 避免"标签写 DeepSeek、实际调 GLM"导致对比失真。

# 三个模型的"身份证"：名字 + 接口地址 + 密钥 + 模型名 + 备注
# base_url 和 model 都是各平台官方文档给的，照抄即可。
MODELS = [
    {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "model": "deepseek-chat",
        "note": "便宜、中文好、有深度推理版(reasoner)",
    },
    {
        "name": "通义千问 (Qwen)",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),
        "model": "qwen-turbo",
        # 申请：阿里云百炼平台 bailian.aliyun.com → 开通百炼 → 拿 API-KEY
        "note": "生态全（阿里系）、有开源模型、价格低",
    },
    {
        "name": "智谱 GLM",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_key": os.environ.get("ZHIPU_API_KEY", ""),
        "model": "glm-4-flash",
        # 申请：open.bigmodel.cn 注册 → API 密钥 → 复制
        "note": "有免费额度(glm-4-flash)、轻量好上手",
    },
]


def gen_response(cfg, question):
    """用 cfg 里的配置问一个问题，返回 (回答文字, 耗时秒, Token数)。

    cfg 是 MODELS 里的某一个字典；换任何模型都走这一个函数。
    """
    url = f"{cfg['base_url']}/chat/completions"

    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": cfg["model"],
        "messages": [{"role": "user", "content": question}],
        "temperature": 0.3,
        "max_tokens": 256,  # 限制长度，测试更公平也更快
    }

    t0 = time.time()
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    elapsed = time.time() - t0

    if resp.status_code != 200:
        # 常见失败原因：key 没配、配错、余额不足
        msg = resp.json().get("error", {}).get("message", "")
        return f"[请求失败 {resp.status_code}] {msg}", elapsed, 0

    data = resp.json()
    answer = data["choices"][0]["message"]["content"]
    tokens = data.get("usage", {}).get("total_tokens", 0)
    return answer, elapsed, tokens


def main():
    # 测试题：和后续的综合项目（无人机能耗预测）直接相关
    question = "请用一句话介绍：无人机能量消耗预测系统是做什么的？"

    print("=" * 60)
    print("Day 14：多模型对比实验")
    print(f"测试问题：{question}")
    print("=" * 60)

    results = []

    for cfg in MODELS:
        print(f"\n> {cfg['name']}（{cfg['model']}）")

        if not cfg["api_key"]:
            # 这个模型没配 key，跳过但保留占位，保证对比表整齐
            print("   跳过：未配置 API Key。")
            print(f"   （{cfg['note']}）")
            results.append((cfg["name"], "未配置", "-", "-"))
            continue

        answer, elapsed, tokens = gen_response(cfg, question)
        print(f"   耗时 {elapsed:.1f}s | Token {tokens}")
        print(f"   回答：{answer[:120]}")
        results.append((cfg["name"], f"{elapsed:.1f}s", tokens, "OK"))

    # 打印选型对比表（:<16 表示左对齐占 16 字符宽，表格才整齐）
    print("\n" + "=" * 60)
    print("选型对比表")
    print("=" * 60)
    print(f"{'模型':<16}{'耗时':<8}{'Token':<8}{'状态'}")
    for name, elapsed, tokens, status in results:
        print(f"{name:<16}{elapsed:<8}{str(tokens):<8}{status}")

    print("""
怎么选（4 个维度）：
   1. 个人学习/小项目  -> 最便宜：deepseek-chat / glm-4-flash / qwen-turbo
   2. 对话类产品       -> 速度快的（首字延迟低）
   3. 复杂推理/代码    -> 效果强的（deepseek-reasoner / qwen-plus / glm-4-air）
   4. 都要看           -> 上下文长度、是否支持 Function Calling（见 day13）
   结论：先用便宜的起步，够用、中文好，等项目跑通再按瓶颈升级。
""")


if __name__ == "__main__":
    main()
