"""
Day 12: Prompt Engineering

演示 4 个控制大模型输出的技巧：
1. system prompt - 给模型定人设
2. few-shot - 给例子让模型照着格式回答
3. JSON 输出 - 让模型返回结构化数据
4. temperature - 控制模型输出的发散程度

场景：无人机飞行数据解读（电压,电流,温度,高度）
"""

import json
from day11_api_wrapper import DeepSeekClient


client = DeepSeekClient()

FLIGHT_DATA = "24.6,1.2,38.5,120"  # 电压24.6V, 电流1.2A, 温度38.5°C, 高度120m


def demo_system_prompt():
    """对比没有 system prompt 和有 system prompt 的回答差异"""
    question = f"请解读这行飞行数据：{FLIGHT_DATA}"

    print("【实验 1】system prompt：没定人设 vs 定了人设\n")

    # 不加 system
    messages_plain = [{"role": "user", "content": question}]
    print(f"> 提示词：{question}")
    answer_plain = client.chat(messages_plain, temperature=0.3)
    print(f"回答：{answer_plain}\n")

    # 加 system 定人设
    messages_expert = [
        {"role": "system", "content": (
            "你是拥有 10 年经验的无人机飞行数据分析专家。"
            "回答必须：1) 先用一句话说结论；2) 再分条说明依据；"
            "3) 最后给一条具体建议。语气专业、简洁。"
        )},
        {"role": "user", "content": question},
    ]
    print(f"> 提示词（+system）：你是无人机数据分析专家...")
    answer_expert = client.chat(messages_expert, temperature=0.3)
    print(f"回答：{answer_expert}\n")

    print("对比：加了 system 后，回答更有结构、更专业\n")


def demo_few_shot():
    """对比不给例子和给例子的输出格式差异"""
    print("=" * 50)
    print("【实验 2】few-shot：给例子让模型照格式回答")
    print("=" * 50)

    question = f"数据：{FLIGHT_DATA}。请判断这条数据是否异常，并说明理由。"

    # 不给例子
    answer_no_shot = client.chat([{"role": "user", "content": question}], temperature=0.3)
    print(f"不给例子的回答：{answer_no_shot}\n")

    # 给 2 个例子
    messages_few_shot = [
        {"role": "system", "content": "你是无人机数据质检员。"},
        # 例子 1
        {"role": "user", "content": "数据：23.1,1.5,42.0,95。请判断是否异常。"},
        {"role": "assistant", "content": "正常。电压 23.1V 高于 22V 警戒线，温度 42°C 在正常范围。"},
        # 例子 2
        {"role": "user", "content": "数据：21.8,3.2,58.0,140。请判断是否异常。"},
        {"role": "assistant", "content": "异常。电压 21.8V 低于 22V 警戒线，电流 3.2A 超 2.5A 上限。"},
        # 真正的问题
        {"role": "user", "content": question},
    ]
    answer_few_shot = client.chat(messages_few_shot, temperature=0.3)
    print(f"给 2 个例子后的回答：{answer_few_shot}\n")

    print("对比：给例子后，回答格式更统一\n")


def demo_json_output():
    """让模型输出 JSON，程序直接解析"""
    print("=" * 50)
    print("【实验 3】JSON 输出：让模型返回结构化数据")
    print("=" * 50)

    # 不要求格式
    answer_free = client.chat(
        [{"role": "user", "content": f"分析这行飞行数据：{FLIGHT_DATA}，给出风险判断。"}],
        temperature=0.3,
    )
    print(f"不要求格式：{answer_free}\n")

    # 要求输出 JSON
    messages_json = [
        {"role": "system", "content": (
            "你是飞行数据解析器。你只输出 JSON，不输出任何其他文字。"
            "JSON 格式固定如下：\n"
            '{"电压": 24.6, "电流": 1.2, "温度": 38.5, "高度": 120, '
            '"风险等级": "低", "建议": "一句话建议"}'
        )},
        {"role": "user", "content": f"数据：{FLIGHT_DATA}"},
    ]
    answer_json = client.chat(messages_json, temperature=0.3)
    print(f"要求输出 JSON：{answer_json}\n")

    # 解析 JSON
    try:
        parsed = json.loads(answer_json)
        print(f"解析成功：电压={parsed['电压']}V, 风险={parsed['风险等级']}, 建议={parsed['建议']}\n")
    except json.JSONDecodeError:
        print("解析失败，模型可能夹带了多余文字\n")


def demo_temperature():
    """对比 temperature=0 和 =1.5 的差异"""
    print("=" * 50)
    print("【实验 4】temperature：稳定 vs 发散")
    print("=" * 50)

    question = "用一句话夸夸无人机飞手这份职业。"

    print("temperature = 0（稳定）：")
    for i in range(2):
        answer = client.chat([{"role": "user", "content": question}], temperature=0)
        print(f"  第{i+1}次：{answer}")

    print("\ntemperature = 1.5（发散）：")
    for i in range(2):
        answer = client.chat([{"role": "user", "content": question}], temperature=1.5)
        print(f"  第{i+1}次：{answer}")

    print("\ntemperature 选法：")
    print("  0.0~0.3 = 事实/结构化任务（要稳）")
    print("  0.5~0.8 = 日常对话（平衡）")
    print("  1.0~1.5 = 创意任务（要浪）\n")


def main():
    print("Day 12: Prompt Engineering 实验\n")
    demo_system_prompt()
    demo_few_shot()
    demo_json_output()
    demo_temperature()
    print("=" * 50)
    print("完成！你学会了 system、few-shot、JSON、temperature 4 个技巧")


if __name__ == "__main__":
    main()