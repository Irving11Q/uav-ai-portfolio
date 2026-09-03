"""
Day 21-C：给 Agent 加记忆 —— 它怎么记住上一轮说了什么？

【这个文件回答一个问题】
    Day21-B 的 Agent 有个致命问题：**每问一次，它就失忆一次。**
    你问"能飞多久"，它查完电池算出 8.4 分钟；
    你接着问"那换 15A 电流呢"，它完全不记得刚才那块电池，
    只能重新问你参数，或者更糟 —— 自己瞎编一个。

    Checkpointer（存档器）就是解决这个的。

【怎么读这个文件】按这个顺序看：
    1. 先看 main() 的三个演示      —— 用真实输出看清"有记忆 vs 没记忆"
    2. 看 make_model()             —— 直接复用 Day21-B 的模型（跨天复用）
    3. 看 run_turn()               —— 怎么统计"这一轮调了几次工具"
    4. 最后看存档那一节            —— 存档里到底存了什么

【运行方式】
    D:/Python-envs/chroma-env/Scripts/python.exe day21c_agent_memory.py

    没配 API Key 也能跑：会自动只跑对比说明，跳过需要模型的演示。

【依赖】
    复用 day21b_agent_tools.py 的 SimpleChatModel / TOOLS / build_agent。
    这就是"跨天复用"的价值：今天只新增了 checkpointer 这一个概念，
    模型和工具一行都没重写。

【本文件只有 2 个新概念】（符合你一次能消化的量）
    ① Checkpointer：图的"存档器"，每走一个节点就存一次
    ② thread_id：会话编号，靠它区分"这是哪一场对话"
"""

# ════════════════════════════════════════════════════════════════
# 第 1 部分：拿工具（重点看：复用了 Day21-B 的哪些东西）
# ════════════════════════════════════════════════════════════════

import os
import sys
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# 【解释】把自己所在目录加进搜索路径，这样下面 import day21b 一定找得到。
#         从别的目录启动这个脚本时，这一行是保命的。

from day21b_agent_tools import (
    SimpleChatModel,
    # 【解释】Day21-B 写的模型适配器，原样复用（因为没装 langchain_openai）。

    TOOLS,
    # 【解释】两个无人机工具，原样复用。

    build_agent,
    # 【解释】搭图的函数，原样复用 —— 今天只多传一个 checkpointer 参数。

    HAS_KEY, API_KEY, BASE_URL, MODEL,
    # 【解释】配置也复用，不重复读一遍 api_config。
)

from langgraph.checkpoint.memory import MemorySaver
# 【解释】MemorySaver = 最简单的存档器，存在内存里。
#         特点：程序一关就没了，但演示和开发够用，零配置。
#         生产环境要换成 SqliteSaver（存文件）或 PostgresSaver（存数据库），
#         那两个需要额外装包：pip install langgraph-checkpoint-sqlite

from langchain_core.messages import HumanMessage


# ════════════════════════════════════════════════════════════════
# 第 2 部分：模型（一行新代码都没有，全是复用）
# ════════════════════════════════════════════════════════════════

def make_model():
    """造一个带工具的模型 —— 完全复用 Day21-B。"""
    return SimpleChatModel(
        model_id=MODEL,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0.2,
    ).bind_tools(TOOLS)


# ════════════════════════════════════════════════════════════════
# 第 3 部分：跑一轮对话，并统计"调了几次工具"
# ════════════════════════════════════════════════════════════════
# 为什么统计这个？
#     因为"有没有记忆"这件事，光看最终答案看不出来 ——
#     得看它**是不是又去查了一遍电池**。这就是可量化的证据。

EXPECTED_CAPACITY = 3.5
# 【解释】这块电池（3.92 / 3.68 / 3.88，标称 5Ah）压差 0.24V，
#         按 Day21-B 的工具逻辑，可用容量应该是 5.0 × 0.70 = 3.5Ah。
#         它就是我们判断"模型有没有记对"的标准答案。


def run_turn(graph, question, thread_id, show=True):
    """跑一轮对话，返回 (工具调用次数, 最终回答, 用到的容量值)。

    为什么要返回"用到的容量"？
        因为"有没有记忆"最关键的证据不是调了几次工具，
        而是**它填的参数对不对** —— 详见 main() 末尾的汇总表。
    """

    config = {"configurable": {"thread_id": thread_id}}
    # 【解释】⭐ thread_id 是今天的核心：
    #         同一个 thread_id = 同一场对话（能接上之前的内容）
    #         不同的 thread_id = 两个互不相干的会话
    #         它放在 config 里，每次 invoke 都要带上。

    tool_call_count = 0
    final_answer = ""
    used_capacity = None
    # 【解释】记录模型给 estimate_endurance 传的容量值。
    #         这是判断它"记没记住"的关键证据。

    for event in graph.stream(
        {"messages": [HumanMessage(content=question)]},
        config=config,
        # 【解释】config 传进来，LangGraph 才知道去哪个存档里读历史。

        stream_mode="values",
    ):
        last = event["messages"][-1]
        kind = type(last).__name__

        if kind == "AIMessage" and getattr(last, "tool_calls", None):
            tool_call_count += 1
            for tc in last.tool_calls:
                if tc["name"] == "estimate_endurance":
                    used_capacity = tc["args"].get("usable_capacity_ah")
                    # 【解释】抓住这个参数 —— 它等于 3.5 才说明模型真记住了。

                if show:
                    print("      🔧 %s(%s)" % (
                        tc["name"],
                        json.dumps(tc["args"], ensure_ascii=False)))

        elif kind == "AIMessage" and last.content:
            final_answer = last.content.strip()

    if show:
        print("      💬 %s" % final_answer[:120])
        # 【解释】只打前 120 个字，太长会刷屏。

    return tool_call_count, final_answer, used_capacity


# ════════════════════════════════════════════════════════════════
# 第 4 部分：三个演示 —— 用事实说话
# ════════════════════════════════════════════════════════════════

Q1 = "我的电池是 3S，三片电压 3.92 / 3.68 / 3.88，标称 5.0Ah，平均电流 20A，能飞多久？"
# 【解释】第一轮：必须查电池健康度（压差 0.24V → 可用容量 3.5Ah）才能算续航。

Q2 = "那如果换个 15A 的电流呢？"
# 【解释】第二轮：这是个"接续问题" —— 它省略了电池参数，
#         因为人默认对方记得。有记忆的 Agent 应该接得住。


def demo_no_memory():
    """演示一：没有存档器 —— 每轮都失忆。"""
    print()
    print("=" * 64)
    print("  演示一：没有记忆（不传 checkpointer）")
    print("=" * 64)

    model = make_model()
    graph = build_agent(model)
    # 【解释】不传 checkpointer → 默认 None → 无记忆。
    #         和 Day21-B 的行为完全一样。

    print()
    print("  第 1 轮：%s" % Q1)
    c1, _, _ = run_turn(graph, Q1, thread_id="no-mem")

    print()
    print("  第 2 轮：%s" % Q2)
    c2, a2, cap2 = run_turn(graph, Q2, thread_id="no-mem")

    print()
    print("  📊 第 2 轮调了 %d 次工具，传给工具的容量是 %s Ah" % (c2, cap2))
    if cap2 != EXPECTED_CAPACITY:
        print("     ⚠️  正确答案是 %s Ah，它填的是 %s —— 这个数是它编的！" % (
            EXPECTED_CAPACITY, cap2))
        print("        失忆的 Agent 不会说「我不知道」，它会**编一个看起来合理的数**。")
        print("        这才是最危险的地方：答案看起来很专业，其实是错的。")
    else:
        print("     · 恰好填对了（这次运气好），但不代表它记得 ——")
        print("       没有存档机制，纯属巧合。")
    return c1, c2, cap2


def demo_with_memory():
    """演示二：有存档器 —— 同一个 thread_id 能接上。"""
    print()
    print("=" * 64)
    print("  演示二：有记忆（MemorySaver + 同一个 thread_id）")
    print("=" * 64)

    model = make_model()
    graph = build_agent(model, checkpointer=MemorySaver())
    # 【解释】⭐⭐ 就是这里多传一个参数，Agent 就有记忆了。
    #         图的节点、边、工具、模型 —— 全都没变。

    print()
    print("  第 1 轮：%s" % Q1)
    c1, _, _ = run_turn(graph, Q1, thread_id="flight-001")

    print()
    print("  第 2 轮：%s" % Q2)
    c2, a2, cap2 = run_turn(graph, Q2, thread_id="flight-001")
    # 【解释】⭐ 同一个 thread_id → 第二轮能读到第一轮的存档。

    print()
    print("  📊 第 1 轮调 %d 次工具，第 2 轮调 %d 次，容量填的是 %s Ah" % (
        c1, c2, cap2))
    if cap2 == EXPECTED_CAPACITY:
        print("     ✅ 完全正确！它从存档里读到了上一轮的 3.5Ah，")
        print("        不需要你重复说一遍电池参数。")
    else:
        print("     ⚠️  填的是 %s（正确答案 %s），这次没接上 —— 模型偶尔会算错，" % (
            cap2, EXPECTED_CAPACITY))
        print("       多跑几次看看，或者把 temperature 调更低。")
    return c1, c2, cap2


def demo_new_thread():
    """演示三：换个 thread_id —— 立刻变成陌生人。"""
    print()
    print("=" * 64)
    print("  演示三：换个 thread_id（同一块电池，但换个会话编号）")
    print("=" * 64)

    model = make_model()
    graph = build_agent(model, checkpointer=MemorySaver())

    print()
    print("  第 1 轮（thread=flight-002）：%s" % Q1)
    c1, _, _ = run_turn(graph, Q1, thread_id="flight-002")

    print()
    print("  第 2 轮（thread=flight-999，换号了）：%s" % Q2)
    c2, _, cap2 = run_turn(graph, Q2, thread_id="flight-999")
    # 【解释】⭐ 换了 thread_id = 全新的会话，完全不认识上一轮。

    print()
    print("  📊 换号后第 2 轮调了 %d 次工具，容量填的是 %s Ah" % (c2, cap2))
    print("     ⚠️  虽然存档器还在，但 thread_id 换了 → 读到的是空存档。")
    print("        👉 thread_id 才是「哪一串记忆」的钥匙，checkpointer 只是仓库。")
    return c1, c2, cap2


def demo_peek_state():
    """演示四：存档里到底存了什么。"""
    print()
    print("=" * 64)
    print("  演示四：打开存档看看里面是什么")
    print("=" * 64)

    model = make_model()
    graph = build_agent(model, checkpointer=MemorySaver())

    run_turn(graph, Q1, thread_id="peek-demo", show=False)
    # 【解释】先跑一轮，制造出存档。show=False 表示不打印过程。

    state = graph.get_state({"configurable": {"thread_id": "peek-demo"}})
    # 【解释】get_state = 把这个 thread 当前的完整状态取出来看。

    print()
    print("  存档里的消息条数：%d" % len(state.values["messages"]))
    print()
    print("  每一条是什么：")
    for i, m in enumerate(state.values["messages"], 1):
        extra = ""
        if getattr(m, "tool_calls", None):
            extra = "（携带 %d 个工具调用请求）" % len(m.tool_calls)
        print("    %d. %-14s %s%s" % (
            i,
            type(m).__name__,
            (m.content or "")[:34],
            extra))
    # 【解释】type(m).__name__ 会显示 HumanMessage / AIMessage / ToolMessage。
    #         你会看到完整的对话链条 —— 这就是"记忆"的实体。

    print()
    print("  ⭐ 记忆不是什么玄学，就是**这一串消息**。")
    print("     下次 invoke，LangGraph 把这串消息原样塞回 state，")
    print("     模型自然就「想起来」了。")


# ════════════════════════════════════════════════════════════════
# 第 5 部分：main()
# ════════════════════════════════════════════════════════════════

def main():
    print()
    print("╔" + "═" * 62 + "╗")
    print("║" + " " * 14 + "Day 21-C：给 Agent 加记忆" + " " * 21 + "║")
    print("╚" + "=" * 62 + "╝")

    print()
    print("  今天只有 2 个新概念：")
    print("    ① Checkpointer = 图的存档器（仓库）")
    print("    ② thread_id   = 会话编号（钥匙）")
    print()
    print("  加记忆的代码改动只有一行：")
    print("      graph = build_agent(model)                      # 无记忆")
    print("      graph = build_agent(model, MemorySaver())       # 有记忆")

    if not HAS_KEY:
        print()
        print("  ⚠️  没读到 API Key，下面需要模型的演示会跳过。")
        print("     但上面那行对比已经说明了全部原理。")
        return

    print()
    print("  （每次调用真实模型，三个演示大约需要 30~60 秒）")

    c_no = demo_no_memory()
    c_yes = demo_with_memory()
    c_new = demo_new_thread()
    demo_peek_state()

    # ── 汇总对比表 ──
    print()
    print("=" * 66)
    print("  ⭐ 关键对比：第 2 轮问的都是「那如果换个 15A 的电流呢？」")
    print("     一句省略了全部电池参数的话。正确答案的容量是 %s Ah。" % EXPECTED_CAPACITY)
    print("=" * 66)
    print()
    print("      %-22s %-12s %-10s" % ("场景", "第2轮填的容量", "判定"))
    print("      " + "-" * 48)

    rows = [
        ("无记忆", c_no[2]),
        ("有记忆（同 thread）", c_yes[2]),
        ("有记忆（换 thread）", c_new[2]),
    ]
    for name, cap in rows:
        if cap == EXPECTED_CAPACITY:
            mark = "✅ 正确"
        elif cap is None:
            mark = "— 没调这个工具"
        else:
            mark = "❌ 编造（差 %+.1f%%）" % (
                (cap - EXPECTED_CAPACITY) / EXPECTED_CAPACITY * 100)
        print("      %-22s %-12s %-10s" % (name, cap, mark))

    print()
    print("  💡 注意：三行都只调了 1 次工具 —— 差别不在调用次数，")
    print("     而在**参数对不对**。没记忆时它不会说「我不知道」，")
    print("     而是编一个像模像样的数（实测编了 4.76，算出 15.2 分钟，")
    print("     比正确答案 11.2 分钟高出 36%）。")
    print()
    print("     👉 这就是 Agent 记忆最实在的价值：不是少问一句，是**少编一个数**。")

    print()
    print("=" * 64)
    print("  【读完之后】Day21-C 你学到了什么")
    print("=" * 64)
    print("""
    ✅ 已掌握
       · 加记忆 = compile(checkpointer=MemorySaver())，图的结构一行都不用改
       · thread_id 决定"读哪一份存档"，同一串对话必须用同一个号
       · 记忆的实体就是消息列表 —— 不是向量，不是数据库，就是那串消息
       · get_state() 可以随时把存档取出来看，调试非常好用

    ⚠️  还要留意的坑
       · MemorySaver 存内存，程序一关就丢。生产要换 SqliteSaver / PostgresSaver
         （需要 pip install langgraph-checkpoint-sqlite，这个环境没装）
       · 消息会越攒越长 → token 越来越贵 → 长对话要做裁剪或摘要
       · thread_id 取错 = 串到别人的会话，多用户系统里要严格隔离
       · 存档会保存工具结果，如果工具返回敏感数据，要注意存储安全

    💡 面试怎么讲（三句话）
       1. Agent 的记忆落地就是 checkpointer，我在 compile 时多传一个参数就开了
       2. thread_id 是会话钥匙，同一个号才能接上上下文，多用户靠它隔离
       3. MemorySaver 是内存版（开发用），生产我了解要换 SqliteSaver / PostgresSaver，
         另外长对话要做消息裁剪，否则 token 成本会线性上涨

    ➡️  下一步
       第 5 周收官。接下来两条路：
       · 把 Day21 发布到作品集（Day1–21 全齐）
       · 或者直接进第 6-7 周的综合项目：无人机能耗预测 AI 系统
    """)


if __name__ == "__main__":
    main()
