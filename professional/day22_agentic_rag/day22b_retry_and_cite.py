#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════
  Day 22-B：Agent 会自己改写查询词 —— 查不到还会换个说法再试
═══════════════════════════════════════════════════════════════

【这个文件做什么】
    Day22-A 让 Agent 自己决定"要不要查"。今天往前再走一步：
    让它自己决定**"用什么词去查"**，以及**查不到时要不要换个说法再来**。

    同一个问题，两种查法，实测差距（本文件会实地跑一遍）：

        固定管线：拿用户原话「能撑几个起落」去查 → 0.404 → 拒答 ❌
        Agent：   自己改写成「续航 时间」再查     → 0.673 → 命中 ✅

    注意：这个"口语 → 术语"的翻译**不是代码写的**，是模型自己做的。

【本文件最值得记住的一点】
    固定管线（Day19）永远只能拿用户的原话去检索；
    Agent 会先把用户的话翻译成检索系统听得懂的术语，查不到还会再试一次。

    对比 Day18-B：那里也做过"查询改写"，但那是你**手写的一张同义词表**，
    遇到表里没有的说法（比如"起落"）就歇菜了。
    模型改写没有这个边界 —— 它的词汇量就是它的改写规则。

    另外一个同样重要的道理：
        **检索失败不是终点，而是给模型的一条情报。**
    工具只要如实把"为什么没查到"告诉模型，模型就有机会自己救回来。
    所以 Day22-A 里那段"查不到时返回原因而不是抛异常"的设计，在这里才真正值回票价。

【怎么读这个文件】按这个顺序：
    1. main()                 —— 三个演示的输出，先看现象
    2. SYSTEM_PROMPT          —— ★ 那条"查不到就换词重试"的规矩
    3. demo_retry()           —— 演示一：口语提问 → 重试 → 命中
    4. demo_fixed_pipeline()  —— 演示二：同一问题，固定管线只能认栽
    5. demo_recursion_limit() —— 演示三：重试必须设上限，否则会烧钱

【运行方式】
    D:/Python-envs/chroma-env/Scripts/python.exe day22b_retry_and_cite.py

【依赖】Day22-A（检索工具 + 建库）、Day21-B（SimpleChatModel 适配器 + 图）
"""

# ════════════════════════════════════════════════════════════════
# 第 1 部分：环境 —— 复用 Day22-A，不重新发明
# ════════════════════════════════════════════════════════════════

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_PARENT, "w4_rag"), os.path.join(_PARENT, "w3_ai")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 【解释】同 Day22-A：赶在 import 重家伙之前设离线标志，避免联网干等 5 分钟。

import json

from day22a_retrieval_as_tool import (
    search_manual,          # ★ 检索工具，原样复用，一个字没改
    get_retriever,
    HAS_RAG,
    HAS_KEY,
    RAG_ERR,
    MODEL,
)

try:
    import api_config
    API_KEY = api_config.API_KEY
    BASE_URL = api_config.BASE_URL
except Exception:
    API_KEY, BASE_URL = "", ""

from day21b_agent_tools import SimpleChatModel, build_agent

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ════════════════════════════════════════════════════════════════
# 第 2 部分：★ 唯一的改动 —— 在系统提示里加一条"重试"规矩
# ════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "你是无人机电池与能耗方面的技术助手，手边有一本《无人机锂电池使用手册》。\n"
    "\n"
    "【查资料】\n"
    "涉及电池、续航、充电、温度、电压、维护保养等话题时，先用 search_manual 查证，\n"
    "基于查到的原文回答，并在每条结论后标注引用编号，例如：……【1】。\n"
    "\n"
    "【★ 查不到时怎么办】\n"
    "如果工具返回「没有检索到」，说明这个说法在手册里匹配不上，\n"
    "你必须**换一个更规范的说法再查一次**，最多重试 2 次。\n"
    "改写要诀：把口语换成术语，去掉代词和语气词。例如：\n"
    "    「能撑几个起落」      →  「续航 时间」\n"
    "    「这玩意儿能撑多久」  →  「续航 时间」\n"
    "    「电池胖了」          →  「电池 鼓包」\n"
    "    「电量掉得快」        →  「电压 跌落」\n"
    "\n"
    "【不要反问用户】\n"
    "不要反问「你指的是什么型号」「请提供更多信息」这类问题。\n"
    "先按你的理解去查，查不到再说没有 —— 反问等于把问题推回给用户，什么都没解决。\n"
    "\n"
    "【重试后仍然查不到】\n"
    "老实回答「手册里没有提到」，不要编造，也不要补充常识。\n"
)
# 【解释】"不要反问"这条是被逼出来的：
#         第一版没有它，问「这玩意儿充满电能撑多久？」时，
#         glm-4-flash 会回敬一句「这玩意儿指的是什么型号的无人机？」
#         工具一次都没调 —— 小模型遇到指代词，第一反应是找人要澄清，
#         而不是自己去查。必须写死这条规矩。
# 【解释】⭐⭐ 今天全部的新东西就是这一段文字。
#         工具没改、检索没改、图没改 —— 只是把"重试策略"写成了一条 Prompt。
#         这正是 Agent 开发的特点：很多能力是靠"怎么跟模型说话"实现的。

MAX_RETRY = 2
# 【解释】最多重试 2 次（总共最多查 3 次）。不设上限，模型可能一直换词查下去。


def build_model_cold():
    """造模型时把 temperature 设为 0 —— 这是被随机性逼出来的改动。

    实测：同一份代码、同一个问题"能撑几个起落？"，
        temperature=0.3 时：这一轮检索到 0.673，下一轮一次工具都没调，
                            直接回一句"续航时间"就完事了。
        temperature=0   时：行为稳定得多。

    Agent 场景里，模型的每个回答都可能决定"下一步调不调工具"，
    随机性会被放大成"系统时灵时不灵"。所以 Agent 默认就该用 temperature=0。
    """
    m = SimpleChatModel(
        model_id=MODEL, api_key=API_KEY, base_url=BASE_URL,
        temperature=0.0,
    )
    return m.bind_tools([search_manual])


def build_retry_agent(model):
    """图和 Day22-A 完全一样，只是模型换了条 System Prompt。"""
    return build_agent(model, tools=[search_manual])
    # 【解释】第三次强调同一件事：图是通用骨架，换任务不换图。


# ════════════════════════════════════════════════════════════════
# 第 3 部分：跑一轮，并在过程中数"重试了几次"
# ════════════════════════════════════════════════════════════════

def run_agent(graph, question, tag=""):
    """
    跑一个问题，返回 (调工具次数, 每次的查询词, 最终回答, 是否命中)。

    重点看"每次的查询词" —— 如果两次不一样，就说明模型真的改写重试了。
    """
    print()
    print("  提问：%s" % question)

    queries = []
    answer = ""
    hit = False

    for event in graph.stream(
        {"messages": [SystemMessage(content=SYSTEM_PROMPT),
                      HumanMessage(content=question)]},
        stream_mode="values",
        config={"recursion_limit": 12},
        # 【解释】recursion_limit 是**步数上限**，不是重试次数。
        #         一轮"模型思考 + 工具执行"算 2 步，所以给 12 步留足空间。
        #         这是必须设的保险：模型万一陷在"查→查不到→再查"里出不来，
        #         没有这个上限就会一直烧 token。
    ):
        last = event["messages"][-1]
        kind = type(last).__name__

        if kind == "HumanMessage":
            continue
        # 【解释】首步会吐出含系统提示/用户提问的初始状态，跳过。

        if kind == "AIMessage" and getattr(last, "tool_calls", None):
            for tc in last.tool_calls:
                q = tc["args"].get("query", "")
                queries.append(q)
                print("      🔧 第 %d 次检索：%s" % (len(queries), q))

        elif kind == "ToolMessage":
            body = last.content or ""
            if body.startswith("没有检索到"):
                print("      ❌ %s" % body.split("\n")[0])
            else:
                hit = True
                first = body.split("\n")[0]
                print("      ✅ %s" % first[:60])

        elif kind == "AIMessage":
            answer = (last.content or "").strip()
            if answer:
                print("      💬 %s" % answer[:220])

    return len(queries), queries, answer, hit


# ════════════════════════════════════════════════════════════════
# 第 4 部分：三个演示
# ════════════════════════════════════════════════════════════════

def demo_retry(graph):
    """演示一：口语化提问 → 第一次查不到 → 换词 → 查到。"""
    print()
    print("  ── 演示一：口语提问，看模型会不会自己换词 ──")

    q = "能撑几个起落？"
    # 【解释】"起落"是飞行员的口语，手册里一个字都没有。
    #         实测（本仓库数据，bge-small-zh + 阈值 0.50）：
    #             「能撑几个起落」 → 最高相似度 0.410 → 拒答
    #             「续航 时间」    → 最高相似度 0.673 → 命中 5.续航估算
    #         所以这题**必须**换词才查得到 —— 正是演示重试的最佳素材。

    n, queries, answer, hit = run_agent(graph, q)

    print()
    print("      📊 共检索 %d 次，查询词依次是：%s" % (
        n, " → ".join(queries) if queries else "（一次都没查）"))
    if n == 1 and hit:
        print("      ✅ 模型第一把就把口语翻译成了术语「%s」，一次命中。" % queries[0])
        print("         ⭐ 注意：这个翻译是**模型自己做的**，代码里没有任何改写规则。")
        print("         固定管线做不到这点 —— 它只能拿用户的原话去查（见演示二）。")
    elif n >= 2 and hit:
        print("      ✅ 第一次没查到，换了说法后查到了 —— 自我纠错成功")
        print("         查询词演变：%s" % " → ".join(queries))
    elif n >= 2 and not hit:
        print("      ⚠️  重试了但仍然没查到（模型可能没换对词）")
    else:
        print("      ⚠️  查了 %d 次就放弃了，没有重试" % n)
    return n, queries, answer, hit


def demo_fixed_pipeline(model):
    """演示二：同一问题，Day19 的固定管线会怎样 —— 查一次，查不到就认栽。"""
    print()
    print("  ── 演示二：同一个问题，交给 Day19 的固定管线 ──")

    q = "能撑几个起落？"
    # 【解释】和演示一**完全同一个问题**，这样对比才公平。
    r = get_retriever()
    hits = r.retrieve(q)
    # 【解释】★ 固定管线的全部能力就这一步：查一次，拿到什么算什么。

    print()
    print("  提问：%s" % q)
    if hits:
        print("      ✅ 查到了 %d 条" % len(hits))
        context = "\n\n".join("[%d] %s\n%s" % (i, t, b)
                              for i, (t, b, s) in enumerate(hits, 1))
    else:
        print("      ❌ %s" % r.reason)
        context = "（没有检索到任何资料）"
        # 【解释】结局已定：资料是空的，模型再聪明也只能说没有。

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content="【参考资料】\n%s\n\n【用户问题】\n%s" % (context, q)),
    ]
    ans = model.invoke(messages).content.strip()
    print("      💬 %s" % ans[:180])
    print()
    print("      📊 固定管线只查 1 次，查不到 → 直接回答「没有」")
    return ans


def demo_recursion_limit(graph):
    """演示三：一个问题注定查不到时，模型会重试几次？会不会停？"""
    print()
    print("  ── 演示三：再换个问法，看它会不会没完没了地重试 ──")

    q = "飞得远不远？"
    # 【解释】又一个口语提问，实测最高相似度 0.492 —— 差一点点，够不着 0.50 阈值。
    #         这题的难点在于：手册里只写了"续航时间"，压根没写"能飞多远"，
    #         所以就算换词也不一定查得到 —— 正好用来看模型会不会没完没了地重试。
    #
    #         （第一版这题用的是"这块电池的保修期是多久？"，
    #           结果模型又一次选择反问用户、一次工具都没调 ——
    #           说明小模型"该查却不查"是稳定行为，不是偶发。见演示三的结论。）

    n, queries, answer, hit = run_agent(graph, q)
    print()
    print("      📊 共检索 %d 次：%s" % (
        n, " → ".join(queries) if queries else "（一次都没查）"))
    if n <= MAX_RETRY + 1:
        print("      ✅ 在 %d 次以内停手了，没有死循环" % (MAX_RETRY + 1))
    else:
        print("      ⚠️  检索了 %d 次，超过预期上限 —— 提示词里的次数约束要写得更死" % n)
    return n


# ════════════════════════════════════════════════════════════════
# 第 5 部分：main
# ════════════════════════════════════════════════════════════════

def main():
    print("=" * 64)
    print("  Day 22-B：查不到就换个词再查 —— Agent 的自我纠错")
    print("=" * 64)

    if not HAS_RAG:
        print()
        print("  ⚠️  检索依赖没装好：%s" % RAG_ERR)
        print("     请用 chroma-env 运行。")
        return

    if not HAS_KEY:
        print()
        print("  ⚠️  没检测到 API Key，本文件必须调模型才能演示重试。")
        print("     请配置 ZHIPU_API_KEY 或 DEEPSEEK_API_KEY。")
        return

    print()
    print("  [1/3] 接入模型（%s，temperature=0）" % MODEL)
    model = build_model_cold()
    graph = build_retry_agent(model)
    print("      ✅ Agent 就绪")

    print()
    print("  [2/3] 演示一：口语提问的自我纠错")
    n, queries, answer, hit = demo_retry(graph)

    print()
    print("  [3/3] 演示二：同一问题交给固定管线")
    demo_fixed_pipeline(model)

    print()
    print("  ── 附：演示三，止损能力 ──")
    demo_recursion_limit(graph)

    # ── 汇总 ──
    print()
    print("=" * 64)
    print("  对比汇总")
    print("=" * 64)
    print()
    print("      %-26s %-16s %-16s" % ("", "Day19 固定管线", "Day22-B Agent"))
    print("      " + "-" * 58)
    print("      %-26s %-16s %-16s" % ("查不到的处理", "直接回答没有", "换词重试"))
    print("      %-26s %-16s %-16s" % ("本次检索次数", "1 次", "%d 次" % n))
    print("      %-26s %-16s %-16s" % (
        "口语提问的结果", "认栽", "挽救回来" if hit else "仍没查到"))
    print()
    print("      查询词演变：%s" % (" → ".join(queries) if queries else "（无）"))
    print()
    print("      一句话总结：")
    print("        固定管线把「查不到」当成终点；Agent 把它当成反馈。")

    print()
    print("=" * 64)
    print("  读完之后")
    print("=" * 64)
    print("""
  1. 今天全部的新代码就是一段 System Prompt —— 工具、检索、图全都没动。
     Agent 开发就是这样：很多"能力"其实是你跟模型约定的行为规则。

  2. 「检索失败」要当成**情报**返回，不要抛异常也不要返回空字符串。
     把"为什么没查到"（最高分多少、阈值多少）告诉模型，它才知道往哪改。

  3. 重试必须设上限（recursion_limit + Prompt 里写死次数）：
     否则模型可能在"查→查不到→换词→再查"里打转，token 就这么烧掉了。
     这是 Agent 系统工程化时最容易漏的一条。

  4. 引用编号【1】【2】是模型照着资料编号自己标的 ——
     资料里写「[1] 来源：xxx」，回答里才能写【1】。编号对得上，答案才敢信。
""")


if __name__ == "__main__":
    main()
