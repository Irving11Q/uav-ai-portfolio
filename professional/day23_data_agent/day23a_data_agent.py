#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════
  Day 23-A：数据分析 Agent —— 让模型「算」，不是让模型「背」
═══════════════════════════════════════════════════════════════

【程序做什么】
  数据是你第 2 周用串口真实采集的飞行日志（23 行：时间/电压/电流/温度/高度）。
  现在可以用人话问它：
      「这次飞行电压最低掉到多少？」
      「温度超过 55 度的有几次？都什么时候？」
      「电流和电压是正相关吗？」
  模型不自己算 —— 它调工具去 pandas 里算，再把数字组织成人话。

【怎么读这个文件（按顺序看这 5 块）】
  1. load_df()          数据从哪来（你 W2 的真实 CSV，23 行）
  2. 四个 @tool         ★ 今天的主角：把「会算数」包成工具
  3. demo_naive()       对照组：把整张表塞进 Prompt 会怎样
  4. demo_agent()       实验组：模型调工具算
  5. build_data_agent() 图一行没变，只换了工具（框架的价值）

【运行方式】
  cd w5_agent
  D:/Python-envs/chroma-env/Scripts/python.exe day23a_data_agent.py

  没配 API Key 也能跑：会演示 4 个工具本身有效，只跳过「模型说话」那一环。

【为什么学这个（面试能讲）】
  这是 Text2SQL / 数据分析 Agent 的最小骨架，AI 应用岗高频考点。
  核心一句话：
      LLM 算数不可靠（会数错、会编），
      但「判断该算什么」非常擅长。
  所以分工是：模型负责决定问什么，代码负责把数算准。

【和 Day22 的区别（重要）】
  Day22 的工具是「检索」：返回一段文本，模型从中摘答案。
  Day23 的工具是「计算」：返回一个数字，这个数字本身就是事实。
  👉 工具返回数字时，幻觉空间被压缩到接近零 —— 因为数不是模型编的。
"""

# ════════════════════════════════════════════════════════════════
# 第 1 部分：环境
# ════════════════════════════════════════════════════════════════

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
_W3 = os.path.join(_PARENT, "w3_ai")
sys.path.insert(0, _HERE)
sys.path.insert(0, _W3)

# 【解释】Key 从 w3_ai/api_config.py 读，不写死在代码里（防 GitHub 泄露）。
# ⚠️ 注意用 import 模块再取属性，别用 from ... import MODEL：
#    api_config 里这个变量叫 MODEL_NAME，不叫 MODEL，
#    写成 from api_config import MODEL 会抛 ImportError，
#    整个 try 一起失败 —— 结果是有 Key 也被当成没配，静默降级。
try:
    import api_config
    API_KEY = api_config.API_KEY
    BASE_URL = api_config.BASE_URL
    MODEL = api_config.MODEL_NAME
    HAS_KEY = bool(API_KEY)
except Exception:
    API_KEY, BASE_URL, MODEL = "", "", "glm-4-flash"
    HAS_KEY = False

import pandas as pd
from langchain_core.tools import tool

try:
    from day21b_agent_tools import SimpleChatModel, build_agent
    HAS_AGENT = True
except Exception as e:                       # 没装 langgraph 时降级
    HAS_AGENT = False
    AGENT_ERR = e

# ════════════════════════════════════════════════════════════════
# 第 2 部分：数据（你 W2 串口采的真实日志）
# ════════════════════════════════════════════════════════════════

# 【解释】数据文件可能在两个地方，按优先级依次找：
#   ① 本文件旁边（发布包自带，保证拷出去能跑）
#   ② 上级目录的 w2_comms（本地学习仓库的原始布局）
# 为什么要两个都支持：GitHub 上每个 day 目录是自包含的，
# 我会把 CSV 拷进 day23_data_agent/w2_comms/，而在本地它本来就在 src/w2_comms/。
# 【坑】别写成 _PARENT/w2_comms 一条路：发布目录里 _PARENT 指向 professional/，会找不到。
_CSV_CANDIDATES = [
    os.path.join(_HERE, "w2_comms", "flight_data_real.csv"),
    os.path.join(_PARENT, "w2_comms", "flight_data_real.csv"),
]
CSV_PATH = next((p for p in _CSV_CANDIDATES if os.path.exists(p)), _CSV_CANDIDATES[0])
_df = None


def load_df() -> pd.DataFrame:
    """惰性加载，只加载一次。"""
    global _df
    if _df is None:
        # 【解释】utf-8-sig：W2 串口存 CSV 时带了 BOM，不加 sig 第一列名会变成 ﻿时间
        _df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    return _df


def _resolve_col(name: str):
    """把模型说的列名模糊匹配到真实列名。

    【解释】这是个很实用却常被忽略的工程细节：模型很可能说「电压」而不是「电压V」。
    不做容错的话，工具直接报 KeyError，模型还以为这列不存在，就开始瞎答了。
    找不到时把「有哪些列」告诉它 —— 等于给模型一次自我纠正的机会。
    """
    df = load_df()
    if name in df.columns:
        return name
    for c in df.columns:
        if name in c or c.replace("V", "").replace("A", "").replace("C", "").replace("m", "") == name:
            return c
    for c in df.columns:
        if name and name[0] == c[0]:
            return c
    return None


# ════════════════════════════════════════════════════════════════
# 第 3 部分：★ 四个工具 —— 把「会算数」包成工具
# ════════════════════════════════════════════════════════════════

@tool
def peek_data(n: int = 5) -> str:
    """看飞行日志的前几行，用来确认有哪些列、数据长什么样。

    Args:
        n: 看几行，默认 5
    """
    df = load_df()
    return "共 %d 行，列：%s\n%s" % (
        len(df), "、".join(df.columns), df.head(n).to_string(index=False)
    )


@tool
def describe_column(col: str) -> str:
    """算某一列的统计值：个数、均值、标准差、最小、25%、中位数、75%、最大。

    问「最高/最低/平均/波动多大」时用这个。

    Args:
        col: 列名，可以写「电压V」也可以写「电压」
    """
    real = _resolve_col(col)
    if real is None:
        return "没有这一列。可用列：%s" % "、".join(load_df().columns)
    s = load_df()[real]
    return ("%s：共 %d 个，均值 %.2f，标准差 %.2f，最小 %.2f，"
            "25%% %.2f，中位数 %.2f，75%% %.2f，最大 %.2f") % (
        real, s.count(), s.mean(), s.std(), s.min(),
        s.quantile(.25), s.median(), s.quantile(.75), s.max(),
    )


@tool
def filter_rows(col: str, op: str, value: float) -> str:
    """按条件筛出符合的行，比如「温度 > 55」「电压 < 23.5」。

    问「有多少次 / 有哪些时刻」时用这个。

    Args:
        col: 列名，如「温度C」
        op: 比较符号，只能是 > >= < <= == != 之一
        value: 阈值
    """
    real = _resolve_col(col)
    if real is None:
        return "没有这一列。可用列：%s" % "、".join(load_df().columns)

    ops = {">": "大于", ">=": "大于等于", "<": "小于",
           "<=": "小于等于", "==": "等于", "!=": "不等于"}
    if op not in ops:
        return "op 只能是 %s" % "、".join(ops.keys())

    df = load_df()
    hit = df[eval("df[real] %s value" % op)]     # 【解释】op 已白名单校验过，安全
    if len(hit) == 0:
        return "%s %s %.2f 的行：0 条（一次都没有）" % (real, ops[op], value)

    head = hit[["时间", real]].head(10).to_string(index=False)
    return "%s %s %.2f 的行：共 %d 条\n%s" % (real, ops[op], value, len(hit), head)


@tool
def correlate(col1: str, col2: str) -> str:
    """算两列的相关系数，判断它们是不是一起涨一起跌。

    问「A 和 B 有关系吗 / 是正相关吗」时用这个。

    Args:
        col1: 第一列，如「电流A」
        col2: 第二列，如「电压V」
    """
    r1, r2 = _resolve_col(col1), _resolve_col(col2)
    if r1 is None or r2 is None:
        return "列名不对。可用列：%s" % "、".join(load_df().columns)

    r = load_df()[r1].corr(load_df()[r2])
    if abs(r) >= 0.7:
        degree = "强"
    elif abs(r) >= 0.4:
        degree = "中等"
    elif abs(r) >= 0.2:
        degree = "弱"
    else:
        degree = "几乎没有"
    direction = "正" if r > 0 else "负"
    return "%s 与 %s 的相关系数 = %.3f（%s%s相关）" % (r1, r2, r, degree, direction)


TOOLS = [peek_data, describe_column, filter_rows, correlate]

SYSTEM_PROMPT = (
    "你是一个无人机飞行日志分析助手。\n"
    "规矩：\n"
    "1. 所有数字必须来自工具返回值，一个都不许自己算或凭印象说。\n"
    "2. 想回答跟数据有关的问题，先调工具；不调工具不许给数字。\n"
    "3. 工具返回「0 条」「没有」时，就如实说没有，不要补充常识。\n"
    "4. 回答要短，先给结论再给数字。"
)

# ════════════════════════════════════════════════════════════════
# 第 4 部分：对照组 —— 把整张表塞进 Prompt 会怎样
# ════════════════════════════════════════════════════════════════

def demo_naive(model):
    """对照组：不给工具，把 23 行原表全塞进 Prompt，让模型自己数。"""
    df = load_df()
    csv_text = df.to_csv(index=False)

    q = "电压低于 23.5 伏的记录有几条？"
    truth = int((df["电压V"] < 23.5).sum())

    print("  ── 对照组：把整张表塞进 Prompt，不给工具 ──")
    print()
    print("  问：%s" % q)
    print("  （正确答案：%d 条 —— 这是 pandas 算的，不是模型算的）" % truth)
    print()

    if model is None:
        print("  ⚠️ 没配 Key，跳过模型环节。上面那张表就是模型看到的所有信息，")
        print("     你可以自己数数看：23 行里电压 < 23.5 的到底有几条。")
        print("     👉 感受一下：让你用眼睛数 23 行，和让代码数，哪个可靠。")
        return

    prompt = ("下面是无人机飞行日志：\n%s\n\n请回答：%s\n只回答数字和一句解释。" % (csv_text, q))
    from langchain_core.messages import HumanMessage, SystemMessage
    resp = model.invoke([SystemMessage(content="你是数据分析助手。"),
                         HumanMessage(content=prompt)])
    print("  模型答：%s" % (resp.content or "").strip())
    print("  正确答案：%d 条" % truth)


# ════════════════════════════════════════════════════════════════
# 第 5 部分：实验组 —— 模型调工具
# ════════════════════════════════════════════════════════════════

def build_data_agent(model):
    """图和 Day21-B / Day22 完全一样，只是工具换成了「会算数」的那套。"""
    return build_agent(model, tools=TOOLS)
    # 【解释】★ 这行是今天最值钱的一行：
    #         Day21 工具=电池体检，Day22 工具=手册检索，Day23 工具=数据分析，
    #         三天的图一行都没改。变的是「模型手上有什么能力」，不是流程。


def run_question(graph, q):
    """问一句，把模型调了什么工具、工具返回了什么、最终怎么答的都打出来。"""
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

    print()
    print("  " + "─" * 58)
    print("  问：%s" % q)

    init = {"messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=q)]}
    seen = set()
    final = ""

    for event in graph.stream(init, stream_mode="values"):
        for m in event["messages"]:
            if id(m) in seen:
                continue
            seen.add(id(m))
            if isinstance(m, ToolMessage):
                print("      📊 工具返回：%s" % str(m.content).replace("\n", " ")[:110])
            elif hasattr(m, "tool_calls") and getattr(m, "tool_calls", None):
                for c in m.tool_calls:
                    print("      🔧 调工具：%s(%s)" % (c["name"], c["args"]))
            elif m.type == "ai" and (m.content or "").strip():
                final = m.content.strip()
                print("      💬 %s" % final)

    return final


def demo_agent(graph):
    """实验组：三个问题，全是自然语言。"""
    print()
    print("  ── 实验组：给模型四个工具，让它自己决定调哪个 ──")
    run_question(graph, "这次飞行电压最低掉到多少？")
    run_question(graph, "温度超过 55 度的有多少次？都大概在什么时候？")
    run_question(graph, "电流和电压是正相关吗？")


# ════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════

def show_truth():
    """先把「正确答案」摆出来，方便你核对模型有没有瞎说。"""
    df = load_df()
    print("  数据：%s（%d 行）" % (os.path.basename(CSV_PATH), len(df)))
    print("  列  ：%s" % "、".join(df.columns))
    print()
    print("  pandas 算出来的真值（用来核对模型答案对不对）：")
    print("    电压V   最小 %.2f  最大 %.2f  均值 %.2f"
          % (df["电压V"].min(), df["电压V"].max(), df["电压V"].mean()))
    print("    温度C   >55 度共 %d 次，最高 %.1f"
          % (int((df["温度C"] > 55).sum()), df["温度C"].max()))
    print("    电流A   最小 %.2f  最大 %.2f  均值 %.2f"
          % (df["电流A"].min(), df["电流A"].max(), df["电流A"].mean()))
    print("    电流A vs 电压V 相关系数 %.3f" % df["电流A"].corr(df["电压V"]))


def main():
    print("=" * 64)
    print("  Day 23-A：数据分析 Agent（自然语言 → 调工具 → 数字）")
    print("=" * 64)
    print()

    if not os.path.exists(CSV_PATH):
        # 【解释】这里必须用非 0 退出码，不能 return。
        #  return 会让脚本「看起来跑完了」（退出码 0），实际一个演示都没执行，
        #  别人看一眼退出码就以为成功了 —— 这种静默失败比直接崩掉更坑。
        print("找不到数据文件，试过这两个位置：")
        for p in _CSV_CANDIDATES:
            print("   -", p)
        print()
        print("把 flight_data_real.csv 放到本目录下的 w2_comms/ 里再跑。")
        sys.exit(1)

    show_truth()
    print()

    # 没 Key 也先把工具演示一遍（工具本身不依赖模型）
    if not HAS_KEY or not HAS_AGENT:
        print("  ⚠️ 没配 API Key 或没装 langgraph，只演示工具本身：")
        print()
        print("     peek_data(3)        →")
        for line in peek_data.invoke({"n": 3}).split("\n"):
            print("       " + line)
        print("     describe_column(电压) →")
        print("       " + describe_column.invoke({"col": "电压"}))
        print("     filter_rows(温度C,>,55) →")
        print("       " + filter_rows.invoke({"col": "温度C", "op": ">", "value": 55}).replace("\n", "\n       "))
        print("     correlate(电流A,电压V) →")
        print("       " + correlate.invoke({"col1": "电流A", "col2": "电压V"}))
        print()
        print("  👉 注意：这四个工具的返回值全是「算出来的数」，")
        print("     不是让模型从文本里读出来的。这就是今天的核心。")
        return

    model = SimpleChatModel(model_id=MODEL, api_key=API_KEY, base_url=BASE_URL)
    graph = build_data_agent(model.bind_tools(TOOLS))

    demo_naive(model)      # 先看不给工具会怎样
    demo_agent(graph)      # 再给工具

    print()
    print("=" * 64)
    print("  今天的三句话")
    print("=" * 64)
    print("  ① 工具返回数字时，模型没有编造空间 —— 数不是它算的。")
    print("  ② 模型真正值钱的能力是「决定该算什么」，不是算。")
    print("  ③ 同一张 LangGraph 图，换工具 = 换一个应用（Day21/22/23 都证明了）。")
    print()
    print("  下一步 Day23-B：再给一个「画图」工具，")
    print("  让模型把算出来的结果画成折线图 —— 这才是完整的数据分析 Agent。")


if __name__ == "__main__":
    main()
