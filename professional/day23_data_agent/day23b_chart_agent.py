#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════
  Day 23-B：给 Agent 加「画图」工具 —— 图是给你看的，不是给模型看的
═══════════════════════════════════════════════════════════════

【程序做什么】
  在 Day23-A 的四个「算数工具」之上，再加一个「画图工具」：
      「把电压随时间的变化画出来」
      「温度和高度画在一张图上，看看有没有关系」
  图会存成 png 存到 _out_day23b/ 目录，你打开就能看。

【★ 今天最重要的一句话】
  模型看不见这张图。
  它调用画图工具，工具把图画好存盘，返回给模型的只是一串**文字摘要**。
  所以模型聊「图上怎么样」的时候，聊的其实是数字，不是像素。

  想让模型真的「看图」，得把图片再喂给视觉模型（多模态），那是另一回事。
  今天先记住这个边界：**工具的输出形态，决定了模型能聊什么。**

【怎么读这个文件】
  1. _setup_chinese_font()  ★ matplotlib 中文乱码的坑（必踩）
  2. plot_series()          ★ 今天的唯一新工具
  3. demo_draw()            让模型画两张图
  4. demo_cannot_see()      ★ 对照：问它「图上最低点在哪」，它只能靠数字答

【运行方式】
  cd w5_agent
  D:/Python-envs/chroma-env/Scripts/python.exe day23b_chart_agent.py

  跑完去 _out_day23b/ 看生成的图。

【依赖】Day23-A（数据 + 四个算数工具）、Day21-B（SimpleChatModel + ReAct 图）
"""

# ════════════════════════════════════════════════════════════════
# 第 1 部分：环境
# ════════════════════════════════════════════════════════════════

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_PARENT, "w3_ai"))

# 【解释】和 Day23-A 踩的是同一个坑：api_config 里叫 MODEL_NAME，不是 MODEL。
#        用 import 模块再取属性，别用 from ... import 具体名字 —— 否则一个名字写错，
#        整个 try 一起失败，有 Key 也被当成没配（那天就是这么静默降级的）。
try:
    import api_config
    API_KEY = api_config.API_KEY
    BASE_URL = api_config.BASE_URL
    MODEL = api_config.MODEL_NAME
    HAS_KEY = bool(API_KEY)
except Exception:
    API_KEY, BASE_URL, MODEL = "", "", "glm-4-flash"
    HAS_KEY = False

import matplotlib
matplotlib.use("Agg")          # 【解释】无界面后端，服务器/脚本里必须这么设
import matplotlib.pyplot as plt
from langchain_core.tools import tool

try:
    from day21b_agent_tools import SimpleChatModel, build_agent
    HAS_AGENT = True
except Exception as e:
    HAS_AGENT = False
    AGENT_ERR = e

from day23a_data_agent import (
    load_df, _resolve_col, TOOLS as DATA_TOOLS, SYSTEM_PROMPT, CSV_PATH,
)

OUT_DIR = os.path.join(_HERE, "_out_day23b")

# ════════════════════════════════════════════════════════════════
# 第 2 部分：★ matplotlib 中文乱码（Windows 必踩的坑）
# ════════════════════════════════════════════════════════════════

def _setup_chinese_font():
    """不设这个，图上的中文会变成一个个方框 □□□。

    【解释】matplotlib 默认字体不含中文。Windows 上指定系统自带的中文字体即可。
    第二个参数 axes.unicode_minus 是连带的坑：不设 False，负号会变成方框。
    """
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


# ════════════════════════════════════════════════════════════════
# 第 3 部分：★ 今天的唯一新工具 —— 画图
# ════════════════════════════════════════════════════════════════

@tool
def plot_series(cols: str, title: str = "") -> str:
    """把一列或多列数据画成折线图（横轴是时间），保存成 png 文件。

    问「画一下XX的变化」「把A和B画在一起」时用这个。
    ⚠️ 注意：图画好是给人看的，模型自己看不到，所以返回值是数据摘要。

    Args:
        cols: 列名，多个用逗号分隔，比如「电压V」或「温度C,高度m」
        title: 图标题，可以留空
    """
    df = load_df()
    names = [c.strip() for c in cols.replace("，", ",").split(",") if c.strip()]
    if not names:
        return "没给列名。可用列：%s" % "、".join(df.columns)

    real = []
    for n in names:
        r = _resolve_col(n)
        if r is None:
            return "没有「%s」这一列。可用列：%s" % (n, "、".join(df.columns))
        real.append(r)

    _setup_chinese_font()
    os.makedirs(OUT_DIR, exist_ok=True)

    x = df["时间"].astype(str)
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=120)
    for r in real:
        ax.plot(x, df[r], marker="o", markersize=3, linewidth=1.4, label=r)

    ax.set_xlabel("时间")
    ax.set_ylabel(" / ".join(real))
    ax.set_title(title or ("、".join(real) + " 随时间变化"))
    ax.legend()
    ax.grid(alpha=.3)
    # 【解释】时间标签密，只显示每隔几个，否则 x 轴糊成一片
    step = max(1, len(x) // 8)
    ax.set_xticks(range(0, len(x), step))
    ax.set_xticklabels(x.iloc[::step], rotation=30, ha="right")
    fig.tight_layout()

    fname = "_".join(real).replace("/", "") + ".png"
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path)
    plt.close(fig)

    # 【解释】★ 返回值里只给数字摘要，不给图。
    #        模型拿到的是下面这串文字，它聊「图」其实是在聊这些数字。
    lines = ["图已保存到：%s" % path, "（图是给你看的，我看不到；以下是我算出来的数字）"]
    for r in real:
        s = df[r]
        lines.append("%s：最小 %.2f，最大 %.2f，均值 %.2f" % (r, s.min(), s.max(), s.mean()))
    return "\n".join(lines)


TOOLS = DATA_TOOLS + [plot_series]

# 【解释】系统提示里补一条：画图时顺手把关键数字也说出来，
#        因为用户可能是在命令行里跑，图得另外打开看。
SYSTEM_PROMPT_2 = SYSTEM_PROMPT + "\n5. 画完图后，用一句话说出最关键的那个数字。"


# ════════════════════════════════════════════════════════════════
# 第 4 部分：演示
# ════════════════════════════════════════════════════════════════

def run_question(graph, q):
    """问一句，把调了什么工具、返回什么、怎么答的都打出来。"""
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

    print()
    print("  " + "─" * 58)
    print("  问：%s" % q)

    init = {"messages": [SystemMessage(content=SYSTEM_PROMPT_2), HumanMessage(content=q)]}
    seen = set()
    final = ""

    for event in graph.stream(init, stream_mode="values"):
        for m in event["messages"]:
            if id(m) in seen:
                continue
            seen.add(id(m))
            if isinstance(m, ToolMessage):
                txt = str(m.content).replace("\n", " ")
                print("      📊 工具返回：%s" % txt[:130])
            elif hasattr(m, "tool_calls") and getattr(m, "tool_calls", None):
                for c in m.tool_calls:
                    print("      🔧 调工具：%s(%s)" % (c["name"], c["args"]))
            elif m.type == "ai" and (m.content or "").strip():
                final = m.content.strip()
                print("      💬 %s" % final)
    return final


def demo_draw(graph):
    """让模型画两张图。"""
    print()
    print("  ── 演示一：让它画图 ──")
    run_question(graph, "把这次飞行的电压随时间变化画出来")
    run_question(graph, "把温度和高度画在同一张图上，我想看看有没有关系")


def demo_cannot_see(graph):
    """★ 对照：问它「图上」怎么样 —— 它只能靠数字答，因为它看不见图。"""
    print()
    print("  ── 演示二：★ 问它「图上」的事，它其实看不见图 ──")
    print("     （注意看：它会去调算数工具，而不是去「看」刚才那张图）")
    run_question(graph, "刚才那张电压图里，最低的那一段大概是什么时候？")


def main():
    print("=" * 64)
    print("  Day 23-B：画图工具 —— 图给你看，数字给模型用")
    print("=" * 64)
    print()
    print("  数据：%s（%d 行）" % (os.path.basename(CSV_PATH), len(load_df())))
    print()

    if not HAS_KEY or not HAS_AGENT:
        print("  ⚠️ 没配 API Key 或没装 langgraph，只演示画图工具本身：")
        print()
        print(plot_series.invoke({"cols": "电压V", "title": "电压随时间变化"}))
        print()
        print("  去 %s 打开 png 看看 —— 中文如果能正常显示，说明字体设对了。" % OUT_DIR)
        return

    model = SimpleChatModel(model_id=MODEL, api_key=API_KEY, base_url=BASE_URL)
    graph = build_agent(model.bind_tools(TOOLS), tools=TOOLS)

    demo_draw(graph)
    demo_cannot_see(graph)

    print()
    print("=" * 64)
    print("  今天的三句话")
    print("=" * 64)
    print("  ① 工具返回什么形态，模型就只能聊什么 —— 返回摘要，它就只能聊数字。")
    print("  ② 「让模型看图」不是画出来就行，得把图片再喂给视觉模型（多模态）。")
    print("  ③ matplotlib 画中文：必须指定中文字体，否则全是方框 □□□。")
    print()
    print("  图都在：%s" % OUT_DIR)
    print()
    print("  Day23 完结。到今天你已经能让模型：")
    print("    查手册（Day22）→ 算数据（Day23-A）→ 出图（Day23-B）")
    print("    下一步 W5 D5：多 Agent 协作（一个调度 + 几个干活的）。")


if __name__ == "__main__":
    main()
