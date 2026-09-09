#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════
  Day 24：多 Agent 协作 —— 一个调度者 + 两个干活的
═══════════════════════════════════════════════════════════════

【程序做什么】
  Day21 是一个 Agent，Day22/23 是给 Agent 加不同工具。
  今天把「多个 Agent 协作」做出来：一个 **调度者（supervisor）** 听问题，
  判断该交给谁，再把问题派给两个 **worker 子图**：
     · data_worker    —— 复用 Day23 的 pandas/画图工具，回答「飞行数据」
     · manual_worker  —— 用 bge 在内存里建索引，检索《电池手册》回答问题
  调度者不直接干活，只负责「派活 + 收活 + 决定结束」。

【怎么读这个文件（5 块）】
  1. 环境 + find_local_model()   bge 离线加载（沿用 Day20 的稳法）
  2. ManualIndex                  ★ manual_worker 的检索器（内存建索引，不落盘）
  3. build_workers()              ★ 两个 worker 子图（都复用 day21b.build_agent）
  4. Supervisor                   ★ 调度图（路由 + 派活 + 收活）
  5. main                        三个问题：1 数据 / 1 手册 / 1 两可

【运行方式】
  cd w5_agent
  D:/Python-envs/chroma-env/Scripts/python.exe day24_multi_agent.py

  没 Key 也能跑：调度者改走「关键词路由」，两个 worker 只演示工具本身。

【为什么学这个（面试能讲）】
  这是「多 Agent 系统 / 调度 Agent」的最小骨架，AI 应用岗高频考点。
  核心一句话：
      难任务不要塞给一个 Agent 硬扛，拆成「专业工」+「调度员」，
      调度员负责分工，工负责把一件事由头做到尾。
  今天你看到的就是 LangGraph 里最经典的 supervisor 模式：
      调度节点 → 条件边 → worker 子图 → 回到调度节点 → 结束。

【和前三天的关系】
  Day21 图（build_agent）被用了 3 次：数据分析 Agent、手册 Agent、今天两个 worker。
  变的是「挂什么工具、谁在调度」，不变的是那张图。
"""

# ════════════════════════════════════════════════════════════════
# 第 1 部分：环境
# ════════════════════════════════════════════════════════════════

import os
import sys
import json
import math

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_PARENT, "w3_ai"))
sys.path.insert(0, os.path.join(_PARENT, "day22_agentic_rag"))   # 找手册原文
sys.path.insert(0, os.path.join(_PARENT, "day23_data_agent"))    # 复用 Day23 工具

try:
    import api_config
    API_KEY = api_config.API_KEY
    BASE_URL = api_config.BASE_URL
    MODEL = api_config.MODEL_NAME
    HAS_KEY = bool(API_KEY)
except Exception:
    API_KEY, BASE_URL, MODEL = "", "", "glm-4-flash"
    HAS_KEY = False

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

try:
    from day21b_agent_tools import SimpleChatModel, build_agent
    HAS_AGENT = True
except Exception as e:
    HAS_AGENT = False
    AGENT_ERR = e


# ── bge 离线加载（照抄 Day20 的 find_local_model，最稳的写法）──
def find_local_model(model_name="BAAI/bge-small-zh-v1.5"):
    """在 HF 缓存里找模型的本地路径。找到返回路径，没找到返回 None。

    【坑】只设 HF_HUB_OFFLINE=1 不够 —— 它照样去请求 huggingface.co 重试 5 次。
    最稳的是把本地快照路径直接喂给 SentenceTransformer，零网络秒开。
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        cache = HF_HUB_CACHE
    except ImportError:
        cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    folder = "models--" + model_name.replace("/", "--")
    snap = os.path.join(cache, folder, "snapshots")
    if not os.path.isdir(snap):
        return None
    subs = [d for d in os.listdir(snap) if os.path.isdir(os.path.join(snap, d))]
    if not subs:
        return None
    return os.path.join(snap, sorted(subs)[-1])


LOCAL_MODEL = find_local_model()
if LOCAL_MODEL:
    os.environ["HF_HUB_OFFLINE"] = "1"
else:
    print("⚠️ 本地没找到 bge 模型，manual_worker 将无法做语义检索（会降级为关键词匹配）。")


# ════════════════════════════════════════════════════════════════
# 第 2 部分：★ manual_worker 的检索器（内存建索引，不落盘）
# ════════════════════════════════════════════════════════════════

_MANUAL_CANDIDATES = [
    os.path.join(_HERE, "uav_battery_manual.md"),                       # 发布目录自带
    os.path.join(_PARENT, "w4_rag", "uav_battery_manual.md"),            # 本地学习仓库（手册原文在这）
    os.path.join(_PARENT, "day22_agentic_rag", "uav_battery_manual.md"),  # 兜底
    os.path.join(_PARENT, "day23_data_agent", "uav_battery_manual.md"),   # 兜底
]
MANUAL_PATH = next((p for p in _MANUAL_CANDIDATES if os.path.exists(p)), _MANUAL_CANDIDATES[0])


class ManualIndex:
    """把《电池手册》切成块，用 bge 在内存里建向量索引；查的时候算余弦相似度。

    【解释】为什么不用 Chroma：Chroma 要落盘建库，发布时那个库目录被 gitignore，
    拷出去就空了。这个场景块数很少（7 段），内存里直接算 cosine 就够了，
    好处是「目录自包含、拷出去就能跑」，符合我们一直追求的零配置。
    """

    def __init__(self):
        self.chunks = []
        self.vectors = []
        self._ready = False

    def _embed(self, texts):
        """懒加载 bge，批量转向量。第一次调用才真正加载模型。"""
        if not hasattr(self, "_model"):
            if not LOCAL_MODEL:
                return None          # 没模型 → 走关键词降级
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(LOCAL_MODEL)
        return self._model.encode(texts, normalize_embeddings=True)

    def build(self):
        if self._ready:
            return
        text = open(MANUAL_PATH, encoding="utf-8").read()
        # 【解释】按 Markdown 的二级标题切块：每「## x. 小节」是一段独立知识。
        parts = text.split("\n## ")
        self.chunks = []
        for i, p in enumerate(parts):
            title = "总览" if i == 0 else p.split("\n", 1)[0].strip()
            body = p if i == 0 else p.split("\n", 1)[1]
            body = body.strip()
            if body:
                self.chunks.append((title, body))

        vecs = self._embed([c[1] for c in self.chunks])
        if vecs is not None:
            self.vectors = vecs
        self._ready = True

    def search(self, query, k=3):
        """返回前 k 段（标题, 正文, 分数）。没 bge 时退化为关键词命中数打分。"""
        self.build()
        if len(self.vectors) > 0:
            qv = self._embed([query])[0]
            scores = [float(qv @ v) for v in self.vectors]
            idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            return [(self.chunks[i][0], self.chunks[i][1], scores[i]) for i in idx]
        # 关键词降级
        qwords = [w for w in query if w.strip()]
        ranked = []
        for title, body in self.chunks:
            hit = sum(body.count(w) for w in qwords)
            ranked.append((title, body, float(hit)))
        ranked.sort(key=lambda x: x[2], reverse=True)
        return ranked[:k]


_MANUAL = ManualIndex()


@tool
def search_manual(query: str) -> str:
    """在《无人机锂电池使用手册》里检索与问题相关的段落。

    问「电池规格 / 电压管理 / 温度限制 / 充电 / 安全预警」等手册里有的事时用这个。
    返回相关段落并标注【引用】编号，方便核对答案出处。

    Args:
        query: 想查的问题，比如「温度超过多少要降落」
    """
    results = _MANUAL.search(query, k=3)
    if not results or results[0][2] <= 0:
        return "手册里没有相关内容。"
    out = []
    for i, (title, body, score) in enumerate(results, 1):
        out.append("【%d】%s\n%s" % (i, title, body))
    return "\n\n".join(out)


MANUAL_TOOLS = [search_manual]


# ════════════════════════════════════════════════════════════════
# 第 3 部分：★ 两个 worker 子图（都复用 Day21-B 的 build_agent）
# ════════════════════════════════════════════════════════════════

# 复用 Day23 的数据工具（+ 画图）
from day23a_data_agent import TOOLS as DATA_TOOLS
try:
    from day23b_chart_agent import plot_series
    DATA_TOOLS = DATA_TOOLS + [plot_series]
except Exception:
    pass

DATA_SYSTEM = (
    "你是无人机飞行日志分析助手。所有数字必须来自工具返回值，不许自己算。"
    "想回答跟数据有关的问题先调工具；回答要短，先结论后数字。"
)
MANUAL_SYSTEM = (
    "你是无人机电池手册问答助手。只依据【引用】里的手册内容回答，"
    "手册没有的就说没有，不许编造。回答要短，先结论后引用编号。"
)


def build_workers(model):
    """给两个 worker 各建一张子图。每张都是 Day21-B 那张图，只换工具 + 系统提示。

    model 为 None（没配 Key）时只返回占位，no-key 分支根本不会调到真正的图。
    """
    if model is None:
        return {"data": (None, DATA_SYSTEM), "manual": (None, MANUAL_SYSTEM)}
    data_graph = build_agent(model.bind_tools(DATA_TOOLS), tools=DATA_TOOLS)
    manual_graph = build_agent(model.bind_tools(MANUAL_TOOLS), tools=MANUAL_TOOLS)
    return {"data": (data_graph, DATA_SYSTEM), "manual": (manual_graph, MANUAL_SYSTEM)}


def run_worker(graph, system, messages):
    """调一个 worker 子图，把它的消息合并进调度者的消息列表。

    【解释】worker 是独立编译的图，直接 .invoke() 跑完一轮，
    返回的消息追加到主状态里 —— 这就是 LangGraph「子图」协作的本质：
    调度者只管传话，工负责把一件事由头做到尾。
    """
    inp = {"messages": [SystemMessage(content=system)] + list(messages)}
    out = graph.invoke(inp)
    return out["messages"]


# ════════════════════════════════════════════════════════════════
# 第 4 部分：★ 调度者（supervisor）
# ════════════════════════════════════════════════════════════════

from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

SUPERVISOR_SYSTEM = (
    "你是调度者，不直接回答问题。你有两个 worker：\n"
    "  - data：分析飞行数据（电压/电流/温度/高度/画图），问题里常含「几次/最低/最高/相关/画一张图」\n"
    "  - manual：查《电池手册》（规格/电压管理/温度限制/充电/安全预警）\n"
    "判断问题该给谁，只输出一行 JSON：{\"next\":\"data\"|\"manual\"|\"FINISH\",\"reason\":\"...\"}。"
    "当某个 worker 已经给出最终答案后，输出 {\"next\":\"FINISH\"}。"
)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    next: str
    step: Annotated[int, operator.add]


def _parse_route(text):
    """从模型输出里抠出 next。容错：找不到 JSON 就按关键词猜。"""
    try:
        obj = json.loads(text.strip().replace("```json", "").replace("```", "").strip())
        n = obj.get("next", "").upper()
        if n in ("DATA", "MANUAL", "FINISH"):
            return n.lower()
    except Exception:
        pass
    for kw, n in (("数据", "data"), ("电压", "data"), ("电流", "data"),
                  ("温度", "data"), ("图", "data"), ("手册", "manual"),
                  ("电池", "manual"), ("充电", "manual"), ("安全", "manual")):
        if kw in text:
            return n
    return "data"


def supervisor(state):
    """调度节点：看目前对话，决定下一步派给谁。"""
    # 防死循环：worker 已经给过最终答案（非调度标记的消息）→ 直接结束
    if any(isinstance(m, AIMessage) and not str(m.content).startswith("[调度]")
           for m in state["messages"]):
        return {"next": "FINISH"}
    if state.get("step", 0) >= 6:
        return {"next": "FINISH"}

    if not HAS_KEY:
        # 没 Key：用关键词路由（演示调度逻辑本身，不耗 API）
        last_q = ""
        for m in reversed(state["messages"]):
            if isinstance(m, HumanMessage):
                last_q = m.content
                break
        nxt = "manual" if any(w in last_q for w in ("电池", "手册", "充电", "安全", "规格", "存放")) else "data"
        print("      🚦 调度者（关键词路由）：→ %s" % nxt)
        return {"next": nxt, "step": 1}

    msgs = [SystemMessage(content=SUPERVISOR_SYSTEM)]
    msgs += [m for m in state["messages"] if not (isinstance(m, SystemMessage))]
    resp = _MODEL.invoke(msgs)
    content = (resp.content or "").strip()
    nxt = _parse_route(content)
    print("      🚦 调度者：→ %s   (%s)" % (nxt, content[:50]))
    return {"next": nxt, "step": 1, "messages": [AIMessage(content="[调度] → " + nxt)]}


def call_data(state):
    if not HAS_KEY:
        s = _demo_worker("data")
        return {"messages": [AIMessage(content="[data_worker] " + s)], "next": "FINISH"}
    out = run_worker(_WORKERS["data"][0], _WORKERS["data"][1], state["messages"])
    return {"messages": out, "next": "FINISH"}


def call_manual(state):
    if not HAS_KEY:
        s = _demo_worker("manual")
        return {"messages": [AIMessage(content="[manual_worker] " + s)], "next": "FINISH"}
    out = run_worker(_WORKERS["manual"][0], _WORKERS["manual"][1], state["messages"])
    return {"messages": out, "next": "FINISH"}


def _route(state):
    return state["next"]


def build_supervisor(model):
    """把调度图拼起来。"""
    global _MODEL, _WORKERS
    _MODEL = model
    _WORKERS = build_workers(model)
    g = StateGraph(State)
    g.add_edge(START, "supervisor")
    g.add_node("supervisor", supervisor)
    g.add_node("call_data", call_data)
    g.add_node("call_manual", call_manual)
    g.add_conditional_edges("supervisor", _route,
                             {"data": "call_data", "manual": "call_manual", "FINISH": END})
    g.add_edge("call_data", "supervisor")
    g.add_edge("call_manual", "supervisor")
    return g.compile()


def _demo_worker(which):
    """没 Key 时，只演示 worker 手上的工具本身（证明它「能干这个活」）。
    返回一句总结，作为该 worker 的「答复」追加进消息，让调度者能正常结束。"""
    if which == "data":
        print("      🔧 data_worker 工具演示：")
        from day23a_data_agent import filter_rows, describe_column, correlate
        print("         filter_rows(温度C,>,55) →")
        print("           " + filter_rows.invoke({"col": "温度C", "op": ">", "value": 55}).replace("\n", "\n           "))
        print("         correlate(电流A,电压V) →")
        print("           " + correlate.invoke({"col1": "电流A", "col2": "电压V"}))
        return "（无 Key 演示）温度>55度共4次；电流与电压几乎不相关(0.161)。"
    else:
        print("      🔧 manual_worker 工具演示（bge 内存检索）：")
        r = search_manual.invoke({"query": "温度超过多少要降落"})
        print("           " + r.replace("\n", "\n           "))
        return "（无 Key 演示）手册规定温度>55℃应降落散热，>60℃会鼓包起火。"


# ════════════════════════════════════════════════════════════════
# 第 5 部分：main —— 三个问题
# ════════════════════════════════════════════════════════════════

QUESTIONS = [
    "这次飞行温度超过 55 度的有几次？",
    "锂电池温度超过多少度必须降落散热？",
    "把电压和电流画在同一张图上，看看它们有没有关系",
]


def ask_one(graph, q):
    print()
    print("  " + "─" * 60)
    print("  用户：%s" % q)
    state = {"messages": [HumanMessage(content=q)], "next": "", "step": 0}
    seen = set()
    for ev in graph.stream(state, stream_mode="values"):
        for m in ev["messages"]:
            mid = getattr(m, "id", None) or id(m)
            if mid in seen:
                continue
            seen.add(mid)
            if isinstance(m, ToolMessage):
                print("      📊 工具返回：%s" % str(m.content).replace("\n", " ")[:90])
            elif isinstance(m, AIMessage) and (m.content or "").strip() and not str(m.content).startswith("[调度]"):
                print("      💬 %s" % m.content.strip())
    print("  " + "─" * 60)


def main():
    print("=" * 64)
    print("  Day 24：多 Agent 协作（supervisor + 2 workers）")
    print("=" * 64)
    if not HAS_AGENT:
        print("⚠️ 没装 langgraph，无法构建图：%s" % AGENT_ERR)
        return

    if not HAS_KEY:
        print("⚠️ 没配 API Key：调度者走关键词路由，worker 只演示工具本身。")
        print("   配置 ZHIPU_API_KEY / DEEPSEEK_API_KEY 后可看完整多 Agent 对话。")
        print()
        graph = build_supervisor(None)
        for q in QUESTIONS:
            ask_one(graph, q)
        print()
        print("=" * 64)
        print("  今天的三句话")
        print("=" * 64)
        print("  ① 难任务拆成「专业工 + 调度员」，调度者只分工不干活。")
        print("  ② 子图协作 = 调度者 .invoke(worker) 把消息合并回来。")
        print("  ③ 同一张 Day21 图被复用了 4 次（21/22/23/今天两个 worker）。")
        return

    model = SimpleChatModel(model_id=MODEL, api_key=API_KEY, base_url=BASE_URL)
    graph = build_supervisor(model)
    for q in QUESTIONS:
        try:
            ask_one(graph, q)
        except Exception as e:
            print("      ⚠️ 这个问题调用模型时出错了：%s" % e)
            print("         （多半是 API 限流 429，稍后重跑即可；其余问题不受影响）")

    print()
    print("=" * 64)
    print("  今天的三句话")
    print("=" * 64)
    print("  ① 难任务拆成「专业工 + 调度员」，调度者只分工不干活。")
    print("  ② 子图协作 = 调度者 .invoke(worker) 把消息合并回来。")
    print("  ③ 同一张 Day21 图被复用了 4 次（21/22/23/今天两个 worker）。")
    print()
    print("  到今天 W5 收官：单 Agent(21) → 加工具(22/23) → 多 Agent 协作(24)。")
    print("  下一步可做：worker 之间直接传中间结果（如数据结论喂给手册核对）。")


if __name__ == "__main__":
    main()
