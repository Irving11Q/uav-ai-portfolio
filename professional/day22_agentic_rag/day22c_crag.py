"""═══════════════════════════════════════════════════════════════
  Day 22-C：给检索加一道「评分器」—— Corrective RAG (CRAG)
═══════════════════════════════════════════════════════════════

【这个文件做什么】
    Day22-A 让 Agent 自己决定「查不查」，Day22-B 让 Agent 自己「换词再查」。
    但这两天的命门是同一句话：**小模型靠自觉不靠谱**。
        —— 问「电池保修多久」，glm-4-flash 会跳过工具，凭常识编「通常 1 年」
        —— 调了工具但捞到的段落答不了，它也可能硬编
    今天引入 CRAG（微软 2024 提出，2026 已成药）：在「检索」和「生成」之间，
    插一个**独立的评分器节点**，用一条规则把「查得好不好」这件事从模型脑子里
    拿出来，变成代码里看得见、可观测、可熔断的一步。

    一句话定位：
        Day22-A 管「要不要查」   → 由模型决策
        Day22-B 管「查不到换词」 → 由模型自觉
        Day22-C 管「查到了管不管用」→ 由评分器判定（不靠模型自觉）

【怎么读这个文件】
    阅读顺序：main() → build_crag_agent()（看图的 4 个节点）→ grade_node()（评分器）
    → rewrite_node()（换词重查）→ run_question()（看打印出来的过程）
    运行方式：D:/Python-envs/chroma-env/Scripts/python.exe day22c_crag.py
    为什么学：这是 Agentic RAG 里最值钱的一块——把「幻觉防线」从 Prompt 玄学
             变成工程节点。面试讲这个，比讲「我调了 search_manual」高一个层次。

【和 Day22-A/B 的关系】
    A/B 的图是 agent ⇄ tools（两节点循环）。今天在 tools 之后插了 grade，
    形成 agent ⇄ tools → grade ⇄ rewrite 的四节点循环。图变了，所以自己建图，
    不复用 day21b.build_agent（那正是「图是骨架、按需求长」的活教材）。
═══════════════════════════════════════════════════════════════"""

import json
import os
import sys

# 离线标志：day17 那组文件第一次跑会去 hugginface 联网校验，国内必卡 5 分半。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_ENDPOINT", "hf-mirror.com")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── 复用 Day22-A 的检索器和工具（一个字不改） ──
from day22a_retrieval_as_tool import (
    search_manual,        # @tool 包装的检索器
    get_retriever,        # 惰性单例
    HAS_KEY,
    HAS_RAG,
    RAG_ERR,
    MODEL,
    API_KEY,              # 复用 day22a 已经拿到的 Key，别自己读环境变量
    BASE_URL,
    SYSTEM_PROMPT,        # Day22-A 那版「必须查 / 不许凭常识 / 查不到说没有」
)
from day21b_agent_tools import SimpleChatModel   # 自研适配器（langchain_openai 没装）

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from typing import Annotated, TypedDict

MAX_RETRY = 2
# 【解释】最多换词重查 2 次。这是 CRAG 的「熔断」——防止 Agent 陷入
#         无限循环（Agentic RAG 的经典故障模式之一）。

# ── 三个角色各自的 System Prompt ──
AGENT_SYS = (
    SYSTEM_PROMPT
    + "\n\n【补充规则】如果对话里出现以「【检索评分】」开头的反馈，说明刚才检索到的"
      "资料不相关、系统已自动换词重查。请基于**最后一条工具返回**的结果回答；"
      "若最后的结果仍然无关，明确告诉用户手册里没有，不要编造。"
)

GRADE_SYS = (
    "你是一个检索质量评分器。下面会给你「用户问题」和「检索到的资料」。\n"
    "请判断这份资料是否足以回答用户问题。\n"
    "只输出一行：relevant（资料能回答用户问题）或 not_relevant（资料不能回答），"
    "并在冒号后用一句话说明原因。"
)
# 【解释】⭐ 这就是 CRAG 的灵魂：把「查得好不好」变成一个**独立的、可被观测的判断**，
#         而不是指望主模型在生成时自己意识到「我查到的不对」。

REWRITE_SYS = (
    "你是查询词改写器。用户用口语化、带代词或语气词的方式提问，但知识库只收录规范术语。\n"
    "请把下面的问题改写成一个更适合向量检索的简短查询词：\n"
    "  - 去掉代词（这玩意儿 / 它）和语气词\n"
    "  - 换成手册可能用的术语（续航、鼓包、电压跌落…）\n"
    "只输出改写后的查询词本身，不要任何解释。"
)


# ════════════════════════════════════════════════════════════════
# 第 1 部分：评分器 + 改写器（两个轻量模型调用，temperature=0 求稳定）
# ════════════════════════════════════════════════════════════════

_grader = None   # 惰性单例：纯对话模型，不绑工具


def _get_grader():
    global _grader
    if _grader is None:
        _grader = SimpleChatModel(model_id=MODEL, api_key=API_KEY, base_url=BASE_URL)
        # 【解释】day22a 的 build_model 已经建了一个 bind_tools 的模型；
        #         评分器不需要工具，单独建一个干净的实例，互不干扰。
        #         用的是和主模型同一个 Key（从 day22a 复用），保证一致可用。
    return _grader


def grade_retrieval(question, doc_text):
    """返回 (verdict, reason)。verdict ∈ {'relevant', 'not_relevant'}。

    没配 Key 时降级为「相似度阈值」判断（从资料文本里抽不出分，
    就直接重查一遍拿最高分）——保证离线也能看到评分器的行为。
    """
    if not HAS_KEY:
        r = get_retriever()
        hits = r.retrieve(question)
        if not hits:
            return "not_relevant", "检索被阈值拒答（手册里没有相关段落）"
        return "relevant", "检索命中，最高相似度 %.3f" % hits[0][2]

    g = _get_grader()
    payload = "用户问题：%s\n\n检索到的资料：\n%s" % (question, (doc_text or "")[:1500])
    try:
        resp = g.invoke([SystemMessage(content=GRADE_SYS),
                         HumanMessage(content=payload)])
    except Exception as e:
        return "relevant", "评分器调用失败（%s），保守放行" % e
    # 【解释】保守策略：评分器挂了就当 relevant，让主模型去答，
    #         而不是卡死流程——生产系统的「 fail open 」原则。
    text = (resp.content or "").lower()
    if "not_relevant" in text:
        return "not_relevant", (resp.content or "").strip()
    if "relevant" in text:
        return "relevant", (resp.content or "").strip()
    return "relevant", "默认放行"


def rewrite_query(question):
    """把口语问题改写成规范查询词。"""
    g = _get_grader()
    resp = g.invoke([SystemMessage(content=REWRITE_SYS),
                     HumanMessage(content=question)])
    return (resp.content or "").strip()


# ════════════════════════════════════════════════════════════════
# 第 2 部分：自定义状态（在 messages 之外，记下评分结论和重写次数）
# ════════════════════════════════════════════════════════════════

class CragState(MessagesState):
    rewrite_count: int                         # 已经换词重查几次
    grade: str                                 # 最近一次评分结论
    need_search: bool                          # 问题是否命中领域词（必须查手册）
    route: str                                 # guard 节点的路由结论


def _agent_node(model):
    """agent 节点：拿全部历史，决定「调工具」还是「直接回答」。"""
    def _run(state):
        return {"messages": [model.invoke(state["messages"])]}
    return _run


def grade_node(state):
    """评分器节点：读最后一条工具返回，判 relevant / not_relevant。"""
    last_tool = None
    for m in reversed(state["messages"]):
        if isinstance(m, ToolMessage):
            last_tool = m
            break
    question = state["messages"][0].content
    verdict, reason = grade_retrieval(question, last_tool.content if last_tool else "")

    if verdict == "relevant":
        # 【解释】相关资料 → 不加任何干扰，流程回到 agent，agent 会基于它生成答案。
        return {"grade": "relevant"}

    # 不相关 → 把结论作为一条反馈消息压回对话，让 agent 最后知道「查不到」。
    feedback = HumanMessage(
        content="【检索评分】本次检索到的资料不相关（%s）。"
                "系统已自动换词重查；若重查后仍无相关结果，请明确说手册里没有，不要编造。"
                % reason)
    return {"grade": "not_relevant", "messages": [feedback]}


def rewrite_node(state):
    """重写节点：评分不相关时，改写成规范查询词，并**强制**再查一次。

    ⭐ 关键：不是让 agent「自觉」换词（Day22-B 的坑就在这），
       而是代码直接构造一条 tool_calls，交给 ToolNode 执行——
       重查由工程保证发生，不靠模型心情。
    """
    question = state["messages"][0].content
    new_q = rewrite_query(question)
    n = state["rewrite_count"] + 1
    print("      🔄 评分为不相关 → 重写查询词为：%s（第 %d 次重查）" % (new_q, n))

    # 构造一条「假」的 agent 决策，让 ToolNode 去执行 search_manual(new_q)。
    ai = AIMessage(content="", tool_calls=[{
        "name": "search_manual",
        "args": {"query": new_q, "top_k": 3},
        "id": "rewrite_%d" % n,
    }])
    return {"messages": [ai], "rewrite_count": n}


# ════════════════════════════════════════════════════════════════
# 第 2.5 部分：关键词兜底（guard 节点）—— 治「模型跳过工具」的病
# ════════════════════════════════════════════════════════════════

def guard_node(state):
    """agent 之后的一道护栏。

    ⭐ 这是 CRAG 的「另一半」：评分器（grade）只管「查到了但不好」，
       管不了「模型干脆不查」。后者靠代码兜底 —— 命中领域词就强制查一次。
       否则 Day22-A 演示三那种「问保修期，模型跳过工具编答案」会原样复现。
    """
    last = state["messages"][-1]
    has_tool_msg = any(isinstance(m, ToolMessage) for m in state["messages"])

    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return {"route": "tools"}              # 模型自己决定查了 → 放行去执行

    if state.get("need_search") and not has_tool_msg:
        # 还没查过、且这题必须查 → 代码强制构造一次检索，不靠模型自觉
        q = ""
        for m in state["messages"]:
            if isinstance(m, HumanMessage) and not (m.content or "").startswith("【检索评分】"):
                q = m.content                       # 取最初的用户问题，不是 System Prompt
                break
        if not q:
            # 【解释】极端情况下取不到用户原话（比如消息全被改写过），
            # 就别硬塞一个空查询给检索器，直接结束，让模型按上下文回答。
            return {"route": "end"}
        ai = AIMessage(content="", tool_calls=[{
            "name": "search_manual",
            "args": {"query": q, "top_k": 3},
            "id": "guard_1",
        }])
        return {"messages": [ai], "route": "tools"}

    return {"route": "end"}                     # 不必查、或已经查过了 → 结束


# ════════════════════════════════════════════════════════════════
# 第 3 部分：把节点连成图
# ════════════════════════════════════════════════════════════════

def build_crag_agent(model):
    """图：agent → guard → tools → grade ⇄ rewrite"""
    builder = StateGraph(CragState)

    builder.add_node("agent", _agent_node(model))
    builder.add_node("guard", guard_node)
    builder.add_node("tools", ToolNode([search_manual]))
    builder.add_node("grade", grade_node)
    builder.add_node("rewrite", rewrite_node)

    builder.add_edge(START, "agent")
    builder.add_edge("agent", "guard")
    # guard：模型已决定查 → tools；命中领域词却没查 → 强制查；否则结束
    builder.add_conditional_edges("guard", lambda s: s["route"],
                                  {"tools": "tools", "end": END})
    builder.add_edge("tools", "grade")
    # grade 相关 → 回 agent 生成；不相关且还能重查 → 去 rewrite；
    # 不相关且重查次数用尽 → 回 agent（它会基于反馈说「没有」）
    builder.add_conditional_edges(
        "grade",
        lambda s: ("agent" if s["grade"] == "relevant"
                   else ("rewrite" if s["rewrite_count"] < MAX_RETRY else "agent")),
        {"agent": "agent", "rewrite": "rewrite"})
    builder.add_edge("rewrite", "tools")
    # 【解释】⭐⭐ grade + rewrite 这两条边，构成了 CRAG 的「纠错循环」：
    #         查 → 评 → 不好就改写重查 → 再评。这是 Day19 固定管线永远做不到的。
    #         guard 节点则是「另一半防线」：保证模型至少查一次。

    return builder.compile()


# ════════════════════════════════════════════════════════════════
# 第 4 部分：跑一轮并展示过程
# ════════════════════════════════════════════════════════════════

_DOMAIN_KW = ["电池", "续航", "充电", "放电", "温度", "电压", "电流",
              "维护", "保养", "安全", "保修", "质保", "鼓包", "容量", "电量"]


def _need_search(q):
    return any(k in q for k in _DOMAIN_KW)


def run_question(graph, question, show=True):
    if show:
        print()
        print("  提问：%s" % question)

    answer = ""
    seen = set()                                # 用消息 id 去重，避免同一条被多个节点 event 重复打印
    init = {
        "messages": [SystemMessage(content=AGENT_SYS), HumanMessage(content=question)],
        "rewrite_count": 0,
        "grade": "",
        "need_search": _need_search(question),
        "route": "",
    }
    for event in graph.stream(init, stream_mode="values"):
        last = event["messages"][-1]
        if getattr(last, "id", None) in seen:
            continue
        seen.add(getattr(last, "id", None))
        kind = type(last).__name__

        if kind == "HumanMessage":
            if (last.content or "").startswith("【检索评分】") and show:
                print("      📊 %s" % last.content)
            continue

        if kind == "AIMessage" and getattr(last, "tool_calls", None):
            for tc in last.tool_calls:
                tid = str(tc.get("id", ""))
                if tid.startswith("rewrite"):
                    tag = "🔄 重写后重查"
                elif tid.startswith("guard"):
                    tag = "🛡️ 强制检索（命中领域词）"
                else:
                    tag = "🔧 调工具"
                if show:
                    print("      %s：%s(%s)" % (
                        tag, tc["name"],
                        json.dumps(tc["args"], ensure_ascii=False)))
        elif kind == "ToolMessage":
            if show:
                print("      📄 工具返回：%s…" % (last.content or "").replace("\n", " ")[:80])
        elif kind == "AIMessage":
            answer = (last.content or "").strip()
            if answer and show:
                print("      💬 %s" % answer[:200])

    return answer


# ════════════════════════════════════════════════════════════════
# 第 5 部分：三个演示（对应 Day22-A 的三个痛点）
# ════════════════════════════════════════════════════════════════

def demo_offline_grade():
    """没配 Key 时，看评分器用阈值怎么判 relevant / not_relevant。"""
    print()
    print("  [离线] 评分器（阈值版）行为演示：")
    r = get_retriever()
    for q in ["冬天低温飞行电池要注意什么", "附近有什么好吃的餐厅"]:
        hits = r.retrieve(q)
        verdict = "relevant" if hits else "not_relevant"
        print("      问题：%s" % q)
        print("         → 检索%s → 评分 %s" % (
            "命中(最高 %.3f)" % hits[0][2] if hits else "拒答", verdict))


def main():
    print("=" * 64)
    print("  Day 22-C：Corrective RAG —— 给检索加一道「评分器」")
    print("=" * 64)

    # 复用 day22a 的建模型（返回 bind_tools 的模型，或 None）
    try:
        from day22a_retrieval_as_tool import build_model
        model = build_model()
    except Exception:
        model = None

    if model is None:
        print()
        print("  ⚠️  未配置 API_KEY（或适配器不可用），运行离线评分演示：")
        demo_offline_grade()
        return

    graph = build_crag_agent(model)
    print()
    print("  ✅ CRAG Agent 就绪（评分器 + 改写器已挂载）")

    print()
    print("  ── 演示一：手册里有的问题 → 一次查中，评分 relevant ──")
    run_question(graph, "冬天低温飞行，电池要注意什么？")

    print()
    print("  ── 演示二：口语「能撑几个起落」→ 查不到，评分不相关 → 改写重查命中 ──")
    run_question(graph, "这块电池充满电能撑几个起落？")

    print()
    print("  ── 演示三：手册根本没有的「保修期」→ 重查两次仍无 → 老实说没有 ──")
    run_question(graph, "这块电池的保修期是多久？")

    print()
    print("=" * 64)
    print("  今天相对 Day22-A 多出来的四道防线：")
    print("    ① 关键词兜底(guard)：命中领域词就强制查，治「模型跳过工具」")
    print("    ② 评分器(grade)：查到的资料「管不管用」由独立节点判定，不靠模型自觉")
    print("    ③ 反馈回路：不相关结论压回对话，模型最后知道「查不到」")
    print("    ④ 熔断：最多重查 %d 次，防 Agent 陷入死循环" % MAX_RETRY)
    print("=" * 64)


if __name__ == "__main__":
    main()
