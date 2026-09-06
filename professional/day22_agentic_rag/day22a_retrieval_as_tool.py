#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════
  Day 22-A：把检索变成一个「工具」—— Agent 自己决定要不要查资料
═══════════════════════════════════════════════════════════════

【这个文件做什么】
    Day15~Day20 你搭了一套 RAG 管线，流程是**写死的**：
        用户提问 → 必定检索 → 拼 Prompt → 模型回答
    不管问的是「电池怎么保养」还是「你好啊」，都要先查一遍库。

    今天把检索包成一个**工具**，塞给 Day21 的 Agent：
        用户提问 → 模型自己决定「这题要查资料吗？」→ 查 or 直接答

【本文件最值得记住的一点】
    Day19 那套管线里，「要不要检索」是你用 if/else 替模型决定的；
    今天这个决定权交给了模型 —— 这是 RAG 和 Agent 的分水岭：

        · 固定管线（Day19）：检索是流程的**第 1 步**，永远执行
        · Agentic RAG（今天）：检索是工具箱里的**一件**，按需取用

    这带来两个真实好处：
        ① 闲聊/常识题不再浪费一次检索（省时间、省 token）
        ② 一次查不够，模型可以换个词再查一次（Day22-B 演示）

【怎么读这个文件】按这个顺序：
    1. main()                  —— 先看三个演示的输出，建立直觉
    2. ManualRetriever         —— 复用 Day18 的检索成品，一行不改
    3. search_manual()         —— ★ 检索器 → 工具的转换（重点看返回格式）
    4. build_rag_agent()       —— 图和 Day21-B 完全一样，只是换了个工具
    5. demo_compare()          —— Day19 固定管线 vs 今天，逐项对比

【运行方式】
    D:/Python-envs/chroma-env/Scripts/python.exe day22a_retrieval_as_tool.py

    第一次会加载中文 BGE 模型（约 1-2 分钟，之后走缓存）。
    没配 API Key 也能跑：会显示检索工具本身有效，只是跳过模型环节。

【依赖】Day18-A（建库）、Day18-B（混合检索+阈值拒答）、
        Day21-B（SimpleChatModel 适配器 + ReAct 图）
"""

# ════════════════════════════════════════════════════════════════
# 第 1 部分：环境 —— 必须在 import 重家伙之前设好
# ════════════════════════════════════════════════════════════════

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
_W4 = os.path.join(_PARENT, "w4_rag")
_W3 = os.path.join(_PARENT, "w3_ai")
# 【解释】w4_rag 里放着 Day18 的检索成品，w3_ai 里放着 api_config。
#         把两条路都加进模块搜索路径，才能 import 到兄弟目录的文件。

for _p in (_HERE, _W4, _W3):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 【解释】⭐ Day17 踩过的坑：BGE 模型传的是模型名，会先联网校验
#         huggingface.co，国内连不上就重试 5 次、白等 5 分钟。
#         模型本地早就有缓存，设离线标志让它直接用本地文件。
#         必须赶在 import 那些库之前设，晚了就已经被读走了。

import json

# ════════════════════════════════════════════════════════════════
# 第 2 部分：复用 Day18 的检索成品（一行不改）
# ════════════════════════════════════════════════════════════════

HAS_RAG = True
RAG_ERR = ""
try:
    from day18a_baseline_eval import build_index, CHUNKS, SECTIONS
    from day18b_recall_tuning import (
        search_with_threshold,
        top_semantic_score,
        SCORE_THRESHOLD,
    )
except Exception as _e:
    # 【解释】缺 chromadb 或 sentence-transformers 时不让整个文件崩掉，
    #         下面会提示你该换哪个环境。
    HAS_RAG = False
    RAG_ERR = "%s: %s" % (type(_e).__name__, _e)


class ManualRetriever:
    """
    无人机手册检索器 —— 复用了 Day18 的全部优化。

    对外只暴露一个方法：
        retrieve(question, top_k) -> [(标题, 正文, 分数), ...]
    """

    def __init__(self, threshold=SCORE_THRESHOLD if HAS_RAG else 0.5, top_k=3):
        self.threshold = threshold
        self.top_k = top_k

        self.coll, self.mode = build_index()
        # 【解释】建 Chroma 库 + 加载 BGE 模型。慢就慢在这一步（约 1-2 分钟）。

        self.refused = False
        self.reason = ""
        # 【解释】把"上一次为什么拒答"记下来，工具要把它告诉模型。

    def retrieve(self, question, top_k=None):
        k = top_k or self.top_k

        picked, refused = search_with_threshold(
            self.coll, question, threshold=self.threshold, top_k=k
        )
        # 【解释】★ 就这一行吃掉了 Day18 那两千行的成果：
        #         查询改写 → 语义 + 字频双路召回 → RRF 融合排序 → 阈值判定。

        self.refused = refused
        if refused:
            _, best = top_semantic_score(self.coll, question)
            self.reason = ("最高相似度只有 %.3f，低于阈值 %.2f"
                           % (best, self.threshold))
            return []

        self.reason = ""
        out = []
        for idx, score in picked:
            title = SECTIONS[idx] if idx < len(SECTIONS) else "第 %d 块" % idx
            out.append((title, CHUNKS[idx]["text"], float(score)))
        return out


_RETRIEVER = None


def get_retriever():
    """惰性单例：只有第一次真的要用时才建库（省掉无谓的 1 分钟等待）。"""
    global _RETRIEVER
    if _RETRIEVER is None:
        print("      （首次调用，正在建库并加载中文模型，约 1-2 分钟…）")
        _RETRIEVER = ManualRetriever()
    return _RETRIEVER


# ════════════════════════════════════════════════════════════════
# 第 3 部分：★ 把检索器包装成一个「工具」
# ════════════════════════════════════════════════════════════════

from langchain_core.tools import tool


@tool
def search_manual(query: str, top_k: int = 3) -> str:
    """
    在《无人机锂电池使用手册》中检索与问题相关的段落。

    什么时候该用：用户问的是手册里才有的内容（电池规格、续航、充电、
    温度、电压、安全预警等），必须用这个工具查到原文再回答。

    什么时候不该用：寒暄、常识题、纯计算题 —— 这些跟手册无关，直接回答即可。

    参数：
        query  —— 检索词，用简洁的短语，不要用完整句子
        top_k  —— 要几条结果，默认 3 条

    返回：编号的段落列表；如果手册里没有相关内容，会明确告诉你"没有检索到"。
    """
    r = get_retriever()
    hits = r.retrieve(query, top_k=top_k)
    # 【解释】工具内部调的还是 Day18 那套检索，一模一样。

    if not hits:
        return ("没有检索到相关段落。%s\n"
                "请换一个说法再试一次；如果换了说法仍然检索不到，"
                "就明确告诉用户手册里没有这个内容，不要编造。"
                % r.reason)
    # 【解释】⭐⭐ 这段话是整个 Day22 最重要的设计：
    #         "查不到"不是错误，而是给模型的**一条情报**。
    #         模型看到它，可以选择换个词再查，也可以直接告诉用户没有。
    #         Day19 的固定管线做不到这点 —— 它查不到就只能返回空，流程就断了。

    lines = []
    for i, (title, body, score) in enumerate(hits, 1):
        lines.append("[%d] 来源：%s（相关度 %.3f）\n%s" % (i, title, score, body))
    # 【解释】给每条资料编号 —— 模型后面才能在回答里说"我引用了第 2 条"。

    lines.append("（回答时请在每一条结论后面用【编号】标注出处，例如【1】）")
    # 【解释】光在 System Prompt 里说一次，小模型经常忘记照做；
    #         在资料末尾再提醒一遍，命中率明显更高（实测有效）。

    return "\n\n".join(lines)


# ════════════════════════════════════════════════════════════════
# 第 4 部分：接上大模型 —— 复用 Day21-B 的适配器和图
# ════════════════════════════════════════════════════════════════

try:
    import api_config
    API_KEY = api_config.API_KEY
    BASE_URL = api_config.BASE_URL
    MODEL = api_config.MODEL_NAME
    HAS_KEY = bool(API_KEY)
except Exception:
    API_KEY, BASE_URL, MODEL, HAS_KEY = "", "", "", False

try:
    from day21b_agent_tools import SimpleChatModel, build_agent
    HAS_AGENT = True
    AGENT_ERR = ""
except Exception as _e:
    HAS_AGENT = False
    AGENT_ERR = "%s: %s" % (type(_e).__name__, _e)
    SimpleChatModel = None

SYSTEM_PROMPT = (
    "你是无人机电池与能耗方面的技术助手，手边有一本《无人机锂电池使用手册》。\n"
    "\n"
    "【必须查手册的情况】\n"
    "只要问题涉及电池、续航、充电、放电、温度、电压、电流、维护保养、安全、\n"
    "保修/质保这类**手册可能覆盖的话题**，就必须先调用 search_manual 查证，\n"
    "再基于查到的原文回答，并在每条结论后标注引用编号，例如：……【1】。\n"
    "这类问题**不允许凭你自己的常识回答** —— 你的常识对这款机型未必成立。\n"
    "\n"
    "【不用查的情况】\n"
    "只有寒暄、和无人机电池完全无关的闲聊、纯数学计算，才直接回答。\n"
    "\n"
    "【查不到时】\n"
    "明确回答「手册里没有提到」，不要补充任何常识性内容。\n"
)
# 【解释】⭐ 三条规则对应三种情况：该查的查 / 不该查的别查 / 查不到的说没有。
#         这三条就是 Day22-A 的全部"智能"来源 —— 代码里没有任何 if/else 判断题型。
#
#         ⚠️ 第一版写得太温和（"回答手册里才有的内容时，先查"），实测结果是：
#            问「保修期多久」，glm-4-flash 会跳过工具，凭常识编一句
#            "通常电池保修 1 年，建议咨询制造商" —— 看似专业，实为幻觉。
#            加上"不允许凭常识回答"这句硬约束后，它才老老实实去查。
#            👉 教训：小模型的工具调用决策很散漫，Prompt 必须写死边界。


def build_model():
    """造一个带检索工具的模型。没配 Key 时返回 None。"""
    if not (HAS_KEY and HAS_AGENT):
        return None
    m = SimpleChatModel(model_id=MODEL, api_key=API_KEY, base_url=BASE_URL)
    return m.bind_tools([search_manual])
    # 【解释】bind_tools 把工具的说明书挂到模型上，模型才知道自己有这个能力。


def build_rag_agent(model):
    """图的结构和 Day21-B 一模一样，只是工具换成了检索器。"""
    return build_agent(model, tools=[search_manual])
    # 【解释】★ 这就是框架的价值：换任务不用改图，只换工具。
    #         Day21-B 是电池体检工具，今天是手册检索工具，图一行没变。
    #
    #         ⚠️ tools= 这个参数是踩坑后才加上去的：
    #            原先 day21b 的 build_agent 里写死 ToolNode(TOOLS)，
    #            结果模型点名要调 search_manual，执行节点却只认得电池工具，
    #            回了句「search_manual is not a valid tool」——
    #            模型拿不到资料，只能凭常识编一个看起来很像的答案。
    #            👉 记住：bind_tools 绑的和 ToolNode 执行的必须是同一套工具。


# ════════════════════════════════════════════════════════════════
# 第 5 部分：跑一轮对话并展示过程
# ════════════════════════════════════════════════════════════════

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


def run_question(graph, question, show=True):
    """
    跑一个问题，返回 (调工具次数, 最终回答)。

    重点看"调工具次数"：
        0 次 = 模型认为这题不用查
        1 次 = 查一次就够
        2 次+= 查了又查（Day22-B 会见到）
    """
    if show:
        print()
        print("  提问：%s" % question)

    steps = []
    called = 0
    answer = ""

    for event in graph.stream(
        {"messages": [HumanMessage(content=question)]},
        stream_mode="values",
    ):
        last = event["messages"][-1]
        kind = type(last).__name__

        if kind == "HumanMessage":
            continue
        # 【解释】stream_mode="values" 第一步会吐出含用户提问的初始状态，
        #         跳过它，否则用户提问会被印成"最终回答"。

        if kind == "AIMessage" and getattr(last, "tool_calls", None):
            called += 1
            if show:
                for tc in last.tool_calls:
                    print("      🔧 第 %d 次调工具：%s(%s)" % (
                        called, tc["name"],
                        json.dumps(tc["args"], ensure_ascii=False)))
            steps.append("tool")

        elif kind == "ToolMessage":
            if show:
                preview = (last.content or "").replace("\n", " ")[:90]
                print("      📄 工具返回：%s…" % preview)

        elif kind == "AIMessage":
            answer = (last.content or "").strip()
            if answer and show:
                print("      💬 %s" % answer[:200])

    return called, answer


# ════════════════════════════════════════════════════════════════
# 第 6 部分：三个演示 —— 该查 / 不该查 / 查不到
# ════════════════════════════════════════════════════════════════

def demo_offline_tool():
    """不调模型，先看工具本身：检索器还是 Day18 那个，行为没变。"""
    print()
    print("  ── 检索工具本体测试（不经过模型）──")

    for q in ["电池能飞多久", "附近有什么好吃的餐厅"]:
        r = get_retriever()
        hits = r.retrieve(q)
        print()
        print("  检索词：%s" % q)
        if not hits:
            print("     ❌ 没有检索到（%s）" % r.reason)
        else:
            for title, body, score in hits:
                print("     ✅ %.3f ｜ %s" % (score, title))
    # 【解释】这两个结果和 Day18 完全一致 —— 说明"包成工具"没有改变检索能力，
    #         变的只是"谁来决定要不要用"。


def demo_three_cases(graph):
    """三个问题，看模型怎么用（或不用）检索工具。"""
    print()
    print("  ── 模型自主决策测试 ──")

    q_need = "冬天低温飞行，电池要注意什么？"
    q_none = "你好，请用一句话介绍一下你自己。"
    q_miss = "这块电池的保修期是多久？"
    # 【解释】三个问题分别对应三种情况：
    #         ① 手册里有 → 应该查
    #         ② 跟手册无关 → 不该查
    #         ③ 像是有、其实手册里没有（保修/售后/托运，手册一概没写）
    #            → 模型会去查，但查完得老实承认"手册里没有"
    #
    #         ⚠️ 第 ③ 题不要用"失物招领处"这种一看就跟手册无关的问题：
    #            实测 glm-4-flash 会直接跳过工具，还自己幻想出一个
    #            "搜索引擎"工具，把伪 tool call 当正文吐出来。
    #            要测拒答，就得用**主题相关但手册确实没写**的问题。

    n1, _ = run_question(graph, q_need)
    print("      → 调工具 %d 次（预期 ≥1：这题手册里有答案）" % n1)

    n2, _ = run_question(graph, q_none)
    print("      → 调工具 %d 次（预期 0：寒暄不用查手册）" % n2)

    n3, a3 = run_question(graph, q_miss)
    REFUSE_MARKS = ("手册里没有", "手册中没有", "资料里没有", "没有提到", "未提到")
    admitted = any(m in a3 for m in REFUSE_MARKS)
    # 【解释】不能简单判断"没有"两个字 —— 模型常说
    #         "您**没有**提供具体型号"，那是反问，不是拒答。
    #         必须匹配"没有提到 / 手册里没有"这类完整说法。
    print("      → 调工具 %d 次；回答里%s承认「手册里没有」" % (
        n3, "有" if admitted else "没有"))
    # 【解释】这才是拒答能力的关键指标：不是"查没查"，
    #         而是"查不到之后敢不敢说没有"。

    if n3 == 0:
        print()
        print("      ⚠️  模型压根没查，直接凭常识回答了 ——")
        print("         Agent 一旦跳过检索，就退化成「裸问」，幻觉立刻回来。")
        print("         这是 Agentic RAG 的真实软肋，演示四给出工程解法。")
    # 【解释】这段不是"程序出错了"，而是今天最值钱的一条经验。
    #         glm-4-flash 这个量级的小模型，工具调用决策相当散漫，
    #         收紧 System Prompt 也不总能管住 —— 所以必须有代码兜底。

    return n1, n2, n3, a3


# ════════════════════════════════════════════════════════════════
# 第 7 部分：演示四 —— Agent 的软肋，以及工程上怎么兜底
# ════════════════════════════════════════════════════════════════

GUARD_KEYWORDS = [
    "电池", "续航", "充电", "放电", "温度", "电压", "电流",
    "保养", "维护", "安全", "保修", "质保", "飞行",
]
# 【解释】领域关键词表。命中其中任意一个，就认为"这题八成要查手册"。


def demo_guardrail(model):
    """
    演示四：给 Agent 加一道代码兜底。

    上面演示三暴露的问题：模型决定不查 → 直接凭常识回答 → 幻觉。
    工程上的解法不是继续调 Prompt，而是**把最关键的判断权收回代码**：
        命中领域关键词 → 强制检索，检索结果直接塞进上下文，
        模型只负责"照着资料说"，不再有"查不查"的选择权。
    """
    q = "这块电池的保修期是多久？"

    print()
    print("  ── 演示四：加代码兜底（关键词命中 → 强制检索）──")
    print()
    print("  提问：%s" % q)

    hit = any(k in q for k in GUARD_KEYWORDS)
    print("      🛡️ 关键词命中：%s → 强制先检索" % ("是（电池）" if hit else "否"))

    hits_text = search_manual.invoke({"query": q, "top_k": 3})
    # 【解释】@tool 装饰过的对象要用 .invoke(字典) 调用，不能直接当普通函数。

    if hits_text.startswith("没有检索到"):
        print("      📄 检索结果：没有检索到相关段落")
    else:
        print("      📄 检索结果：拿到资料")

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=(
            "【参考资料】\n%s\n\n"
            "（资料已由系统检索完毕，不要再调用工具，直接基于以上资料回答。）\n\n"
            "【用户问题】\n%s" % (hits_text, q)
        )),
    ]
    # 【解释】⭐ 这一步其实又变回了 Day19 的固定管线 ——
    #         但只用在"命中领域关键词"的问题上。
    #         👉 真实工程不是"固定管线 vs Agent"二选一，而是**混合**：
    #            高风险领域用固定管线保底，开放场景交给 Agent 自主。

    ans = model.invoke(messages).content.strip()
    print("      💬 %s" % ans[:220])

    REFUSE_MARKS = ("手册里没有", "手册中没有", "资料里没有", "没有提到", "未提到")
    ok = any(m in ans for m in REFUSE_MARKS)
    print("      → %s" % ("✅ 老实承认手册里没有（幻觉被兜住了）" if ok
                          else "⚠️ 仍然在编 —— 需要更强的模型或更严的约束"))

    if ok and not hits_text.startswith("没有检索到"):
        print("      （检索其实捞到了段落，但模型看过后判断里面没有保修内容 ——")
        print("       这就是 Day17 学过的「检索到的 ≠ 能回答的」，两道防线各管一段。）")
    return ans


def demo_compare(n_need, n_none, n_miss):
    """Day19 固定管线 vs 今天的 Agent，逐项对比。"""
    print()
    print("=" * 64)
    print("  Day19 固定管线  vs  Day22 Agentic RAG")
    print("=" * 64)
    print()
    print("      %-22s %-18s %-18s" % ("问题类型", "Day19 固定管线", "Day22 Agent"))
    print("      " + "-" * 58)
    print("      %-22s %-18s %-18s" % (
        "手册里有答案", "检索 1 次", "检索 %d 次" % n_need))
    print("      %-22s %-18s %-18s" % (
        "跟手册无关（寒暄）", "仍然检索 1 次（浪费）",
        "检索 %d 次" % n_none + ("（省下了）" if n_none == 0 else "")))
    print("      %-22s %-18s %-18s" % (
        "手册里没有答案", "检索 1 次后拒答", "检索 %d 次后拒答" % n_miss))
    print()
    print("      Day19 里「要不要检索」是代码写死的 if/else；")
    print("      今天这个决定由模型做 —— 代码里一行判断都没有。")


# ════════════════════════════════════════════════════════════════
# 第 7 部分：main
# ════════════════════════════════════════════════════════════════

def main():
    print("=" * 64)
    print("  Day 22-A：把检索变成工具，让 Agent 自己决定要不要查")
    print("=" * 64)

    if not HAS_RAG:
        print()
        print("  ⚠️  检索依赖没装好：%s" % RAG_ERR)
        print("     请用 chroma-env 运行（系统 Python 没有 chromadb）。")
        return

    if not HAS_AGENT:
        print()
        print("  ⚠️  没能复用 Day21-B 的适配器：%s" % AGENT_ERR)
        return

    print()
    print("  [1/3] 检索工具本体测试 —— 不经过模型")
    demo_offline_tool()

    if not HAS_KEY:
        print()
        print("  ⚠️  没检测到 API Key，跳过模型环节。")
        print("     上面的检索结果是真实有效的 —— 工具本身已经能工作了。")
        print("     配上 ZHIPU_API_KEY 或 DEEPSEEK_API_KEY 就能看到模型自主决策。")
        return

    print()
    print("  [2/3] 接入模型（%s）" % MODEL)
    model = build_model()
    graph = build_rag_agent(model)
    print("      ✅ Agent 就绪，工具：search_manual")

    print()
    print("  [3/4] 三个问题，看模型怎么用工具")
    n1, n2, n3, _ = demo_three_cases(graph)

    print()
    print("  [4/4] 演示四：给 Agent 加一道代码兜底")
    demo_guardrail(model)

    demo_compare(n1, n2, n3)

    print()
    print("=" * 64)
    print("  读完之后")
    print("=" * 64)
    print("""
  1. 检索器包成工具，核心就三件事：
       ① @tool 装饰 + 写明"什么时候该用/不该用"的 docstring（模型靠它判断）
       ② 返回值是**字符串**（工具结果进的是 ToolMessage.content）
       ③ 查不到时返回"没有检索到"+ 原因，而不是抛异常

  2. 今天踩到的两个真坑（面试讲这两个，比讲原理值钱）：
       坑一：bind_tools 绑的工具，必须和 ToolNode 执行的**是同一套**。
             否则模型点名要调，执行节点却不认得，只能回一句
             「xxx is not a valid tool」，然后模型被迫凭常识编答案。
       坑二：模型会**跳过工具**，直接凭常识回答（演示三稳定复现）。
             收紧 System Prompt 也未必管得住 ——
             解法是把判断权收回来：领域关键词命中就强制检索（演示四）。

  3. 真实工程不是「固定管线 vs Agent」二选一，而是混合：
       高风险领域用固定管线保底，开放场景交给 Agent 自主决策。

  4. 下一步（Day22-B）：让模型在查不到时**换个词再查一次**，
     并把引用编号落到最终回答 —— 那是固定管线永远做不到的。
""")


if __name__ == "__main__":
    main()
