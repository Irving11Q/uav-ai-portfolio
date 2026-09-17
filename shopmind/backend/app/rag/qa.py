"""RAG 问答核心：混合检索知识库片段 + 多轮历史 + 通义千问生成答案，并回显引用来源。

企业级增强（阶段4）：
- 混合检索：向量语义召回 + 关键词 BM25 召回 -> RRF 融合 -> gte-rerank 重排（见 retrieval.py）
- 语义缓存：近义问题直接复用答案，省一次大模型调用（见 cache.py）
- 并发控制：大模型调用经信号量限并发，避免打满配额（见 ratelimit.py）
- 可观测：记录总时延 / 检索时延 / 大模型时延 / 缓存命中，供 /api/stats 展示（见 stats.py）
"""
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .. import appconfig, ratelimit, stats
from ..config import settings
from . import cache, retrieval
from .embeddings import get_llm

SYSTEM_PROMPT = """你是一个电商商品知识库问答助手。请仅依据下方提供的「知识库参考片段」回答用户问题。
要求：
1. 回答要准确、专业、面向消费者，使用中文。
2. 必须在答案中通过 [n] 引用对应的参考片段编号（例如 [1]、[2]）。
3. 如果参考片段中没有相关信息，明确告知「知识库中未找到相关内容」，不要编造。
4. 不要输出参考片段原文之外的无关内容。

知识库参考片段：
{context}
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{history}\n\n当前问题：{question}"),
    ]
)


def _format_context(refs: list[dict]) -> str:
    lines = []
    for i, r in enumerate(refs, 1):
        snippet = r["text"].strip().replace("\n", " ")
        lines.append(f"[{i}] 来源：{r['source']}\n{snippet}")
    return "\n\n".join(lines) if lines else "（无）"


def _format_history(messages: list[dict]) -> str:
    """把历史消息拼成 human 可读的多轮对话。"""
    if not messages:
        return "（无历史对话）"
    parts = []
    for m in messages:
        role = "用户" if m["role"] == "user" else "助手"
        parts.append(f"{role}：{m['content']}")
    return "\n".join(parts)


def _to_references(refs: list[dict]) -> list[dict]:
    """把检索结果裁剪为引用结构（snippet 截断避免超长）。"""
    return [
        {
            "doc_id": r["doc_id"],
            "source": r["source"],
            "snippet": r["text"].strip()[:500],
            "score": r["score"],
        }
        for r in refs
    ]


def answer(question: str, history: list[dict] | None = None) -> dict:
    """执行一次 RAG 问答（带缓存 / 混合检索 / 埋点）。

    返回 {answer, references:[{doc_id, source, snippet, score}], cached: bool}
    references 与 prompt 中的 [1][2] 编号一一对应。
    """
    history = history or []
    t_start = time.perf_counter()

    # 1) 语义缓存：命中则直接返回，跳过检索与大模型
    if appconfig.get("CACHE_ENABLED", settings.CACHE_ENABLED):
        hit = cache.get_cached(question)
        if hit:
            stats.record_query(
                latency_ms=(time.perf_counter() - t_start) * 1000, cached=True
            )
            return {
                "answer": hit["answer"],
                "references": hit["references"],
                "cached": True,
            }

    # 2) 混合检索（向量 + 关键词 -> RRF 融合 -> Rerank）
    #    top_k / 候选池 / 是否重排 都由运行时配置决定（见 retrieval.hybrid_search）
    t_retrieval = time.perf_counter()
    retrieval_error = False
    try:
        refs = retrieval.hybrid_search(question)
    except Exception:  # noqa: BLE001
        retrieval_error = True
        refs = []
    retrieval_ms = (time.perf_counter() - t_retrieval) * 1000

    context = _format_context(refs)
    history_text = _format_history(history)
    references = _to_references(refs)

    # 3) 大模型生成（经并发信号量，避免瞬时高并发打满配额）
    t_llm = time.perf_counter()
    chain = PROMPT | get_llm() | StrOutputParser()
    try:
        with ratelimit.llm_slot():
            answer_text = chain.invoke(
                {"context": context, "history": history_text, "question": question}
            ).strip()
    except Exception:  # noqa: BLE001
        # 失败也要埋点，便于 /api/stats 观察错误率
        stats.record_query(
            latency_ms=(time.perf_counter() - t_start) * 1000,
            retrieval_ms=retrieval_ms,
            llm_ms=(time.perf_counter() - t_llm) * 1000,
            retrieval_error=retrieval_error,
            llm_error=True,
        )
        raise
    llm_ms = (time.perf_counter() - t_llm) * 1000

    # 4) 写语义缓存 + 埋点
    cache.put(question, answer_text, references)
    stats.record_query(
        latency_ms=(time.perf_counter() - t_start) * 1000,
        retrieval_ms=retrieval_ms,
        llm_ms=llm_ms,
        cached=False,
        retrieval_error=retrieval_error,
    )
    return {"answer": answer_text, "references": references, "cached": False}


def answer_stream(question: str, history: list[dict] | None = None):
    """流式版问答：逐块产出**事件字典**，由路由层编码成 SSE 推给前端。

    与 `answer()` 的分工：
    - 这里只管「检索 + 生成 + 埋点」，**不落库、不写审计**；
      消息 id 与请求上下文只有路由层才知道，交回去做更自然。
    - 返回引用的时机提前到「检索完成、生成开始之前」——检索本来就先跑完，
      先把引用推给前端，用户能立刻看到「引用了哪些片段」，不用干等模型吐字。

    事件序列（顺序固定）：
        {"type": "meta",       "cached": bool}         检索前，告知是否命中缓存
        {"type": "references", "references": [...]}     检索后立即（命中缓存时同样发）
        {"type": "delta",      "text": str}            0..n 次，答案增量
        {"type": "done",       "cached", "latency_ms", "ttft_ms"}

    缓存命中时不切片伪装流式 —— 命中本来就该「秒回」，一次性吐全文更诚实。
    生成中途异常会直接向上抛，由路由层决定怎么把错误推给前端（这里只保证埋点已记）。
    """
    history = history or []
    t_start = time.perf_counter()
    ttft_ms: float | None = None

    # 1) 语义缓存：命中则一次性返回整段答案，跳过检索与大模型
    if appconfig.get("CACHE_ENABLED", settings.CACHE_ENABLED):
        hit = cache.get_cached(question)
        if hit:
            cost_ms = round((time.perf_counter() - t_start) * 1000, 1)
            stats.record_query(latency_ms=cost_ms, cached=True)
            yield {"type": "meta", "cached": True}
            yield {"type": "references", "references": hit["references"]}
            if hit["answer"]:
                yield {"type": "delta", "text": hit["answer"]}
            yield {"type": "done", "cached": True, "latency_ms": cost_ms, "ttft_ms": cost_ms}
            return

    # 2) 混合检索（向量 + 关键词 -> RRF 融合 -> Rerank）
    t_retrieval = time.perf_counter()
    retrieval_error = False
    try:
        refs = retrieval.hybrid_search(question)
    except Exception:  # noqa: BLE001
        retrieval_error = True
        refs = []
    retrieval_ms = (time.perf_counter() - t_retrieval) * 1000

    references = _to_references(refs)
    context = _format_context(refs)
    history_text = _format_history(history)

    # 引用先发：这时候生成还没开始，用户已经能看到溯源卡片了
    yield {"type": "meta", "cached": False}
    yield {"type": "references", "references": references}

    # 3) 流式生成（经并发信号量，避免瞬时高并发打满配额）
    chain = PROMPT | get_llm() | StrOutputParser()
    parts: list[str] = []
    t_llm = time.perf_counter()
    try:
        with ratelimit.llm_slot():
            for chunk in chain.stream(
                {"context": context, "history": history_text, "question": question}
            ):
                if not chunk:
                    continue
                if ttft_ms is None:
                    # 首字延迟（TTFT）：前端「开始出字」的真实耗时，比总耗时更能反映体感
                    ttft_ms = (time.perf_counter() - t_start) * 1000
                parts.append(chunk)
                yield {"type": "delta", "text": chunk}
    except Exception:  # noqa: BLE001
        stats.record_query(
            latency_ms=(time.perf_counter() - t_start) * 1000,
            retrieval_ms=retrieval_ms,
            llm_ms=(time.perf_counter() - t_llm) * 1000,
            retrieval_error=retrieval_error,
            llm_error=True,
        )
        raise
    llm_ms = (time.perf_counter() - t_llm) * 1000

    # 4) 写语义缓存 + 埋点（与同步版口径一致）
    answer_text = "".join(parts).strip()
    cache.put(question, answer_text, references)
    stats.record_query(
        latency_ms=(time.perf_counter() - t_start) * 1000,
        retrieval_ms=retrieval_ms,
        llm_ms=llm_ms,
        cached=False,
        retrieval_error=retrieval_error,
    )

    yield {
        "type": "done",
        "cached": False,
        "latency_ms": round((time.perf_counter() - t_start) * 1000, 1),
        "ttft_ms": round(ttft_ms, 1) if ttft_ms is not None else None,
    }
