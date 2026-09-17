"""混合检索：向量语义召回 + 关键词 BM25 召回 -> RRF 融合 -> (可选) Rerank 重排。

- 向量检索擅长语义泛化，关键词检索擅长精确词面匹配，二者互补显著提升召回质量。
- RRF(Reciprocal Rank Fusion) 无需归一化分数即可融合多路召回。
- Rerank 用通义千问 gte-rerank 对融合候选二次精排；失败时优雅退化为融合排序。
"""
import logging

from ..config import settings
from . import keyword, vectorstore

logger = logging.getLogger("rag")


def _rrf(lists: list[list[dict]], k: int = 60) -> list[dict]:
    """多路召回按排名融合。"""
    fused: dict[int, float] = {}
    info: dict[int, dict] = {}
    for lst in lists:
        for rank, item in enumerate(lst):
            cid = item.get("chunk_id")
            if cid is None:
                continue
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (rank + k)
            info.setdefault(cid, item)
    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    out = []
    for cid, sc in ranked:
        it = dict(info[cid])
        it["score"] = round(sc, 4)
        out.append(it)
    return out


def _rerank(query: str, candidates: list[dict]) -> list[dict]:
    """用通义千问 gte-rerank 对候选精排；不可用时退化为原顺序。"""
    from .. import appconfig

    if not (appconfig.get("RERANK_ENABLED", settings.RERANK_ENABLED) and settings.DASHSCOPE_API_KEY):
        return candidates
    if not candidates:
        return candidates
    try:
        import requests

        top_n = int(appconfig.get("RERANK_TOP_N", settings.RERANK_TOP_N))
        documents = [c["text"] for c in candidates[:top_n]]
        resp = requests.post(
            # ⚠️ DashScope 的文本重排端点是**两段式**路径，少了 /text-rerank/text-rerank
            # 后缀会返回 400「No static resource api/v1/rerank」。曾因此静默降级很久。
            "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
            headers={
                "Authorization": f"Bearer {settings.DASHSCOPE_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.RERANK_MODEL,
                "input": {"query": query, "documents": documents},
                "parameters": {"return_documents": False},
            },
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json().get("output", {}).get("results", [])
        if not results:
            return candidates
        reranked = []
        for r in sorted(results, key=lambda x: x["relevance_score"], reverse=True):
            item = dict(candidates[r["index"]])
            # 用重排相关性分数覆盖融合分，前端引用卡片展示更直观（0~1，越大越相关）
            item["score"] = round(float(r["relevance_score"]), 4)
            reranked.append(item)
        return reranked
    except Exception as e:  # noqa: BLE001
        logger.warning("rerank 失败，退化为融合排序: %s", e)
        return candidates


def hybrid_search(query: str, top_k: int | None = None) -> list[dict]:
    """执行混合检索，返回最终交给大模型的片段列表。

    top_k / 候选池大小都可由管理员在后台调整（运行时配置），改完立即生效。
    """
    from .. import appconfig

    top_k = top_k or int(appconfig.get("RETRIEVE_TOP_K", settings.RETRIEVE_TOP_K))
    pool = int(appconfig.get("HYBRID_POOL_SIZE", settings.HYBRID_POOL_SIZE))

    try:
        vec = vectorstore.search(query, top_k=pool)
    except Exception as e:  # noqa: BLE001
        logger.warning("向量检索失败: %s", e)
        vec = []

    try:
        kw = keyword.search(query, top_k=pool)
    except Exception as e:  # noqa: BLE001
        logger.warning("关键词检索失败: %s", e)
        kw = []

    if not vec and not kw:
        return []

    lists = [lst for lst in (vec, kw) if lst]
    fused = _rrf(lists)
    reranked = _rerank(query, fused)
    return reranked[:top_k]
