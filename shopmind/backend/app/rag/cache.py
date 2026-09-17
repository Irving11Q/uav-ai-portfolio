"""语义缓存 + 问题池：既复用答案降本降延迟，也为「相似问题推荐 / 热问榜」沉淀题库。

## 两层职责（刻意分开，别混）
1. **缓存复用**：余弦相似度 ≥ 阈值（默认 0.96，即几乎同一句话）时直接返回历史答案，
   跳过检索与大模型。受运行时配置 `CACHE_ENABLED` 控制。
2. **问题池**：每次真实问答都把「问题 + 答案 + 引用」写进同一个集合。
   **不受 `CACHE_ENABLED` 影响** —— 因为「相似问题推荐」需要的是历史问题的向量，
   关掉缓存复用不代表想把题库清空。两件事分开，语义更清楚。

## 相似度口径统一为余弦
本模块对外输出的所有「相似度」都是**余弦相似度（0~1）**，由 `_cosine_from_distance` 换算。
原先直接用 `1 - L2平方距离`，数值被系统性压小、不相关问句还会算出负数（详见该函数注释）。

## 修掉的一个隐患
原来文档 id 用 `abs(hash(question))` 生成，而 Python 对字符串的 hash **每次进程启动都加盐随机**，
导致同一句问题在重启后会被算成不同 id、重复入库。改用 sha1，稳定可复现。
"""
import hashlib
import json
import logging

from ..config import settings
from .embeddings import get_embeddings
from .vectorstore import get_chroma_client

logger = logging.getLogger("rag")

_coll = None
_build_failed = False

# 相似问题推荐用的默认阈值（余弦相似度口径）：比缓存阈值低得多——
# 缓存要求「几乎同一个问题」才敢复用答案，而推荐只是想找「相关的问题」给用户点。
SIMILAR_DEFAULT_THRESHOLD = 0.5


def _cosine_from_distance(distance: float) -> float:
    """把 Chroma 的 L2 平方距离换算成余弦相似度（0~1）。

    ## 为什么要换算（原来这里是错的）
    Chroma 默认用 L2，`similarity_search_with_score` 返回的是**平方**欧氏距离。
    对归一化向量有恒等式：‖a-b‖² = 2(1 - cos)，所以 `cos = 1 - dist/2`。
    原实现写的是 `sim = 1 - dist`，导致：
      - 数值被系统性压小：dist=0.64（其实 cos=0.68）被算成 0.36；
      - 不相关的问句会算出**负数**（实测出现过 -0.14），当作「相似度」毫无意义。
    换成这个公式后，所有阈值都是「余弦相似度」口径，含义一致、可解释、可调。
    """
    return max(0.0, min(1.0, 1.0 - float(distance) / 2.0))


def _collection(force: bool = False):
    """取 Chroma 集合。

    force=True：即使缓存开关关闭也返回集合（写问题池 / 查相似问题用）。
    """
    global _coll, _build_failed
    if _coll is not None:
        return _coll
    if not settings.DASHSCOPE_API_KEY:
        return None
    if not force:
        # 普通读缓存：开关关了就别白花一次向量查询
        from .. import appconfig

        if not appconfig.get("CACHE_ENABLED", settings.CACHE_ENABLED):
            return None
        if _build_failed:
            return None
    try:
        from langchain_chroma import Chroma

        _coll = Chroma(
            client=get_chroma_client(),
            collection_name=settings.CACHE_COLLECTION,
            embedding_function=get_embeddings(),
        )
        _build_failed = False
        return _coll
    except Exception as e:  # noqa: BLE001
        logger.warning("语义缓存不可用: %s", e)
        _build_failed = True
        return None


def _doc_id(question: str) -> str:
    """稳定 id：同一句问题永远映射到同一条记录。"""
    return "q-" + hashlib.sha1(question.strip().encode("utf-8")).hexdigest()[:16]


def get_cached(question: str) -> dict | None:
    """命中返回 {answer, references}，否则 None。"""
    from .. import appconfig

    if not appconfig.get("CACHE_ENABLED", settings.CACHE_ENABLED):
        return None
    coll = _collection()
    if coll is None:
        return None
    threshold = float(
        appconfig.get("CACHE_SIMILARITY_THRESHOLD", settings.CACHE_SIMILARITY_THRESHOLD)
    )
    try:
        res = coll.similarity_search_with_score(question, k=1)
        if not res:
            return None
        doc, distance = res[0]
        sim = _cosine_from_distance(distance)
        if sim < threshold:
            return None
        meta = doc.metadata or {}
        refs = meta.get("references_json")
        return {
            "answer": meta.get("answer"),
            "references": json.loads(refs) if refs else [],
        }
    except Exception as e:  # noqa: BLE001
        logger.warning("读语义缓存失败: %s", e)
        return None


def put(question: str, answer: str, references: list[dict]) -> None:
    """把一次问答沉淀进问题池（同时成为可复用的缓存项）。"""
    coll = _collection(force=True)
    if coll is None:
        return
    try:
        coll.add_texts(
            texts=[question],
            metadatas=[
                {
                    "question": question,
                    "answer": answer,
                    "references_json": json.dumps(references, ensure_ascii=False),
                }
            ],
            ids=[_doc_id(question)],
        )
    except Exception as e:  # noqa: BLE001
        # 重复 id 会报错，但那说明该问题已在池中，属于正常情况
        logger.debug("写问题池跳过: %s", e)


def similar_questions(
    question: str, k: int = 3, threshold: float | None = None
) -> list[dict]:
    """相似问题推荐：在历史问题池里找语义相近的问题。

    返回 [{"question": str, "score": float}]，按相似度降序；
    会自动排除与当前问题完全相同的那条，避免推荐出「你自己刚问的这句」。
    无题库 / 无 Key / 异常时返回空列表，不抛错。
    """
    coll = _collection(force=True)
    if coll is None:
        return []
    thr = SIMILAR_DEFAULT_THRESHOLD if threshold is None else float(threshold)
    target = question.strip()
    try:
        res = coll.similarity_search_with_score(question, k=max(k + 2, 4))
    except Exception as e:  # noqa: BLE001
        logger.warning("相似问题推荐查询失败: %s", e)
        return []

    out: list[dict] = []
    for doc, distance in res:
        meta = doc.metadata or {}
        q = (meta.get("question") or doc.page_content or "").strip()
        if not q or q == target:
            continue
        score = _cosine_from_distance(distance)
        if score < thr:
            continue
        out.append({"question": q, "score": round(score, 4)})
        if len(out) >= k:
            break
    return out
