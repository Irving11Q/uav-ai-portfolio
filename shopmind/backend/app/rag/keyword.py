"""关键词检索（纯 Python BM25，零额外依赖）。

- 把每个文档分块的分词结果建内存索引，问答时按 BM25 打分召回。
- 与向量检索互补：向量擅长语义泛化，关键词擅长精确词面匹配（型号、规格、专名）。
- 索引在首次查询时惰性构建，文档变更后由 documents 模块调用 invalidate() 失效重建。
"""
import math
import re
import threading

from .. import models
from ..config import settings
from ..db import SessionLocal

# 英文/数字词 + 单个 CJK 汉字作为 token（中文按字切，简单但足够做关键词命中）
_token_re = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]")

BM25_K1 = 1.5
BM25_B = 0.75

_index_lock = threading.Lock()
_index = None  # {"docs": {chunk_id: {...}}, "df": {...}, "n": int, "avgdl": float}


def _tokenize(text: str) -> list[str]:
    toks: list[str] = []
    for m in _token_re.findall((text or "").lower()):
        if re.match(r"[a-zA-Z0-9]+", m):
            toks.append(m)
        else:
            # CJK：逐字作为 token
            toks.extend(list(m))
    return toks


def invalidate() -> None:
    """文档增删改后调用，使内存索引失效，下次查询重建。"""
    global _index
    with _index_lock:
        _index = None


def _build() -> None:
    global _index
    database = SessionLocal()
    try:
        rows = database.query(models.Chunk).all()
        docs: dict[int, dict] = {}
        df: dict[str, int] = {}
        for c in rows:
            toks = _tokenize(c.content or "")
            docs[c.id] = {
                "doc_id": c.doc_id,
                "source": c.source,
                "text": c.content or "",
                "tokens": toks,
            }
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        n = len(docs)
        avgdl = (sum(len(d["tokens"]) for d in docs.values()) / n) if n else 0
        _index = {"docs": docs, "df": df, "n": n, "avgdl": avgdl}
    finally:
        database.close()


def _ensure() -> None:
    global _index
    if _index is None:
        with _index_lock:
            if _index is None:
                _build()


def search(query: str, top_k: int = 20) -> list[dict]:
    """返回 [{chunk_id, doc_id, source, text, score}, ...]，score 为 BM25 分。"""
    _ensure()
    if not _index or _index["n"] == 0:
        return []
    idx = _index
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []
    scores: dict[int, float] = {}
    n = idx["n"]
    avgdl = idx["avgdl"]
    for qt in set(q_tokens):
        if qt not in idx["df"]:
            continue
        idf = math.log((n - idx["df"][qt] + 0.5) / (idx["df"][qt] + 0.5) + 1)
        for cid, doc in idx["docs"].items():
            f = doc["tokens"].count(qt)
            if f == 0:
                continue
            dl = len(doc["tokens"])
            denom = f + BM25_K1 * (1 - BM25_B + BM25_B * dl / avgdl)
            scores[cid] = scores.get(cid, 0.0) + idf * (f * (BM25_K1 + 1)) / denom
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {
            "chunk_id": cid,
            "doc_id": idx["docs"][cid]["doc_id"],
            "source": idx["docs"][cid]["source"],
            "text": idx["docs"][cid]["text"],
            "score": round(sc, 4),
        }
        for cid, sc in ranked
    ]
