"""Chroma 向量库封装（基于 langchain_chroma）。

- 单 collection 存全部知识库片段，metadata 带 doc_id / source 便于按文档删除与回显。
- client 持久化到 settings.CHROMA_DIR，无需独立服务。
"""
from functools import lru_cache

from langchain_chroma import Chroma

from ..config import settings
from .embeddings import get_embeddings


def get_chroma_client():
    """返回持久化的 Chroma 客户端（进程内，无需独立服务）。"""
    import chromadb

    settings.ensure_dirs()
    return chromadb.PersistentClient(path=str(settings.CHROMA_DIR))


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    client = get_chroma_client()
    return Chroma(
        client=client,
        collection_name=settings.VECTOR_COLLECTION,
        embedding_function=get_embeddings(),
    )


def add_document_chunks(
    doc_id: int, source: str, chunk_ids: list[int], chunks: list[str]
) -> int:
    """把一篇文档的分块写入向量库，返回写入条数。

    chunk_ids 为 SQLite chunks 表的主键，写入 metadata 以便混合检索按 chunk_id 融合。
    """
    vs = get_vectorstore()
    metadatas = [
        {
            "doc_id": doc_id,
            "source": source,
            "chunk_index": i,
            "chunk_id": chunk_ids[i],
        }
        for i in range(len(chunks))
    ]
    ids = [f"doc-{doc_id}-{chunk_ids[i]}" for i in range(len(chunks))]
    vs.add_texts(texts=chunks, metadatas=metadatas, ids=ids)
    return len(chunks)


def delete_document_chunks(doc_id: int) -> None:
    """删除某篇文档的全部分块。"""
    vs = get_vectorstore()
    try:
        vs.delete(where={"doc_id": doc_id})
    except Exception:
        # 集合可能为空，忽略
        pass


def search(query: str, top_k: int | None = None) -> list[dict]:
    """语义检索，返回 [{text, doc_id, source, chunk_index, score}, ...]。

    score 为余弦距离（越小越相关），这里转为「相似度」= 1 - distance 便于前端展示。
    """
    vs = get_vectorstore()
    top_k = top_k or settings.RETRIEVE_TOP_K
    results = vs.similarity_search_with_score(query, k=top_k)
    out = []
    for doc, distance in results:
        meta = doc.metadata or {}
        out.append(
            {
                "text": doc.page_content,
                "doc_id": meta.get("doc_id", 0),
                "source": meta.get("source", ""),
                "chunk_index": meta.get("chunk_index", 0),
                # chunk_id 为 SQLite chunks 表主键，混合检索按它与关键词召回融合
                "chunk_id": meta.get("chunk_id"),
                "score": round(1.0 - float(distance), 4),
            }
        )
    return out
