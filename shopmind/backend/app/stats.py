"""可观测性埋点：累计查询数、语义缓存命中、各阶段时延，供 /api/stats 与管理员面板展示。

数据存内存，服务重启清零（毕设规模足够；后续可换 Redis/InfluxDB 持久化）。
"""
import threading

_lock = threading.Lock()
_stats = {
    "total_queries": 0,
    "cache_hits": 0,
    "retrieval_errors": 0,
    "llm_errors": 0,
    "sum_latency_ms": 0.0,
    "sum_retrieval_ms": 0.0,
    "sum_llm_ms": 0.0,
}


def record_query(
    latency_ms: float,
    retrieval_ms: float = 0.0,
    llm_ms: float = 0.0,
    cached: bool = False,
    retrieval_error: bool = False,
    llm_error: bool = False,
) -> None:
    with _lock:
        _stats["total_queries"] += 1
        _stats["sum_latency_ms"] += latency_ms
        _stats["sum_retrieval_ms"] += retrieval_ms
        _stats["sum_llm_ms"] += llm_ms
        if cached:
            _stats["cache_hits"] += 1
        if retrieval_error:
            _stats["retrieval_errors"] += 1
        if llm_error:
            _stats["llm_errors"] += 1


def snapshot() -> dict:
    with _lock:
        s = dict(_stats)
    total = s["total_queries"]
    s["cache_hit_rate"] = round(s["cache_hits"] / total, 4) if total else 0.0
    s["avg_latency_ms"] = round(s["sum_latency_ms"] / total, 1) if total else 0.0
    s["avg_retrieval_ms"] = round(s["sum_retrieval_ms"] / total, 1) if total else 0.0
    s["avg_llm_ms"] = round(s["sum_llm_ms"] / total, 1) if total else 0.0
    return s
