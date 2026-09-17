"""限流与并发控制（企业级稳定性）。

- llm_slot：大模型调用的并发信号量，避免瞬时高并发打满 API 配额。
- check_rate_limit：每用户滑动窗口限流（默认每分钟 N 次问答）。
"""
import threading
import time

from .config import settings

# 大模型调用并发信号量（同步端点运行在线程池，用线程信号量即可）
_llm_sem = threading.BoundedSemaphore(max(1, settings.LLM_MAX_CONCURRENCY))


def llm_slot() -> threading.BoundedSemaphore:
    """上下文管理器：with ratelimit.llm_slot(): 包裹大模型调用。"""
    return _llm_sem


_rl_lock = threading.Lock()
_user_hits: dict[int, list[float]] = {}


def check_rate_limit(user_id: int, limit: int | None = None) -> bool:
    """返回 True 表示放行；False 表示已超限（调用方应返回 429）。

    采用 60 秒滑动窗口统计该用户的问答次数。
    限额可由管理员在后台调整（运行时配置），改完立即生效。
    """
    if limit is None:
        from . import appconfig

        limit = int(
            appconfig.get("RATE_LIMIT_PER_USER_PER_MIN", settings.RATE_LIMIT_PER_USER_PER_MIN)
        )
    now = time.time()
    with _rl_lock:
        hits = _user_hits.get(user_id, [])
        hits = [t for t in hits if now - t < 60]
        if len(hits) >= limit:
            _user_hits[user_id] = hits
            return False
        hits.append(now)
        _user_hits[user_id] = hits
        return True
