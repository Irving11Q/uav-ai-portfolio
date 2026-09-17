"""知识库统计仪表盘（管理员）。

指标分两个口径，接口里显式标注，避免看数的人误读：
- **DB 口径**：文档/分块/用户/会话/消息/提问数、反馈数 —— 永久累积，重启不丢。
- **内存口径**：查询数、缓存命中、平均时延、错误数 —— 来自 `stats.py` 的进程内埋点，**服务重启清零**。
  返回体里用 `memory_scope_keys` 明确列出哪些是内存口径，前端据此显示「重启清零」提示。
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy import func

from .. import audit, auth, insights, models, schemas, stats as stats_mod
from ..db import SessionLocal

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

# 内存埋点口径的字段（前端用来加「重启清零」标记）
_MEMORY_KEYS = [
    "total_queries",
    "cache_hits",
    "cache_hit_rate",
    "avg_latency_ms",
    "avg_retrieval_ms",
    "avg_llm_ms",
    "retrieval_errors",
    "llm_errors",
]


def _count(database, model, *filters) -> int:
    query = database.query(func.count(model.id))
    for f in filters:
        query = query.filter(f)
    return int(query.scalar() or 0)


@router.get("", response_model=schemas.DashboardStats)
def dashboard(
    _: models.User = Depends(auth.require_admin),
    database: SessionLocal = Depends(auth.get_db),
):
    """仪表盘核心指标（管理员）。"""
    documents = _count(database, models.Document)
    ready = _count(database, models.Document, models.Document.status == "ready")
    failed = _count(database, models.Document, models.Document.status == "failed")
    char_count = int(
        database.query(func.coalesce(func.sum(models.Document.char_count), 0)).scalar() or 0
    )
    chunks = _count(database, models.Chunk)
    users = _count(database, models.User)
    sessions = _count(database, models.Session)
    messages = _count(database, models.Message)
    questions = _count(database, models.Message, models.Message.role == "user")

    fb_up = _count(database, models.Feedback, models.Feedback.rating == "up")
    fb_down = _count(database, models.Feedback, models.Feedback.rating == "down")
    fb_total = fb_up + fb_down

    mem = stats_mod.snapshot()

    return schemas.DashboardStats(
        documents=documents,
        ready_documents=ready,
        failed_documents=failed,
        chunks=chunks,
        char_count=char_count,
        users=users,
        sessions=sessions,
        messages=messages,
        questions=questions,
        total_queries=int(mem.get("total_queries", 0)),
        cache_hits=int(mem.get("cache_hits", 0)),
        cache_hit_rate=float(mem.get("cache_hit_rate", 0.0)),
        avg_latency_ms=float(mem.get("avg_latency_ms", 0.0)),
        avg_retrieval_ms=float(mem.get("avg_retrieval_ms", 0.0)),
        avg_llm_ms=float(mem.get("avg_llm_ms", 0.0)),
        retrieval_errors=int(mem.get("retrieval_errors", 0)),
        llm_errors=int(mem.get("llm_errors", 0)),
        feedback_up=fb_up,
        feedback_down=fb_down,
        satisfaction_rate=round(fb_up / fb_total, 4) if fb_total else 0.0,
        memory_scope_keys=_MEMORY_KEYS,
    )


@router.get("/trend", response_model=list[schemas.TrendPoint])
def trend(
    _: models.User = Depends(auth.require_admin),
    database: SessionLocal = Depends(auth.get_db),
    days: int = Query(7, ge=1, le=90),
):
    """近 N 天提问量趋势（管理员）。"""
    return [schemas.TrendPoint(**p) for p in insights.question_trend(database, days=days)]


@router.get("/hot-questions", response_model=list[schemas.HotQuestion])
def hot(
    _: models.User = Depends(auth.require_admin),
    database: SessionLocal = Depends(auth.get_db),
    limit: int = Query(10, ge=1, le=50),
    days: int = Query(30, ge=1, le=365),
):
    """热问榜（管理员口径，可调时间窗）。"""
    return [schemas.HotQuestion(**h) for h in insights.hot_questions(database, limit=limit, days=days)]


@router.get("/documents")
def document_breakdown(
    _: models.User = Depends(auth.require_admin),
    database: SessionLocal = Depends(auth.get_db),
):
    """文档级明细：每篇文档的分块数与字数，用于仪表盘下钻。"""
    rows = (
        database.query(
            models.Document.id,
            models.Document.filename,
            models.Document.status,
            models.Document.chunk_count,
            models.Document.char_count,
            models.Document.created_at,
        )
        .order_by(models.Document.id.asc())
        .all()
    )
    return [
        {
            "id": r[0],
            "filename": r[1],
            "status": r[2],
            "chunk_count": r[3] or 0,
            "char_count": r[4] or 0,
            "created_at": r[5],
        }
        for r in rows
    ]


@router.get("/actions")
def action_breakdown(
    _: models.User = Depends(auth.require_admin),
    database: SessionLocal = Depends(auth.get_db),
):
    """按动作类型统计审计日志条数（谁在用、在干什么，一眼看出使用重心）。"""
    rows = (
        database.query(models.AuditLog.action, func.count(models.AuditLog.id))
        .group_by(models.AuditLog.action)
        .order_by(func.count(models.AuditLog.id).desc())
        .all()
    )
    return [
        {"action": a, "label": audit.ACTION_LABELS.get(a, a), "count": c} for a, c in rows
    ]
