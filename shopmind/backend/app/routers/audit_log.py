"""审计日志查询（管理员）：谁在何时做了什么，支持按人/动作/关键词/时间范围过滤。

命名用 audit_log.py 而非 audit.py，刻意与写入模块 `app.audit` 区分开，避免读代码时混淆。
"""
import csv
import io
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy import func

from .. import audit, auth, models, schemas
from ..db import SessionLocal

router = APIRouter(prefix="/api/audit-logs", tags=["audit"])


def _apply_filters(query, username, action, keyword, days):
    if username:
        query = query.filter(models.AuditLog.username == username)
    if action:
        query = query.filter(models.AuditLog.action == action)
    if days:
        since = datetime.utcnow() - timedelta(days=days)
        query = query.filter(models.AuditLog.created_at >= since)
    if keyword:
        like = f"%{keyword}%"
        query = query.filter(
            (models.AuditLog.question.like(like))
            | (models.AuditLog.detail.like(like))
            | (models.AuditLog.target.like(like))
        )
    return query


def _row_to_dict(r: models.AuditLog) -> dict:
    return {
        "id": r.id,
        "user_id": r.user_id,
        "username": r.username,
        "role": r.role,
        "action": r.action,
        "action_label": audit.ACTION_LABELS.get(r.action, r.action),
        "detail": r.detail,
        "target": r.target,
        "question": r.question,
        "ip": r.ip,
        "created_at": r.created_at,
    }


@router.get("", response_model=dict)
def list_logs(
    _: models.User = Depends(auth.require_admin),
    database: SessionLocal = Depends(auth.get_db),
    username: str | None = Query(None, description="按用户名精确过滤"),
    action: str | None = Query(None, description="按动作过滤，如 ask / login"),
    keyword: str | None = Query(None, description="在问题内容/描述/对象里模糊搜索"),
    days: int | None = Query(None, ge=1, le=365, description="只看最近 N 天"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
):
    """分页查询审计日志（时间倒序）。"""
    query = _apply_filters(
        database.query(models.AuditLog), username, action, keyword, days
    )
    total = query.count()
    rows = (
        query.order_by(models.AuditLog.created_at.desc(), models.AuditLog.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": [_row_to_dict(r) for r in rows],
    }


@router.get("/actions", response_model=list)
def action_options(
    _: models.User = Depends(auth.require_admin),
    database: SessionLocal = Depends(auth.get_db),
):
    """动作下拉选项（只列出实际出现过的动作）。"""
    rows = (
        database.query(models.AuditLog.action, func.count(models.AuditLog.id))
        .group_by(models.AuditLog.action)
        .order_by(func.count(models.AuditLog.id).desc())
        .all()
    )
    return [
        {"action": a, "label": audit.ACTION_LABELS.get(a, a), "count": c} for a, c in rows
    ]


@router.get("/users", response_model=list)
def user_options(
    _: models.User = Depends(auth.require_admin),
    database: SessionLocal = Depends(auth.get_db),
):
    """用户名下拉选项。"""
    rows = (
        database.query(models.AuditLog.username, func.count(models.AuditLog.id))
        .group_by(models.AuditLog.username)
        .order_by(func.count(models.AuditLog.id).desc())
        .all()
    )
    return [{"username": u or "(未登录)", "count": c} for u, c in rows]


@router.get("/export")
def export_logs(
    _: models.User = Depends(auth.require_admin),
    database: SessionLocal = Depends(auth.get_db),
    username: str | None = Query(None),
    action: str | None = Query(None),
    keyword: str | None = Query(None),
    days: int | None = Query(None, ge=1, le=365),
    max_rows: int = Query(5000, ge=1, le=50000),
):
    """把当前过滤条件下的审计日志导成 CSV（留档 / 给答辩材料做证据）。

    用 utf-8-sig 编码，Excel 直接双击打开不会中文乱码。
    """
    query = _apply_filters(
        database.query(models.AuditLog), username, action, keyword, days
    )
    rows = (
        query.order_by(models.AuditLog.created_at.desc())
        .limit(max_rows)
        .all()
    )

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["时间", "用户", "角色", "动作", "问题内容", "描述", "对象", "IP"])
    for r in rows:
        writer.writerow(
            [
                r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else "",
                r.username or "",
                r.role or "",
                audit.ACTION_LABELS.get(r.action, r.action),
                r.question or "",
                r.detail or "",
                r.target or "",
                r.ip or "",
            ]
        )

    data = buf.getvalue().encode("utf-8-sig")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Response(
        content=data,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="audit_logs_{stamp}.csv"',
            "X-Total-Rows": str(len(rows)),
        },
    )
