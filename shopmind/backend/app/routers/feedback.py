"""答案反馈：赞/踩 + 原因 + 备注，沉淀 bad case 供调优。

## 为什么要把 question/answer 冗余存一份
看 bad case 的人需要「问题 + 答案 + 引用」一起看，才能判断是检索错了还是生成错了。
如果每次列表查询都去连 messages → sessions，不但慢，而且用户删掉会话后
bad case 也跟着消失 —— 而 bad case 恰恰是最不该丢的调优素材。所以下单时就快照一份。
"""
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import func

from .. import audit, auth, models, schemas
from ..db import SessionLocal

logger = logging.getLogger("rag")

router = APIRouter(prefix="/api/feedback", tags=["feedback"])

# 预设的「踩」原因，前端下拉直接取这份，避免前后端各写一套
DOWN_REASONS = [
    "答案不准确",
    "答非所问",
    "引用不相关",
    "知识库中没有相关内容",
    "内容过时",
    "格式混乱",
    "其他",
]


def _find_question(database, session_id: int, assistant_msg_id: int) -> str | None:
    """找出这条助手回答对应的用户问题：取同一会话里它之前最近的一条 user 消息。"""
    row = (
        database.query(models.Message)
        .filter(
            models.Message.session_id == session_id,
            models.Message.id < assistant_msg_id,
            models.Message.role == "user",
        )
        .order_by(models.Message.id.desc())
        .first()
    )
    return row.content if row else None


@router.get("/reasons", response_model=list[str])
def down_reasons(_: models.User = Depends(auth.get_current_user)):
    """「踩」的可选原因（前端下拉用）。"""
    return DOWN_REASONS


@router.post("", response_model=schemas.FeedbackOut, status_code=status.HTTP_200_OK)
def submit_feedback(
    body: schemas.FeedbackCreate,
    request: Request,
    user: models.User = Depends(auth.get_current_user),
    database: SessionLocal = Depends(auth.get_db),
):
    """提交或修改反馈。同一条消息同一个用户只保留一条记录（改判即覆盖）。"""
    rating = (body.rating or "").strip().lower()
    if rating not in ("up", "down"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="rating 只能是 up 或 down")

    msg = (
        database.query(models.Message)
        .filter(models.Message.id == body.message_id)
        .first()
    )
    if not msg:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="消息不存在")
    if msg.role != "assistant":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="只能对助手回答做反馈")

    sess = database.query(models.Session).filter(models.Session.id == msg.session_id).first()
    if not sess:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")
    # 只有会话属主（或 admin）能评价这条回答
    if sess.user_id != user.id and user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无权评价该回答")

    row = (
        database.query(models.Feedback)
        .filter(
            models.Feedback.message_id == body.message_id,
            models.Feedback.user_id == user.id,
        )
        .first()
    )
    question = _find_question(database, sess.id, msg.id)

    if row is None:
        row = models.Feedback(message_id=body.message_id, user_id=user.id)
        database.add(row)

    row.session_id = sess.id
    row.rating = rating
    row.reason = (body.reason or "").strip() or None
    row.comment = (body.comment or "").strip() or None
    row.question = question
    row.answer = msg.content
    row.references_json = msg.references_json
    # 改判为「赞」时清掉处理标记，避免已修复的 bad case 混在待处理列表里
    if rating == "up":
        row.resolved = 0
        row.resolved_note = None
    database.commit()
    database.refresh(row)

    if rating == "down":
        audit.log(
            "feedback",
            user=user,
            detail=f"标记回答不满意（{row.reason or '未填原因'}）",
            target=f"message#{row.message_id}",
            question=question,
            ip=audit.client_ip(request),
            database=database,
        )

    out = schemas.FeedbackOut.model_validate(row)
    out.username = user.username
    return out


@router.get("/summary", response_model=schemas.FeedbackSummary)
def summary(
    _: models.User = Depends(auth.require_admin),
    database: SessionLocal = Depends(auth.get_db),
):
    """满意度汇总（管理员）。"""
    up = database.query(func.count(models.Feedback.id)).filter(models.Feedback.rating == "up").scalar() or 0
    down = (
        database.query(func.count(models.Feedback.id))
        .filter(models.Feedback.rating == "down")
        .scalar()
        or 0
    )
    total = up + down
    breakdown_rows = (
        database.query(models.Feedback.reason, func.count(models.Feedback.id))
        .filter(models.Feedback.rating == "down")
        .group_by(models.Feedback.reason)
        .order_by(func.count(models.Feedback.id).desc())
        .all()
    )
    return schemas.FeedbackSummary(
        up=up,
        down=down,
        total=total,
        satisfaction_rate=round(up / total, 4) if total else 0.0,
        reason_breakdown=[
            {"reason": r or "未填原因", "count": c} for r, c in breakdown_rows
        ],
    )


@router.get("/bad-cases", response_model=dict)
def bad_cases(
    _: models.User = Depends(auth.require_admin),
    database: SessionLocal = Depends(auth.get_db),
    rating: str | None = Query(None, description="up / down，留空表示全部"),
    keyword: str | None = Query(None, description="按问题或备注模糊搜索"),
    only_unresolved: bool = Query(False, description="只看未处理的"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """反馈列表（管理员），默认按时间倒序。bad case 调优主要看 rating=down。"""
    query = database.query(models.Feedback)
    if rating in ("up", "down"):
        query = query.filter(models.Feedback.rating == rating)
    if only_unresolved:
        query = query.filter(models.Feedback.resolved == 0)
    if keyword:
        like = f"%{keyword}%"
        query = query.filter(
            (models.Feedback.question.like(like))
            | (models.Feedback.comment.like(like))
            | (models.Feedback.answer.like(like))
        )

    total = query.count()
    rows = (
        query.order_by(models.Feedback.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    # 补用户名：一次查出本页涉及的用户，避免 N+1
    user_ids = {r.user_id for r in rows if r.user_id}
    names = {}
    if user_ids:
        names = {
            u.id: u.username
            for u in database.query(models.User).filter(models.User.id.in_(user_ids)).all()
        }

    items = []
    for r in rows:
        item = schemas.FeedbackOut.model_validate(r).model_dump()
        item["username"] = names.get(r.user_id)
        item["resolved"] = bool(r.resolved)
        item["resolved_note"] = r.resolved_note
        items.append(item)

    return {"total": total, "page": page, "page_size": page_size, "items": items}


@router.post("/{feedback_id}/resolve", response_model=dict)
def resolve(
    feedback_id: int,
    body: dict | None = None,
    _: models.User = Depends(auth.require_admin),
    database: SessionLocal = Depends(auth.get_db),
):
    """标记 bad case 已处理（可带处理备注），让反馈形成闭环。"""
    row = database.query(models.Feedback).filter(models.Feedback.id == feedback_id).first()
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="反馈不存在")
    body = body or {}
    row.resolved = 1 if body.get("resolved", True) else 0
    row.resolved_note = body.get("note") or row.resolved_note
    database.commit()
    return {"msg": "已更新", "id": feedback_id, "resolved": bool(row.resolved)}
