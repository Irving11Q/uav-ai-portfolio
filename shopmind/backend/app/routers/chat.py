"""会话与问答（任意登录用户）：多会话管理、多轮问答、引用回显、历史持久化、
答案反馈回显、相似问题推荐、热问榜、对话导出、审计埋点。
"""
import datetime as _dt
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from .. import appconfig, audit, auth, demo, insights, models, ratelimit, schemas
from ..config import settings
from ..db import SessionLocal
from ..export import content_disposition, markdown_to_pdf_bytes, session_to_markdown
from ..rag import cache as qa_cache, qa

logger = logging.getLogger("rag")

router = APIRouter(prefix="/api/chat", tags=["chat"])


def _get_owned_session(session_id: int, user: models.User, database: Session):
    sess = database.query(models.Session).filter(models.Session.id == session_id).first()
    if not sess:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")
    if sess.user_id != user.id and user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无权访问该会话")
    return sess


def _sse(payload: dict) -> str:
    """把一个事件字典编码成 SSE 帧。

    只发 `data:` 行、靠 JSON 里的 `type` 字段区分事件类型，不额外用 `event:` 行 ——
    前端解析更简单，也不依赖 EventSource（我们要 POST + 自定义鉴权头，只能用 fetch 读流）。
    `json.dumps` 会把换行转义成 `\\n`，所以帧内不会出现裸换行，SSE 分帧是安全的。
    """
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def _refs_from_json(raw: str | None) -> list[schemas.ReferenceItem]:
    if not raw:
        return []
    try:
        return [schemas.ReferenceItem(**r) for r in json.loads(raw)]
    except Exception:  # noqa: BLE001
        return []


def _feedback_map(database: Session, session_id: int, user_id: int) -> dict[int, str]:
    """本会话里当前用户给出的反馈：{message_id: rating}，用于历史重载时回显赞踩状态。"""
    rows = (
        database.query(models.Feedback.message_id, models.Feedback.rating)
        .filter(
            models.Feedback.session_id == session_id,
            models.Feedback.user_id == user_id,
        )
        .all()
    )
    return {mid: rating for mid, rating in rows}


def _feedback_reason_map(database: Session, session_id: int, user_id: int) -> dict[int, str]:
    rows = (
        database.query(models.Feedback.message_id, models.Feedback.reason)
        .filter(
            models.Feedback.session_id == session_id,
            models.Feedback.user_id == user_id,
        )
        .all()
    )
    return {mid: reason for mid, reason in rows if reason}


@router.post("/sessions", response_model=schemas.ChatSessionOut, status_code=status.HTTP_201_CREATED)
def create_session(
    request: Request,
    body: schemas.ChatSessionCreate | None = None,
    user: models.User = Depends(auth.get_current_user),
    database: SessionLocal = Depends(auth.get_db),
):
    sess = models.Session(
        user_id=user.id, title=(body.title if body else None) or "新会话"
    )
    database.add(sess)
    database.commit()
    database.refresh(sess)
    audit.log(
        "session_create",
        user=user,
        detail=f"新建会话 #{sess.id}（{sess.title}）",
        target=f"session#{sess.id}",
        ip=audit.client_ip(request),
        database=database,
    )
    return sess


@router.get("/sessions", response_model=list[schemas.ChatSessionOut])
def list_sessions(
    user: models.User = Depends(auth.get_current_user),
    database: SessionLocal = Depends(auth.get_db),
):
    query = database.query(models.Session)
    if user.role != "admin":
        query = query.filter(models.Session.user_id == user.id)
    return query.order_by(models.Session.updated_at.desc()).all()


@router.delete("/sessions/{session_id}", status_code=status.HTTP_200_OK)
def delete_session(
    session_id: int,
    request: Request,
    user: models.User = Depends(auth.get_current_user),
    database: SessionLocal = Depends(auth.get_db),
):
    sess = _get_owned_session(session_id, user, database)
    title = sess.title
    database.query(models.Message).filter(models.Message.session_id == session_id).delete()
    database.query(models.Feedback).filter(models.Feedback.session_id == session_id).delete()
    database.delete(sess)
    database.commit()
    audit.log(
        "session_delete",
        user=user,
        detail=f"删除会话 #{session_id}（{title}）及其消息与反馈",
        target=f"session#{session_id}",
        ip=audit.client_ip(request),
        database=database,
    )
    return {"msg": "已删除", "id": session_id}


@router.get("/sessions/{session_id}/messages", response_model=list[schemas.ChatMessageOut])
def get_messages(
    session_id: int,
    user: models.User = Depends(auth.get_current_user),
    database: SessionLocal = Depends(auth.get_db),
):
    _get_owned_session(session_id, user, database)
    msgs = (
        database.query(models.Message)
        .filter(models.Message.session_id == session_id)
        .order_by(models.Message.created_at.asc(), models.Message.id.asc())
        .all()
    )
    fb = _feedback_map(database, session_id, user.id)
    fb_reason = _feedback_reason_map(database, session_id, user.id)
    return [
        schemas.ChatMessageOut(
            id=m.id,
            role=m.role,
            content=m.content,
            references=_refs_from_json(m.references_json),
            created_at=m.created_at,
            feedback=fb.get(m.id),
            feedback_reason=fb_reason.get(m.id),
        )
        for m in msgs
    ]


@router.post("/sessions/{session_id}/ask", response_model=schemas.ChatAskResult)
def ask(
    session_id: int,
    body: schemas.ChatAsk,
    request: Request,
    user: models.User = Depends(auth.get_current_user),
    database: SessionLocal = Depends(auth.get_db),
):
    sess = _get_owned_session(session_id, user, database)
    question = (body.question or "").strip()
    if not question:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="问题不能为空")

    # 每用户滑动窗口限流（企业级稳定性）：超限直接 429，不消耗大模型配额
    if not ratelimit.check_rate_limit(user.id):
        limit = int(
            appconfig.get("RATE_LIMIT_PER_USER_PER_MIN", settings.RATE_LIMIT_PER_USER_PER_MIN)
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"提问过于频繁，每分钟最多 {limit} 次，请稍后再试",
        )

    # 演示模式：公开体验站点的总额度上限。放在限流之后、调用大模型之前，
    # 避免额度用完还在花钱。（命中语义缓存的提问不占额度，见下方 consume）
    demo.ensure_available(database)

    # 取历史（不含当前这一轮），用于多轮上下文
    history_rows = (
        database.query(models.Message)
        .filter(models.Message.session_id == session_id)
        .order_by(models.Message.created_at.asc(), models.Message.id.asc())
        .all()
    )
    history = [
        {"role": m.role, "content": m.content}
        for m in history_rows
        if m.role in ("user", "assistant")
    ]

    # 保存用户消息
    user_msg = models.Message(session_id=session_id, role="user", content=question)
    database.add(user_msg)
    database.commit()

    # 调用 RAG 问答
    try:
        result = qa.answer(question, history=history)
    except RuntimeError as e:
        database.rollback()
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:  # noqa: BLE001
        database.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"问答失败：{type(e).__name__}: {e}",
        )

    # 保存助手消息 + 引用
    refs_json = [
        {"doc_id": r["doc_id"], "source": r["source"], "snippet": r["snippet"], "score": r["score"]}
        for r in result["references"]
    ]
    assistant_msg = models.Message(
        session_id=session_id,
        role="assistant",
        content=result["answer"],
        references_json=json.dumps(refs_json, ensure_ascii=False),
    )
    database.add(assistant_msg)
    sess.title = question[:30] if sess.title == "新会话" else sess.title
    sess.updated_at = _dt.datetime.utcnow()
    database.commit()
    database.refresh(assistant_msg)

    # 演示模式：只有真的调用了大模型才计数；命中缓存是复用已有答案，不该占额度
    if not result.get("cached"):
        demo.consume()

    # 审计：需求明确要「谁在何时问了什么」→ 问题原文落库
    audit.log(
        "ask",
        user=user,
        detail=(
            f"提问（命中缓存={bool(result.get('cached'))}，"
            f"引用 {len(refs_json)} 条，模型={appconfig.get('LLM_MODEL')}）"
        ),
        target=f"session#{session_id}",
        question=question,
        ip=audit.client_ip(request),
        extra={"cached": bool(result.get("cached")), "references": len(refs_json)},
        database=database,
    )

    return schemas.ChatAskResult(
        answer=result["answer"],
        references=[
            schemas.ReferenceItem(**r) for r in result["references"]
        ],
        session_id=session_id,
        message_id=assistant_msg.id,
        cached=result.get("cached", False),
    )


@router.post("/sessions/{session_id}/ask/stream")
def ask_stream(
    session_id: int,
    body: schemas.ChatAsk,
    request: Request,
    user: models.User = Depends(auth.get_current_user),
    database: SessionLocal = Depends(auth.get_db),
):
    """流式问答（SSE）：引用先推、答案逐字推。

    与同步版 `/ask` **完全等价** —— 同一套限流、演示额度、审计、落库、语义缓存、埋点，
    区别只是答案边生成边推给前端，用户不用干等 8 秒才看到第一个字。

    事件序列（前端按 `type` 字段分派）：
        {"type":"meta",       "cached":bool}
        {"type":"references", "references":[...]}       ← 检索完立即推，早于答案
        {"type":"delta",      "text":"..."}             × n
        {"type":"done",       "cached","latency_ms","ttft_ms"}
        {"type":"final",      "message_id":int,...}     ← 落库成功后才有（前端拿它绑定反馈按钮）
        {"type":"error",      "message":"..."}
    """
    sess = _get_owned_session(session_id, user, database)
    question = (body.question or "").strip()
    if not question:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="问题不能为空")

    # 限流 / 额度：与同步版一致，且都放在「开始流」之前 —— 一旦开始推流就没法再改状态码了
    if not ratelimit.check_rate_limit(user.id):
        limit = int(
            appconfig.get("RATE_LIMIT_PER_USER_PER_MIN", settings.RATE_LIMIT_PER_USER_PER_MIN)
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"提问过于频繁，每分钟最多 {limit} 次，请稍后再试",
        )
    demo.ensure_available(database)

    history_rows = (
        database.query(models.Message)
        .filter(models.Message.session_id == session_id)
        .order_by(models.Message.created_at.asc(), models.Message.id.asc())
        .all()
    )
    history = [
        {"role": m.role, "content": m.content}
        for m in history_rows
        if m.role in ("user", "assistant")
    ]

    user_msg = models.Message(session_id=session_id, role="user", content=question)
    database.add(user_msg)
    database.commit()

    # 只带「标量」进生成器：流式期间请求级 session 的生命周期不可控，
    # ORM 实例一旦 detach，再访问属性会抛 DetachedInstanceError。
    user_id = user.id
    client_ip = audit.client_ip(request)

    def gen():
        # 生成器自己开 session —— 不复用请求级的那个，避免它被提前关闭
        db = SessionLocal()
        parts: list[str] = []
        refs: list[dict] = []
        done: dict = {}
        try:
            for ev in qa.answer_stream(question, history=history):
                kind = ev["type"]
                if kind == "delta":
                    parts.append(ev["text"])
                elif kind == "references":
                    refs = ev["references"]
                elif kind == "done":
                    done = ev
                yield _sse(ev)

            # 流干净结束后才落库：助手消息 + 引用（与同步版字段完全一致）
            refs_json = [
                {"doc_id": r["doc_id"], "source": r["source"], "snippet": r["snippet"], "score": r["score"]}
                for r in refs
            ]
            assistant_msg = models.Message(
                session_id=session_id,
                role="assistant",
                content="".join(parts).strip(),
                references_json=json.dumps(refs_json, ensure_ascii=False),
            )
            db.add(assistant_msg)
            row = db.query(models.Session).filter(models.Session.id == session_id).first()
            if row:
                if row.title == "新会话":
                    row.title = question[:30]
                row.updated_at = _dt.datetime.utcnow()
            db.commit()
            db.refresh(assistant_msg)

            # 命中缓存不占演示额度（复用已有答案，没花钱）
            if not done.get("cached"):
                demo.consume()

            audit.log(
                "ask",
                user=db.get(models.User, user_id),
                detail=(
                    f"流式提问（命中缓存={bool(done.get('cached'))}，引用 {len(refs_json)} 条，"
                    f"模型={appconfig.get('LLM_MODEL')}，首字 {done.get('ttft_ms')}ms）"
                ),
                target=f"session#{session_id}",
                question=question,
                ip=client_ip,
                extra={
                    "cached": bool(done.get("cached")),
                    "references": len(refs_json),
                    "stream": True,
                    "ttft_ms": done.get("ttft_ms"),
                    "latency_ms": done.get("latency_ms"),
                },
                database=db,
            )

            yield _sse(
                {
                    "type": "final",
                    "message_id": assistant_msg.id,
                    "cached": bool(done.get("cached")),
                    "latency_ms": done.get("latency_ms"),
                    "ttft_ms": done.get("ttft_ms"),
                }
            )
        except Exception as e:  # noqa: BLE001
            # 已经推出去的内容收不回，只能把错误当最后一个事件发出去让前端提示
            db.rollback()
            logger.warning("流式问答失败: %s", e)
            yield _sse({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            db.close()

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # 关键：告诉 nginx / 云平台网关「别缓冲」。
            # 少了这个头，网关会把整条流攒完再一次性吐，流式等于白做。
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/similar-questions", response_model=list[schemas.SimilarQuestion])
def similar_questions(
    question: str = Query(..., min_length=1, description="以此为基准找相似历史问题"),
    k: int = Query(3, ge=1, le=10),
    _: models.User = Depends(auth.get_current_user),
):
    """相似问题推荐：在历史问题池里做语义近邻查询（无数据时返回空列表）。"""
    items = qa_cache.similar_questions(question, k=k)
    return [schemas.SimilarQuestion(**it) for it in items]


@router.get("/hot-questions", response_model=list[schemas.HotQuestion])
def hot_questions(
    _: models.User = Depends(auth.get_current_user),
    database: SessionLocal = Depends(auth.get_db),
    limit: int = Query(8, ge=1, le=50),
    days: int = Query(30, ge=1, le=365),
):
    """热问榜（所有登录用户可见，帮助用户快速找到常见问题）。"""
    return [schemas.HotQuestion(**h) for h in insights.hot_questions(database, limit=limit, days=days)]


@router.get("/sessions/{session_id}/export")
def export_session(
    session_id: int,
    request: Request,
    fmt: str = Query("md", alias="format", pattern="^(md|pdf)$"),
    user: models.User = Depends(auth.get_current_user),
    database: SessionLocal = Depends(auth.get_db),
):
    """导出整段对话为 Markdown 或 PDF。"""
    sess = _get_owned_session(session_id, user, database)
    rows = (
        database.query(models.Message)
        .filter(models.Message.session_id == session_id)
        .order_by(models.Message.created_at.asc(), models.Message.id.asc())
        .all()
    )
    if not rows:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="该会话还没有内容，无法导出")

    fb = _feedback_map(database, session_id, user.id)
    fb_reason = _feedback_reason_map(database, session_id, user.id)
    messages = [
        {
            "role": m.role,
            "content": m.content,
            "references": [r.model_dump() for r in _refs_from_json(m.references_json)],
            "feedback": fb.get(m.id),
            "feedback_reason": fb_reason.get(m.id),
        }
        for m in rows
    ]

    markdown_text = session_to_markdown(sess, messages)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = (sess.title or "会话").replace("/", "_").replace("\\", "_")[:30]

    if fmt == "pdf":
        try:
            pdf_bytes = markdown_to_pdf_bytes(sess.title or "会话导出", markdown_text)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"PDF 生成失败：{type(e).__name__}: {e}",
            )
        audit.log(
            "session_export",
            user=user,
            detail=f"导出会话 #{session_id} 为 PDF（{len(pdf_bytes)} 字节）",
            target=f"session#{session_id}",
            ip=audit.client_ip(request),
            database=database,
        )
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": content_disposition(f"{safe_title}_{stamp}.pdf")
            },
        )

    audit.log(
        "session_export",
        user=user,
        detail=f"导出会话 #{session_id} 为 Markdown（{len(markdown_text)} 字）",
        target=f"session#{session_id}",
        ip=audit.client_ip(request),
        database=database,
    )
    return Response(
        content=markdown_text.encode("utf-8"),
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": content_disposition(f"{safe_title}_{stamp}.md")
        },
    )
