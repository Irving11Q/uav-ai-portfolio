"""审计日志写入（best-effort）。

原则：**审计不能反过来搞挂主流程**。写日志失败只记 warning，绝不抛异常影响业务。

用法二选一：
- 传 `database`（复用请求里的 session）→ 与业务同事务提交，最省连接；
- 不传 → 自己开一个短 session 提交，适用于登录/注册这类已有自己事务的场景。
"""
import json
import logging
from typing import Any

logger = logging.getLogger("rag")

# 动作 -> 中文标签（前端审计页直接展示，避免前端再维护一份映射）
ACTION_LABELS: dict[str, str] = {
    "login": "登录",
    "login_failed": "登录失败",
    "register": "注册",
    "change_password": "修改密码",
    "ask": "提问",
    "session_create": "新建会话",
    "session_delete": "删除会话",
    "session_export": "导出会话",
    "upload": "上传文档",
    "delete_doc": "删除文档",
    "reindex": "重索引",
    "download_doc": "下载文档",
    "feedback": "答案反馈",
    "config_update": "修改配置",
    "config_reset": "恢复默认配置",
    "ocr": "OCR 预处理",
}


def log(
    action: str,
    user: Any = None,
    detail: str = "",
    target: str | None = None,
    question: str | None = None,
    ip: str | None = None,
    extra: dict | None = None,
    database: Any = None,
) -> None:
    """写一条审计记录。`user` 传 models.User 实例（可为 None，如登录失败）。"""
    try:
        from . import models

        row = models.AuditLog(
            user_id=getattr(user, "id", None),
            username=getattr(user, "username", None),
            role=getattr(user, "role", None),
            action=action,
            detail=detail,
            target=target,
            question=question,
            ip=ip,
            extra_json=json.dumps(extra, ensure_ascii=False) if extra else None,
        )

        if database is not None:
            database.add(row)
            database.commit()
            return

        from .db import SessionLocal

        own = SessionLocal()
        try:
            own.add(row)
            own.commit()
        finally:
            own.close()
    except Exception as e:  # noqa: BLE001
        logger.warning("写审计日志失败（忽略）: %s", e)


def client_ip(request: Any) -> str | None:
    """尽力取客户端 IP（兼容反向代理头）。取不到返回 None，不报错。"""
    try:
        if request is None:
            return None
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else None
    except Exception:  # noqa: BLE001
        return None
