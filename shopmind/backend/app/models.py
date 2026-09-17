from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from .db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, index=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    role = Column(String(16), default="user", nullable=False)  # admin | user
    created_at = Column(DateTime, default=datetime.utcnow)


# ---- 后续阶段使用的表，先建好结构，避免日后迁移 ----
class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"))
    filename = Column(String(255), nullable=False)
    file_type = Column(String(32))
    status = Column(String(32), default="processing")  # processing|ready|failed
    chunk_count = Column(Integer, default=0)
    char_count = Column(Integer, default=0)
    error = Column(Text)
    meta_json = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Chunk(Base):
    """文档分块后的文本片段，单独落库以支持关键词(BM25)检索与混合检索融合。"""

    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)
    doc_id = Column(Integer, ForeignKey("documents.id"), index=True)
    chunk_index = Column(Integer)
    source = Column(String(255))
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    title = Column(String(255), default="新会话")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), index=True)
    role = Column(String(16))  # user | assistant | system
    content = Column(Text)
    references_json = Column(Text)  # 引用的知识库片段（JSON 数组）
    created_at = Column(DateTime, default=datetime.utcnow)


# ---- 阶段6：反馈 / 审计 / 运行时配置 ----
# 注意：这三张表都是「新增表」，不改动上面任何已有表结构。
# 因为库里已经有真实数据（用户、会话、消息、文档、分块），不能靠删库重建来迁移。
class Feedback(Base):
    """答案反馈（赞/踩）：沉淀 bad case 用于调优。

    设计要点：
    - 挂在 message_id 上（助手消息），一个用户对同一条消息只有一条有效反馈（改判时覆盖）。
    - question / answer / references_json 冗余存一份：bad case 列表要连着问题与答案一起看，
      冗余后列表查询不用连 messages、sessions 两张表，也不怕会话被删。
    """

    __tablename__ = "feedbacks"

    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(Integer, ForeignKey("messages.id"), index=True, nullable=False)
    session_id = Column(Integer, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    rating = Column(String(8), nullable=False)  # up | down
    reason = Column(String(64))  # 预设原因：答案不准确 / 引用不相关 / 答非所问 / 未找到内容 / 其他
    comment = Column(Text)  # 用户补充说明
    question = Column(Text)  # 冗余：对应的问题原文
    answer = Column(Text)  # 冗余：被评价的答案
    references_json = Column(Text)  # 冗余：当时的引用片段
    resolved = Column(Integer, default=0)  # 0 未处理 / 1 已处理（bad case 闭环标记）
    resolved_note = Column(Text)  # 处理备注（例如「已在 v2 分块后修复」）
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AuditLog(Base):
    """审计日志：谁在何时做了什么。

    需求「admin 查看谁在何时问了什么」→ 提问类动作把问题原文落在 question 字段，
    便于按关键词直接检索问题内容，不用去 messages 里反查。
    """

    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    username = Column(String(64), index=True)
    role = Column(String(16))
    action = Column(String(32), index=True)  # login|register|ask|upload|delete_doc|...
    detail = Column(Text)  # 人类可读描述
    target = Column(String(255))  # 目标对象（会话 id / 文档名等）
    question = Column(Text)  # 仅提问类动作填写
    ip = Column(String(64))
    extra_json = Column(Text)  # 其他上下文（如提问耗时、是否命中缓存）
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class AppSetting(Base):
    """运行时配置覆盖层：DB 里只存「被管理员改过的值」，未改的用 .env 默认值。"""

    __tablename__ = "app_settings"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(64), unique=True, index=True, nullable=False)
    value = Column(Text)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
