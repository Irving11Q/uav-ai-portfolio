from datetime import datetime

from pydantic import BaseModel, ConfigDict


class UserCreate(BaseModel):
    username: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


class ChangePassword(BaseModel):
    old_password: str
    new_password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    role: str
    created_at: datetime | None = None


# ---- 知识库文档 ----
class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    file_type: str | None = None
    status: str
    chunk_count: int
    char_count: int
    error: str | None = None
    # 解析元信息（JSON 字符串）：前端据此显示「经 OCR 识别」等标记
    meta_json: str | None = None
    created_at: datetime | None = None


class DocStats(BaseModel):
    total_documents: int
    ready_documents: int
    total_chunks: int


# ---- 会话与问答 ----
class ChatSessionCreate(BaseModel):
    title: str | None = None


class ChatSessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    title: str
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ReferenceItem(BaseModel):
    doc_id: int
    source: str
    snippet: str
    score: float


class ChatMessageOut(BaseModel):
    id: int
    role: str
    content: str
    references: list[ReferenceItem] = []
    created_at: datetime | None = None
    # 当前用户对该消息的反馈（up/down/None），用于刷新后仍能回显赞踩状态
    feedback: str | None = None
    feedback_reason: str | None = None


class ChatAsk(BaseModel):
    question: str


class ChatAskResult(BaseModel):
    answer: str
    references: list[ReferenceItem] = []
    session_id: int
    message_id: int
    cached: bool = False  # True 表示命中语义缓存（未调用大模型）


# ---- 阶段6：答案反馈 ----
class FeedbackCreate(BaseModel):
    message_id: int
    rating: str  # up | down
    reason: str | None = None
    comment: str | None = None


class FeedbackOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    message_id: int
    session_id: int | None = None
    user_id: int | None = None
    username: str | None = None
    rating: str
    reason: str | None = None
    comment: str | None = None
    question: str | None = None
    answer: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class FeedbackSummary(BaseModel):
    up: int = 0
    down: int = 0
    total: int = 0
    satisfaction_rate: float = 0.0  # up / total，0~1
    reason_breakdown: list[dict] = []  # [{"reason": str, "count": int}]


# ---- 阶段6：问题推荐 ----
class SimilarQuestion(BaseModel):
    question: str
    score: float


class HotQuestion(BaseModel):
    question: str
    count: int
    ratio: float = 0.0  # 该问题占全部提问的比例


# ---- 阶段6：统计仪表盘 ----
class TrendPoint(BaseModel):
    date: str
    count: int


class DashboardStats(BaseModel):
    # 知识库
    documents: int = 0
    ready_documents: int = 0
    failed_documents: int = 0
    chunks: int = 0
    char_count: int = 0
    # 使用者
    users: int = 0
    sessions: int = 0
    messages: int = 0
    questions: int = 0  # 用户提问总条数（DB 口径，永久）
    # 质量与性能（内存埋点口径，服务重启清零）
    total_queries: int = 0
    cache_hits: int = 0
    cache_hit_rate: float = 0.0
    avg_latency_ms: float = 0.0
    avg_retrieval_ms: float = 0.0
    avg_llm_ms: float = 0.0
    retrieval_errors: int = 0
    llm_errors: int = 0
    # 反馈
    feedback_up: int = 0
    feedback_down: int = 0
    satisfaction_rate: float = 0.0
    # 哪些指标属于内存口径（前端据此提示「重启清零」）
    memory_scope_keys: list[str] = []


# ---- 阶段6：运行时配置 ----
class SettingItem(BaseModel):
    key: str
    type: str
    label: str
    desc: str = ""
    group: str = "其他"
    options: list[str] | None = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    default: object | None = None
    value: object | None = None
    overridden: bool = False


class SettingsOut(BaseModel):
    items: list[SettingItem] = []
    llm_model_options: list[str] = []


class SettingsUpdate(BaseModel):
    values: dict


# ---- 阶段6：审计日志 ----
class AuditLogOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int | None = None
    username: str | None = None
    role: str | None = None
    action: str
    action_label: str | None = None
    detail: str | None = None
    target: str | None = None
    question: str | None = None
    ip: str | None = None
    created_at: datetime | None = None
