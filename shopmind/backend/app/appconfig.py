"""运行时配置：管理员可在后台切换大模型、调整检索参数，改完立即生效，无需重启服务。

## 设计
- **覆盖层**：默认值来自 `config.settings`（即 `.env`）；DB 表 `app_settings` 只存「被管理员改过」的值。
  这样 `.env` 依然是唯一的事实来源，后台改动是叠加在它之上的一层。
- **进程内缓存**：问答链路每次请求都要读好几个参数，全部查库不可接受。
  首次读库后缓存住，写配置时主动失效（`invalidate()`），因此读操作接近零成本。
- **白名单 + 范围校验**：只认 `SPEC` 里的键，且逐个校验类型与上下限，
  避免把 `top_k` 调成 999 或把缓存阈值调到 0.1 之类把系统调坏。
- **失败不致命**：读库异常（表还没建、库被锁）时回退到默认值，不让问答链路挂掉。
"""
import logging
import threading
from typing import Any

from .config import settings

logger = logging.getLogger("rag")

_lock = threading.RLock()
_cache: dict[str, Any] | None = None

# 可选大模型（DashScope OpenAI 兼容模式）。前端下拉允许自定义输入，这里只是常用项。
LLM_MODEL_OPTIONS = [
    "qwen-turbo",
    "qwen-plus",
    "qwen-max",
    "qwen-long",
    "qwen3-max",
]

# 规格表：所有可配置项的真相来源。前端设置页据此自动渲染表单。
SPEC: dict[str, dict[str, Any]] = {
    "LLM_MODEL": {
        "type": "enum",
        "default": settings.LLM_MODEL,
        "options": LLM_MODEL_OPTIONS,
        "label": "大模型",
        "desc": "生成答案用的模型。turbo 最快最省，plus 均衡，max 质量最好但更贵。",
        "group": "模型",
    },
    "LLM_TEMPERATURE": {
        "type": "float",
        "default": 0.3,
        "min": 0.0,
        "max": 2.0,
        "step": 0.1,
        "label": "温度 (temperature)",
        "desc": "越低越稳定保守，越高越发散。知识库问答建议 0.1~0.5。",
        "group": "模型",
    },
    "LLM_MAX_TOKENS": {
        "type": "int",
        "default": 1200,
        "min": 128,
        "max": 8192,
        "label": "答案最大长度 (max_tokens)",
        "desc": "单次回答的 token 上限。太小会把答案截断。",
        "group": "模型",
    },
    "RETRIEVE_TOP_K": {
        "type": "int",
        "default": settings.RETRIEVE_TOP_K,
        "min": 1,
        "max": 20,
        "label": "检索片段数 (top-k)",
        "desc": "最终送给大模型的片段条数。调大召回更全但更慢、更费 token。",
        "group": "检索",
    },
    "HYBRID_POOL_SIZE": {
        "type": "int",
        "default": settings.HYBRID_POOL_SIZE,
        "min": 5,
        "max": 100,
        "label": "混合检索候选池",
        "desc": "向量与关键词各自召回的候选数量，融合重排后再截取 top-k。",
        "group": "检索",
    },
    "RERANK_ENABLED": {
        "type": "bool",
        "default": settings.RERANK_ENABLED,
        "label": "启用重排 (Rerank)",
        "desc": "用 gte-rerank 对候选二次精排，通常显著提升相关性；关闭可省一次调用。",
        "group": "检索",
    },
    "RERANK_TOP_N": {
        "type": "int",
        "default": settings.RERANK_TOP_N,
        "min": 1,
        "max": 50,
        "label": "送重排的候选数",
        "desc": "融合后取前 N 条送重排。N 太大会拖慢响应。",
        "group": "检索",
    },
    "CACHE_ENABLED": {
        "type": "bool",
        "default": settings.CACHE_ENABLED,
        "label": "启用语义缓存",
        "desc": "近义问题直接复用历史答案，省一次大模型调用。",
        "group": "缓存与限流",
    },
    "CACHE_SIMILARITY_THRESHOLD": {
        "type": "float",
        "default": settings.CACHE_SIMILARITY_THRESHOLD,
        "min": 0.5,
        "max": 1.0,
        "step": 0.01,
        "label": "缓存相似度阈值（余弦）",
        "desc": "按余弦相似度比较，越高越保守（只有几乎同一句才命中）。建议 0.95~0.99；调低会更容易复用历史答案，但答偏的风险上升。",
        "group": "缓存与限流",
    },
    "RATE_LIMIT_PER_USER_PER_MIN": {
        "type": "int",
        "default": settings.RATE_LIMIT_PER_USER_PER_MIN,
        "min": 1,
        "max": 1000,
        "label": "每人每分钟提问上限",
        "desc": "滑动窗口限流，超限返回 429。防止刷接口把大模型配额打满。",
        "group": "缓存与限流",
    },
}


def _coerce(key: str, raw: Any) -> Any:
    """按 SPEC 把值转成正确类型并校验范围；非法值抛 ValueError。"""
    spec = SPEC[key]
    kind = spec["type"]
    try:
        if kind == "bool":
            if isinstance(raw, bool):
                return raw
            s = str(raw).strip().lower()
            if s in ("true", "1", "yes", "on"):
                return True
            if s in ("false", "0", "no", "off"):
                return False
            raise ValueError
        if kind == "int":
            val = int(float(raw))
        elif kind == "float":
            val = float(raw)
        else:  # enum / 自由字符串
            val = str(raw).strip()
            if not val or len(val) > 64:
                raise ValueError
            return val
    except (TypeError, ValueError):
        raise ValueError(f"{key} 的取值不合法：{raw!r}")

    lo, hi = spec.get("min"), spec.get("max")
    if lo is not None and val < lo:
        raise ValueError(f"{key} 不能小于 {lo}")
    if hi is not None and val > hi:
        raise ValueError(f"{key} 不能大于 {hi}")
    return val


def _defaults() -> dict[str, Any]:
    return {k: v["default"] for k, v in SPEC.items()}


def _load_from_db() -> dict[str, Any]:
    """读取 DB 覆盖层；任何异常都退回默认值（不能让配置层成为单点故障）。"""
    values = _defaults()
    try:
        from . import models
        from .db import SessionLocal

        database = SessionLocal()
        try:
            rows = database.query(models.AppSetting).all()
        finally:
            database.close()
        for row in rows:
            if row.key not in SPEC:
                continue  # 忽略已废弃的键
            try:
                values[row.key] = _coerce(row.key, row.value)
            except ValueError:
                logger.warning("忽略非法配置 %s=%r，使用默认值", row.key, row.value)
    except Exception as e:  # noqa: BLE001
        logger.warning("读取运行时配置失败，使用默认值: %s", e)
    return values


def _snapshot() -> dict[str, Any]:
    global _cache
    with _lock:
        if _cache is None:
            _cache = _load_from_db()
        return _cache


def invalidate() -> None:
    """让缓存失效，下次读取重新查库。"""
    global _cache
    with _lock:
        _cache = None


def get(key: str, default: Any = None) -> Any:
    """读一个配置项（带兜底：未知键返回 default 或 .env 默认值）。"""
    if key not in SPEC:
        return default
    return _snapshot().get(key, SPEC[key]["default"])


def all_values() -> dict[str, Any]:
    """当前全部生效值。"""
    return dict(_snapshot())


def describe() -> dict[str, Any]:
    """给前端设置页用的元信息（标签/说明/类型/范围/默认值）+ 当前值。"""
    current = _snapshot()
    items = []
    for key, spec in SPEC.items():
        items.append(
            {
                "key": key,
                "type": spec["type"],
                "label": spec["label"],
                "desc": spec.get("desc", ""),
                "group": spec.get("group", "其他"),
                "options": spec.get("options"),
                "min": spec.get("min"),
                "max": spec.get("max"),
                "step": spec.get("step"),
                "default": spec["default"],
                "value": current.get(key, spec["default"]),
                # 是否被管理员改过（与默认值不同即为改过）
                "overridden": current.get(key, spec["default"]) != spec["default"],
            }
        )
    return {"items": items, "llm_model_options": LLM_MODEL_OPTIONS}


def update(values: dict[str, Any]) -> dict[str, Any]:
    """批量更新配置：先全部校验通过才落库，避免出现「改了一半」的状态。"""
    if not isinstance(values, dict) or not values:
        raise ValueError("没有需要更新的配置项")

    cleaned: dict[str, Any] = {}
    for key, raw in values.items():
        if key not in SPEC:
            raise ValueError(f"不支持的配置项：{key}")
        cleaned[key] = _coerce(key, raw)

    from . import models
    from .db import SessionLocal

    database = SessionLocal()
    try:
        for key, val in cleaned.items():
            row = (
                database.query(models.AppSetting)
                .filter(models.AppSetting.key == key)
                .first()
            )
            stored = "true" if val is True else "false" if val is False else str(val)
            if row:
                row.value = stored
            else:
                database.add(models.AppSetting(key=key, value=stored))
        database.commit()
    finally:
        database.close()

    invalidate()
    return all_values()


def reset() -> dict[str, Any]:
    """清空覆盖层，全部回到 `.env` 默认值。"""
    from . import models
    from .db import SessionLocal

    database = SessionLocal()
    try:
        database.query(models.AppSetting).delete()
        database.commit()
    finally:
        database.close()

    invalidate()
    return all_values()
