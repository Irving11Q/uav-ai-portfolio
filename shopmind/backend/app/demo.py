"""演示模式的全局问答额度控制。

【为什么需要这一层】
在线体验版调用的是真实的大模型 API（按 token 计费）。一旦链接被转发给他人、
或者被爬虫扫到，如果没有总量上限，账单就是不可控的。
所以演示模式给「整个站点」设一个总次数上限，并且：

1. **计数落库**（`app_settings` 表），服务重启 / 重新部署都不会重置，无法靠重启绕过。
2. **命中语义缓存的提问不计次** —— 缓存复用不产生 API 费用，不该占额度。
3. **额度只从环境变量读，不做成后台可调** —— 若登录后台就能把上限改大，
   这层保护形同虚设。只有部署者能改动它。

【已知的宽松之处】
并发突发时，多个请求可能同时通过前置检查，导致实际调用数略微超过上限
（超出量 ≤ 并发数）。在「每用户每分钟 20 次」的限流之下，这个偏差可以接受，
换来的是不必为计数加锁、不影响正常问答的响应速度。
"""

from __future__ import annotations

import logging

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from . import db, models
from .config import settings

logger = logging.getLogger("rag")

# 计数在 app_settings 里的键名（与 admin 可调的配置项共用同一张表，互不干扰）
_QUOTA_KEY = "demo_quota_used"


def enabled() -> bool:
    """当前是否为演示模式（由 DEMO_MODE 环境变量决定）。"""
    return settings.DEMO_MODE


def limit() -> int:
    return settings.DEMO_MAX_QUESTIONS


def used(database: Session) -> int:
    """已消耗的真实大模型调用次数。"""
    row = (
        database.query(models.AppSetting)
        .filter(models.AppSetting.key == _QUOTA_KEY)
        .first()
    )
    if row is None:
        return 0
    try:
        return int(row.value or 0)
    except (TypeError, ValueError):
        return 0


def remaining(database: Session) -> int:
    """剩余额度；-1 表示不限量（非演示模式）。"""
    if not enabled():
        return -1
    return max(0, limit() - used(database))


def quota_info(database: Session) -> dict:
    """给前端展示的额度信息。"""
    if not enabled():
        return {"enabled": False, "limit": 0, "used": 0, "remaining": -1}
    return {
        "enabled": True,
        "limit": limit(),
        "used": used(database),
        "remaining": remaining(database),
    }


def ensure_available(database: Session) -> None:
    """额度用尽则抛 429（在调用大模型之前拦截，避免白花钱）。"""
    if not enabled():
        return
    if remaining(database) <= 0:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"这里是公开演示环境，大模型调用额度（共 {limit()} 次）已经用完了。"
                "你仍可浏览知识库、查看仪表盘与审计日志；"
                "如需继续提问，请下载压缩包内的「免安装本地版」在本机运行。"
            ),
        )


def consume() -> int:
    """真实调用一次大模型后计数 +1，返回累计值。

    自开短会话，不掺和请求会话的事务；任何异常只 warning，
    绝不因为「计数失败」而让用户的问答失败。
    """
    if not enabled():
        return 0
    database = db.SessionLocal()
    try:
        row = (
            database.query(models.AppSetting)
            .filter(models.AppSetting.key == _QUOTA_KEY)
            .first()
        )
        if row is None:
            row = models.AppSetting(key=_QUOTA_KEY, value="0")
            database.add(row)
            database.flush()
        current = int(row.value or 0)
        row.value = str(current + 1)
        database.commit()
        return current + 1
    except Exception as e:  # noqa: BLE001
        database.rollback()
        logger.warning("演示额度计数失败（不影响问答本身）: %s", e)
        return 0
    finally:
        database.close()
