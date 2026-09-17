"""统计洞察：热问榜与提问趋势。

放在单独模块的原因：仪表盘（admin）和问答页的热问榜（所有用户）都要用同一套口径，
两边各写一份 SQL 迟早会算出不同的数字。

## 口径说明（回答答辩时会被问）
热问榜按**问题文本的归一化形式**聚合：去掉首尾空白、去掉结尾的标点符号，
让「连衣裙怎么选码？」和「连衣裙怎么选码」算作同一问。
它**不做语义聚类** —— 那是更重的做法（要跑向量聚类），
所以「裙子和连衣裙怎么挑」这类改写仍会被算成两个问题。这在文档里如实标注。
"""
import re
from collections import Counter
from datetime import datetime, timedelta

from . import models

# 结尾标点（中英文），归一化时剥掉
_TRAILING_PUNCT = re.compile(r"[\s，。？！?!,.;；：:、~～]+$")


def normalize_question(text: str) -> str:
    """问题归一化：统一空白 + 去尾标点，用于热问榜聚合。"""
    s = re.sub(r"\s+", " ", (text or "").strip())
    return _TRAILING_PUNCT.sub("", s).strip()


def _user_questions(database, days: int | None = None, scan_limit: int = 5000):
    """取用户提问（role=user），按时间倒序最多 scan_limit 条。"""
    query = database.query(models.Message.content, models.Message.created_at).filter(
        models.Message.role == "user"
    )
    if days:
        since = datetime.utcnow() - timedelta(days=days)
        query = query.filter(models.Message.created_at >= since)
    return query.order_by(models.Message.id.desc()).limit(scan_limit).all()


def hot_questions(database, limit: int = 10, days: int | None = 30) -> list[dict]:
    """热问榜 Top N。返回 [{"question", "count", "ratio"}]，按次数降序。"""
    rows = _user_questions(database, days=days)
    if not rows:
        return []

    counter: Counter = Counter()
    display: dict[str, str] = {}
    for content, _created in rows:
        key = normalize_question(content or "")
        if not key:
            continue
        counter[key] += 1
        # 展示用第一次（最新）出现的原始写法，比归一化结果更自然
        display.setdefault(key, (content or "").strip())

    total = sum(counter.values())
    out = []
    for key, count in counter.most_common(limit):
        out.append(
            {
                "question": display.get(key, key),
                "count": count,
                "ratio": round(count / total, 4) if total else 0.0,
            }
        )
    return out


def question_trend(database, days: int = 7) -> list[dict]:
    """近 N 天每天的用户提问量（含今天）。没有提问的日期补 0，前端画图不断线。"""
    since = (datetime.utcnow() - timedelta(days=days - 1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    rows = (
        database.query(models.Message.created_at)
        .filter(models.Message.role == "user", models.Message.created_at >= since)
        .all()
    )

    buckets: Counter = Counter()
    for (created,) in rows:
        if created:
            buckets[created.strftime("%Y-%m-%d")] += 1

    out = []
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    for i in range(days):
        day = (today - timedelta(days=days - 1 - i)).strftime("%Y-%m-%d")
        out.append({"date": day, "count": buckets.get(day, 0)})
    return out
