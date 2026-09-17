"""生产模式：由 FastAPI 直接托管前端构建产物，实现「一个进程 = 一个端口 = 完整系统」。

【为什么需要它】
开发态是双进程（Vite dev server 负责热更新与 /api 代理 + uvicorn 负责后端），
但对外交付时不可能要求使用者装 Node。把 `frontend/dist` 拷到 `app/static/` 后，
后端一个进程就能同时提供 API 与页面，使用者只需一个 Python 环境。

【为什么不会影响开发】
只有当 `app/static/index.html` 真实存在时才挂载；开发态没有这个目录，
本模块直接跳过，前端照旧走 Vite dev server。

【路径安全】
SPA 兜底路由会拼接收到的路径，必须校验解析结果仍在静态目录内，
否则 `../` 之类的构造可以读到目录外的文件（路径穿越）。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import settings

logger = logging.getLogger("rag")


def static_dir() -> Path:
    """前端构建产物目录。默认 `backend/app/static`，可用环境变量 STATIC_DIR 覆盖。"""
    raw = os.getenv("STATIC_DIR")
    return Path(raw).resolve() if raw else settings.BACKEND_DIR / "app" / "static"


def mount_frontend(app: FastAPI) -> bool:
    """把前端构建产物挂到根路径。返回是否挂载成功。

    必须在所有 API 路由注册**之后**调用：
    FastAPI 按注册顺序匹配，兜底路由 (`/{full_path:path}`) 放最后，
    才不会把 `/api/...`、`/docs`、`/openapi.json` 抢走。
    """
    directory = static_dir()
    index = directory / "index.html"
    if not index.is_file():
        logger.info(
            "未发现前端构建产物(%s)，跳过静态托管 —— 当前为开发态双进程模式", directory
        )
        return False

    assets = directory / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    # 根路径 `/` 在 main.py 里已经注册了一个返回 JSON 的路由，它注册更早、
    # 优先级更高，会把首页抢走（实测：GET / 返回 {"msg": "..."} 而不是页面）。
    # 既然前端产物存在，根路径就该交给 SPA —— 把那条旧路由摘掉。
    app.router.routes = [
        r
        for r in app.router.routes
        if not (
            getattr(r, "path", None) == "/"
            and "GET" in (getattr(r, "methods", None) or set())
        )
    ]

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str):
        """静态文件命中就返回文件，否则一律回 index.html（交给前端路由处理）。"""
        if full_path:
            candidate = (directory / full_path).resolve()
            # 路径穿越防护：解析后必须仍在静态目录内
            if candidate.is_file() and candidate.is_relative_to(directory):
                return FileResponse(candidate)
        return FileResponse(index, media_type="text/html")

    logger.info("已挂载前端构建产物：%s —— 单端口访问 http://127.0.0.1:<port>/", directory)
    return True
