"""启动入口 —— 单端口同时提供 API 与前端页面。

【端口】
优先读环境变量 `PORT`（云平台会注入），否则用 8000。
绑定 `0.0.0.0`，这样容器 / 局域网 / 反向代理都能访问。

【双击启动（免安装本地版）】
设置 `AUTO_OPEN_BROWSER=1` 时，服务起来后自动打开浏览器，
使用者双击 bat 即可，不用自己敲地址。

【为什么先手动加载 .env】
config.py 要等 `app.main` 被 import 时才读 .env，
而 run.py 在启动前就需要 PORT / AUTO_OPEN_BROWSER，所以这里先自己读一遍。
"""

from __future__ import annotations

import os
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn

BASE_DIR = Path(__file__).resolve().parent


def _load_env() -> None:
    """尽力加载同目录 .env；python-dotenv 缺失时静默跳过（不影响启动）。"""
    try:
        from dotenv import load_dotenv
    except Exception:  # pragma: no cover
        return
    load_dotenv(BASE_DIR / ".env")


def _open_browser_later(port: int, delay: float = 3.0) -> None:
    """等 uvicorn 把端口监听起来再开浏览器，避免打开一个「无法访问」的空白页。"""

    def _run() -> None:
        time.sleep(delay)
        try:
            webbrowser.open(f"http://127.0.0.1:{port}")
        except Exception:  # pragma: no cover
            pass

    threading.Thread(target=_run, daemon=True).start()


def main() -> None:
    _load_env()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    banner = f"""
============================================================
  RAG 知识库问答系统
  访问地址：http://127.0.0.1:{port}
  接口文档：http://127.0.0.1:{port}/docs
  停止服务：在本窗口按 Ctrl+C
============================================================
"""
    print(banner)

    if os.getenv("AUTO_OPEN_BROWSER", "0") == "1":
        _open_browser_later(port)

    uvicorn.run("app.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
