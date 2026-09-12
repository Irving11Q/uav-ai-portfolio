#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════
  Day 26：桌面端上位机调用 AI 服务（PySide6 + HTTP 客户端）
═══════════════════════════════════════════════════════════════

【程序做什么】
  Day25 把 Agent 包成了 HTTP 服务，但服务自己是没人用的 —— 得有客户端。
  今天写 W2 那套 PySide6 上位机的「升级版」：界面上点一下，背后发 HTTP 请求，
  把 AI 的结论显示在窗口里。五个按钮，一一对应 Day25 的五个端点：

      【刷新状态】    GET  /health         看服务活着没、Key 配没配、数据几行
      【检索手册】    POST /manual/search  问《电池手册》：温度超过多少要降落
      【过滤数据】    POST /data/filter    温度 > 55 的行，pandas 真算
      【统计摘要】    POST /data/summary   电压这一列的均值/最小/最大
      【AI 提问】     POST /ask            自然语言，Agent 自己选工具

  而且它**会自己把服务拉起来**（起一个 uvicorn 子进程，轮询 /health 等到就绪），
  不用你先开一个黑框手动跑服务。

【怎么读这个文件（7 块）】
  1. 环境探测       没 PySide6 就只跑命令行自测，不许崩
  2. ServiceManager  拉起/停止 uvicorn 子进程 + 轮询 /health 等就绪
  3. call_api()      ★ 统一出口：把 HTTP 结果翻译成「成功 / 业务错 / 配置错 / 连不上」
  4. 两个 QThread    ★ 等服务和发请求都必须放线程，否则窗口「未响应」
  5. MainWindow      三个功能页（手册 / 数据 / AI）
  6. self_test()     不弹窗口、不依赖 Qt，命令行把五个端点全打一遍
  7. main()          有 Qt 就开窗口，没 Qt 就自测

【运行方式】
  A. 开窗口（推荐）：
       cd w6_service
       D:/Python-envs/chroma-env/Scripts/python.exe day26_client_app.py
       （点【启动服务】，等状态灯变绿，bge 预热约 25 秒）
  B. 命令行自测（不弹窗口，发布/CI 用这个）：
       D:/Python-envs/chroma-env/Scripts/python.exe day26_client_app.py --selftest
  C. 服务已在别处跑着（比如你手动 uvicorn 起了）：
       set DAY26_BASE=http://127.0.0.1:8000
       D:/Python-envs/chroma-env/Scripts/python.exe day26_client_app.py

【为什么学这个（面试能讲）】
  Day25 是「把能力包成服务」，今天是「让服务真的被人用」。中间隔着三个坑：
    ① ★ GUI 里绝不能同步发 HTTP。主线程一卡，窗口就是「未响应」。
       等 bge 预热 25 秒更是灾难 —— 连「等服务起来」都要放线程。
    ② ★ 错误要分三层，不能一句「请求失败」糊过去：
       422/404 = 业务层（你传的参数有问题，服务端已告诉你原因，显示 detail 即可）
       503     = 配置层（Key 没配 / 能力不可用，告诉用户去配什么）
       连不上  = 运维层（服务压根没起，提示去点【启动服务】）
       这三层处理方式完全不同，混在一起用户根本不知道该干嘛。
    ③ ★ 客户端要「自己能活」：不要让用户先去开黑框跑服务。
       程序自己 Popen 起子进程 + 轮询 /health，这才叫交付给别人用。

【和前几天的关系】
  W1  PySide6 界面      → 今天的窗口布局（信号槽、布局、表格）
  W2  串口上位机        → 同样的「后台收数据 → 主线程刷新界面」套路
  W3  requests 调 LLM   → 今天调的是自己的服务，不是厂商 API
  Day25 HTTP 服务       → 今天的服务端；两边靠 Pydantic 契约对齐
"""

import os
import sys
import json
import time
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
# 【解释】Day25 的服务端脚本跟本文件放同一目录（发布时会一起带上）
SERVICE_SCRIPT = os.path.join(HERE, "day25_api_service.py")
# 端口避开常见的 8000，防止和你手动起的服务打架
PORT = int(os.environ.get("DAY26_PORT", "8123"))
BASE_URL = os.environ.get("DAY26_BASE", "http://127.0.0.1:%d" % PORT)
TIMEOUT = (5, 90)   # (连接超时, 读取超时)。/ask 要走 Agent 多轮，给足 90 秒

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QLineEdit, QTextEdit, QComboBox,
        QDoubleSpinBox, QTabWidget, QGroupBox, QMessageBox,
    )
    from PySide6.QtCore import Qt, QThread, Signal
    from PySide6.QtGui import QFont
    HAS_GUI = True
except ImportError:
    HAS_GUI = False


# ════════════════════════════════════════════════════════════════
# 第 1 部分：ServiceManager —— 自己把服务拉起来
# ════════════════════════════════════════════════════════════════
class ServiceManager:
    """管理 uvicorn 子进程：启动、等就绪、停止。

    【解释】为什么要轮询等就绪？
        服务启动要加载 bge（约 25 秒）。在它就绪之前，/health 是连不上的。
        如果客户端一上来就发请求，只会拿到 ConnectionError。
        所以必须「反复探 /health，直到 200 或超时」—— 这叫 readiness probe，
        生产环境里 K8s 的健康检查就是同一个道理。
    """

    def __init__(self, on_log=None):
        self.proc = None
        self.on_log = on_log or (lambda s: None)

    def _log(self, msg):
        self.on_log(msg)

    def is_running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, wait_sec=150):
        """启动服务并等到 /health 返回 200。

        【解释】cwd 必须是服务端脚本所在目录，否则 uvicorn 找不到
                day25_api_service 模块（它会连带 import day23a / day21b）。
        """
        if not os.path.exists(SERVICE_SCRIPT):
            self._log("✗ 找不到服务端脚本：%s" % SERVICE_SCRIPT)
            return False
        if self.is_running():
            self._log("· 服务已在运行")
            return True

        cmd = [sys.executable, "-m", "uvicorn", "day25_api_service:app",
               "--host", "127.0.0.1", "--port", str(PORT)]
        self._log("▶ 启动服务：%s" % " ".join(cmd[1:]))
        try:
            self.proc = subprocess.Popen(
                cmd, cwd=HERE,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            self._log("✗ 启动失败：%r" % e)
            return False

        # 轮询等就绪
        t0 = time.time()
        last_tip = 0
        while time.time() - t0 < wait_sec:
            try:
                r = requests.get(BASE_URL + "/health", timeout=3)
                if r.status_code == 200:
                    self._log("✅ 服务就绪（耗时 %.1f 秒，bge 预热占大头）" % (time.time() - t0))
                    return True
            except Exception:
                pass
            if self.proc.poll() is not None:
                self._log("✗ 服务进程已退出（可能是 fastapi/uvicorn 没装）")
                return False
            elapsed = time.time() - t0
            if elapsed - last_tip >= 10:      # 每 10 秒报一次进度，别让用户以为死了
                last_tip = elapsed
                self._log("   … 仍在加载（%.0f 秒），首次要加载中文向量模型" % elapsed)
            time.sleep(1)
        self._log("✗ 等待超时（%d 秒）" % wait_sec)
        return False

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self._log("■ 停止服务")
            self.proc.terminate()
            try:
                self.proc.wait(timeout=8)
            except Exception:
                self.proc.kill()
        self.proc = None


# ════════════════════════════════════════════════════════════════
# 第 2 部分：call_api —— 把 HTTP 结果翻译成「该干什么」
# ════════════════════════════════════════════════════════════════
def call_api(method, path, payload=None):
    """统一出口，返回 (level, text, raw)。

    level 决定客户端该怎么提示用户（★ 这就是「错误分三层」的落点）：
        "ok"       成功，text 是已格式化好的展示文本
        "business" 4xx —— 参数有问题，text 直接显示服务端给的 detail
        "config"   503 —— 能力没配好（比如没 API Key），text 说清去配什么
        "offline"  连不上 —— 服务没起，text 提示去点【启动服务】
        "timeout"  超时
    raw 是原始 json（成功时），失败为 None。
    """
    url = BASE_URL + path
    try:
        if method == "GET":
            r = requests.get(url, timeout=TIMEOUT)
        else:
            r = requests.post(url, json=payload or {}, timeout=TIMEOUT)
    except requests.exceptions.ConnectionError:
        return "offline", "连不上服务（%s）。点上面的【启动服务】再试。" % BASE_URL, None
    except requests.exceptions.Timeout:
        return "timeout", "请求超时（%d 秒）。/ask 要走 Agent 多轮，慢是正常的，可再试一次。" % TIMEOUT[1], None
    except Exception as e:
        return "offline", "请求异常：%r" % e, None

    try:
        body = r.json()
    except Exception:
        body = {"detail": r.text[:300]}

    if r.status_code == 200:
        return "ok", _fmt(path, body), body
    detail = body.get("detail", json.dumps(body, ensure_ascii=False)[:300])

    if r.status_code == 503:
        return "config", "【服务不可用 503】%s" % detail, None
    return "business", "【%d】%s" % (r.status_code, detail), None


def _fmt(path, body):
    """把不同端点的返回体格式化成人能读的一段文字。"""
    if path == "/health":
        return ("状态 %s ｜ 模型已配：%s ｜ 数据工具：%s\n"
                "飞行数据 %s 行 ｜ 手册切块 %s ｜ 检索后端 %s ｜ 已运行 %.1f 秒"
                % (body.get("status"), body.get("has_llm"), body.get("has_data"),
                   body.get("data_rows"), body.get("manual_chunks"),
                   body.get("index_backend"), body.get("uptime_sec", 0)))
    if path == "/manual/search":
        hits = body.get("hits", [])
        if not hits:
            return "（手册里没检索到相关内容）"
        lines = ["检索后端：%s，命中 %d 段" % (body.get("backend"), len(hits))]
        for i, h in enumerate(hits, 1):
            lines.append("\n【%d】%s   相似度 %.3f" % (i, h.get("title"), h.get("score", 0)))
            lines.append("    " + h.get("snippet", "").replace("\n", " "))
        return "\n".join(lines)
    if path == "/data/filter":
        return "%s %s %s 的结果：\n%s" % (body.get("col"), body.get("op"),
                                       body.get("value"), body.get("result"))
    if path == "/data/summary":
        return "%s\n%s" % (body.get("col"), body.get("result"))
    if path == "/ask":
        return "%s\n\n—— %s（模式：%s）" % (body.get("question"), body.get("answer"), body.get("mode"))
    return json.dumps(body, ensure_ascii=False, indent=2)


# ════════════════════════════════════════════════════════════════
# 第 3 部分：两个 QThread —— 凡是要等的活，都不许放主线程
# ════════════════════════════════════════════════════════════════
if HAS_GUI:
    class BootWorker(QThread):
        """后台启动服务。★ 等 25 秒预热绝不能放主线程，否则窗口直接「未响应」."""
        log = Signal(str)
        done = Signal(bool)

        def __init__(self, mgr):
            super().__init__()
            self.mgr = mgr

        def run(self):
            # 【解释】ServiceManager 的 on_log 是构造时传进来的普通函数，
            #         它跑在子线程里，直接改界面会崩。所以这里换成桥接函数，
            #         把日志交给 _boot_log_sink（主窗口的 log 方法）去处理。
            self.mgr.on_log = on_log_bridge
            ok = self.mgr.start()
            self.done.emit(ok)

    # 【解释】模块级桥接：子线程产生的日志 → 经槽函数 → 主窗口日志框。
    _boot_log_sink = None

    def on_log_bridge(msg):
        if _boot_log_sink:
            _boot_log_sink(msg)

    class HttpWorker(QThread):
        """后台发一个 HTTP 请求，结果用信号回主线程刷新界面。"""
        done = Signal(str, str)      # (level, text)
        failed = Signal(str, str)

        def __init__(self, method, path, payload=None):
            super().__init__()
            self.method, self.path, self.payload = method, path, payload

        def run(self):
            level, text, _ = call_api(self.method, self.path, self.payload)
            self.done.emit(level, text)


# ════════════════════════════════════════════════════════════════
# 第 4 部分：MainWindow
# ════════════════════════════════════════════════════════════════
if HAS_GUI:
    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.mgr = ServiceManager(on_log=self.log)
            self.worker = None
            self.booter = None
            self._build_ui()
            self.setWindowTitle("Day26 · 无人机 AI 上位机（调 Day25 的 HTTP 服务）")
            self.resize(900, 640)
            self.log("提示：先点【启动服务】，首次要加载中文向量模型约 25 秒。")

        # ── 界面 ──
        def _build_ui(self):
            root = QWidget()
            self.setCentralWidget(root)
            v = QVBoxLayout(root)

            # 顶部：服务状态条
            top = QGroupBox("服务")
            th = QHBoxLayout(top)
            self.lamp = QLabel("●")
            self.lamp.setStyleSheet("color:#c0392b; font-size:20px;")
            self.lamp_txt = QLabel("未启动")
            self.btn_start = QPushButton("启动服务")
            self.btn_stop = QPushButton("停止服务")
            self.btn_health = QPushButton("刷新状态")
            self.btn_stop.setEnabled(False)
            self.btn_health.setEnabled(False)
            self.btn_start.clicked.connect(self.on_start)
            self.btn_stop.clicked.connect(self.on_stop)
            self.btn_health.clicked.connect(self.on_health)
            for w in (self.lamp, self.lamp_txt, self.btn_start, self.btn_stop,
                      self.btn_health, QLabel("地址：" + BASE_URL)):
                th.addWidget(w)
            th.addStretch(1)
            v.addWidget(top)

            # 中部：三个功能页
            self.tabs = QTabWidget()
            self.tabs.addTab(self._page_manual(), "手册检索")
            self.tabs.addTab(self._page_data(), "飞行数据")
            self.tabs.addTab(self._page_ask(), "AI 提问")
            v.addWidget(self.tabs, stretch=3)

            # 结果区
            gb = QGroupBox("结果")
            gh = QVBoxLayout(gb)
            self.out = QTextEdit()
            self.out.setReadOnly(True)
            self.out.setFont(QFont("Consolas", 10))
            gh.addWidget(self.out)
            v.addWidget(gb, stretch=3)

            # 底部日志
            self.logbox = QTextEdit()
            self.logbox.setReadOnly(True)
            self.logbox.setMaximumHeight(110)
            v.addWidget(self.logbox)

        def _page_manual(self):
            w = QWidget()
            h = QHBoxLayout(w)
            self.in_manual = QLineEdit("温度超过多少要降落散热")
            b = QPushButton("检索手册")
            b.clicked.connect(self.on_manual)
            h.addWidget(QLabel("问题："))
            h.addWidget(self.in_manual, stretch=1)
            h.addWidget(b)
            return w

        def _page_data(self):
            w = QWidget()
            h = QHBoxLayout(w)
            self.cb_col = QComboBox()
            self.cb_col.addItems(["温度C", "电压V", "电流A", "高度m"])
            self.cb_op = QComboBox()
            self.cb_op.addItems([">", ">=", "<", "<=", "==", "!="])
            self.sp_val = QDoubleSpinBox()
            self.sp_val.setRange(-9999, 9999)
            self.sp_val.setValue(55.0)
            b1 = QPushButton("过滤数据")
            b2 = QPushButton("统计摘要")
            b1.clicked.connect(self.on_filter)
            b2.clicked.connect(self.on_summary)
            h.addWidget(QLabel("列："))
            h.addWidget(self.cb_col)
            h.addWidget(self.cb_op)
            h.addWidget(self.sp_val)
            h.addWidget(b1)
            h.addWidget(b2)
            h.addStretch(1)
            return w

        def _page_ask(self):
            w = QWidget()
            h = QHBoxLayout(w)
            self.in_ask = QLineEdit("这次飞行温度超过 55 度的有几次？")
            b = QPushButton("问 AI")
            b.clicked.connect(self.on_ask)
            h.addWidget(QLabel("问题："))
            h.addWidget(self.in_ask, stretch=1)
            h.addWidget(b)
            return w

        # ── 行为 ──
        def log(self, msg):
            self.logbox.append(msg)

        def set_busy(self, busy, tip="请求中…"):
            self.tabs.setEnabled(not busy)
            self.btn_start.setEnabled(not busy)
            if busy:
                self.out.setPlainText(tip)

        def on_start(self):
            global _boot_log_sink
            _boot_log_sink = self.log          # 让后台线程的日志回到主线程
            self.btn_start.setEnabled(False)
            self.out.setPlainText("正在启动服务并等待就绪（首次约 25 秒）…")
            self.booter = BootWorker(self.mgr)
            self.booter.done.connect(self.on_boot_done)
            self.booter.start()

        def on_boot_done(self, ok):
            self.lamp.setStyleSheet("color:%s; font-size:20px;" % ("#27ae60" if ok else "#c0392b"))
            self.lamp_txt.setText("已就绪" if ok else "启动失败")
            self.btn_start.setEnabled(not ok)
            self.btn_stop.setEnabled(ok)
            self.btn_health.setEnabled(ok)
            if ok:
                self.on_health()
            else:
                self.out.setPlainText("服务没起来，看下面日志。常见原因：fastapi / uvicorn 没装。")

        def on_stop(self):
            self.mgr.stop()
            self.lamp.setStyleSheet("color:#c0392b; font-size:20px;")
            self.lamp_txt.setText("已停止")
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.btn_health.setEnabled(False)

        def _run(self, method, path, payload, tip):
            """所有请求共用一个出口：建线程 → 绑信号 → 起线程。"""
            self.set_busy(True, tip)
            self.worker = HttpWorker(method, path, payload)
            self.worker.done.connect(self.on_result)
            self.worker.start()

        def on_result(self, level, text):
            self.set_busy(False)
            color = {"ok": "#1a1a1a", "business": "#b9770e",
                     "config": "#b9770e", "offline": "#c0392b",
                     "timeout": "#c0392b"}.get(level, "#1a1a1a")
            self.out.setHtml('<span style="color:%s">%s</span>'
                             % (color, text.replace("\n", "<br>").replace(" ", "&nbsp;")))
            self.log("← %s" % level)

        def on_health(self):
            self._run("GET", "/health", None, "正在查服务状态…")

        def on_manual(self):
            q = self.in_manual.text().strip()
            if not q:
                return
            self._run("POST", "/manual/search", {"query": q, "top_k": 3}, "正在检索手册…")

        def on_filter(self):
            self._run("POST", "/data/filter",
                      {"col": self.cb_col.currentText(),
                       "op": self.cb_op.currentText(),
                       "value": float(self.sp_val.value())}, "正在用 pandas 过滤…")

        def on_summary(self):
            self._run("POST", "/data/summary",
                      {"col": self.cb_col.currentText()}, "正在统计…")

        def on_ask(self):
            q = self.in_ask.text().strip()
            if not q:
                return
            self._run("POST", "/ask", {"question": q}, "Agent 思考中（可能要十几秒）…")

        def closeEvent(self, e):
            self.mgr.stop()
            super().closeEvent(e)


# ════════════════════════════════════════════════════════════════
# 第 5 部分：命令行自测（不弹窗口、不依赖 Qt）
# ════════════════════════════════════════════════════════════════
def self_test():
    """把五个端点 + 三类错误全打一遍，返回通过项数。

    【解释】发布前跑这个就能证明「客户端和服务端契约对得上」，
           而且不用起窗口、不用人点，可以直接进 CI。
    """
    print("=" * 64)
    print("  Day26 自测：拉起服务 → 打全部端点 → 校验三类错误")
    print("=" * 64)
    if not HAS_REQUESTS:
        print("✗ 缺 requests，无法自测")
        return 0

    logs = []
    mgr = ServiceManager(on_log=logs.append)
    if not mgr.start():
        print("\n".join(logs))
        return 0
    print("\n".join(logs))

    passed = 0
    checks = [
        ("GET", "/health", None, None, "服务体检"),
        ("POST", "/manual/search", {"query": "温度超过多少要降落散热", "top_k": 3}, None, "手册检索"),
        ("POST", "/data/filter", {"col": "温度C", "op": ">", "value": 55}, None, "数据过滤"),
        ("POST", "/data/summary", {"col": "电压V"}, None, "列统计"),
        ("POST", "/data/filter", {"col": "温度C", "op": "大于", "value": 55}, "business", "非法比较符→422"),
        ("POST", "/data/filter", {"col": "不存在的列", "op": ">", "value": 1}, "business", "乱写列名→404"),
    ]
    for method, path, payload, expect, name in checks:
        level, text, _ = call_api(method, path, payload)
        ok = (level == "ok") if expect is None else (level == expect)
        print("\n%s %s → %s" % ("✅" if ok else "❌", name, level))
        print("     " + text.replace("\n", "\n     ")[:400])
        passed += 1 if ok else 0

    # /ask 单独处理：没 Key 是 503（config），算合理降级，不算失败
    level, text, _ = call_api("POST", "/ask", {"question": "温度超过 55 度的有几次？"})
    ok = level in ("ok", "config")
    print("\n%s AI 提问 → %s" % ("✅" if ok else "❌", level))
    print("     " + text.replace("\n", "\n     ")[:300])
    passed += 1 if ok else 0

    # 断网模拟：把 BASE_URL 指到一个没人监听的端口，验证 offline 分支
    global BASE_URL
    old = BASE_URL
    BASE_URL = "http://127.0.0.1:59999"
    level, text, _ = call_api("GET", "/health", None)
    BASE_URL = old
    ok = level == "offline"
    print("\n%s 服务不可达 → %s（应识别为 offline，提示去启动服务）" % ("✅" if ok else "❌", level))
    passed += 1 if ok else 0

    mgr.stop()
    print()
    print("=" * 64)
    print("  自测完成：%d / %d 项通过" % (passed, len(checks) + 2))
    print("  开窗口看效果：python day26_client_app.py")
    print("=" * 64)
    return passed


def main():
    if "--selftest" in sys.argv or not HAS_GUI:
        if not HAS_GUI:
            print("（本机没有 PySide6，自动切到命令行自测模式）")
        n = self_test()
        sys.exit(0 if n >= 6 else 1)

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
