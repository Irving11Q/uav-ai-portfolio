#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════
  Day 29b：实时监控面板（PySide6 定时轮询 Day28 的 /live/* 服务）
═══════════════════════════════════════════════════════════════

【程序做什么】
  Day29a 把「能测的核」写完了（服务管理 / 请求 / 快照格式化 / 采样缓冲）。
  今天是**薄薄一层界面**：把那些纯函数的结果贴到控件上，再给它一个心跳。

  窗口一开，每秒自动问一次 /live/status，实时显示：
      电压 / 电流 / 温度 / 高度（大字）
      窗口统计（最近 30 帧的 min/mean/max）
      续航预测（R² + 预计几秒后跌破安全线）
      电压曲线（红虚线＝安全线，橙点＝外推的跌破时刻）
      告警时间线（按服务端计数增量追加）
      轮询开销（服务端帧数 vs 客户端采样点 → 漏了多少帧）

  这条链到今天就合上了：
      Day27 规则+预测  →  Day28 服务（后台线程常驻）  →  ★ 今天：屏幕
  「板子到屏幕」的闭环，W6 到此收口。

【怎么读这个文件（4 块）】
  1. 界面骨架     服务条 + 巡检条 + 上半（数值|曲线）+ 下半（告警|日志）
  2. 三个 QThread 启动服务一个、轮询一个、发控制指令一个 —— 主线程只管刷界面
  3. ★ 心跳 tick()  QTimer 回调，核心是「防重入」
  4. apply()      把一份快照铺到所有控件上（展示逻辑全在 day29a 的 fmt_* 里）

【运行方式】
  cd w6_service
  A. 开面板（推荐）：
       D:/Python-envs/chroma-env/Scripts/python.exe day29b_dashboard.py
       然后：【启动服务】→ 选数据源 → 【开始巡检】
  B. 命令行自测（不弹窗口，实际跑的是 day29a 的自测）：
       D:/Python-envs/chroma-env/Scripts/python.exe day29b_dashboard.py --selftest
  C. 服务已在别处跑着：
       set DAY29_BASE=http://127.0.0.1:8000
       D:/Python-envs/chroma-env/Scripts/python.exe day29b_dashboard.py

【为什么学这个（面试能讲）】
  ① ★ 轮询必须防重入。定时器一秒一跳，可服务偶尔要两秒才回 ——
     不设「在飞标志」，请求就会一帧叠一帧：界面显示的其实是好几秒前的旧数据，
     而且请求会越堆越多。生产里这是定时器的头号坑
     （in-flight 挡住 back-pressure），代码只有三行，缺了就是事故。
  ② ★ 定时器只在「采集真的开始了」之后才启动。
     on_ctrl_result 里先看服务端有没有回 started，再 poll_timer.start() ——
     不要让界面空转，也别让「服务没起」和「采集没开始」这两种状态混在一起。
  ③ ★ 界面里不写业务判断。这里没有一句 if 电压 < 23 ——
     全是 fmt_* 的输出。所以改文案、加字段、写测试，都不需要动这个文件。
     判断「GUI 分层做没做好」有个土办法：看窗口类里有没有出现业务数字。
  ④ ★ 曲线用 setData 复用同一个曲线对象，绝不重调 plot()。
     重调 plot(clear=True) 会把阈值线和橙点标记一起清掉 ——
     表现出来就是「曲线有，安全线跑了一次就没了」，很难一眼看出来。

【和前几天的关系】
  Day26 客户端调端点      → 今天的骨架（ServiceManager / call_api）
  Day29a 内核             → 今天 import 它，一行业务逻辑都不重写
  Day28 /live/* 服务      → 今天的被轮询方

【读完之后】
  · 想练 SSE：把轮询换成 text/event-stream，QTimer 直接删掉，缓冲区留用
  · 想练 PySide6 进阶：曲线加多 Y 轴、滚动窗口、导出 PNG、深色主题
  · 想练工程：把 R² 门槛和轮询间隔改成 /health 下发，客户端不再硬编码
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# 【解释】★ 今天最要紧的一行：内核全部来自 day29a，本文件只做界面。
import day29a_live_client as core          # noqa: E402

try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QPushButton, QLabel, QComboBox, QDoubleSpinBox, QGroupBox, QPlainTextEdit,
        QSplitter, QSpinBox,
    )
    from PySide6.QtCore import Qt, QThread, Signal, QTimer
    from PySide6.QtGui import QFont
    HAS_GUI = True
except ImportError:
    HAS_GUI = False

try:
    import pyqtgraph as pg
    HAS_PG = True
except ImportError:
    HAS_PG = False

POLL_MS = 1000          # 默认轮询间隔（毫秒）


# ════════════════════════════════════════════════════════════════
# 第 1 部分：三个 QThread —— 凡是要等的活，都不许放主线程
# ════════════════════════════════════════════════════════════════
if HAS_GUI:
    _log_sink = None

    def _log_bridge(msg):
        # 【解释】子线程里产生的日志不能直接改控件，得绕回主线程。
        if _log_sink:
            _log_sink(msg)

    class BootWorker(QThread):
        """后台启动服务（探 /health 可能要几十秒）。"""
        done = Signal(bool)

        def __init__(self, mgr):
            super().__init__()
            self.mgr = mgr

        def run(self):
            self.mgr.on_log = _log_bridge
            self.done.emit(self.mgr.start())

    class PollWorker(QThread):
        """★ 一轮轮询：GET /live/status。"""
        got = Signal(str, object, str)         # (level, data, text)

        def run(self):
            level, data, text = core.call_api("GET", "/live/status")
            self.got.emit(level, data, text)

    class CtrlWorker(QThread):
        """★ 发一次控制指令：/live/start 或 /live/stop。"""
        got = Signal(str, object, str)

        def __init__(self, path, payload):
            super().__init__()
            self.path, self.payload = path, payload

        def run(self):
            level, data, text = core.call_api("POST", self.path, self.payload)
            self.got.emit(level, data, text)


# ════════════════════════════════════════════════════════════════
# 第 2 部分：MainWindow —— QTimer 心跳 + 曲线缓冲
# ════════════════════════════════════════════════════════════════
if HAS_GUI:
    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.mgr = core.ServiceManager(on_log=self.log)
            self.poll_worker = None            # ★ 在飞标志：非 None 说明还没回来
            self.ctrl_worker = None
            self.booter = None
            self._ctrl_is_start = False
            self.buf = core.SampleBuffer()     # 采样缓冲（来自 day29a）
            self.alerts_seen = 0
            self.poll_timer = QTimer(self)
            self.poll_timer.timeout.connect(self._tick)
            self._build_ui()
            self.setWindowTitle("Day29 · 无人机实时监控面板（轮询 Day28 的 /live/* 服务）")
            self.resize(1180, 780)
            self.log("提示：先【启动服务】，再选数据源点【开始巡检】。")
            self.log("默认 replay + 1x：服务端 1 帧/秒，正好和 1 秒轮询对齐（不漏帧）。")

        # ── 界面 ──
        def _build_ui(self):
            root = QWidget()
            self.setCentralWidget(root)
            v = QVBoxLayout(root)

            # 服务条
            svc = QGroupBox("服务")
            sh = QHBoxLayout(svc)
            self.lamp = QLabel("●")
            self.lamp.setStyleSheet("color:#7f8c8d; font-size:22px;")
            self.lamp_txt = QLabel("未启动")
            self.btn_start = QPushButton("启动服务")
            self.btn_stop = QPushButton("停止服务")
            self.btn_stop.setEnabled(False)
            self.btn_start.clicked.connect(self.on_start)
            self.btn_stop.clicked.connect(self.on_stop)
            for w in (self.lamp, self.lamp_txt, self.btn_start, self.btn_stop,
                      QLabel("地址：" + core.BASE_URL)):
                sh.addWidget(w)
            sh.addStretch(1)
            v.addWidget(svc)

            # 巡检条
            ctl = QGroupBox("巡检")
            ch = QHBoxLayout(ctl)
            self.cb_src = QComboBox()
            self.cb_src.addItems(["replay", "synth", "real"])
            self.sp_speed = QDoubleSpinBox()
            self.sp_speed.setRange(0, 20)
            self.sp_speed.setValue(1.0)
            self.sp_speed.setSingleStep(0.5)
            self.sp_poll = QSpinBox()
            self.sp_poll.setRange(200, 5000)
            self.sp_poll.setSingleStep(100)
            self.sp_poll.setValue(POLL_MS)
            self.sp_poll.setSuffix(" ms")
            self.btn_live_on = QPushButton("开始巡检")
            self.btn_live_off = QPushButton("停止巡检")
            self.btn_clear = QPushButton("清空曲线")
            self.btn_live_off.setEnabled(False)
            self.btn_live_on.clicked.connect(self.on_live_start)
            self.btn_live_off.clicked.connect(self.on_live_stop)
            self.btn_clear.clicked.connect(self.on_clear)
            self.sp_poll.valueChanged.connect(self._apply_poll_interval)
            ch.addWidget(QLabel("数据源："))
            ch.addWidget(self.cb_src)
            ch.addWidget(QLabel("倍速："))
            ch.addWidget(self.sp_speed)
            ch.addWidget(QLabel("轮询："))
            ch.addWidget(self.sp_poll)
            ch.addWidget(self.btn_live_on)
            ch.addWidget(self.btn_live_off)
            ch.addWidget(self.btn_clear)
            ch.addStretch(1)
            v.addWidget(ctl)

            split = QSplitter(Qt.Vertical)
            top = QWidget()
            th = QHBoxLayout(top)

            box = QGroupBox("当前状态")
            g = QGridLayout(box)
            self.lbl_val = {}
            for i, (name, _x) in enumerate(core.fmt_frame({})):
                lab = QLabel(name)
                val = QLabel("—")
                val.setFont(QFont("Consolas", 20, QFont.Weight.Bold))
                val.setStyleSheet("color:#2471a3;")
                g.addWidget(lab, 0, i * 2)
                g.addWidget(val, 0, i * 2 + 1)
                self.lbl_val[name] = val
            self.lbl_stats = QLabel("（窗口还没填满）")
            self.lbl_stats.setFont(QFont("Consolas", 9))
            self.lbl_pred = QLabel("—")
            self.lbl_pred.setFont(QFont("Consolas", 10))
            self.lbl_pred.setWordWrap(True)
            self.lbl_count = QLabel("—")
            self.lbl_count.setFont(QFont("Consolas", 9))
            for title, w in (("窗口统计（最近 30 帧）", self.lbl_stats),
                             ("续航预测", self.lbl_pred),
                             ("轮询开销", self.lbl_count)):
                t = QLabel(title)
                t.setStyleSheet("color:#7f8c8d; font-weight:bold;")
                g.addWidget(t, g.rowCount(), 0, 1, 8)
                g.addWidget(w, g.rowCount(), 0, 1, 8)
            th.addWidget(box, stretch=2)

            if HAS_PG:
                self.plot = pg.PlotWidget()
                self.plot.setBackground("w")
                self.plot.showGrid(x=True, y=True, alpha=0.3)
                self.plot.setLabel("left", "电压 (V)")
                self.plot.setLabel("bottom", "客户端采样点")
                self.plot.setTitle("电压曲线（红虚线＝安全线，橙点＝外推的跌破时刻）")
                self.curve = self.plot.plot([], [], pen=pg.mkPen("#2471a3", width=2))
                self.thr = pg.InfiniteLine(angle=0, pen=pg.mkPen(
                    "#c0392b", width=1, style=Qt.PenStyle.DashLine))
                self.plot.addItem(self.thr)
                self.marker = pg.ScatterPlotItem(size=13, brush=pg.mkBrush("#e67e22"))
                self.plot.addItem(self.marker)
                th.addWidget(self.plot, stretch=3)
            else:
                self.plot = None
                th.addWidget(QLabel("（没装 pyqtgraph，曲线不可用；数值面板照常）"),
                             stretch=3)

            split.addWidget(top)

            bottom = QWidget()
            bh = QHBoxLayout(bottom)
            ab = QGroupBox("告警时间线")
            ah = QVBoxLayout(ab)
            self.alerts_box = QPlainTextEdit()
            self.alerts_box.setReadOnly(True)
            self.alerts_box.setFont(QFont("Consolas", 9))
            ah.addWidget(self.alerts_box)
            bh.addWidget(ab, stretch=3)

            lb = QGroupBox("日志")
            lh = QVBoxLayout(lb)
            self.logbox = QPlainTextEdit()
            self.logbox.setReadOnly(True)
            self.logbox.setFont(QFont("Consolas", 9))
            self.logbox.setMaximumBlockCount(500)
            lh.addWidget(self.logbox)
            bh.addWidget(lb, stretch=2)

            split.addWidget(bottom)
            split.setSizes([430, 300])
            v.addWidget(split, stretch=1)

        # ── 小工具 ──
        def log(self, msg):
            self.logbox.appendPlainText(msg)

        def set_lamp(self, color, text):
            self.lamp.setStyleSheet("color:%s; font-size:22px;" % color)
            self.lamp_txt.setText(text)

        def _apply_poll_interval(self, ms):
            if self.poll_timer.isActive():
                self.poll_timer.setInterval(ms)
                self.log("· 轮询间隔改为 %d ms" % ms)

        def on_clear(self):
            self.buf.clear()
            self._redraw()

        # ── 服务 ──
        def on_start(self):
            global _log_sink
            _log_sink = self.log
            self.btn_start.setEnabled(False)
            self.booter = BootWorker(self.mgr)
            self.booter.done.connect(self.on_boot_done)
            self.booter.start()

        def on_boot_done(self, ok):
            self.btn_start.setEnabled(not ok)
            self.btn_stop.setEnabled(ok)
            # 【解释】不在这里禁用【开始巡检】：服务可能在别处跑着（DAY29_BASE），
            #         禁用了那些用户就用不了。服务真没起的话点一下会拿到 offline，
            #         日志里会告诉他去点【启动服务】。
            if not ok:
                self.log("✗ 服务没起来。常见原因：fastapi / uvicorn 没装，"
                         "或端口 %d 被占。" % core.PORT)

        def on_stop(self):
            self.poll_timer.stop()
            self.mgr.stop()
            self.set_lamp("#7f8c8d", "已停止")
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.btn_live_on.setEnabled(True)
            self.btn_live_off.setEnabled(False)

        # ── 采集控制 ──
        def on_live_start(self):
            src = self.cb_src.currentText()
            speed = float(self.sp_speed.value())
            self.log("▶ 请求开始巡检：source=%s speed=%g" % (src, speed))
            self._ctrl("/live/start", {"source": src, "speed": speed}, is_start=True)

        def on_live_stop(self):
            self.log("■ 请求停止巡检")
            self._ctrl("/live/stop", {}, is_start=False)

        def _ctrl(self, path, payload, is_start):
            # 【解释】把「这一次是开始还是停止」记在窗口上 —— 信号只能带固定参数，
            #         再靠回包内容猜（比如 "start" in str(data)）既脆又难读。
            self._ctrl_is_start = is_start
            self.ctrl_worker = CtrlWorker(path, payload)
            self.ctrl_worker.got.connect(self.on_ctrl_result)
            self.ctrl_worker.start()

        def on_ctrl_result(self, level, data, text):
            self.ctrl_worker = None
            if level != "ok":
                self.log("✗ %s：%s" % (level, text))
                return
            if self._ctrl_is_start and data.get("started"):
                self.log("✅ 已开始采集：%s" % data.get("source"))
                self.btn_live_on.setEnabled(False)
                self.btn_live_off.setEnabled(True)
                self.alerts_seen = 0
                self.alerts_box.clear()
                self.on_clear()
                # ★ 只在采集真的开始之后才让心跳跑起来
                self.poll_timer.start(self.sp_poll.value())
                self.log("· 轮询已启动（%d ms）" % self.sp_poll.value())
            else:
                self.log("✅ 已停止采集（%s）" % data.get("reason", "ok"))
                self.btn_live_on.setEnabled(True)
                self.btn_live_off.setEnabled(False)

        # ── ★ 心跳 ──
        def _tick(self):
            """QTimer 回调。★ 防重入：上一次还没回来就直接跳过这一拍。"""
            if self.poll_worker is not None:
                self.log("· 上一轮还没回来，跳过这一拍（防请求堆积）")
                return
            self.poll_worker = PollWorker()
            self.poll_worker.got.connect(self.on_poll)
            self.poll_worker.start()

        def on_poll(self, level, data, text):
            self.poll_worker = None              # ★ 先放行，再处理
            if level == "offline":
                self.poll_timer.stop()
                self.log("✗ 服务掉线，轮询已停：%s" % text)
                self.set_lamp("#c0392b", "服务掉线")
                return
            if level != "ok":
                self.log("✗ 轮询失败 %s：%s" % (level, text))
                return
            self._apply(data)

        def _apply(self, d):
            """一份快照 → 铺满所有控件。业务判断全在 day29a 的 fmt_* 里。"""
            color, txt = core.fmt_lamp(d)
            self.set_lamp(color, txt)

            for name, val in core.fmt_frame(d):
                self.lbl_val[name].setText(val)
            self.lbl_stats.setText(core.fmt_stats(d))
            self.lbl_pred.setText(core.fmt_prediction(d))

            self.buf.push(d)                     # ★ 客户端自己攒点
            self.lbl_count.setText(core.fmt_counters(d, self.buf.n))

            fresh, self.alerts_seen, missed = core.new_alerts(self.alerts_seen, d)
            if missed:
                self.alerts_box.appendPlainText(
                    "  … 另有 %d 条未显示（一拍里涨太多，见 /live/alerts）" % missed)
            for a in fresh:
                mark = "🔴" if a.get("level") == "alarm" else "🟡"
                self.alerts_box.appendPlainText(
                    "%s 帧%-4s %s" % (mark, a.get("frame"), a.get("msg")))

            if not d.get("running") and d.get("finished"):
                self.poll_timer.stop()
                self.btn_live_on.setEnabled(True)
                self.btn_live_off.setEnabled(False)
                self.log("· 数据源放完了，轮询自动停止")

            self._redraw(d)

        def _redraw(self, d=None):
            if not self.plot:
                return
            # 【解释】★ 用 setData 复用同一个 curve，不能重调 plot()：
            #        plot(clear=True) 会把阈值线和橙点标记一起清掉。
            xs, ys = self.buf.points()
            self.curve.setData(xs, ys)
            thr = 23.0
            if d:
                thr = (d.get("prediction") or {}).get("threshold_v", thr)
            self.thr.setValue(thr)
            # ★ 把外推结果画出来：橙点 = 预计跌破的时刻
            p = (d or {}).get("prediction") or {}
            if p.get("trustworthy") and self.buf.n:
                self.marker.setData([self.buf.n + p.get("seconds_to_threshold", 0)], [thr])
            else:
                self.marker.setData([], [])

        def closeEvent(self, e):
            self.poll_timer.stop()
            self.mgr.stop()
            super().closeEvent(e)


# ════════════════════════════════════════════════════════════════
# 第 3 部分：main
# ════════════════════════════════════════════════════════════════
def main():
    if "--selftest" in sys.argv or not HAS_GUI:
        if not HAS_GUI:
            print("（本机没有 PySide6，自动切到命令行自测模式）")
        n = core.self_test()
        sys.exit(0 if n >= 11 else 1)

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
