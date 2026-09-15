#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════
  Day 28：把实时巡检变成服务（后台采集线程 + /live/* 端点）
═══════════════════════════════════════════════════════════════

【程序做什么】
  Day27 的巡检员是「一个脚本」：它自己读串口、自己打印、自己写报告。
  问题是 —— 只能有一个巡检员，而且只能在那台跑脚本的电脑上看。

  今天把它**拆成服务**：
      · 采集跑在服务进程的**后台线程**里，持续进行（不等人问）
      · 客户端随时 GET /live/status 拿到「此刻」的快照
      · 谁来问都一样 —— 上位机、小程序、网页都能看同一个飞行状态

  端点（5 个）：
      POST /live/start     开始采集  {source: replay|synth|real, speed: 倍数}
      POST /live/stop      停止采集
      GET  /live/status    ★ 当前快照：最新帧 / 窗口统计 / 续航预测 / 最近告警
      GET  /live/alerts    完整告警时间线（?limit=N）
      GET  /live/report    当前巡检报告的 Markdown 文本
      GET  /health         服务与采集状态

【怎么读这个文件（5 块）】
  1. 复用层      直接 import day27 的数据源/规则/窗口/预测，一行都不重写
  2. Collector   ★ 后台采集线程 + 锁 + 快照（本文件的核心）
  3. app + lifespan  启动/关闭时管好线程
  4. 五个端点   都是「读快照」，只有 start/stop 改状态
  5. self_test   用 TestClient 真起服务、真采集、真读状态

【运行方式】
  A. 自测（推荐先跑，不占端口）：
       cd w6_service
       D:/Python-envs/chroma-env/Scripts/python.exe day28_live_api.py
  B. 真起服务，然后浏览器开 http://127.0.0.1:8000/docs 点着玩：
       D:/Python-envs/chroma-env/Scripts/python.exe -m uvicorn day28_live_api:app --port 8000
  C. 起服务后手动跑一段：
       curl -X POST localhost:8000/live/start -H "Content-Type: application/json" -d "{\\"source\\":\\"synth\\",\\"speed\\":5}"
       curl localhost:8000/live/status

【为什么学这个（面试能讲）】
  ① ★ 状态必须住在服务进程里，不能放客户端。
     飞行的「当前状态」只有一个（就是那架飞机），十个客户端来看都得看到同一份。
     如果采集放客户端，等于每个客户端都在看自己编的数据。
  ② ★ 后台线程 + 锁，是因为 FastAPI 的端点跑在**线程池**里：
     采集线程在写、请求线程在读，同一个 window 对象被两个线程摸 —— 不加锁就是玄学崩溃。
     本项目用「写的时候加锁、读的时候返回快照副本」的简单策略。
  ③ ★ 轮询优先，别一上来就 WebSocket。
     /live/status 一秒问一次，简单、无状态、能用 curl 调、断线自动恢复；
     对「人看的仪表盘」这个场景完全够。SSE / WebSocket 是流量上万之后的升级项，
     不是第一天就该上的东西 —— 面试时能说清「为什么先不上」比会用更难。
  ④ ★ 线程生命周期要有人管：start 必须幂等（点两次不能起两个采集线程）、
     stop 要 join、lifespan 关闭时兜底停（否则 uvicorn 退出后线程还赖着）。

【和前几天的关系】
  Day25  FastAPI 服务化基础（lifespan / Pydantic / 降级）
  Day26  桌面客户端调端点（轮询这套它已经会了）
  Day27  实时巡检逻辑（数据源 / 规则 / 预测 / AI 巡检）← 今天直接复用
  今天把两者接上：巡检逻辑常驻服务，客户端变成「看一眼快照」
"""

import os
import sys
import time
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal

HERE = os.path.dirname(os.path.abspath(__file__))
# 【解释】复用 Day27：它已经把「数据源 / 规则 / 滑窗 / 预测 / AI 巡检」都写好了。
#         今天的重点是**服务化**，不是重写一份巡检逻辑。
for _p in (HERE, os.path.join(os.path.dirname(HERE), "w3_ai"),
           os.path.join(os.path.dirname(HERE), "w5_agent")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import day27_live_monitor as core          # noqa: E402

try:
    from fastapi import FastAPI, HTTPException, Query   # noqa: E402
    from pydantic import BaseModel, Field               # noqa: E402
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

OUT_DIR = os.path.join(HERE, "_out_day28")
WINDOW_SIZE = 30        # 和 Day27 保持一致


# ════════════════════════════════════════════════════════════════
# 第 1 部分：Collector —— 后台采集线程 + 共享快照
# ════════════════════════════════════════════════════════════════
def parse_frame(line, t, index):
    """把一行串口文本变成帧字典。和 Day27 主循环里那段是同款逻辑。

    【解释】两种帧格式都认：真串口/回放是「时间,4个数」，合成轨迹没有时间戳。
    """
    parts = [p for p in line.strip().replace("\t", ",").split(",") if p != ""]
    if len(parts) >= 5:
        num = parts[1:5]
        stamp = parts[0]
    elif len(parts) == 4:
        num = parts[0:4]
        stamp = str(index)
    else:
        return None
    try:
        vals = [float(x) for x in num]
    except ValueError:
        return None
    return {"t": t, "vals": vals, "m": dict(zip(core.COLS, vals)), "raw": stamp}


class Collector:
    """持续采集 + 规则判断 + 降频 AI 巡检，并把「此刻的状态」暴露成快照。

    【解释】★ 为什么读要返回「快照」而不是直接给 window 对象？
        把内部对象交出去，调用方可能一边遍历一边被采集线程改 —— 那是竞态。
        复制一份出来（快照），锁的范围就只在复制那一瞬间，简单且安全。
    """

    def __init__(self, window_size=WINDOW_SIZE):
        self._lock = threading.Lock()
        self._thread = None
        self._stop = threading.Event()
        self._window_size = window_size
        self._reset_state()

    def _reset_state(self):
        self.window = core.Window(self._window_size)
        self.frames = 0
        self.latest = None
        self.alert_log = []           # [{"frame","level","msg","t"}]
        self.inspections = []      # [{"frame","text","origin"}]
        self.source_name = "-"
        self.speed = 0.0
        self.started_at = None
        self.last_error = None
        self.finished = False

    # ── 控制 ──
    def is_running(self):
        t = self._thread
        return bool(t and t.is_alive())

    def start(self, kind="synth", speed=1.0):
        """启动采集。★ 幂等：已经在跑就直接返回，不会起第二个线程。"""
        with self._lock:
            if self.is_running():
                return False, "已经在采集了（先 /live/stop）"
            try:
                src, name = core.open_source(kind)
            except Exception as e:
                return False, "数据源打不开：%r" % e
            self._reset_state()
            self._src = src
            self.source_name = name
            self.speed = float(speed)
            self.started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._stop.clear()
            # 【解释】daemon=True：万一忘了 stop，进程退出时线程也不赖着。
            #         但正常路径还是要 lifespan 里显式 stop（见下）。
            self._thread = threading.Thread(target=self._loop, daemon=True,
                                            name="day28-collector")
            self._thread.start()
        return True, name

    def stop(self, wait=True):
        self._stop.set()
        t = self._thread
        if t and wait and t.is_alive():
            t.join(timeout=10)      # 等它把这一帧处理完，别硬杀
        with self._lock:
            self._thread = None
        return True

    # ── 采集主循环（跑在后台线程里）──
    def _loop(self):
        period = float(getattr(self._src, "period", 1.0))
        # 【解释】speed=0 → 全速跑（自测用）；否则按采样周期折算真实节奏。
        delay = 0.0 if self.speed <= 0 else period / self.speed
        n = 0
        try:
            while not self._stop.is_set():
                line = self._src.readline()
                if not line:
                    break           # 回放/合成数据放完了
                n += 1
                fr = parse_frame(line, (n - 1) * period, n)
                if fr is None:
                    n -= 1
                    continue
                with self._lock:                       # ★ 写共享状态：持锁
                    self.window.push(fr)
                    self.latest = fr
                    self.frames = n
                    for lv, msg in core.evaluate(fr, self.window):
                        self.alert_log.append({"frame": n, "level": lv, "msg": msg,
                                            "t": fr["t"]})
                # AI 巡检放在锁外面：它要几秒，不能抱着锁等模型
                if (core.HAS_LLM and n % core.INSPECT_EVERY == 0
                        and len(self.window) >= 3):
                    with self._lock:
                        alerts_now = [a for a in self.alert_log if a["frame"] == n]
                        snap = self.window
                    text, origin = core.inspect_ai(
                        snap, [(a["level"], a["msg"]) for a in alerts_now], n)
                    with self._lock:
                        self.inspections.append({"frame": n, "text": text,
                                                 "origin": origin})
                if delay:
                    time.sleep(delay)
        except Exception as e:
            with self._lock:
                self.last_error = repr(e)[:200]
        finally:
            try:
                self._src.close()
            except Exception:
                pass
            with self._lock:
                self.finished = True

    # ── 读状态 ──
    def status(self):
        """返回一份快照（dict）。端点直接把它丢给 JSON 序列化。"""
        with self._lock:
            st = self.window.stats()
            sec, slope, cur = self.window.predict_endurance()
            r2 = self.window.fit_r2()
            cur_volt = self.latest["m"]["电压V"] if self.latest else None
            return {
                "running": self.is_running(),
                "source": self.source_name,
                "speed": self.speed,
                "started_at": self.started_at,
                "frames": self.frames,
                "window_len": len(self.window),
                "latest": (dict(self.latest["m"]) if self.latest else None),
                "latest_t": (self.latest["t"] if self.latest else None),
                "stats": st,
                "current_voltage": cur_volt,
                "energy_ah": self.window.energy_used_ah(),
                "prediction": {
                    "seconds_to_threshold": sec,
                    "slope_v_per_s": slope,
                    "r2": r2,
                    "threshold_v": core.VOLT_ALARM,
                    "trustworthy": bool(sec is not None),
                },
                "alerts_total": len(self.alert_log),
                "alerts_alarm": sum(1 for a in self.alert_log if a["level"] == "alarm"),
                "alerts_recent": self.alert_log[-5:],
                "inspections_recent": self.inspections[-2:],
                "ai_enabled": bool(core.HAS_LLM),
                "finished": self.finished,
                "last_error": self.last_error,
            }

    def alerts_tail(self, limit=50):
        with self._lock:
            return self.alert_log[-limit:]

    def report_markdown(self):
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, "report_%s.md"
                            % datetime.now().strftime("%Y%m%d_%H%M%S"))
        with self._lock:
            kw = dict(meta={"source": self.source_name, "frames": self.frames},
                      window=self.window, alert_log=self.alert_log,
                      inspections=self.inspections)
        core.write_markdown(path, **kw)
        with open(path, "r", encoding="utf-8") as f:
            return path, f.read()


# ════════════════════════════════════════════════════════════════
# 第 2 部分：app + lifespan
# ════════════════════════════════════════════════════════════════
COLLECTOR = Collector()


@asynccontextmanager
async def lifespan(app):
    """服务启动时不自动采集（等人 /live/start），关闭时兜底停线程。

    【解释】★ 为什么关闭一定要 stop？
        uvicorn 退出时若采集线程还在跑，进程会拖到线程结束才走；
        更糟的是真串口模式下端口一直被占着，下次启动就打不开了。
    """
    print("  ▶ Day28 就绪。POST /live/start 开始采集。")
    yield
    if COLLECTOR.is_running():
        print("  ■ 服务关闭，停止采集线程")
        COLLECTOR.stop()


if HAS_FASTAPI:
    app = FastAPI(title="Day28 实时飞行巡检服务", version="1.0", lifespan=lifespan)

    class StartIn(BaseModel):
        # 【解释】Literal 让非法来源在 Pydantic 层就被挡成 422（Day25 学过的套路）
        source: Literal["replay", "synth", "real"] = Field(
            "synth", description="replay=回放历史数据 synth=合成下降轨迹 real=真串口")
        speed: float = Field(1.0, ge=0, le=100,
                             description="播放倍速，0=全速跑（自测用）")

    @app.get("/health", summary="服务与采集状态")
    def health():
        return {"status": "ok", "version": "1.0",
                "collecting": COLLECTOR.is_running(),
                "ai_enabled": bool(core.HAS_LLM),
                "window_size": WINDOW_SIZE}

    @app.post("/live/start", summary="开始采集")
    def live_start(body: StartIn):
        ok, info = COLLECTOR.start(body.source, body.speed)
        if not ok:
            raise HTTPException(status_code=409, detail=info)   # 409 = 状态冲突
        return {"started": True, "source": info, "speed": body.speed}

    @app.post("/live/stop", summary="停止采集")
    def live_stop():
        if not COLLECTOR.is_running():
            return {"stopped": False, "reason": "当前没有在采集"}
        COLLECTOR.stop()
        return {"stopped": True}

    @app.get("/live/status", summary="★ 当前快照")
    def live_status():
        return COLLECTOR.status()

    @app.get("/live/alerts", summary="告警时间线")
    def live_alerts(limit: int = Query(50, ge=1, le=500)):
        return {"count": len(COLLECTOR.alerts_tail(limit)),
                "alerts": COLLECTOR.alerts_tail(limit)}

    @app.get("/live/report", summary="巡检报告（Markdown 文本）")
    def live_report():
        path, text = COLLECTOR.report_markdown()
        return {"path": os.path.basename(path), "markdown": text}


# ════════════════════════════════════════════════════════════════
# 第 3 部分：自测（不占端口）
# ════════════════════════════════════════════════════════════════
def self_test():
    if not HAS_FASTAPI:
        print("✗ 缺 fastapi，无法自测")
        return 0
    from fastapi.testclient import TestClient
    checks = []
    print("=" * 64)
    print("  Day28 自测：起服务 → 后台采集 → 读快照/告警/报告 → 停")
    print("=" * 64)

    def check(name, cond, extra=""):
        print("  %s %s%s" % ("✅" if cond else "❌", name, ("  " + extra) if extra else ""))
        checks.append(bool(cond))

    with TestClient(app) as c:
        r = c.get("/health")
        check("GET /health", r.status_code == 200 and r.json()["collecting"] is False,
              "初始应为未采集")

        # ── 第一轮：回放真实数据，验证「告警真的被记下并能通过端点读到」
        r = c.post("/live/start", json={"source": "replay", "speed": 5})
        check("POST /live/start（replay）", r.status_code == 200, r.json().get("source", ""))
        time.sleep(2.5)                      # 5 帧/秒 × 2.5 秒 ≈ 12 帧

        d = c.get("/live/status").json()
        check("GET /live/status", d["frames"] > 0 and d["running"],
              "frames=%s running=%s" % (d["frames"], d["running"]))
        check("告警链路（真实数据温度>55℃）", d["alerts_alarm"] > 0,
              "alarm=%s total=%s" % (d["alerts_alarm"], d["alerts_total"]))

        # ★ 幂等：再点一次必须 409，而不是起第二个线程
        r2 = c.post("/live/start", json={"source": "replay", "speed": 5})
        check("重复 start 被拒（幂等）", r2.status_code == 409, r2.json().get("detail", ""))

        r = c.post("/live/start", json={"source": "不是来源"})
        check("非法 source → 422", r.status_code == 422)

        # ── 第二轮：合成下降轨迹，验证「预测真的触发」（R² 门槛放行）
        c.post("/live/stop")
        r = c.post("/live/start", json={"source": "synth", "speed": 5})
        check("重新 start（synth）", r.status_code == 200, r.json().get("source", ""))
        time.sleep(2.5)
        p = c.get("/live/status").json()["prediction"]
        check("预测触发（合成轨迹）", bool(p["trustworthy"]) and p["seconds_to_threshold"] > 0,
              "R²=%.3f → 约 %.0f 秒后跌破 %.1fV"
              % (p["r2"], p["seconds_to_threshold"] or -1, p["threshold_v"]))

        r = c.get("/live/alerts?limit=5")
        check("GET /live/alerts", r.status_code == 200, "count=%s" % r.json()["count"])

        r = c.get("/live/report")
        txt = r.json().get("markdown", "")
        check("GET /live/report", r.status_code == 200 and "巡检报告" in txt,
              "%d 字符" % len(txt))

        r = c.post("/live/stop")
        check("POST /live/stop", r.status_code == 200 and r.json()["stopped"])
        time.sleep(0.3)
        d = c.get("/live/status").json()
        check("停止后 running=False", d["running"] is False)

    passed = sum(1 for x in checks if x)
    print("=" * 64)
    print("  自测完成：%d / %d 项通过" % (passed, len(checks)))
    print("  起服务：python -m uvicorn day28_live_api:app --port 8000")
    print("=" * 64)
    return passed


if __name__ == "__main__":
    if "--selftest" in sys.argv or not HAS_FASTAPI:
        if not HAS_FASTAPI:
            print("（缺 fastapi，自动切到自测模式）")
        n = self_test()
        sys.exit(0 if n >= 8 else 1)
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
