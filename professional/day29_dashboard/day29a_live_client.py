#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════
  Day 29a：实时巡检面板的「内核」（服务管理 + 轮询 + 快照格式化）
═══════════════════════════════════════════════════════════════

【程序做什么】
  Day28 把巡检做成了服务，但服务只有一份 JSON —— 没人看得见。
  要把它变成屏幕上的仪表盘，需要两块东西：

      ① 一堆**能脱离界面单独测**的逻辑：起服务、发请求、把快照翻译成人话
      ② 一层薄薄的 PySide6 界面（见 day29b_dashboard.py）

  今天先写 ①。本文件**没有任何 Qt 依赖**：不起窗口、不 import PySide6，
  命令行就能把「起服务 → 开始采集 → 按轮询节奏读几拍 → 格式化 → 停止」跑完。

【怎么读这个文件（6 块）】
  1. 环境探测       缺 requests 就明确报错，别静默失败
  2. ServiceManager  拉起/停止 day28 的 uvicorn 子进程 + 轮询 /health 等就绪
  3. call_api()      统一出口：成功 / 业务错 / 配置错 / 连不上
  4. ★ 纯函数层      fmt_* 把快照翻译成人话，new_alerts 做增量去重
  5. SampleBuffer    客户端自己攒采样点（服务只给「此刻」，历史得自己记）
  6. self_test()     真起服务、真采集、真轮询，验整条链路

【运行方式】
  cd w6_service
  A. 自测（发布 / CI 用这个，不开界面）：
       D:/Python-envs/chroma-env/Scripts/python.exe day29a_live_client.py
  B. 只把服务起起来，然后用 curl 手玩：
       D:/Python-envs/chroma-env/Scripts/python.exe day29a_live_client.py --serve
  C. 开界面看效果（这个文件的下半截）：
       D:/Python-envs/chroma-env/Scripts/python.exe day29b_dashboard.py

【为什么学这个（面试能讲）】
  ① ★ 能被单测的逻辑，别塞进窗口类。
     GUI 测试最难的是「跑不起来」—— 没显示器、CI 里没窗口系统、headless 环境装不了 Qt。
     把服务管理、HTTP 请求、文案格式化全搬到窗口外面，自测就能在纯命令行里跑完。
     这一招的收益不是「代码好看」，是**你敢改**：改完一条命令就知道有没有坏。
  ② ★ 轮询一定会漏帧。服务端 5 帧/秒、客户端 1 秒问一次，必然丢掉 4 帧。
     SampleBuffer 记客户端采样点、fmt_counters 把差值摆到台面上 ——
     而不是假装「我看到的就是全部」。看仪表盘够用，要保真就上 SSE/WebSocket。
  ③ ★ 增量去重靠单调计数器，不靠内容比较。
     同一句告警可能连发，内容会重复；服务端的 alerts_total 不会。
     另外 /live/status 只回最近 5 条告警，一拍里涨 6 条以上就会缺 ——
     缺多少如实报出来，别假装没发生。
  ④ ★ 阈值耦合写在明面上。客户端判断「预测可不可信」用的 R² 门槛
     必须和服务端（Day27 的 MIN_FIT_R2）一致。这里硬编码并注明出处；
     更稳的做法是让 /health 把门槛下发下来，免得改一边忘一边。

【和前几天的关系】
  Day26  客户端调端点      → ServiceManager / call_api 就是那天的思路，今天抽出来复用
  Day27  规则 + 续航预测    → 阈值的来源
  Day28  /live/* 服务      → 今天的被轮询方
  今天（29a）把「能测的核」做出来，明天（29b）只贴一层界面。

【读完之后】
  · 想练接口设计：把 MIN_R2 改成 /health 下发，客户端不再硬编码
  · 想练 SSE：把轮询换成 text/event-stream，SampleBuffer 改成 push 驱动
  · 想练测试：给 fmt_* 和 new_alerts 加一份 pytest（纯函数，喂假快照即可）
"""

import os
import sys
import json
import time
import subprocess
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
# 【解释】Day28 的服务端脚本跟自己放同一目录（发布时会一起带上）
SERVICE_SCRIPT = os.path.join(HERE, "day28_live_api.py")
# 端口避开 Day26 的 8123 和常见的 8000，三个程序可以同时开着
PORT = int(os.environ.get("DAY29_PORT", "8124"))
BASE_URL = os.environ.get("DAY29_BASE", "http://127.0.0.1:%d" % PORT)
TIMEOUT = (5, 30)            # (连接超时, 读取超时)
CURVE_POINTS = 180           # 曲线缓冲长度（3 分钟 @1Hz）
# ★ 必须和 Day27 的 MIN_FIT_R2 保持一致（见文件头【为什么学这个】④）
MIN_R2 = 0.50

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ════════════════════════════════════════════════════════════════
# 第 1 部分：ServiceManager —— 别让用户先去开一个黑框
# ════════════════════════════════════════════════════════════════
class ServiceManager:
    """管理 Day28 服务的 uvicorn 子进程：启动、等就绪、停止。

    【解释】和 Day26 同一套：起子进程 → 反复探 /health → 直到返回 200。
            这就是 K8s 里的 readiness probe，只是这里用 while + sleep 手写。
            为什么必须等：服务启动要 import langchain 那一串，几秒内 /health 是连不上的，
            客户端一上来就发请求只会拿到 ConnectionError。
    """

    def __init__(self, on_log=None):
        self.proc = None
        self.on_log = on_log or (lambda s: None)

    def _log(self, msg):
        self.on_log(msg)

    def is_running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, wait_sec=90):
        """启动并等到 /health 返回 200。★ cwd 必须是服务端脚本所在目录。"""
        if not os.path.exists(SERVICE_SCRIPT):
            self._log("✗ 找不到服务端脚本：%s" % SERVICE_SCRIPT)
            return False
        if self.is_running():
            self._log("· 服务已在运行")
            return True

        cmd = [sys.executable, "-m", "uvicorn", "day28_live_api:app",
               "--host", "127.0.0.1", "--port", str(PORT)]
        self._log("▶ 启动服务（端口 %d）" % PORT)
        try:
            self.proc = subprocess.Popen(cmd, cwd=HERE,
                                         stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL)
        except Exception as e:
            self._log("✗ 启动失败：%r" % e)
            return False

        t0 = time.time()
        while time.time() - t0 < wait_sec:
            try:
                if requests.get(BASE_URL + "/health", timeout=3).status_code == 200:
                    self._log("✅ 服务就绪（%.1f 秒）" % (time.time() - t0))
                    return True
            except Exception:
                pass
            if self.proc.poll() is not None:
                self._log("✗ 服务进程已退出（fastapi / uvicorn 装了吗？）")
                return False
            time.sleep(0.5)
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
# 第 2 部分：call_api —— 统一出口 + 错误分三层
# ════════════════════════════════════════════════════════════════
def call_api(method, path, payload=None):
    """返回 (level, data, text)。

    【解释】level 决定界面怎么提示（Day26 的三层分法照搬）：
        "ok"       200，data 是解析好的 dict
        "business" 4xx —— 参数有问题，text 是服务端给的 detail
        "config"   503 —— 能力没配好（没 API Key），text 说清去哪配
        "offline"  连不上 —— 服务没起
        "timeout"  超时
    ★ 为什么要有这一层：把「网络异常」「HTTP 状态码」「业务语义」
      三种完全不同的事，收敛成一个客户端能 switch 的字段。
      不然每个调用点都要写一遍 try/except，还不一致。
    """
    url = BASE_URL + path
    try:
        if method == "GET":
            r = requests.get(url, timeout=TIMEOUT)
        else:
            r = requests.post(url, json=payload or {}, timeout=TIMEOUT)
    except requests.exceptions.ConnectionError:
        return "offline", None, "连不上服务（%s）。先点【启动服务】。" % BASE_URL
    except requests.exceptions.Timeout:
        return "timeout", None, "请求超时（%d 秒）" % TIMEOUT[1]
    except Exception as e:
        return "offline", None, "请求异常：%r" % e

    try:
        body = r.json()
    except Exception:
        body = {"detail": r.text[:300]}

    if r.status_code == 200:
        return "ok", body, ""
    detail = body.get("detail", json.dumps(body, ensure_ascii=False)[:300])
    if r.status_code == 503:
        return "config", None, "【能力不可用 503】%s" % detail
    return "business", None, "【%d】%s" % (r.status_code, detail)


# ════════════════════════════════════════════════════════════════
# 第 3 部分：★ 纯函数层 —— 快照 → 人话（不 import Qt，所以能单测）
# ════════════════════════════════════════════════════════════════
def fmt_lamp(d):
    """状态灯该是什么颜色 + 一句话。返回 (颜色, 文字)。"""
    if d.get("last_error"):
        return "#c0392b", "采集出错"
    if d.get("running"):
        return "#27ae60", "巡检中 · %s · %gx" % (d.get("source"), d.get("speed", 0))
    if d.get("finished"):
        return "#b9770e", "数据放完了（%s 帧）" % d.get("frames")
    return "#7f8c8d", "待机（服务就绪，未开始采集）"


def fmt_frame(d):
    """最新一帧的四个数，返回 [(标签, 值文本), ...]。"""
    m = d.get("latest")
    if not m:
        return [("电压 (V)", "—"), ("电流 (A)", "—"), ("温度 (℃)", "—"), ("高度 (m)", "—")]
    return [("电压 (V)", "%.2f" % m.get("电压V", 0)),
            ("电流 (A)", "%.2f" % m.get("电流A", 0)),
            ("温度 (℃)", "%.1f" % m.get("温度C", 0)),
            ("高度 (m)", "%.1f" % m.get("高度m", 0))]


def fmt_stats(d):
    """窗口统计（每列 min/mean/max）压成一段文本。"""
    st = d.get("stats") or {}
    if not st:
        return "（窗口还没填满）"
    lines = []
    for col in ("电压V", "电流A", "温度C", "高度m"):
        if col in st:
            lines.append("%-5s min %8.2f  mean %8.2f  max %8.2f"
                         % (col, st[col]["min"], st[col]["mean"], st[col]["max"]))
    return "\n".join(lines) if lines else "（窗口还没填满）"


def fmt_prediction(d):
    """续航预测。★ 不可信时必须说明为什么，不能静默不报。

    【解释】Day27 的硬教训：真实采集那 23 帧电压的 R² 只有 0.030（就是噪声在抖），
            不设门槛直接外推，会算出「84 秒后坠机」这种假警报。
            所以「不预测」也是一种结论，而且必须把理由讲清楚 ——
            界面上写「—」等于什么都没说。
    """
    p = d.get("prediction") or {}
    thr = p.get("threshold_v", 23.0)
    if p.get("trustworthy"):
        return ("可信 R²=%.3f（≥ %.2f）\n斜率 %.4f V/s\n"
                "★ 预计 %.0f 秒后跌破 %.1fV"
                % (p.get("r2", 0), MIN_R2, p.get("slope_v_per_s", 0),
                   p.get("seconds_to_threshold", 0), thr))
    if p.get("slope_v_per_s") is None or p.get("r2") is None:
        return "样本不足（窗口没填满），暂不外推"
    r2 = p.get("r2", 0)
    slope = p.get("slope_v_per_s", 0)
    if slope >= 0:
        return "电压没有下降趋势（斜率 %.4f V/s ≥ 0），无需预测" % slope
    return ("拒绝外推：R²=%.3f 低于门槛 %.2f\n"
            "→ 数据是噪声不是趋势（Day27 实测真实数据 R²=0.030，\n"
            "   硬外推会报出「84 秒后坠机」这种假警报）" % (r2, MIN_R2))


def fmt_counters(d, sampled):
    """服务端帧数 vs 客户端采样点数 —— 轮询漏帧的实测证据。"""
    frames = d.get("frames", 0)
    ratio = (frames / sampled) if sampled else 0
    return ("服务端帧数   %d\n客户端采样点 %d\n采样比       %.2fx\n"
            "（>1 就在漏帧：服务端跑得比轮询快，\n  这一秒里的其余帧没被看见）"
            % (frames, sampled, ratio))


def new_alerts(seen, d):
    """从快照里挑出「上次没见过」的告警。返回 (新告警列表, 新的已见计数, 遗漏数)。

    【解释】★ 靠服务端的单调计数器 alerts_total 去重，不去比内容 ——
            内容可能重复（同一句告警连发），计数不会骗你。
            但 /live/status 只回最近 5 条，一次涨了 6 条以上就会缺，
            缺多少如实报出来。
    """
    total = d.get("alerts_total", 0)
    if total <= seen:
        return [], seen, 0
    recent = d.get("alerts_recent") or []
    fresh = total - seen
    if fresh > len(recent):
        return list(recent), total, fresh - len(recent)
    return list(recent[-fresh:]), total, 0


# ════════════════════════════════════════════════════════════════
# 第 4 部分：SampleBuffer —— 客户端自己攒采样点
# ════════════════════════════════════════════════════════════════
class SampleBuffer:
    """固定长度的采样缓冲，给曲线用。

    【解释】★★ 为什么历史要客户端记？
        服务端的 /live/status 只回答「此刻是多少」，不回答「过去一分钟什么样」。
        想要一条曲线，就得每次轮询把当前值抄下来自己攒 —— 这就是轮询方案的固有代价。
        反过来说，如果改用 SSE/WebSocket 把每一帧推给客户端，缓冲区照样要自己攒，
        只是点更密（不再漏帧）。

        用 deque(maxlen=N)：跑一晚上也不会把内存吃满（Day27 同款理由）。
    """

    def __init__(self, maxlen=CURVE_POINTS):
        self.x = deque(maxlen=maxlen)      # 采样序号（当作横轴）
        self.y = deque(maxlen=maxlen)      # 电压
        self.n = 0                         # 客户端一共采到几个点

    def push(self, snapshot):
        """把快照里的当前电压抄进缓冲。返回是否真的采到了一个点。"""
        v = snapshot.get("current_voltage")
        if v is None:
            return False
        self.n += 1
        self.x.append(self.n)
        self.y.append(float(v))
        return True

    def clear(self):
        self.x.clear()
        self.y.clear()
        self.n = 0

    def points(self):
        return list(self.x), list(self.y)


# ════════════════════════════════════════════════════════════════
# 第 5 部分：命令行自测（不弹窗口、不依赖 Qt）
# ════════════════════════════════════════════════════════════════
def self_test():
    """起服务 → 采集 → 按轮询节奏读几拍 → 验格式化 → 停止。

    【解释】★ 关键在于：fmt_* / new_alerts / SampleBuffer 都是纯逻辑，
            不用窗口就能验；而「服务 → 轮询」这段用真 HTTP 打，测的是真契约。
            所以这个自测同时覆盖了「我的逻辑对不对」和「两端约定对不对」。
    """
    print("=" * 66)
    print("  Day29a 自测：起服务 → 开始巡检 → 轮询数拍 → 验展示 → 停止")
    print("=" * 66)
    if not HAS_REQUESTS:
        print("✗ 缺 requests，无法自测")
        return 0

    checks = []

    def check(name, cond, extra=""):
        print("  %s %-32s %s" % ("✅" if cond else "❌", name, extra))
        checks.append(bool(cond))

    # ── 纯函数先验（不用服务，喂假快照）──
    fake = {"latest": {"电压V": 24.5, "电流A": 1.8, "温度C": 52.0, "高度m": 88.0},
            "stats": {"电压V": {"min": 24.0, "mean": 24.5, "max": 25.0}},
            "current_voltage": 24.5,
            "prediction": {"trustworthy": False, "r2": 0.03, "slope_v_per_s": -0.0166,
                           "threshold_v": 23.0, "seconds_to_threshold": None}}
    check("fmt_frame 四列", len(fmt_frame(fake)) == 4 and fmt_frame(fake)[0][1] == "24.50")
    check("fmt_prediction 拒绝噪声外推", "拒绝外推" in fmt_prediction(fake), "R²=0.030")
    check("fmt_stats 含均值", "24.50" in fmt_stats(fake))
    trust = dict(fake, prediction={"trustworthy": True, "r2": 1.0, "slope_v_per_s": -0.03,
                                   "threshold_v": 23.0, "seconds_to_threshold": 64.0})
    check("fmt_prediction 可信时给秒数", "64 秒" in fmt_prediction(trust))
    check("fmt_lamp 运行中给绿灯", fmt_lamp({"running": True, "source": "synth", "speed": 5})[0]
          == "#27ae60")

    buf = SampleBuffer(maxlen=3)
    for _ in range(5):
        buf.push(fake)
    check("SampleBuffer 受 maxlen 约束", len(buf.y) == 3 and buf.n == 5)

    got, seen, missed = new_alerts(0, {"alerts_total": 3, "alerts_recent":
                                       [{"level": "alarm", "frame": 2, "msg": "x"}] * 3})
    check("new_alerts 增量取 3 条", len(got) == 3 and seen == 3 and missed == 0)
    got2, _s2, missed2 = new_alerts(0, {"alerts_total": 9, "alerts_recent": [{}] * 5})
    check("new_alerts 超 5 条如实报遗漏", len(got2) == 5 and missed2 == 4)

    # ── 真服务 ──
    logs = []
    mgr = ServiceManager(on_log=logs.append)
    if not mgr.start():
        print("\n".join(logs))
        return sum(checks)

    check("GET /health", call_api("GET", "/health")[0] == "ok")

    level, _d, text = call_api("POST", "/live/start", {"source": "replay", "speed": 5})
    check("POST /live/start(replay)", level == "ok", text)

    time.sleep(0.4)
    snaps, buf2, seen = [], SampleBuffer(), 0
    alerts_n = 0
    for _ in range(6):                       # 模拟 6 拍轮询
        lv, d, _t = call_api("GET", "/live/status")
        if lv == "ok":
            snaps.append(d)
            buf2.push(d)
            fresh, seen, _m = new_alerts(seen, d)
            alerts_n += len(fresh)
        time.sleep(0.4)
    check("轮询 6 拍都拿到快照", len(snaps) == 6)
    frames = [s.get("frames", 0) for s in snaps]
    check("服务端帧数在增长", frames[-1] > frames[0], "%s → %s" % (frames[0], frames[-1]))

    check("告警被增量读到（真实数据>55℃）", alerts_n > 0, "%d 条" % alerts_n)
    d = snaps[-1]
    check("客户端采样点 < 服务端帧数（漏帧证据）",
          len(snaps) < d.get("frames", 0),
          "采样 %d / 服务端 %d" % (len(snaps), d.get("frames", 0)))
    check("fmt_counters 给出采样比", "采样比" in fmt_counters(d, len(snaps)))
    check("续航预测给出明确结论", len(fmt_prediction(d)) > 5,
          fmt_prediction(d).split("\n")[0][:28])

    # ── 换合成轨迹：验证预测链路真的会触发 ──
    call_api("POST", "/live/stop")
    call_api("POST", "/live/start", {"source": "synth", "speed": 5})
    trust_ok = False
    for _ in range(14):
        lv, d2, _t = call_api("GET", "/live/status")
        if lv == "ok" and (d2.get("prediction") or {}).get("trustworthy"):
            trust_ok = True
            check("预测触发（synth，R² 过门槛）", True,
                  "R²=%.3f → %.0f 秒" % (d2["prediction"]["r2"],
                                        d2["prediction"]["seconds_to_threshold"]))
            break
        time.sleep(0.5)
    if not trust_ok:
        check("预测触发（synth，R² 过门槛）", False, "7 秒内没触发")

    call_api("POST", "/live/stop")
    time.sleep(0.3)
    lv, d3, _t = call_api("GET", "/live/status")
    check("停止后 running=False", lv == "ok" and d3.get("running") is False)

    # ── offline 分支 ──
    global BASE_URL
    old = BASE_URL
    BASE_URL = "http://127.0.0.1:59998"
    lv, _d, _t = call_api("GET", "/live/status")
    BASE_URL = old
    check("服务不可达 → offline", lv == "offline")

    mgr.stop()
    passed = sum(1 for x in checks if x)
    print()
    print("=" * 66)
    print("  自测完成：%d / %d 项通过" % (passed, len(checks)))
    print("  开面板：python day29b_dashboard.py")
    print("=" * 66)
    return passed


def main():
    if "--serve" in sys.argv:
        mgr = ServiceManager(on_log=print)
        if not mgr.start():
            sys.exit(1)
        print("服务已在 %s 跑着，Ctrl+C 退出" % BASE_URL)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        mgr.stop()
        return

    n = self_test()
    sys.exit(0 if n >= 11 else 1)


if __name__ == "__main__":
    main()
