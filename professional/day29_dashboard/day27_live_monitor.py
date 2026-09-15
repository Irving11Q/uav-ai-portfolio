#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════
  Day 27：实时飞行数据巡检（串口 → 规则告警 → 续航预测 → AI 总结）
═══════════════════════════════════════════════════════════════

【程序做什么】
  前几天的数据都是「静态 CSV」：Day23 一次性读全表算，Day25/26 由人点按钮问。
  但真实飞行是**一帧一帧流进来的**：串口每 1 秒吐一行「24.7,1.7,59.1,101」，
  你要在**当下**就知道 —— 温度是不是超了？电压还够飞多久？

  今天把 W2 的串口采集和 W5 的 Agent 缝在一起，做成一个「实时巡检员」：

      ① 逐帧读串口（真板子 / 回放历史数据 / 合成下降轨迹，三种来源一个接口）
      ② 规则引擎    ← 毫秒级、零成本，越线立刻报警
      ③ 滑动窗口    ← 只留最近 30 帧，长时间跑不涨内存
      ④ 在线预测    ← 用最近电压的下降斜率，外推「还有几秒跌破安全线」
      ⑤ AI 巡检     ← ★ 降频触发：每 10 帧才把窗口统计交给模型写成人话
      ⑥ 报告落盘    ← 逐帧 jsonl + 结束时的 Markdown 巡检报告

【怎么读这个文件（6 块）】
  1. 数据源       ★ RealSerial / ReplaySerial / SynthSerial，三档同一接口
  2. 规则引擎     阈值越线判断（阈值来自 Day22 手册 + W2 实测分布）
  3. 窗口 + 预测   deque 滑窗、统计量、★ 电压斜率外推 + R² 可信度门槛
  4. AI 巡检      降频调用模型；没 Key 就本地拼装结论，链路不断
  5. 落盘         逐帧 jsonl + 结束时 Markdown 报告
  6. main         逐帧主循环、退出收尾

【运行方式】（先 cd w6_service）
  A. 回放 W2 真实采集的数据（推荐先跑，零硬件零成本）：
       python day27_live_monitor.py --source=replay
  B. 合成「电压持续下降」轨迹，看续航预测真被触发：
       python day27_live_monitor.py --source=synth
  C. 真板子（洋桃1号 STM32，115200）：
       python day27_live_monitor.py --source=real
  D. 自测（发布/CI 用，全程不调模型）：
       python day27_live_monitor.py --selftest
  （python = D:/Python-envs/chroma-env/Scripts/python.exe）

【为什么学这个（面试能讲）】
  这是「AI 应用落地」最容易被问倒的地方 —— 面试官会问：
      「你 1Hz 的传感器数据，难道每秒调一次大模型？」
  正确答案就是今天这套**两级架构**：
    ① 规则引擎管「实时」：越线判断是算术，微秒级、零成本、断网也能用。
       ★ 别把安全告警交给大模型 —— 它慢、贵、还可能胡说。
    ② 大模型管「洞察」：把规则结论 + 窗口统计翻译成人能读的建议。
       ★ 所以必须**降频**（每 10 帧一次），否则成本和延迟都失控。
    ③ 中间的桥是「滑动窗口 + 统计摘要」：模型不需要 30 帧原始数据，
       它需要 min/max/均值/斜率这些已经算好的特征 —— 这也是在省 token。

  今天要记住的四句话：
    ① 数据源要抽象：真串口 / 回放 / 合成，上层代码一行都不用改
    ② 实时性用规则保证，洞察力用模型保证，两者分频
    ③ 预测不一定要模型：线性外推就能回答「还有多久跌破安全线」，
       ★ 但外推必须带可信度（R²）—— 噪声外推出来的全是假警报
    ④ 窗口必须是固定长度的 deque，否则跑一晚上内存就爆了

【和前几天的关系】
  W2  day7/day9 串口采集   → 今天的 RealSerial（接口一模一样）
  Day22 电池手册           → 今天的阈值来源（55℃ 降落、低电压报警）
  Day23 数据分析 Agent     → 那是一次性全表分析，今天是流式增量
  Day25/26 服务 + 客户端   → 今天的报告可再包成端点，供上位机轮询
"""

import os
import sys
import csv, json, time
from collections import deque
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "_out_day27")

# 数据文件候选：源目录与发布目录两种布局
_CSV_CANDIDATES = [os.path.join(HERE, "w2_comms", "flight_data_real.csv"),
                   os.path.join(os.path.dirname(HERE), "w2_comms", "flight_data_real.csv")]
CSV_PATH = next((p for p in _CSV_CANDIDATES if os.path.exists(p)), _CSV_CANDIDATES[0])

COLS = ["电压V", "电流A", "温度C", "高度m"]

# ════════════════════════════════════════════════════════════════
# 第 1 部分：阈值与规则（来自 Day22《电池手册》+ W2 实测数据分布）
# ════════════════════════════════════════════════════════════════
# 【解释】把阈值写成常量 + 出处，别散落在 if 里。
#         手册原文：「超过 55℃ 应尽快降落散热」「电压跌落过快应触发返航」。
#         实测数据是 6S 电池，正常 23~25V，所以低电压线定在 23.0V。
TEMP_ALARM = 55.0        # ℃   （手册：超过 55℃ 尽快降落）
TEMP_WARN = 50.0         # ℃   （手册：温度骤升超 50℃ 触发返航）
VOLT_ALARM = 23.0        # V
VOLT_DROP_RATE = 0.5     # V/s  （跌得这么快说明异常，手册的「跌落过快」）
CURR_WARN = 3.0          # A
# ★ 外推的最低拟合可信度（R²）。
# 【解释】实测对比过：真实那 23 帧电压 R² 只有 0.030（就是噪声在抖），
#         合成下降轨迹 R² = 1.000（真的是直线在掉）。
#         不管 R² 就外推，噪声会被放大成「84 秒后坠机」这种假警报 —— 所以必须设门槛。
MIN_FIT_R2 = 0.50


def evaluate(fr, window):
    """规则引擎：一帧数据进去，告警列表出来。纯算术，不碰模型。"""
    m = fr["m"]          # 【解释】按列名取值（m = 由 vals 数组和 COLS 拼出来的字典）
    alerts = []
    if m["温度C"] > TEMP_ALARM:
        alerts.append(("alarm", "温度 %.1f℃ 超过 %.0f℃ 安全线 → 尽快降落散热"
                       % (m["温度C"], TEMP_ALARM)))
    elif m["温度C"] > TEMP_WARN:
        alerts.append(("warn", "温度 %.1f℃ 超过预警线 %.0f℃" % (m["温度C"], TEMP_WARN)))

    if m["电压V"] < VOLT_ALARM:
        alerts.append(("alarm", "电压 %.2fV 低于 %.1fV 安全线" % (m["电压V"], VOLT_ALARM)))
    if m["电流A"] > CURR_WARN:
        alerts.append(("warn", "电流 %.2fA 超过 %.1fA" % (m["电流A"], CURR_WARN)))

    # ★ 骤降告警用「窗口斜率」而不是「和上一帧比」。
    # 【坑】第一版写的是 (上一帧电压 - 这一帧电压) / dt，结果真实数据里噪声抖动 0.6V
    #       立刻被判成 0.6V/s 骤降，23 帧报了 10 条误报。
    #       单点差分会把噪声放大成告警；滑窗拟合看的是**趋势**，天然抗噪。
    #       实测：真实数据窗口斜率约 -0.013V/s（正常放电），不再触发。
    slope = window.volt_slope()
    if slope < -VOLT_DROP_RATE:
        alerts.append(("alarm", "电压持续下降 %.2fV/s（超过 %.1fV/s）"
                       % (slope, VOLT_DROP_RATE)))
    return alerts


# ════════════════════════════════════════════════════════════════
# 第 2 部分：数据源 —— 三档同一个接口 readline()
# ════════════════════════════════════════════════════════════════
class ReplaySerial:
    """回放历史 CSV，冒充串口。接口和真串口一致：readline() 返回一行字符串。

    【解释】这就是 W2 里 MockSerial / RealSerial 的同一招：
            上层只管 readline()，不知道背后是板子还是文件。
            好处是「没硬件也能开发、能测、能进 CI」。
    """

    def __init__(self, path=CSV_PATH, rounds=1, delay=0.0):
        self.rows = []
        with open(path, "r", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                try:
                    self.rows.append(["%s" % (r.get("时间", "").strip() or "00:00:00")] +
                                     ["%.2f" % float(r[c]) for c in COLS])
                except (KeyError, ValueError):
                    continue          # 脏行跳过，别让一行坏数据搞死整个采集
        self.rounds = rounds
        self.delay = delay
        self.period = 1.0        # 采样周期（秒）：W2 板子是 1Hz，一帧一秒
        self._i = 0
        self._total = len(self.rows) * rounds

    def readline(self):
        if self._i >= self._total:
            return ""                 # 空串 = 数据放完了（真串口里代表超时）
        self._i += 1
        if self.delay:
            time.sleep(self.delay)
        return ",".join(self.rows[(self._i - 1) % len(self.rows)]) + "\n"

    def close(self):
        pass    # 回放/合成没什么要关的，留着是为了和真串口接口一致


class SynthSerial:
    """合成一条「电压持续下降 + 温度升高」的轨迹，专门用来验证续航预测。

    【解释】真实采的那 23 秒电压很稳（跌不下去），预测逻辑永远不触发。
            想看预测真的生效，就得造一条注定会跌破安全线的曲线。
    """

    def __init__(self, n=60, v0=25.2, slope=0.03, delay=0.0):
        self.n = n
        self.v0 = v0
        self.slope = slope      # 每帧下降多少伏（0.03 → 60 帧掉 1.8V，仍在安全线上）
        self.delay = delay
        self.period = 1.0       # 同上：合成数据也按 1Hz 的逻辑时间走
        self._i = 0

    def readline(self):
        if self._i >= self.n:
            return ""
        i = self._i
        self._i += 1
        if self.delay:
            time.sleep(self.delay)
        v = self.v0 - self.slope * i
        a = 1.5 + 0.02 * i
        t = 32.0 + 0.6 * i          # 温度一路涨，最终超过 55℃
        h = 120.0 - 0.8 * i
        return "%.2f,%.2f,%.2f,%.2f\n" % (v, a, t, h)

    def close(self):
        pass    # 回放/合成没什么要关的，留着是为了和真串口接口一致


class RealSerial:
    """真串口（洋桃1号 STM32，115200，逗号分隔 + 换行）。接口同 ReplaySerial。"""

    def __init__(self, port=None, baudrate=115200, timeout=1.0):
        import serial                                    # 延迟 import：没装不影响其他模式
        if port is None:
            from serial.tools import list_ports
            cands = list(list_ports.comports())
            if not cands:
                raise RuntimeError("没找到任何串口 —— 板子插了吗？驱动装了吗？")
            port = cands[0].device
        self.port = port
        self.period = 1.0       # 板子固件 1Hz 发一帧
        self._ser = serial.Serial(port, baudrate, timeout=timeout)
        print("  已打开串口 %s @ %d" % (port, baudrate))

    def readline(self):
        return self._ser.readline().decode("utf-8", "ignore").strip()

    def close(self):
        try:
            self._ser.close()
        except Exception:
            pass


def open_source(kind):
    """按名字造数据源，失败就降级 —— 采集端崩了不能拖垮整个巡检。"""
    if kind == "real":
        try:
            return RealSerial(), "real"
        except Exception as e:
            print("  ⚠️ 真串口打不开（%s），降级为回放数据" % str(e)[:60])
            return ReplaySerial(), "replay(fallback)"
    if kind == "synth":
        return SynthSerial(delay=0.0), "synth"
    return ReplaySerial(), "replay"


# ════════════════════════════════════════════════════════════════
# 第 3 部分：滑动窗口 + 在线续航预测
# ════════════════════════════════════════════════════════════════
class Window:
    """固定长度的滑动窗口，存最近的帧。

    【解释】★ 为什么必须固定长度？
        实时系统要跑几小时，如果 frames 用 list.append 无限增长，
        内存会一直涨（内存泄漏），遍历统计也会越来越慢。
        deque(maxlen=N) 满了自动丢最老的 —— 这才是流式处理的正解。
    """

    def __init__(self, size=30):
        self.frames = deque(maxlen=size)

    def push(self, fr):
        self.frames.append(fr)

    def __len__(self):
        return len(self.frames)

    def stats(self):
        """窗口内每列的 min / mean / max。给模型看的是这些特征，不是原始 30 帧。"""
        out = {}
        for i, c in enumerate(COLS):
            vals = [f["vals"][i] for f in self.frames]
            if vals:
                out[c] = {"min": min(vals), "mean": sum(vals) / len(vals), "max": max(vals)}
        return out

    def volt_slope(self):
        """最小二乘拟合电压对时间的斜率（V/s）。负值 = 在掉电。

        【解释】★ 预测不一定要模型。直线拟合是最古老也最好解释的外推法：
            斜率 = Σ(x-x̄)(y-ȳ) / Σ(x-x̄)²
            拿到斜率，就能回答「还有多久跌破安全线」。
            只有 2 帧没法拟合，所以 len<3 直接返回 0。
        """
        n = len(self.frames)
        if n < 3:
            return 0.0
        xs = [f["t"] for f in self.frames]
        ys = [f["vals"][0] for f in self.frames]        # 0 = 电压V
        mx = sum(xs) / n
        my = sum(ys) / n
        den = sum((x - mx) ** 2 for x in xs)
        if den == 0:
            return 0.0
        return sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / den

    def fit_r2(self):
        """拟合可信度 R²（0~1）。★ 外推之前必须先看它。

        【解释】R² 回答的是「这些点真的有线性关系吗」：
            1.0 = 完美直线（电压真的在稳定下降 → 外推可信）
            0.0 = 纯噪声（电压只是上下抖 → 外推等于算命）
            实测：真实采集数据 R²≈0.03，合成下降轨迹 R²≈1.00。
        """
        n = len(self.frames)
        if n < 3:
            return 0.0
        xs = [f["t"] for f in self.frames]
        ys = [f["vals"][0] for f in self.frames]
        mx = sum(xs) / n
        my = sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        syy = sum((y - my) ** 2 for y in ys)
        sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
        if sxx == 0 or syy == 0:
            return 0.0
        return (sxy * sxy) / (sxx * syy)

    def predict_endurance(self, threshold=VOLT_ALARM):
        """外推「还有几秒跌破安全线」。返回 (秒数 or None, 斜率, 当前电压)。

        【解释】四种情况都要说清楚，不能只报好消息：
            已低于阈值        → 已经到了（0 秒）
            斜率 ≥ 0          → 电压没在掉，不预测（None）
            ★ R² < 阈值       → 掉的那点量只是噪声，不做外推（None）
            否则              → (当前 - 阈值) / 下降速率
        """
        if not self.frames:
            return None, 0.0, 0.0
        cur = self.frames[-1]["vals"][0]
        slope = self.volt_slope()
        if cur < threshold:
            return 0.0, slope, cur
        if slope >= -1e-6:
            return None, slope, cur
        if self.fit_r2() < MIN_FIT_R2:      # ★ 噪声不外推
            return None, slope, cur
        return (cur - threshold) / (-slope), slope, cur

    def energy_used_ah(self):
        """用电流对时间积分估算已消耗电量（Ah）。

        【解释】Ah = ∫ I dt。等间隔采样就是 Σ I × dt；
                真实电池管理芯片（BMS）算剩余电量，用的就是这个思路的升级版。
        """
        f = self.frames
        return sum(f[i]["vals"][1] * (f[i]["t"] - f[i - 1]["t"]) / 3600.0
                   for i in range(1, len(f)))


# ════════════════════════════════════════════════════════════════
# 第 4 部分：AI 巡检（降频调用 + 无 Key 降级）
# ════════════════════════════════════════════════════════════════
MODEL = API_KEY = BASE_URL = None
HAS_LLM = False
try:
    # 【解释】api_config 在源目录 w3_ai/、day21b 在 w5_agent/，发布目录里都在同层 —— 都加上
    for _p in (HERE, os.path.join(os.path.dirname(HERE), "w3_ai"),
               os.path.join(os.path.dirname(HERE), "w5_agent")):
        sys.path.insert(0, _p)
    import api_config
    MODEL = getattr(api_config, "MODEL_NAME", None)
    API_KEY = getattr(api_config, "API_KEY", "")
    BASE_URL = getattr(api_config, "BASE_URL", "")
    if MODEL and API_KEY:
        from day21b_agent_tools import SimpleChatModel
        HAS_LLM = True
except Exception:
    HAS_LLM = False

INSPECT_EVERY = 10      # ★ 每 10 帧才问一次模型（降频，控成本控延迟）

# 【解释】★ 末句必须有：第一版漏了它，模型就只泛泛说「注意电压」，丢了外推结论。
SYS_PROMPT = (
    "你是无人机飞行安全巡检员。用户会给你一段飞行窗口的统计特征和已触发的告警。"
    "请用中文给出一段不超过 120 字的巡检结论：先说整体状态（正常/注意/危险），"
    "再给 1-2 条可执行建议。若特征里给了「约 X 秒后跌破 Y V」，必须原样写进结论。"
    "不要罗列数字，不要重复原始数据，直接说人话。"
)


def _fallback_text(window, alerts):
    """没配 Key 时的本地结论 —— 链路不断，只是不够「像人话」。"""
    st = window.stats()
    if not st:
        return "（窗口还没有数据）"
    sec, slope, _ = window.predict_endurance()
    level = "危险" if any(a[0] == "alarm" for a in alerts) else ("注意" if alerts else "正常")
    parts = ["【本地规则结论·未接入模型】整体状态：%s。" % level,
             "窗口内温度最高 %.1f℃，电压最低 %.2fV。"
             % (st["温度C"]["max"], st["电压V"]["min"])]
    if sec is not None:
        parts.append("按当前下降速率（%.3fV/s）外推，约 %.0f 秒后跌破 %.1fV。"
                     % (slope, sec, VOLT_ALARM))
    if alerts:
        parts.append("本窗口触发 %d 条告警，建议优先处理最严重的一条。" % len(alerts))
    return "".join(parts)


def inspect_ai(window, alerts, sample_n):
    """把「窗口统计 + 告警」交给模型写成人话。降频由调用方控制。"""
    if not HAS_LLM:
        return _fallback_text(window, alerts), "local"
    st = window.stats()
    sec, slope, cur = window.predict_endurance()
    feat = ["窗口：最近 %d 帧，累计 %d 帧" % (len(window), sample_n)]
    for c in COLS:
        if c in st:
            feat.append("%s：min %.2f / 均值 %.2f / max %.2f"
                        % (c, st[c]["min"], st[c]["mean"], st[c]["max"]))
    feat.append("电压斜率 %.4f V/s，拟合可信度 R²=%.3f；已耗电 %.5f Ah"
                % (slope, window.fit_r2(), window.energy_used_ah()))
    if sec is not None:
        feat.append("按此速率，约 %.0f 秒后跌破 %.1fV" % (sec, VOLT_ALARM))
    else:
        feat.append("电压无显著下降趋势（R² 不足），不做续航外推")
    feat.append("本窗口告警：" + ("；".join(a[1] for a in alerts) if alerts else "无"))
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        model = SimpleChatModel(model_id=MODEL, api_key=API_KEY, base_url=BASE_URL)
        resp = model.invoke([SystemMessage(content=SYS_PROMPT),
                             HumanMessage(content="\n".join(feat))])
        return (resp.content or "").strip(), "ai"
    except Exception as e:
        # 【解释】模型抽风（429/超时）不能让巡检停摆 —— 退回本地结论，标清来源。
        return _fallback_text(window, alerts) + "（模型调用失败：%s）" % str(e)[:60], "local"


# ════════════════════════════════════════════════════════════════
# 第 5 部分：落盘（逐帧 jsonl + 结束时 Markdown 报告）
# ════════════════════════════════════════════════════════════════
def write_markdown(path, meta, window, alert_log, inspections):
    st = window.stats()
    sec, slope, _ = window.predict_endurance()
    r2 = window.fit_r2()
    L = ["# 无人机飞行巡检报告", "",
         "- 数据源：%s" % meta["source"],
         "- 生成时间：%s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
         "- 总帧数：%d ｜ 窗口长度：%d" % (meta["frames"], len(window)), "",
         "## 窗口统计（最近 %d 帧）" % len(window), "",
         "| 指标 | min | 均值 | max |", "|---|---|---|---|"]
    for c in COLS:
        if c in st:
            L.append("| %s | %.2f | %.2f | %.2f |"
                     % (c, st[c]["min"], st[c]["mean"], st[c]["max"]))
    L += ["", "## 在线续航预测", "",
          "- 电压斜率：%.4f V/s（负值代表掉电）" % slope,
          "- 拟合可信度 R²：%.3f（低于 %.2f 视为噪声，不做外推）" % (r2, MIN_FIT_R2),
          "- 估算已耗电：%.5f Ah" % window.energy_used_ah()]
    if sec is None:
        L.append("- 结论：电压无显著下降趋势（R²=%.3f），**不触发**续航告警" % r2)
    elif sec == 0:
        L.append("- 结论：**电压已低于 %.1fV 安全线**" % VOLT_ALARM)
    else:
        L.append("- 结论：按当前速率约 **%.0f 秒**后跌破 %.1fV，建议提前返航"
                 % (sec, VOLT_ALARM))
    L += ["", "## 告警时间线", ""]
    L += ["- 第 %d 帧 [%s] %s" % (a["frame"], a["level"], a["msg"]) for a in alert_log] \
        or ["- 全程无越线，未触发任何告警"]
    L += ["", "## AI 巡检结论", ""]
    L += ["- **第 %d 帧**（来源：%s）：%s" % (i["frame"], i["origin"], i["text"])
          for i in inspections] or ["- （未触发巡检）"]
    L.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return path


# ════════════════════════════════════════════════════════════════
# 第 6 部分：主循环
# ════════════════════════════════════════════════════════════════
def run_live(kind, max_frames=0, use_ai=True, quiet=False):
    src, src_name = open_source(kind)
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = os.path.join(OUT_DIR, "frames_%s_%s.jsonl" % (kind, stamp))
    md_path = os.path.join(OUT_DIR, "巡检报告_%s_%s.md" % (kind, stamp))
    # 【解释】★ 帧时间用「逻辑时间」= 帧序号 × 采样周期，而不是墙钟。
    #        为什么？回放和合成数据一秒内就跑完 60 帧，用墙钟算出来的
    #        斜率会是 -120V/s 这种毫无意义的数；真实串口是 1Hz，两者才近似相等。
    #        算变化率、做外推，必须先有正确的 dt。
    period = float(getattr(src, "period", 1.0))

    window = Window(size=30)
    alert_log, inspections = [], []

    n = 0
    t0 = time.time()
    jf = open(jsonl_path, "w", encoding="utf-8")

    print("=" * 64)
    print("  Day27 实时巡检 ｜ 数据源：%s ｜ AI 巡检：%s"
          % (src_name, "开" if (use_ai and HAS_LLM) else "关/降级"))
    print("=" * 64)

    try:
        while True:
            line = src.readline()
            if line is None:
                continue
            line = line.strip()
            if not line:
                break                      # 回放型数据源：读空 = 结束
            parts = line.replace("\t", ",").split(",")
            # 【解释】兼容两种帧格式：真串口/回放是「时间,4个数」（5 段），
            #         合成轨迹没有时间戳（4 段）。多留一档容错，换数据源不用改这里。
            if len(parts) >= 5:
                stamp_s, num_s = parts[0], parts[1:5]
            elif len(parts) == 4:
                stamp_s, num_s = "", parts[0:4]
            else:
                continue                   # 真串口可能有噪声行
            try:
                vals = [float(x) for x in num_s]
            except ValueError:
                continue
            n += 1
            fr = {"t": (n - 1) * period, "vals": vals,
                  "m": dict(zip(COLS, vals)),          # 列名 → 值，读起来像人话
                  "raw": stamp_s if stamp_s else str(n)}
            fr_json = {"frame": n, "电压V": vals[0], "电流A": vals[1],
                       "温度C": vals[2], "高度m": vals[3]}
            jf.write(json.dumps(fr_json, ensure_ascii=False) + "\n")
            jf.flush()                     # 【解释】flush：崩了也不丢已采的帧

            # 顺序：先入窗，再判规则 —— 骤降告警要看「包含当前帧」的窗口趋势
            window.push(fr)
            alerts = evaluate(fr, window)
            for lv, msg in alerts:
                alert_log.append({"frame": n, "level": lv, "msg": msg})

            if not quiet:
                mark = "🔥" if any(a[0] == "alarm" for a in alerts) else (
                    "⚠️" if alerts else "✅")
                print("  [%3d] 电压 %.2fV  电流 %.2fA  温度 %.1fC  高度 %.0fm  %s"
                      % (n, vals[0], vals[1], vals[2], vals[3], mark))
                for lv, msg in alerts:
                    print("        %s %s" % ("🔥" if lv == "alarm" else "⚠️", msg))

            # ★ 降频巡检：每 INSPECT_EVERY 帧（且窗口有内容）才动一次模型
            if use_ai and n % INSPECT_EVERY == 0 and len(window) >= 3:
                text, origin = inspect_ai(window, alerts, n)
                inspections.append({"frame": n, "text": text, "origin": origin})
                print("\n  ── 🔍 第 %d 帧巡检（%s）──" % (n, origin))
                print("     " + text.replace("\n", "\n     ") + "\n")

            if max_frames and n >= max_frames:
                break

    except KeyboardInterrupt:
        print("\n  （用户中断）")
    finally:
        jf.close()
        src.close()

    report = write_markdown(md_path, {"source": src_name, "frames": n},
                            window, alert_log, inspections)
    print("=" * 64)
    print("  采集 %d 帧 ｜ 告警 %d 条 ｜ 巡检 %d 次" % (n, len(alert_log), len(inspections)))
    print("  逐帧数据：%s\n  巡检报告：%s" % (jsonl_path, report))
    print("=" * 64)
    return {"frames": n, "alarms": sum(1 for a in alert_log if a["level"] == "alarm"),
            "alerts": len(alert_log), "inspections": len(inspections),
            "report": report}


def self_test():
    """自测：回放真实数据验规则 + 合成下降轨迹验预测，全程不调模型。"""
    print("\n【自测 1/2】回放 W2 真实采集数据（23 帧），验证规则引擎")
    r1 = run_live("replay", use_ai=False, quiet=True)
    ok1 = r1["frames"] == 23 and r1["alarms"] >= 4      # 真值：温度 >55℃ 共 4 条

    print("\n【自测 2/2】合成一条电压下降轨迹（60 帧），验证续航预测触发")
    r2 = run_live("synth", use_ai=False, quiet=True)
    # 合成轨迹 25.2V 起、每帧 -0.06V → 必然跌破 23.0V，预测应给出正秒数
    import re
    txt = open(r2["report"], encoding="utf-8").read()
    m = re.search(r"约 \*\*(\d+) 秒\*\*后跌破", txt)
    ok2 = r2["frames"] == 60 and m is not None and int(m.group(1)) > 0

    print("\n" + "=" * 64)
    print("  自测结果：")
    print("   %s 规则引擎：%d 帧 / %d 条告警（温度>55℃ 应为 4 条）"
          % ("✅" if ok1 else "❌", r1["frames"], r1["alarms"]))
    print("   %s 续航预测：%d 帧 / 预测 %s"
          % ("✅" if ok2 else "❌", r2["frames"],
             ("约 %s 秒后跌破 23V" % m.group(1)) if m else "未触发"))
    passed = (1 if ok1 else 0) + (1 if ok2 else 0)
    print("   通过 %d / 2" % passed)
    print("=" * 64)
    return passed


def main():
    o = {"source": "replay", "frames": 0, "no_ai": False, "selftest": False}
    for a in sys.argv[1:]:
        if a == "--selftest":
            o["selftest"] = True
        elif a == "--no-ai":
            o["no_ai"] = True
        elif a.startswith("--source="):
            o["source"] = a.split("=", 1)[1]
        elif a.startswith("--frames="):
            o["frames"] = int(a.split("=", 1)[1])
    if o["selftest"]:
        sys.exit(0 if self_test() == 2 else 1)
    run_live(o["source"], max_frames=o["frames"], use_ai=not o["no_ai"])


if __name__ == "__main__":
    main()
