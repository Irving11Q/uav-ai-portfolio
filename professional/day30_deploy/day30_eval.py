#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════
  Day 30：把 W6 整套服务「部署化 + 评测化」（收官）
═══════════════════════════════════════════════════════════════

【程序做什么】
  Day25 把数据分析 + 手册检索包成了 HTTP 服务，Day28 把实时巡检包成了服务，
  Day29 做了轮询面板。但「能跑」和「能交付」之间还差两块——也正是我求职自评里
  写的两块短板：

      ① 部署化：别人 clone 下来，一条命令就能把整套服务跑起来（Docker）
      ② 评测化：系统上线前，得有一套能跑出分数的评测集，证明「它确实答得对」

  今天补上这两块，W6 收口：
      - Dockerfile + docker-compose.yml  → 一条 `docker compose up` 起 Day25(8000) + Day28(8001)
      - day30_eval.py（本文件）          → 一套评测集，跑完给 pass/total 分数 + JSON 报告

【评测集覆盖四个维度】
  1. 预测 R² 门槛  ← 直接调 Day27 的 Window.predict_endurance（同一份逻辑，不重写）
  2. 规则告警      ← 直接调 Day27 的 evaluate（温度越线必须报 alarm）
  3. 检索召回      ← 打 Day25 的 /manual/search，看《电池手册》是否召回正确答案
  4. 服务契约冒烟  ← 打两个服务的 /health、/data/summary、/live/*，看端点是否按约定工作

【怎么读这个文件】
  1. 纯逻辑检查（不依赖 fastapi，评测/CI 零成本）：check_prediction_* / check_rule_*
  2. 服务级评测：eval_retrieval / eval_smoke（用 TestClient 在进程内打，不占端口）
  3. 远程模式：--base-url 打真部署（Docker / 服务器）
  4. main：--selftest 只跑纯逻辑；默认跑完整评测并出报告

【运行方式】（先 cd 到发布目录，python = chroma-env）
  A. 纯逻辑自测（最快，CI 用）：
       python day30_eval.py --selftest
  B. 完整评测（进程内起两个服务，证明确实能跑）：
       python day30_eval.py
  C. 打真部署（Docker 起好后）：
       python day30_eval.py --base-url http://127.0.0.1:8000 http://127.0.0.1:8001

【为什么学这个（面试能讲）】
  这是「AI 工程岗」和「调 API 玩玩」的分水岭：
    - 部署化：服务要能被别人一键拉起，健康检查、端口、依赖都得写清楚
    - 评测化：RAG 召回率、Agent 工具命中率、规则误报率——没有量化你不知道改没改好
    - ★ 评测要调「真逻辑」：预测门槛直接复用 Day27 的函数，而不是另写一份「期望答案」，
      否则评测和代码各说各话，等于没测
"""

import os
import sys
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# 【解释】评测不该等 60 秒预热：设 DAY25_PREHEAT=0，首次 /manual/search 再懒建索引。
#         必须在 import day25 之前设好（day25 在模块加载时就读这个环境变量）。
os.environ.setdefault("DAY25_PREHEAT", "0")
sys.path.insert(0, HERE)

# ── 纯逻辑依赖：只 import Day27，不碰 fastapi（评测集核心能离线跑）──
import day27_live_monitor as core  # noqa: E402

COLS = core.COLS
OUT_DIR = os.path.join(HERE, "_out_day30")


# ════════════════════════════════════════════════════════════════
# 第 1 部分：构造测试帧的辅助函数（评测集的「 fixtures 」）
# ════════════════════════════════════════════════════════════════
def make_frame(t, vals):
    """造一帧：和 Day27 的 parse_frame 产出结构完全一致（t / vals / m / raw）。

    【解释】评测要喂「可控的假数据」，才能断言「该触发时必须触发」。
    结构对齐真实帧，Day27 的 evaluate / Window 才认得。
    """
    return {"t": float(t), "vals": list(vals),
            "m": dict(zip(COLS, vals)), "raw": "%.3f" % t}


def _clean_decline(n=30, v0=25.5, v1=24.0):
    """标准线性下降轨迹：电压从 v0 稳稳掉到 v1（终点仍在 23V 报警线之上）。

    【解释】终点必须高于报警线，否则 predict_endurance 会返回『0 秒（已到）』，
    那就测不出『还有多少秒』了。R² 应≈1.0，外推可信。
    """
    step = (v1 - v0) / (n - 1)
    return [make_frame(float(i), [v0 + step * i, 1.8, 40.0, 100.0]) for i in range(n)]


def _noisy_flat(n=30, vmean=24.0, amp=0.15):
    """围绕均值上下抖的噪声序列：像真实采集数据，R² 应≈0，外推=算命，必须拒绝。"""
    import random
    random.seed(7)
    return [make_frame(float(i), [vmean + amp * (random.random() * 2 - 1),
                                   1.8, 40.0, 100.0]) for i in range(n)]


# ════════════════════════════════════════════════════════════════
# 第 2 部分：纯逻辑检查（不依赖服务，--selftest 跑的就是这些）
# ════════════════════════════════════════════════════════════════
def check_prediction_clean():
    """干净直线：必须外推出『还有多少秒跌破 23V』，且 R²≈1.0。"""
    w = core.Window(30)
    for fr in _clean_decline():
        w.push(fr)
    secs, slope, cur = w.predict_endurance()
    ok = (secs is not None) and (secs > 0) and (slope < 0) and (w.fit_r2() > 0.99)
    return ok, "直线下降 seconds=%.1f slope=%.4f r²=%.3f（期望：有秒数且 r²≈1）" % (
        secs or 0.0, slope, w.fit_r2())


def check_prediction_noisy():
    """噪声序列：R² 太低，predict_endurance 必须返回 None（拒绝外推）。"""
    w = core.Window(30)
    for fr in _noisy_flat():
        w.push(fr)
    secs, slope, cur = w.predict_endurance()
    ok = secs is None
    return ok, "噪声抖动 seconds=%s r²=%.3f（期望：None，不外推）" % (secs, w.fit_r2())


def check_rule_hot():
    """温度 60℃ > 55℃：规则引擎必须报 alarm。安全告警不能靠模型，必须毫秒级命中。"""
    fr = make_frame(0.0, [24.0, 2.0, 60.0, 100.0])
    w = core.Window(30)
    w.push(fr)
    levels = [lv for lv, _ in core.evaluate(fr, w)]
    return "alarm" in levels, "温度60℃ 触发等级=%s（期望含 alarm）" % levels


def check_rule_normal():
    """全正常帧：不应有任何告警（误报率要低）。"""
    fr = make_frame(0.0, [24.5, 1.5, 35.0, 100.0])
    w = core.Window(30)
    w.push(fr)
    n = len(core.evaluate(fr, w))
    return n == 0, "正常帧 告警数=%d（期望 0）" % n


# ════════════════════════════════════════════════════════════════
# 第 3 部分：服务级评测（TestClient 在进程内打，或 requests 打远程）
# ════════════════════════════════════════════════════════════════
def _load_questions():
    """读评测集（检索维度的问题 + 期望关键词）。"""
    p = os.path.join(HERE, "eval_questions.json")
    if not os.path.exists(p):
        return {"retrieval": []}
    return json.load(open(p, encoding="utf-8"))


def eval_retrieval(client, post, questions):
    """对 Day25 的 /manual/search 做召回评测。

    【解释】给一句自然语言问题，看《电池手册》召回的片段里有没有「标准答案关键词」。
    这是 RAG 评测最朴素也最硬的指标：答非所问 = 检索失败。
    """
    results = []
    for c in questions.get("retrieval", []):
        q, expect = c["q"], c["expect"]
        try:
            r = post(client, "/manual/search", {"query": q, "top_k": 3})
            hits = (r.json().get("hits") or []) if r.status_code == 200 else []
        except Exception as e:
            results.append((False, "检索「%s」异常：%s" % (q, e)))
            continue
        blob = " ".join((h.get("title", "") + " " + h.get("snippet", "")) for h in hits)
        hit = any(kw in blob for kw in expect)
        results.append((hit, "检索「%s」命中关键词%s：%s（top%d）" % (
            q, "✓" if hit else "✗", expect, len(hits))))
    return results


def eval_smoke(agent, live, get, post):
    """打两个服务的核心端点，验证「部署起来确实能用」。

    【解释】★ 这里特意把 synth 源跑起来再查 /live/status：
        合成轨迹是干净直线，predict_endurance 必然 trustworthy=True 且有秒数；
        这等于在部署层级再验一次 Day27 的 R² 门槛——端点通、逻辑也对。
    """
    results = []

    # ① 两个 /health 必须 200
    for name, cli in (("agent", agent), ("live", live)):
        try:
            r = get(cli, "/health")
            results.append((r.status_code == 200, "%s /health → %d" % (name, r.status_code)))
        except Exception as e:
            results.append((False, "%s /health 异常：%s" % (name, e)))

    # ② Day25 /data/summary：问「电压V」列的统计，必须回 min/mean/max
    try:
        r = post(agent, "/data/summary", {"col": "电压V"})
        ok = r.status_code == 200 and "result" in r.json()
        results.append((ok, "agent /data/summary 电压V → %d" % r.status_code))
    except Exception as e:
        results.append((False, "agent /data/summary 异常：%s" % e))

    # ③ Day28 实时巡检：起 synth → 轮询到足够帧 → 断言预测可信 → 停
    try:
        r = post(live, "/live/start", {"source": "synth", "speed": 0})
        if r.status_code != 200:
            results.append((False, "live /live/start → %d" % r.status_code))
        else:
            pred = None
            for _ in range(150):           # 最多等 15 秒攒够帧
                time.sleep(0.1)
                st = get(live, "/live/status").json()
                if st.get("frames", 0) >= 8:
                    pred = st.get("prediction") or {}
                    break
            ok = bool(pred) and pred.get("trustworthy") is True and \
                (pred.get("seconds_to_threshold") or 0) > 0
            results.append((ok, "live synth 预测 trustworthy=%s seconds=%.1f" % (
                pred.get("trustworthy"), pred.get("seconds_to_threshold") or 0.0)))
            post(live, "/live/stop", {})
    except Exception as e:
        results.append((False, "live 巡检链路异常：%s" % e))

    return results


# ════════════════════════════════════════════════════════════════
# 第 4 部分：报告与入口
# ════════════════════════════════════════════════════════════════
def _report(title, cases):
    passed = sum(1 for ok, _ in cases if ok)
    total = len(cases)
    print("  ┌─ %s（%d/%d 通过）" % (title, passed, total))
    for ok, msg in cases:
        print("  │  %s %s" % ("✓" if ok else "✗", msg))
    return passed, total


def run_selftest():
    """纯逻辑评测：预测门槛 + 规则告警，不依赖任何服务，CI 零成本。"""
    print("═" * 62)
    print("  Day30 评测（纯逻辑 / --selftest）")
    print("═" * 62)
    all_cases = []
    all_cases += [check_prediction_clean(), check_prediction_noisy(),
                  check_rule_hot(), check_rule_normal()]
    p, t = _report("纯逻辑：预测门槛 + 规则告警", all_cases)
    print("─" * 62)
    print("  合计：%d/%d 通过" % (p, t))
    return p, t


def _make_clients():
    """进程内用 TestClient 起两个服务（不占端口、能进 CI、离线可跑）。"""
    from fastapi.testclient import TestClient  # noqa
    import day25_api_service as s25  # noqa
    import day28_live_api as s28  # noqa
    a = TestClient(s25.app)
    l = TestClient(s28.app)
    return a, l


def run_full():
    """完整评测：纯逻辑 + 进程内起服务打检索 + 冒烟。"""
    print("═" * 62)
    print("  Day30 评测（完整 / TestClient 进程内起服务）")
    print("═" * 62)
    q = _load_questions()
    pure = [check_prediction_clean(), check_prediction_noisy(),
            check_rule_hot(), check_rule_normal()]
    p1, t1 = _report("纯逻辑：预测门槛 + 规则告警", pure)

    try:
        agent, live = _make_clients()
    except Exception as e:
        print("  ✗ 起服务失败（缺 fastapi/langchain？）：%s" % e)
        return p1, t1

    def getf(cli, path):
        return cli.get(path)

    def postf(cli, path, body):
        return cli.post(path, json=body)

    with agent, live:
        retr = eval_retrieval(agent, postf, q)
        p2, t2 = _report("检索召回（Day25 /manual/search）", retr)
        smoke = eval_smoke(agent, live, getf, postf)
        p3, t3 = _report("服务契约冒烟（两服务核心端点）", smoke)

    total_p = p1 + p2 + p3
    total_t = t1 + t2 + t3
    print("─" * 62)
    print("  合计：%d/%d 通过" % (total_p, total_t))
    _write_report({
        "pure": pure, "retrieval": retr, "smoke": smoke,
        "passed": total_p, "total": total_t,
    })
    return total_p, total_t


def run_remote(agent_url, live_url):
    """打真部署（Docker 起好后）：用 requests，逻辑和进程内版一致。"""
    import requests  # noqa
    print("═" * 62)
    print("  Day30 评测（远程：%s / %s）" % (agent_url, live_url))
    print("═" * 62)
    q = _load_questions()
    pure = [check_prediction_clean(), check_prediction_noisy(),
            check_rule_hot(), check_rule_normal()]
    p1, t1 = _report("纯逻辑：预测门槛 + 规则告警", pure)

    s = requests.Session()
    getf = lambda cli, path: s.get(cli + path, timeout=10)  # noqa
    postf = lambda cli, path, body: s.post(cli + path, json=body, timeout=30)  # noqa

    retr = eval_retrieval(agent_url, postf, q)
    p2, t2 = _report("检索召回（Day25 /manual/search）", retr)
    smoke = eval_smoke(agent_url, live_url, getf, postf)
    p3, t3 = _report("服务契约冒烟", smoke)

    total_p, total_t = p1 + p2 + p3, t1 + t2 + t3
    print("─" * 62)
    print("  合计：%d/%d 通过" % (total_p, total_t))
    return total_p, total_t


def _write_report(groups):
    os.makedirs(OUT_DIR, exist_ok=True)
    flat = []
    for k, cases in groups.items():
        if k in ("passed", "total"):
            continue
        for ok, msg in cases:
            flat.append({"dimension": k, "passed": ok, "detail": msg})
    json.dump({"passed": groups["passed"], "total": groups["total"], "cases": flat},
              open(os.path.join(OUT_DIR, "eval_report.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("  · 报告已写：%s" % os.path.join(OUT_DIR, "eval_report.json"))


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Day30 评测集（部署化 + 评测化）")
    ap.add_argument("--selftest", action="store_true", help="只跑纯逻辑（不等服务/不加载模型）")
    ap.add_argument("--base-url", nargs=2, metavar=("AGENT", "LIVE"),
                    help="打真部署，例如 --base-url http://127.0.0.1:8000 http://127.0.0.1:8001")
    a = ap.parse_args()

    if a.selftest:
        p, t = run_selftest()
    elif a.base_url:
        p, t = run_remote(a.base_url[0], a.base_url[1])
    else:
        p, t = run_full()
    sys.exit(0 if p == t else 1)


if __name__ == "__main__":
    main()
