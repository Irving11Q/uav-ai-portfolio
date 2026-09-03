#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════
  Day 19-C：效果评估 + bad case 分析 —— 第 4 周最后一块
═══════════════════════════════════════════════════════════════

【这个文件做什么】
    Day19-A/B 让系统能跑了。但「能跑」和「能用」之间还差一件事：
    **它到底有多准？答错的题为什么不準？**

    今天把 Day18-A 的评测集固化成可复跑的脚本，跑完自动：
      ① 算出 Top-1 / Top-3 命中率、离题拒答率
      ② 把答错的题挑出来，自动归因（召回失败？排序失败？误拒？）
      ③ 写一份 md 报告，可以直接拿去当面试材料

【为什么这一环不能省】
    面试官问「你怎么证明你的 RAG 有效」：
        答「我感觉还行」                                    → 零分
        答「6 道题评测集上 Top-1 命中 4/6，bad case 集中在两类…」→ 满分
    这份报告本身就是「评测集工程化」能力的证据 ——
    而这恰恰是很多人做 RAG 项目时唯一缺的一环。

【怎么读这个文件】
    1. 先跑一遍看输出（--threshold 换个值能看阈值的影响）
    2. evaluate_one()   —— 单题怎么判对错
    3. diagnose_case()  —— 错题怎么自动归因（本文件最有价值的部分）
    4. write_report()   —— 报告怎么生成

【运行方式】
    D:/Python-envs/chroma-env/Scripts/python.exe day19c_eval_badcase.py
    D:/Python-envs/chroma-env/Scripts/python.exe day19c_eval_badcase.py --threshold 0.45
    （不加 --threshold 就用 Day18-B 定下的 0.50）

【产物】评测报告_Day19.md
【依赖】Day18-A（评测集）、Day18-B（默认阈值）、Day19-B（真检索链路）
"""

# ════════════════════════════════════════════════════════════════
# 第 1 部分：环境 —— 全复用，只写"评估"这一层新东西
# ════════════════════════════════════════════════════════════════

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from day18a_baseline_eval import EVAL_SET, OFF_TOPIC_QUESTIONS
# 【解释】复用 Day18-A 的评测集：6 道在题（带期望章节）+ 3 道离题。
#         不重新造轮子，也保证今天的数字能和 Day18 直接对比。

from day18b_recall_tuning import SCORE_THRESHOLD
from day19b_rag_backbone import RealRetriever
# 【解释】用 Day19-B 搭好的真检索链路来评测，测的就是成品本身，而不是简化版。


def pad(s, width):
    """中文按 2 格宽补齐，让控制台表格对得齐。"""
    # 【解释】终端里一个汉字占两列，直接按 len() 补空格会歪。
    w = sum(2 if ord(c) > 0x2E80 else 1 for c in s)
    return s + " " * max(0, width - w)


# ════════════════════════════════════════════════════════════════
# 第 2 部分：单题评估 —— 对错到底怎么定
# ════════════════════════════════════════════════════════════════

def evaluate_one(retriever, question, expected=None):
    """
    跑一道题，返回结果字典。

    expected 为 None 表示这是离题问题，期望行为是「拒答」。
    """
    refs = retriever.retrieve(question)
    titles = [t for t, body, score in refs]
    scores = [score for t, body, score in refs]
    # 【解释】把 (标题, 正文, 分数) 三元组拆成两个列表，判定和展示都要用。

    if expected is None:
        # ── 离题问题：拒答了才算对 ──
        return {
            "question": question,
            "kind": "离题",
            "expected": "拒答",
            "titles": titles,
            "scores": scores,
            "refused": retriever.last_refused,
            "reason": retriever.last_reason,
            "ok": retriever.last_refused,
        }

    hit1 = bool(titles) and expected in titles[0]
    hit3 = any(expected in t for t in titles)
    # 【解释】Top-1 命中看第一条；Top-3 命中看前三条里有没有。
    #         用 in 而不是 ==，因为标题可能带前后缀。

    return {
        "question": question,
        "kind": "在题",
        "expected": expected,
        "titles": titles,
        "scores": scores,
        "refused": retriever.last_refused,
        "reason": retriever.last_reason,
        "hit1": hit1,
        "hit3": hit3,
        "ok": hit1,
        # 【解释】只有 Top-1 命中才算对 —— 因为生成环节最看重第 1 条，
        #         排在第 2、3 位的资料对答案的影响要小得多。
    }


# ════════════════════════════════════════════════════════════════
# 第 3 部分：跑全量 + 汇总指标
# ════════════════════════════════════════════════════════════════

def run_all(retriever):
    """跑完整评测集，返回 (逐题明细列表, 汇总字典)。"""
    rows = []
    for question, expected in EVAL_SET:
        rows.append(evaluate_one(retriever, question, expected))
    for question in OFF_TOPIC_QUESTIONS:
        rows.append(evaluate_one(retriever, question, None))
    # 【解释】在题和离题走同一个评估函数，区别只在 expected 是不是 None。

    in_topic = [r for r in rows if r["kind"] == "在题"]
    off_topic = [r for r in rows if r["kind"] == "离题"]
    # 【解释】分两组统计 —— 两组关心的指标根本不是一回事。

    summary = {
        "top1": sum(1 for r in in_topic if r.get("hit1")),
        "top3": sum(1 for r in in_topic if r.get("hit3")),
        "in_total": len(in_topic),
        "refuse_ok": sum(1 for r in off_topic if r["ok"]),
        "off_total": len(off_topic),
        "wrong_reject": sum(1 for r in in_topic if r["refused"]),
        # 【解释】误拒 = 手册里明明有答案，却被阈值挡掉了。
        #         这个数字是"阈值设太高"的直接信号。
        "threshold": retriever.threshold,
        "mode": retriever.mode,
    }
    return rows, summary


# ════════════════════════════════════════════════════════════════
# 第 4 部分：自动归因 —— 本文件最有价值的部分
# ════════════════════════════════════════════════════════════════

def diagnose_case(row):
    """
    给一道错题自动归因，返回 (类型, 说明, 建议)。

    ★ 这一步是把「感觉不准」变成「知道该改哪」的关键。
      RAG 调优最怕的就是瞎调参数 —— 先归因，再动手。
    """
    if row["kind"] == "离题":
        best = max(row["scores"]) if row["scores"] else 0.0
        return ("该拒答却没拒答",
                "手册里没有相关内容，但最高相似度 %.3f 仍然过了阈值 %.2f"
                % (best, row["threshold"]),
                "调高 --threshold；或者加一道大模型二次确认再决定答不答")

    if row["refused"]:
        return ("误拒",
                "手册里有「%s」，却被阈值挡掉了（%s）" % (row["expected"], row["reason"]),
                "调低 --threshold；若一调低就开始漏拒，说明得从查询改写入手，"
                "而不是继续拧阈值")

    if not row.get("hit3"):
        best = max(row["scores"]) if row["scores"] else 0.0
        actual = row["titles"][0] if row["titles"] else "无"
        return ("召回失败",
                "期望「%s」，前 3 条里都没有；实际最高分 %.3f 给的是「%s」"
                % (row["expected"], best, actual),
                "多半是措辞对不上 —— 看 Day18-B 的查询改写能否覆盖这种说法")

    pos = next((i + 1 for i, t in enumerate(row["titles"])
                if row["expected"] in t), None)
    # 【解释】找到期望项实际排在第几位。next(...) 取第一个满足的下标。
    return ("排序失败",
            "期望「%s」确实被召回了，但排在第 %s 位，没进第 1" % (row["expected"], pos),
            "考虑 Day18-C 的重排序；或把块切小一点，让每块主题更聚焦")


def collect_badcases(rows, threshold):
    """把答错的题连同归因一起收出来。"""
    bad = []
    for row in rows:
        row = dict(row)
        row["threshold"] = threshold
        # 【解释】补一个 threshold 进字典，diagnose_case 里要用它拼说明文字。

        if row["kind"] == "在题" and not row.get("hit1"):
            kind, why, fix = diagnose_case(row)
        elif row["kind"] == "离题" and not row["ok"]:
            kind, why, fix = diagnose_case(row)
        else:
            continue
            # 【解释】答对的题跳过。

        row["bad_kind"] = kind
        row["why"] = why
        row["fix"] = fix
        bad.append(row)
    return bad


# ════════════════════════════════════════════════════════════════
# 第 5 部分：输出到控制台 + 写成 md 报告
# ════════════════════════════════════════════════════════════════

def print_detail(rows):
    print("\n【逐题明细】")
    for i, row in enumerate(rows, 1):
        flag = "  OK" if row["ok"] else "  ✗ "
        actual = row["titles"][0] if row["titles"] else "（拒答）"
        score = "%.3f" % row["scores"][0] if row["scores"] else "  -  "
        print("  %s %2d. %s 期望:%s 实际:%s %s"
              % (flag, i, pad(row["question"], 30), pad(row["expected"], 20),
                 pad(actual, 20), score))


def print_badcases(bad):
    if not bad:
        print("\n【Bad Case】本次没有错题。")
        return
    print("\n【Bad Case 归因】共 %d 道" % len(bad))
    for i, row in enumerate(bad, 1):
        print("  %d. 「%s」→ %s" % (i, row["question"], row["bad_kind"]))
        print("     %s" % row["why"])
        print("     建议：%s" % row["fix"])


def write_report(rows, summary, bad, path):
    """写成 md 报告 —— 这份东西可以直接拿去面试。"""
    L = []
    L.append("# Day19 评测报告：无人机手册问答系统")
    L.append("")
    L.append("> 自动生成于 %s" % time.strftime("%Y-%m-%d %H:%M"))
    L.append("> 配置：%s ｜ 拒答阈值 %.2f" % (summary["mode"], summary["threshold"]))
    L.append("")

    L.append("## 一、汇总指标")
    L.append("")
    L.append("| 指标 | 结果 | 说明 |")
    L.append("|---|---|---|")
    L.append("| Top-1 命中率 | %d/%d | 检索第 1 条就是正确章节 |"
             % (summary["top1"], summary["in_total"]))
    L.append("| Top-3 命中率 | %d/%d | 前 3 条里含正确章节 |"
             % (summary["top3"], summary["in_total"]))
    L.append("| 离题拒答率 | %d/%d | 手册没有的问题能说「不知道」 |"
             % (summary["refuse_ok"], summary["off_total"]))
    L.append("| 误拒次数 | %d | 手册有答案却被阈值挡掉（越少越好） |"
             % summary["wrong_reject"])
    L.append("")

    L.append("## 二、逐题明细")
    L.append("")
    L.append("| # | 问题 | 类型 | 期望 | 实际 Top-1 | 分数 | 结果 |")
    L.append("|---|---|---|---|---|---|---|")
    for i, row in enumerate(rows, 1):
        actual = row["titles"][0] if row["titles"] else "（拒答）"
        score = "%.3f" % row["scores"][0] if row["scores"] else "-"
        L.append("| %d | %s | %s | %s | %s | %s | %s |"
                 % (i, row["question"], row["kind"], row["expected"],
                    actual, score, "通过" if row["ok"] else "**未通过**"))
    L.append("")

    L.append("## 三、Bad Case 分析")
    L.append("")
    if not bad:
        L.append("本次评测全部通过。")
    else:
        L.append("共 %d 道错题，逐条归因如下：" % len(bad))
        L.append("")
        for i, row in enumerate(bad, 1):
            L.append("### %d. %s" % (i, row["question"]))
            L.append("")
            L.append("- **问题类型**：%s" % row["bad_kind"])
            L.append("- **期望**：%s" % row["expected"])
            L.append("- **实际召回**：%s"
                     % ("、".join("%s(%.3f)" % (t, s)
                                  for t, s in zip(row["titles"], row["scores"]))
                        or "无（触发拒答）"))
            L.append("- **原因分析**：%s" % row["why"])
            L.append("- **处理建议**：%s" % row["fix"])
            L.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    # 【解释】用 utf-8 写，否则 Windows 上中文会乱码。
    return path


def main():
    threshold = SCORE_THRESHOLD
    if "--threshold" in sys.argv:
        i = sys.argv.index("--threshold")
        threshold = float(sys.argv[i + 1])
    # 【解释】阈值做成命令行参数，方便你反复跑、对比不同阈值的效果。

    print("=" * 62)
    print("  Day 19-C：效果评估 + bad case 分析")
    print("=" * 62)

    retriever = RealRetriever(threshold=threshold)
    rows, summary = run_all(retriever)

    print("\n【汇总】阈值 %.2f ｜ %s" % (summary["threshold"], summary["mode"]))
    print("  Top-1 命中 %d/%d ｜ Top-3 命中 %d/%d ｜ 离题拒答 %d/%d ｜ 误拒 %d"
          % (summary["top1"], summary["in_total"],
             summary["top3"], summary["in_total"],
             summary["refuse_ok"], summary["off_total"],
             summary["wrong_reject"]))

    print_detail(rows)

    bad = collect_badcases(rows, threshold)
    print_badcases(bad)

    report = write_report(rows, summary, bad, os.path.join(_HERE, "评测报告_Day19.md"))
    print("\n✅ 报告已写入：%s" % report)
    print("   换个阈值再跑一次对比：--threshold 0.45")


if __name__ == "__main__":
    main()
