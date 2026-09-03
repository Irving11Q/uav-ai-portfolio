#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════
  Day 19-B：接上真检索 —— 把假的那套换成 Day17/18 的成品
═══════════════════════════════════════════════════════════════

【这个文件做什么】
    Day19-A 的界面已经能跑，但答案是假的。今天只做一件事：
    把 FakeRetriever 换成真检索，把 fake_generate 换成真大模型。
    **界面代码一行都不用改** —— 因为两者守住了同一个接口。

【本文件最值得记住的一点】
    Day19-A 里我们约定了一个接口：
        retrieve(question, top_k) -> [(标题, 正文, 分数), ...]
    只要守住它，换实现就是改一行的事。
    这不是设计模式的炫技，是实实在在省下几小时调试时间。

【怎么读这个文件】按这个顺序：
    1. 先跑 --cli 模式（不启动界面，最快看到效果）
       D:/Python-envs/chroma-env/Scripts/python.exe day19b_rag_backbone.py --cli "电池能飞多久"
    2. RealRetriever.retrieve() —— 一行复用 Day18-B 的全部优化
    3. real_generate()          —— 复用 Day17-C 调过的那段 System Prompt
    4. main()                   —— 怎么把真检索塞进 Day19-A 的界面

【运行方式】
    # 命令行自测（推荐先跑这个，几秒出结果）
    D:/Python-envs/chroma-env/Scripts/python.exe day19b_rag_backbone.py --cli "电池能飞多久"
    # 启动完整界面
    D:/Python-envs/chroma-env/Scripts/python.exe day19b_rag_backbone.py

【依赖】Day17-A（embedding）、Day18-A（建库与块数据）、Day18-B（混合检索+阈值拒答）、
        Day17-C（调大模型）、Day19-A（界面）
"""

# ════════════════════════════════════════════════════════════════
# 第 1 部分：环境 —— 全部复用，不重新发明
# ════════════════════════════════════════════════════════════════

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
# 【解释】把 w4_rag 加进模块搜索路径，才能 import 到 day17/18/19a 这些兄弟文件。

from day18a_baseline_eval import build_index, CHUNKS, SECTIONS
# 【解释】build_index 建 Chroma 库；CHUNKS 是 8 块手册原文；SECTIONS 是章节名。
#         注意：build_index 是函数，import 时不会执行，所以这一行很快。

from day18b_recall_tuning import (
    search_with_threshold,
    top_semantic_score,
    SCORE_THRESHOLD,
)
# 【解释】Day18-B 的最终成果：RRF 融合排序 + 语义相似度阈值拒答。
#         top_semantic_score 用来在拒答时解释"最高分是多少"。

from day17c_rag_pipeline import ask_model, HAS_KEY
# 【解释】ask_model 封装了大模型调用；HAS_KEY 告诉我们有没有配 Key。

from day19a_ui import MainWindow
# 【解释】★ 界面直接拿来用，一行不改 —— 这就是守住接口的回报。


# ════════════════════════════════════════════════════════════════
# 第 2 部分：RealRetriever —— 接口和假的一模一样
# ════════════════════════════════════════════════════════════════

class RealRetriever:
    """
    真检索：Chroma + 混合检索 + 阈值拒答。

    接口与 Day19-A 的 FakeRetriever 逐条对齐：
        retrieve(question, top_k) -> [(标题, 正文, 分数), ...]
        self.mode_name            -> 显示给用户的模式说明
    """

    def __init__(self, threshold=SCORE_THRESHOLD, top_k=3):
        self.threshold = threshold
        self.top_k = top_k
        # 【解释】阈值和 Top-K 做成可配参数，方便你现场调着看效果差异。

        self.coll, self.mode = build_index()
        # 【解释】建库。第一次会加载 BGE 模型（慢几秒），之后走缓存。
        #         返回 (collection 对象, embedding 模式名)。

        self.mode_name = "真检索 · %s · 阈值 %.2f" % (self.mode, threshold)
        # 【解释】这句话会显示成界面的第一句欢迎语，让你一眼知道当前配置。

        self.last_refused = False
        self.last_reason = ""
        # 【解释】把"上一次为什么拒答"记下来，界面和 --cli 都要用它解释。

    def retrieve(self, question, top_k=None):
        k = top_k or self.top_k
        # 【解释】允许调用时临时改 K；不传就用默认的。

        picked, refused = search_with_threshold(
            self.coll, question, threshold=self.threshold, top_k=k
        )
        # 【解释】★ 就这一行，复用了 Day18-B 的全部优化：
        #         查询改写 → 语义 + 字频双路召回 → RRF 融合排序 → 语义相似度阈值判定。
        #         Day18 那两千行代码的成果，在这里被一行吃掉。

        self.last_refused = refused
        if refused:
            idx, best = top_semantic_score(self.coll, question)
            # 【解释】拒答时把"最高分是多少"查出来，好解释为什么不答。
            self.last_reason = "最高相似度只有 %.3f，低于阈值 %.2f" % (best, self.threshold)
            return []
            # 【解释】返回空列表 = 一条资料都不给模型，从源头掐断幻觉。

        self.last_reason = ""
        out = []
        for idx, score in picked:
            title = SECTIONS[idx] if idx < len(SECTIONS) else "第 %d 块" % idx
            # 【解释】用章节名当标题；取不到就退回块号，保证不崩。
            out.append((title, CHUNKS[idx]["text"], float(score)))
        return out


# ════════════════════════════════════════════════════════════════
# 第 3 部分：real_generate —— 照着资料回答
# ════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "你是无人机电池与能耗方面的技术助手。\n"
    "请严格根据【参考资料】回答用户问题。\n"
    "如果资料里没有相关信息，直接说「资料中没有提到」，不要编造。\n"
    "回答时请注明你引用了哪一条资料。"
)
# 【解释】这段 Prompt 从 Day17-C 原样搬来 —— 那是你实测过、能压住幻觉的版本。
#         工程经验：验证过的 Prompt 不要重新发明，直接复用。


def real_generate(question, refs):
    """真生成：拼 Prompt → 调大模型。没配 Key 时给出明确提示。"""
    if not refs:
        return "资料里没有相关内容，我不应该硬答。（触发了拒答阈值）"

    context = "\n\n".join(
        "[%d] 来源：%s\n%s" % (i, title, body)
        for i, (title, body, score) in enumerate(refs, 1)
    )
    # 【解释】给每条资料编号，模型才能在回答里说"我引用了第 2 条"。

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": "【参考资料】\n%s\n\n【用户问题】\n%s" % (context, question)},
    ]
    # 【解释】System 定规矩，User 给资料和问题 —— Day12 学过的两段式。

    answer = ask_model(messages)
    # 【解释】复用 Day17-C 的调用封装，不用重写 requests 那套。

    if answer is None:
        # 【解释】没配 Key 时 ask_model 返回 None，这里如实说明而不是假装成功。
        return ("⚠️ 没检测到可用的 API Key，跳过生成环节。\n\n"
                "但检索这一步是真实有效的 —— 右边那几条就是真检索的结果。\n"
                "配上 ZHIPU_API_KEY 或 DEEPSEEK_API_KEY 后即可看到完整回答。")
    return answer


# ════════════════════════════════════════════════════════════════
# 第 4 部分：两种入口 —— 命令行自测 & 完整界面
# ════════════════════════════════════════════════════════════════

def run_cli(question):
    """命令行自测：不启动界面，直接看检索和生成的结果。"""
    print("=" * 62)
    print("  Day19-B 命令行自测")
    print("=" * 62)

    r = RealRetriever()
    print("\n【检索】%s" % r.mode_name)

    refs = r.retrieve(question)
    if not refs:
        print("  触发拒答：%s" % r.last_reason)
        print("  → 不给模型任何资料，从源头掐断幻觉")
    else:
        for i, (title, body, score) in enumerate(refs, 1):
            print("  %d. %-18s %.3f" % (i, title, score))
            print("     %s" % body.replace("\n", " ")[:66])

    print("\n【生成】")
    print(real_generate(question, refs))
    print()


def main():
    if "--cli" in sys.argv:
        idx = sys.argv.index("--cli")
        question = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "电池能飞多久"
        run_cli(question)
        return
        # 【解释】--cli 不启动界面，几秒就能验证链路通不通，调试时非常省时间。

    from PySide6.QtWidgets import QApplication
    # 【解释】延迟导入：只有真要开界面时才 import PySide6，
    #         这样 --cli 模式在没有 Qt 的环境里也能跑。

    app = QApplication(sys.argv)

    retriever = RealRetriever()
    # 【解释】★ 对比 Day19-A 的 main()：唯一的区别就是这一行。
    #         界面、线程、渲染全都复用，一天的工作压成一次替换。

    win = MainWindow(retriever, real_generate)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
