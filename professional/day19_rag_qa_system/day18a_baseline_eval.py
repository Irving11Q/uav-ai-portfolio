#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════
  Day18-a：先量出基线 —— 优化之前，得先有一把尺子
═══════════════════════════════════════════════════════════════

【这个文件回答什么问题】
  「Day17 做完了，检索到底准不准？不准在哪？」
  不准就急着改，等于闭着眼睛调参。今天先造一把尺子。

【怎么读这个文件】按顺序读这 3 部分，每部分都能单独跑：

  第 1 部分：环境与复用（把 Day17 当工具箱 import 进来）
  第 2 部分：评测集（6 道在题 + 3 道离题，期望答案逐条对着手册核对过）
  第 3 部分：基线复现（语义检索几成准？字频检索几成准？）

【运行方式】
    cd w4_rag
    python day18a_baseline_eval.py

【为什么学这个】
  面试被问「RAG 检索不准你怎么优化」，第一个动作不是上技巧，
  而是问一句：「你的评测集是什么？基线多少？」
  答不上来，后面说的所有优化都没有可信度。
  ⚠️ 没有评测集的优化，都是自我安慰。

【前置】依赖 Day17（整包版 day17_vector_db.py，或拆分版 17a/17b/17c 均可）。
【下一步】day18b_recall_tuning.py —— 知道差在哪，才开始救。
"""

# ════════════════════════════════════════════════════════════════
# 第 1 部分：环境与复用 —— 把 Day17 当成工具箱
# ════════════════════════════════════════════════════════════════

import os
# 【解释】os = 操作系统相关功能，拼路径、建目录要用。

import sys
# 【解释】sys = Python 解释器相关，改模块搜索路径要用。

# 【拆分时清理】原 Day18 里 import 了 re 和 math，但：
#               · re 只在第 7 部分（重排序）用 —— 拆到 18c 后会自己重新 import
#               · math 全文压根没用上（MMR 的点积是用纯 Python 的
#                 sum(a * b for a, b in zip(...)) 算的）
#               这种「僵尸导入」在真实项目里非常常见：代码改了几轮之后，
#               导入还留着。清理掉，别让后来的读者去找一个不存在的用法。

import requests
# 【解释】requests = 发 HTTP 请求，调大模型 API 用（和 Day10 一样）。

# ── 让 Python 找得到 Day17 ──
# 【解释】今天不重复造轮子：Day17 的加载、切块、embedding、Chroma 检索全部复用。
#         这就是真实项目的写法 —— 昨天写的代码，今天是你的工具箱。
_HERE = os.path.dirname(os.path.abspath(__file__))
# 【解释】本文件所在目录（w4_rag）。
for _p in (_HERE, os.path.join(_HERE, "..", "professional", "day17_vector_db")):
    # 【解释】两种可能的位置：本地学习目录（同目录）或 GitHub 发布版（兄弟目录）。
    if _p not in sys.path:
        # 【解释】避免重复插入。
        sys.path.insert(0, _p)
        # 【解释】插到最前面，优先搜索。

try:
    # 【解释】优先用 Day17 的整包版本（发布到 GitHub 的是这一版）。
    import day17_vector_db as d17
    # 【解释】导入 Day17 整个模块，用 d17.xxx 调用里面的函数。
    D17_SOURCE = "day17_vector_db.py（整包版）"
    # 【解释】记下来路，运行时会打印，方便排查。
except Exception:
    # 【解释】整包版不在 —— 比如你把 Day17 拆成了 17a/17b/17c 三个文件。
    #         那就从拆分版把今天用到的三个函数捞出来，拼成一个同名模块来用。
    #         这就是真实项目里的"依赖降级"：接口不变，来源可以换。
    import types
    # 【解释】types = 动态创建简单对象的工具，这里用它现场造一个模块壳。
    try:
        from day17a_embedding import build_embedding_function
        # 【解释】embedding 三档降级逻辑在 17a。
        from day17b_chroma_store import load_chunks
        # 【解释】读取 chunks_parsed.json 的函数在 17b。
        from day17c_rag_pipeline import day15_keyword_search
        # 【解释】Day15 的字频检索（今天要当第二路召回用）在 17c。

        d17 = types.ModuleType("d17")
        # 【解释】造一个空模块，名字叫 d17。
        d17.build_embedding_function = build_embedding_function
        # 【解释】挂上函数 1。
        d17.load_chunks = load_chunks
        # 【解释】挂上函数 2。
        d17.day15_keyword_search = day15_keyword_search
        # 【解释】挂上函数 3。
        D17_SOURCE = "day17a/17b/17c（拆分版）"
        # 【解释】记下来路。
    except Exception as e:
        # 【解释】两种来源都没有。
        print(f"⚠️  没能导入 Day17（{type(e).__name__}），程序无法继续")
        # 【解释】今天强依赖 Day17。
        print("    请把 day17_vector_db.py 放在同目录，")
        print("    或把 day17a/17b/17c 三个文件放在同目录")
        # 【解释】给出两条修复路径。
        sys.exit(1)
        # 【解释】退出程序。

# ── 让 Python 找得到 w3_ai 的 api_config.py（跨周复用，和 Day15/17 一致）──
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "w3_ai"))
# 【解释】w4_rag 的上级目录是 src，再拼 w3_ai 就是配置所在处。
try:
    import api_config
    # 【解释】读 API 配置（BASE_URL / MODEL_NAME / API_KEY）。
    HAS_KEY = bool(api_config.API_KEY)
    # 【解释】有 key 才能调大模型。
except Exception:
    # 【解释】没找到配置或配置报错（比如没设环境变量）。
    HAS_KEY = False
    # 【解释】降级：不调大模型，只跑离线部分。

DB_DIR_18 = os.path.join(_HERE, "chroma_db_day18")
# 【解释】Day18 单独用一个库目录，不污染 Day17 的 chroma_db。
#         实际项目里也是这样：不同实验用不同库，免得互相干扰。



# ════════════════════════════════════════════════════════════════
# 第 2 部分：评测集 —— 优化之前，先有一把尺子
# ════════════════════════════════════════════════════════════════

# 【解释】⚠️ 这是今天最容易被忽略、却最重要的一步：
#         没有评测集，"优化"就变成了"我觉得变好了"。
#         工业界做 RAG，第一件事就是攒评测集（哪怕只有 20 条）。

EVAL_SET = [
    # 【解释】格式：(用户提问, 期望命中的章节名)
    #         前 4 条沿用 Day17，方便直接对比今天提升了多少；
    #         后 2 条是新加的，避免"只挑对自己有利的题"。
    #
    #         ⚠️ 每条期望都对着 chunks_parsed.json 的原文核对过：
    ("电池能飞多久", "5. 续航估算"),
    # 【解释】手册原文："理论续航 = 容量 ÷ 平均电流 ≈ 25 分钟"。
    ("多少伏就必须降落了", "2. 电压管理"),
    # 【解释】手册原文："电压低于 9.6V 时必须尽快降落"。
    ("冬天飞行掉电特别快是为什么", "4. 温度管理"),
    # 【解释】手册原文："低温环境（低于 0℃）时电池内阻增大，续航会明显下降"。
    ("电池胀起来了还能继续用吗", "6. 充电与维护"),
    # 【解释】手册原文："电池鼓包、漏液应立即停止使用"。
    #         注意不是第 7 章"安全预警" —— Day17 就在这里标错过一次。
    ("充电电流应该调多大", "6. 充电与维护"),
    # 【解释】手册原文："建议 1C 电流充电（即 5.2A）"。
    ("悬停的时候电流大概多少", "3. 电流与功耗"),
    # 【解释】手册原文："悬停状态下整机电流约 12A"。
]

# ── 手册里没有答案的提问（用来演示阈值拒答）──
OFF_TOPIC_QUESTIONS = [
    "遥控器怎么对频",
    # 【解释】手册是电池手册，没有遥控器对频的内容。
    "螺旋桨怎么更换",
    # 【解释】同样不在手册里。
    "相机云台怎么校准",
    # 【解释】也不在手册里。
]

CHUNKS = d17.load_chunks()
# 【解释】加载 Day16 切好的 8 个块（复用 Day17 的函数）。
TEXTS = [c["text"] for c in CHUNKS]
# 【解释】块的正文列表。
IDS = [str(c["id"]) for c in CHUNKS]
# 【解释】块的 ID 列表。
#         ⚠️ 为什么要 str()？因为 Chroma 规定 ID 必须是字符串，
#            而 Day16 切块时用的是数字编号（0、1、2...），
#            直接传进去会报 ValueError: Expected ID to be a str。
#            这是跨模块复用时很典型的"接口约定不一致"问题。
ID2IDX = {cid: i for i, cid in enumerate(IDS)}
# 【解释】ID → 下标的反查表。两种检索返回的都是 ID，要换成下标才好比较和融合。
SECTIONS = [c["text"].strip().splitlines()[0].replace("#", "").strip() for c in CHUNKS]
# 【解释】每块的章节名（取正文第一行，去掉 # 号），打印结果时用。



# ════════════════════════════════════════════════════════════════
# 第 3 部分：基线复现 —— 先量出 Day17 的水平
# ════════════════════════════════════════════════════════════════

def build_index():
    """
    建 Day18 自己的 Chroma 库（内容和 Day17 一样，只是换个地方存）。

    返回 (collection, embedding模式名)。
    """
    # 【解释】今天所有"语义检索"都走这个 collection。

    embed_fn, mode_name, is_semantic, is_chinese = d17.build_embedding_function()
    # 【解释】复用 Day17 的三档降级逻辑（中文BGE → 英文ONNX → 本地哈希）。
    #         拿到四件套：函数、模式名、是否真语义、是否支持中文。
    print(f"   当前 embedding 模式：{mode_name}")
    # 【解释】打印出来，让读者知道当前跑的是哪一档。

    import chromadb
    # 【解释】延迟导入：确认要用才 import。

    client = chromadb.PersistentClient(path=DB_DIR_18)
    # 【解释】持久化客户端，数据写到 chroma_db_day18 目录。

    coll = client.get_or_create_collection(
        name="uav_manual_d18",
        # 【解释】集合名和 Day17 区分开，避免两边互相干扰。
        metadata={"hnsw:space": "cosine"},
        # 【解释】余弦空间，和 Day17 一致，相似度 = 1 - distance。
        embedding_function=embed_fn,
        # 【解释】指定自定义 embedding（Day17 实现了 Chroma 1.x 要求的全套接口）。
    )

    if coll.count() != len(CHUNKS):
        # 【解释】库是空的或数量不对 → 重建。
        #         为什么要判断 count？因为 get_or_create 可能拿到一个旧的、
        #         用别的 embedding 建过的库，那样向量维度会对不上。
        try:
            client.delete_collection("uav_manual_d18")
            # 【解释】删掉旧库重建，保证干净。
        except Exception:
            pass
            # 【解释】本来就不存在，删了会报错，忽略即可。
        coll = client.get_or_create_collection(
            name="uav_manual_d18",
            metadata={"hnsw:space": "cosine"},
            embedding_function=embed_fn,
        )
        coll.add(
            ids=IDS,
            # 【解释】每块的唯一 ID。
            documents=TEXTS,
            # 【解释】正文（Chroma 会自动调 embedding_function 算向量）。
            metadatas=[{"section": s} for s in SECTIONS],
            # 【解释】metadata 存章节名，方便按章节过滤和结果展示。
        )
        print(f"   已写入 {coll.count()} 条向量 → {DB_DIR_18}")
        # 【解释】告诉用户建库完成。
    else:
        print(f"   复用已有库（{coll.count()} 条向量）")
        # 【解释】已存在且数量对得上，直接用，省一次 embedding 计算。

    return coll, mode_name
    # 【解释】返回集合 + 模式名。


def semantic_scores(coll, question, n_results=8):
    """
    语义检索：返回 {块下标: 相似度}。

    统一返回"下标 → 分数"的字典，方便后面几种方案互相融合。
    """
    # 【解释】统一数据结构是融合的前提：不管哪路检索，出来都是同一格式。
    res = coll.query(query_texts=[question], n_results=min(n_results, len(CHUNKS)))
    # 【解释】向 Chroma 提问，取前 n 条。min() 防止要的比总数还多。

    out = {}
    # 【解释】结果字典。
    if not res["ids"] or not res["ids"][0]:
        # 【解释】没查到任何东西。
        return out
        # 【解释】返回空字典。
    for cid, dist in zip(res["ids"][0], res["distances"][0]):
        # 【解释】把 ID 和距离配对。
        out[ID2IDX[cid]] = 1.0 - dist
        # 【解释】余弦距离 → 相似度（因为建库时设了 hnsw:space = cosine）。
    return out
    # 【解释】返回 {下标: 相似度}。


def keyword_scores(question, n_results=8):
    """
    字频检索（Day15 的老办法）：返回 {块下标: 相似度}。

    今天它有了新用途 —— 不是被淘汰，而是当"第二路召回"。
    """
    # 【解释】这是今天第二个重要认知：
    #         字频检索不是"落后的技术"，它擅长抓专有名词、型号、数字，
    #         正好补语义检索的短板。真实系统里两者是搭档，不是替代关系。
    hits = d17.day15_keyword_search(question, CHUNKS, top_k=n_results)
    # 【解释】复用 Day17 里移植过来的 Day15 字频检索。
    out = {}
    # 【解释】结果字典。
    for h in hits:
        # 【解释】逐条命中。
        #         ⚠️ 注意这里的字段名：Day17 的 day15_keyword_search 返回的是
        #         "index"（块下标），而 search_chroma 返回的是 "id"（块ID）。
        #         两路召回的返回结构不一致，是跨模块复用时最常见的坑 ——
        #         所以才要在这里统一换算成"下标"，后面才好融合。
        idx = h["index"]
        # 【解释】直接拿到块下标。
        out[idx] = h["similarity"]
        # 【解释】记下 {下标: 相似度}。
    return out
    # 【解释】返回统一结构，方便和语义检索的结果融合。


def topk(score_dict, k=3):
    """把 {下标: 分数} 按分数从高到低取前 k 个，返回 [(下标, 分数), ...]。"""
    # 【解释】小工具函数，后面所有方案都要用。
    items = sorted(score_dict.items(), key=lambda kv: -kv[1])
    # 【解释】按分数倒序排列。key=lambda kv: -kv[1] 表示"取每对里的第 2 个值，取负号"，
    #         这样大的排前面（分数越高越靠前）。
    return items[:k]
    # 【解释】切前 k 个。


def evaluate(name, search_fn, k=3, verbose=True):
    """
    用评测集给某个检索方案打分。

    search_fn 是一个"提问 → [(下标, 分数), ...]"的函数。
    返回 (Top-1 命中数, Top-3 命中数, 题目总数)。
    """
    # 【解释】这是今天的核心工具：所有方案用同一把尺子量。

    hit1 = 0
    # 【解释】Top-1 命中计数：第 1 名就是期望章节。
    hit3 = 0
    # 【解释】Top-3 命中计数：前 3 名里包含期望章节。
    #         为什么看 Top-3？因为真实 RAG 把前 3 块一起喂给大模型，
    #         答案在里面就能答对 —— 这才是实际业务的口径。

    if verbose:
        # 【解释】需要打印明细时才打印。
        print(f"{'提问':<24}{'结果':<6}命中的块")
        # 【解释】表头。
        print("─" * 68)
        # 【解释】分隔线。

    for q, expect in EVAL_SET:
        # 【解释】逐题测试。
        res = search_fn(q)[:3]
        # 【解释】跑检索，取前 3。
        names = [SECTIONS[i] for i, _ in res]
        # 【解释】把下标换成章节名，方便阅读。
        ok1 = bool(names) and names[0].startswith(expect.split(".")[0])
        # 【解释】第 1 名是不是期望章节。
        #         用章节号（"5. 续航估算" → "5."）比对，比比对全名更稳。
        ok3 = any(n.startswith(expect.split(".")[0]) for n in names)
        # 【解释】前 3 名里有没有期望章节。
        hit1 += ok1
        # 【解释】累计（True 当 1，False 当 0）。
        hit3 += ok3
        # 【解释】累计。

        if verbose:
            # 【解释】打印这一题的结果。
            mark = "✅" if ok1 else ("🔶" if ok3 else "❌")
            # 【解释】✅ = 第1名就对；🔶 = 在前3名里；❌ = 没找到。
            print(f"{q:<24}{mark:<6}" + " / ".join(names))
            # 【解释】打印提问、判定、命中的章节。

    total = len(EVAL_SET)
    # 【解释】题目总数。
    if verbose:
        print("─" * 68)
        # 【解释】分隔线。
        print(f"   Top-1 命中 {hit1}/{total}     Top-3 命中 {hit3}/{total}")
        # 【解释】汇总。
    return hit1, hit3, total
    # 【解释】返回三项数据，第 9 部分画总表要用。



# ════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════

def main():
    """按 步骤 1→4 跑完"量基线"。"""

    print("=" * 66)
    print("  Day18-a：先量出基线 —— 优化之前，得先有一把尺子")
    print("=" * 66)
    print(f"  复用的 Day17：{D17_SOURCE}")
    # 【解释】打印 Day17 是从哪来的，方便排查路径问题。
    print(f"  评测集：{len(EVAL_SET)} 道在题 + {len(OFF_TOPIC_QUESTIONS)} 道离题")
    # 【解释】说明尺子的构成：不光要有"该答对"的，还要有"该拒答"的。

    # ── 步骤 1：建索引 ──
    print("\n" + "═" * 66)
    print("  步骤 1 / 4：建索引（复用 Day17 的 embedding + Chroma）")
    print("═" * 66)
    coll, mode = build_index()
    # 【解释】建 Day18 自己的库，不污染 Day17 的 chroma_db。

    # ── 步骤 2：语义检索基线 ──
    print("\n" + "═" * 66)
    print("  步骤 2 / 4：基线① —— Day17 的语义检索，几成准？")
    print("═" * 66)
    s1, s3, total = evaluate("语义检索", lambda q: topk(semantic_scores(coll, q)))
    # 【解释】不做任何优化，原提问直接查 —— 这就是 Day17 的真实水平。

    # ── 步骤 3：字频检索基线 ──
    print("\n" + "═" * 66)
    print("  步骤 3 / 4：基线② —— Day15 的字频检索，几成准？")
    print("═" * 66)
    print("  说明：字频检索不是「被淘汰的老办法」，它擅长抓专有名词、型号、数字，")
    # 【解释】先把认知摆正：今天它是第二路召回，不是垃圾。
    k1, k3, _ = evaluate("字频检索", lambda q: topk(keyword_scores(q)))
    # 【解释】同一把尺子量第二路。

    # ── 步骤 4：两路分数摆在一起看 ──
    print("\n" + "═" * 66)
    print("  步骤 4 / 4：两路分数摆在一起 —— 问题出在哪，一眼就看见")
    print("═" * 66)
    print("  同一道题，两条路给出的分数根本不在一个量级：")
    print("\n" + f"  {'提问':<24}{'字频最高分':<14}{'语义最高分':<14}")
    # 【解释】表头。
    print("  " + "─" * 54)
    # 【解释】分隔线。
    kw_all, se_all = [], []
    # 【解释】收集两边的分数，最后要算区间。
    for q, _ in EVAL_SET:
        # 【解释】逐题。
        kw_hit = topk(keyword_scores(q), 1)
        # 【解释】字频路第 1 名。
        se_hit = topk(semantic_scores(coll, q), 1)
        # 【解释】语义路第 1 名。
        kwv = kw_hit[0][1] if kw_hit else 0.0
        # 【解释】取分数，没命中就记 0。
        sev = se_hit[0][1] if se_hit else 0.0
        # 【解释】同上。
        kw_all.append(kwv)
        # 【解释】记录。
        se_all.append(sev)
        # 【解释】记录。
        print(f"  {q:<24}{kwv:<14.3f}{sev:<14.3f}")
        # 【解释】打印对比。
    print("  " + "─" * 54)
    # 【解释】分隔线。

    if kw_all and se_all:
        # 【解释】两边都有数据才谈得上区间。
        print(f"  {'分数区间':<24}{min(kw_all):.3f}~{max(kw_all):.3f}    "
              f"{min(se_all):.3f}~{max(se_all):.3f}")
        # 【解释】把两个区间摆出来。
        print("\n  📌 这张表直接给出了 18b 要解决的第一个问题：")
        # 【解释】承上启下。
        print("     两路分数不可比 —— 字频普遍 0.1~0.4，语义普遍 0.5~0.7。")
        # 【解释】点出矛盾。
        print("     所以融合时不能把分数加权平均（那样字频永远赢不了），")
        # 【解释】否定错误做法。
        print("     得用 RRF 这类「只看名次、不看分数」的融合法。")
        # 【解释】引出正确做法。

    if s1 is not None and k1 is not None:
        print("\n  📊 基线汇总：")
        # 【解释】汇总两路成绩。
        print(f"     语义检索   Top-1 {s1}/{total}   Top-3 {s3}/{total}")
        # 【解释】语义路。
        print(f"     字频检索   Top-1 {k1}/{total}   Top-3 {k3}/{total}")
        # 【解释】字频路。
        print("\n     ⚠️ 记住这两个数字：后面每加一个技巧，都要回头跟它比。")
        # 【解释】强调基线的作用。

    # ── 收尾 ──
    print("\n" + "═" * 66)
    print("  【读完之后】Day18-a 你学到了什么")
    print("═" * 66)
    print("""
  ✅ 会攒一份最小可用的评测集（在题 + 离题两类都要有）
  ✅ 会用同一把尺子量不同的检索方案（Top-1 / Top-3 两个口径）
  ✅ 看懂了"两路分数不可比"这个融合前必须解决的坑
  ✅ 拿到了基线数字 —— 后面所有优化都要跟它比

  一句话：不知道基线是多少，就别谈优化了多少。

  下一步 → day18b_recall_tuning.py
           解决两个病：该召回的没召回、不该答的乱答。
""")
    # 【解释】学习路线图。


if __name__ == "__main__":
    # 【解释】这行守卫：直接运行本文件时才跑 main()，被 import 时不跑。
    #         18b / 18c / 18d 都会 import 本文件，所以必须有这一行。
    main()
    # 【解释】启动主流程。
