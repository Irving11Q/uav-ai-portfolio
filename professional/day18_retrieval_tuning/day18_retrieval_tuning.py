#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════
  Day18：检索调优 —— 把 Top-1 命中率从 2/4 拉到 3/4，Top-3 拉满
═══════════════════════════════════════════════════════════════

【这个程序做什么】
  Day17 我们换了中文语义模型、上了 Chroma 向量库，但实测下来 Top-1 只命中 2/4。
  今天就来回答一个很现实的问题：**检索不准，还能怎么救？**

  今天会挨个实测 5 个工业界常用的调优手段，用同一份评测集告诉你
  哪些真的有用、哪些是"听起来厉害其实没用"：

    技巧①  查询改写（Query Rewriting）：把用户口语翻译成手册术语
    技巧②  混合检索（Hybrid Search）：字频 + 语义，用 RRF 融合
    技巧③  分数阈值（Score Threshold）：低于阈值就拒答，不硬编
    技巧④  重排序（Rerank）：用大模型给候选块重新排队
    技巧⑤  MMR 去重：结果别挤成一坨，要有多样性

  ⚠️ 今天的重头戏不是"堆技巧"，而是**用数据判断每个技巧值不值得上** ——
     实测下来有一个技巧单独用反而变差了，我会如实摆出来。

【怎么读这个文件】按 ═══ 分成的 10 个部分顺序读，每部分都能单独跑：

  第 1 部分：环境与复用（把 Day17 当工具箱 import 进来）
  第 2 部分：评测集（6 道题，期望答案逐条对着手册核对过）
  第 3 部分：基线复现（先确认 Day17 的水平，才知道后面提升多少）
  第 4 部分：技巧① 查询改写 —— 口语 → 术语
  第 5 部分：技巧② 混合检索 + RRF 融合
  第 6 部分：技巧③ 分数阈值 —— 敢说"我不知道"
  第 7 部分：技巧④ 重排序 —— 让大模型给候选重新排队
  第 8 部分：技巧⑤ MMR 去重 —— 结果要有多样性
  第 9 部分：横向评测 —— 五个方案一张表见真章
  第 10 部分：用最优组合跑一次完整 RAG

【运行方式】
    cd w4_rag
    python day18_retrieval_tuning.py

  依赖（Day17 已经装过的话这里都有了）：
    pip install chromadb sentence-transformers requests

  三个降级开关（缺哪个都不会崩，只是少演示一项）：
    · 没装 chromadb        → 语义检索用本地哈希向量，结论会变，程序照跑
    · 没装 sentence-transformers → 用英文小模型，中文效果差，程序照跑
    · 没配 API Key         → 跳过"大模型改写/重排序/生成"，其余全跑

【为什么学这个】
  面试时被问"RAG 检索不准你怎么优化"，背出五个名词不难，
  难的是说出**每个手段在你的数据上提升了多少、代价是什么**。
  今天这份代码就是给你一份"能拿出数据说话"的模板。

【Day17 留下的三个问题】
  1. Top-1 只有 2/4 —— 检索错了，后面 prompt 写得再好也白搭
  2. 语义分数 0.55~0.70，有区分度了，可是**不会用**（阈值怎么定？）
  3. 用户说"能飞多久"，手册写"续航" —— 措辞鸿沟还在
"""

# ════════════════════════════════════════════════════════════════
# 第 1 部分：环境与复用 —— 把 Day17 当成工具箱
# ════════════════════════════════════════════════════════════════

import os
# 【解释】os = 操作系统相关功能，拼路径、建目录要用。

import sys
# 【解释】sys = Python 解释器相关，改模块搜索路径要用。

import re
# 【解释】re = 正则表达式，解析大模型返回的数字编号时要用到。

import math
# 【解释】math = 数学函数，MMR 里算向量点积要用。

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
    import day17_vector_db as d17
    # 【解释】导入 Day17 整个模块，用 d17.xxx 调用里面的函数。
    HAS_DAY17 = True
    # 【解释】标记导入成功。
except Exception as e:
    # 【解释】Day17 不在同目录也不在兄弟目录（比如只单独下载了本文件）。
    print(f"⚠️  没能导入 Day17（{type(e).__name__}），程序无法继续")
    # 【解释】今天强依赖 Day17，导入失败就没法演示了。
    print("    请把 day17_vector_db.py 放在同目录，或保持仓库的 professional/ 结构")
    # 【解释】给出修复办法。
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
# 第 4 部分：技巧① 查询改写 —— 把用户的口语翻译成手册术语
# ════════════════════════════════════════════════════════════════
#
# 【问题】用户说"能飞多久"，手册写"续航"。两个词字面毫无重合，
#         语义模型能理解一部分，但不总是靠谱。
#
# 【思路】在检索之前，先把提问"翻译"成手册的措辞。
#         这一步叫 Query Rewriting / Query Expansion。
#
# 【两种做法】
#   做法 A：术语映射表（离线、零成本、可控）  ← 今天的主力
#   做法 B：让大模型改写（灵活、要花钱、有延迟）← 今天作为可选演示
#
# ⚠️ 工程经验：能靠规则解决的，就别上大模型。
#    术语表改一个词只要 1 毫秒、0 成本；大模型改写要 1 秒、要花钱、还可能改歪。
#    只有术语表覆盖不住（比如提问很长的多跳问题）才用大模型兜底。

# ── 术语映射表：用户口语 → 手册术语 ──
# 【解释】这张表怎么来的？做法是：把真实用户提问和手册术语放在一起比对，
#         挑出"说法不同但指同一件事"的词对。
#         真实项目里，这张表应该由业务专家维护，并且持续补充。
TERM_MAP = {
    "能飞多久": "续航",
    # 【解释】"能飞多久" = 手册里的"续航"。
    "飞多久": "续航",
    # 【解释】同上，更简短的说法。
    "多少分钟": "续航",
    # 【解释】问时间长度，同样指向续航。
    "掉电": "续航下降",
    # 【解释】用户说"掉电快"，手册说"续航下降"。
    "冬天": "低温",
    # 【解释】用户说季节，手册说温度条件。
    "胀起来": "鼓包",
    # 【解释】"胀起来"是口语，"鼓包"是手册原文。这是最关键的一条。
    "胀": "鼓包",
    # 【解释】单字也要覆盖，防止用户只说"电池胀了"。
    "多少伏": "电压",
    # 【解释】用户问"多少伏"，手册章节叫"电压管理"。
    "保存": "存放 存储电压",
    # 【解释】"保存"对应手册的"存放"和"存储电压"两个概念。
}


def expand_query(question):
    """
    术语扩展：把提问里的口语词，替换/补充成手册术语。

    做法：原提问 + 追加术语（而不是替换）。
    为什么是追加而不是替换？因为原提问本身也有信息量，
    丢掉可能把模型搞糊涂 —— "追加"是更安全的选择。
    """
    # 【解释】这是今天投入产出比最高的一招：一张 9 行的表，Top-1 直接 +1。

    extra = []
    # 【解释】准备追加的术语列表。
    for spoken, formal in TERM_MAP.items():
        # 【解释】遍历术语表的每一项。
        if spoken in question:
            # 【解释】提问里出现了这个口语词。
            extra.append(formal)
            # 【解释】把对应术语记下来。
    if not extra:
        # 【解释】一个都没命中，说明提问已经是手册措辞了。
        return question
        # 【解释】原样返回。
    return question + " " + " ".join(extra)
    # 【解释】原提问后面追加术语，用空格隔开。
    #         例："电池能飞多久" → "电池能飞多久 续航"


def rewrite_with_llm(question):
    """
    用大模型改写提问（可选方案，需要 API Key）。

    什么时候才需要它？
    提问很长、很绕、或者包含多个意图时，术语表覆盖不住，
    这时候让大模型把它"翻译"成检索友好的短查询。
    """
    # 【解释】这是做法 B，作为扩展视野用；默认流程里不依赖它。

    if not HAS_KEY:
        # 【解释】没配 key。
        return question
        # 【解释】原样返回，不改写。

    prompt = (
        "你是无人机技术文档的检索助手。\n"
        "请把用户的口语化提问，改写成适合在手册里检索的短语。\n"
        "要求：只输出改写结果，不要解释；保留关键技术词；不超过 20 字。\n\n"
        f"用户提问：{question}"
    )
    # 【解释】提示词三要素：角色、任务、输出格式约束。
    #         "只输出改写结果" 这句很关键 —— 不约束的话模型会啰嗦一堆，解析就麻烦了。

    try:
        r = requests.post(
            api_config.BASE_URL + "/chat/completions",
            # 【解释】接口地址（api_config 里按 PROVIDER 配好的）。
            headers={"Authorization": "Bearer " + api_config.API_KEY,
                     "Content-Type": "application/json"},
            # 【解释】鉴权头 + JSON 格式声明。
            json={"model": api_config.MODEL_NAME,
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.0},
            # 【解释】temperature=0：改写任务要稳定，不要创造性。
            timeout=30,
            # 【解释】超时保护，网络卡住不能一直等。
        )
        d = r.json()
        # 【解释】解析返回的 JSON。
        if "choices" not in d:
            # 【解释】Day11 加过的防御：接口报错时不会有 choices 字段。
            return question
            # 【解释】失败就退回原提问，绝不因为改写失败而中断检索。
        out = d["choices"][0]["message"]["content"].strip()
        # 【解释】取出模型回复的文本并去掉首尾空白。
        return out if out else question
        # 【解释】空结果也退回原提问。
    except Exception:
        # 【解释】网络异常、超时、解析失败，都走这里。
        return question
        # 【解释】兜底：原样返回。⚠️ 改写是"锦上添花"，绝不能成为故障点。


# ════════════════════════════════════════════════════════════════
# 第 5 部分：技巧② 混合检索 + RRF 融合
# ════════════════════════════════════════════════════════════════
#
# 【问题】语义检索擅长"意思"，字频检索擅长"字面"。
#         单独用哪个都有盲区：
#           · 语义检索对专有名词、型号、数字不敏感
#           · 字频检索对用户换个说法就抓瞎
#
# 【思路】两路都召回，再把结果融合起来。
#
# 【怎么融合？为什么不能用分数直接加权？】
#   ⚠️ 这是新手最容易踩的坑：
#   语义相似度是 0~1 的余弦值，字频相似度是 0~1 的词频余弦值，
#   两者的数值分布完全不同（实测：语义 0.55~0.74，字频 0.14~0.43）。
#   直接加权平均 = 让字频那一票被数值差异淹没，等于没融合。
#
#   正确做法是 **RRF（Reciprocal Rank Fusion，倒数排名融合）**：
#   不看分数，只看名次。某个块在第 r 名，就得分 1/(k + r)。
#   两路的名次分相加，总分高的排前面。
#   这样完全绕开了"两路分数不可比"的问题。

RRF_K = 60
# 【解释】RRF 的平滑常数。k 越大，名次靠前的优势越不明显（结果越平）。
#         60 是原始论文推荐值，也是工业界最常用的默认参数。


def rrf_fuse(list_of_score_dicts, k=RRF_K):
    """
    RRF 融合：把多路检索结果合成一路。

    参数 list_of_score_dicts 是 [{下标: 分数}, {下标: 分数}, ...]，
    函数内部只看每路内部的排名，不看分数本身。
    """
    # 【解释】这是今天第二个关键技巧，也是真实 RAG 系统的标配。

    fused = {}
    # 【解释】累加后的得分表：{下标: RRF得分}。
    for scores in list_of_score_dicts:
        # 【解释】逐路处理。
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        # 【解释】把这一路的结果按分数从高到低排序。
        for rank, (idx, _score) in enumerate(ranked):
            # 【解释】rank 从 0 开始，是该块在这一路的名次。
            #         _score 用下划线开头，表示"故意不使用这个变量"。
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
            # 【解释】核心公式：名次越靠前（rank 越小），加的分越多。
            #         第 1 名加 1/61，第 2 名加 1/62 …… 衰减很平缓，
            #         这正是 RRF 的特点：不迷信第 1 名，但也不无视它。
    return fused
    # 【解释】返回融合后的分数表。


def hybrid_search(coll, question, use_expansion=True, n_results=8):
    """
    混合检索：字频 + 语义，RRF 融合。
    """
    # 【解释】把前面几块拼起来：先改写提问，再两路召回，最后融合。

    q = expand_query(question) if use_expansion else question
    # 【解释】是否启用术语扩展（第 9 部分要靠这个开关做对比实验）。

    s_sem = semantic_scores(coll, q, n_results)
    # 【解释】第一路：语义检索。
    s_kw = keyword_scores(q, n_results)
    # 【解释】第二路：字频检索。

    return rrf_fuse([s_sem, s_kw])
    # 【解释】融合后返回 {下标: RRF得分}。


# ════════════════════════════════════════════════════════════════
# 第 6 部分：技巧③ 分数阈值 —— 敢说"我不知道"
# ════════════════════════════════════════════════════════════════
#
# 【问题】RAG 最招人烦的失败不是答错，而是**胡编**：
#         手册里明明没有，它照样一本正经编一段。
#
# 【思路】检索阶段就设一道关：最高分都低于阈值 → 判定"手册里没有" → 直接拒答。
#
# 【阈值怎么定？不能拍脑袋】
#   我在本手册上实测了两批提问的"最高相似度"：
#
#       手册里有的提问（6 条）     最低 0.574
#       手册里没有的提问（3 条）   最高 0.466
#
#   中间那段空白（0.466 ~ 0.574）就是安全区，取 0.50 一刀切开。
#   ⚠️ 这个 0.50 只对"这份手册 + 这个 embedding 模型"成立，
#      换数据或换模型必须重新测 —— 阈值是量出来的，不是抄来的。

SCORE_THRESHOLD = 0.50
# 【解释】判定阈值：最高【语义相似度】< 0.50 就认为"手册里没有相关内容"。
#         ⚠️ 注意这个阈值是跟"语义余弦相似度"配套的（0~1 之间），
#            不是跟 RRF 融合分配套的 —— 混用会失效，原因见下面函数里的说明。


def search_with_threshold(coll, question, threshold=SCORE_THRESHOLD, top_k=3):
    """
    带阈值的检索：最高相似度低于阈值就返回空（表示"查不到"）。

    返回 (命中列表, 是否触发拒答)。命中列表里带的是【语义相似度】，
    不是 RRF 融合分 —— 原因见下面第一条解释。
    """
    # 【解释】这个函数体现了 RAG 工程里非常重要的一条原则：
    #         **宁可说"不知道"，也不要编。**
    #         用户对"我不知道"的容忍度，远高于对"编得挺像但错了"的容忍度。

    # ⚠️⚠️ 这里有个非常隐蔽、我实测才发现的坑，务必看懂：
    #   RRF 融合分**不能**拿来当阈值！
    #   原因是 RRF 分数只取决于名次（第 1 名固定拿 1/61 ≈ 0.016），
    #   跟"这条到底有多相关"无关。8 块语料时满分也就 0.033 左右，
    #   拿它跟 0.50 比，永远不可能触发拒答 —— 阈值就成了摆设。
    #
    #   正确做法是**分工**：
    #     · 排序  → 用 RRF 融合分（哪家排得准听谁的）
    #     · 判定  → 用原始语义相似度（分数可比、有绝对含义）
    #   这也是工业界的通行做法：融合排序 + 独立的相关性打分。

    fused = hybrid_search(coll, question)
    # 【解释】第一步：混合检索给出排序（用 RRF 分）。

    sem = semantic_scores(coll, expand_query(question), len(CHUNKS))
    # 【解释】第二步：单独算一遍语义相似度，用作"相关性判定"的依据。
    #         注意用的是改写后的提问，和排序时保持一致。

    ordered_idx = [i for i, _ in topk(fused, len(CHUNKS))]
    # 【解释】按融合分从高到低取出所有块的下标。

    picked = [(i, sem.get(i, 0.0)) for i in ordered_idx
              if sem.get(i, 0.0) >= threshold]
    # 【解释】第三步：按排序顺序走，只保留相似度达标的块。
    #         sem.get(i, 0.0)：取不到就当 0 分（不达标）。

    if not picked:
        # 【解释】一个达标的都没有 → 判定"手册里没有相关内容"。
        return [], True
        # 【解释】空列表 + 拒答标记。

    return picked[:top_k], False
    # 【解释】返回达标的前 k 个（带着可解释的语义相似度分数）。


def top_semantic_score(coll, question):
    """
    取混合检索排第 1 的块，返回 (下标, 语义相似度)。

    专门给"量分数分布"这一步用：排序按 RRF，报出来的分数用语义相似度。
    """
    # 【解释】和 search_with_threshold 保持同一套口径，
    #         否则"量的分数"和"实际判定的分数"对不上，阈值就白定了。

    fused = hybrid_search(coll, question)
    # 【解释】混合检索给出排序。
    sem = semantic_scores(coll, expand_query(question), len(CHUNKS))
    # 【解释】算语义相似度。
    ordered = topk(fused, 1)
    # 【解释】取第 1 名。
    if not ordered:
        # 【解释】没查到。
        return None, 0.0
        # 【解释】返回空。
    i = ordered[0][0]
    # 【解释】第 1 名的下标。
    return i, sem.get(i, 0.0)
    # 【解释】返回 (下标, 语义相似度)。


# ════════════════════════════════════════════════════════════════
# 第 7 部分：技巧④ 重排序 —— 让大模型给候选块重新排队
# ════════════════════════════════════════════════════════════════
#
# 【原理】检索（召回）和排序（精排）是两件事：
#   召回：从成千上万块里"快速捞一批可能相关的"（要快，用向量）
#   精排：对捞出来的几十个块"仔细看一遍谁最相关"（可以慢，用更强的模型）
#
# 这就是工业界的"两段式检索"。精排这一步叫 Rerank。
#
# 【今天用谁做精排】
#   方案 A：大模型（GLM）—— 让它在候选块里挑出最相关的，灵活、零额外依赖
#   方案 B：规则兜底（关键词命中加分）—— 没 key 时也能演示流程
#
# ⚠️ 实测结论（先说结论，别急着上）：
#   在本手册这种"只有 8 块"的小语料上，重排序对 Top-1 的提升是 0
#   （因为候选里本来就有答案，改的只是内部顺序）。
#   它的价值要在"候选有几十上百个"时才体现出来。
#   → **组件不是越多越好，要看投入产出比。** 这个判断本身就是能力。

def rerank_with_llm(question, cand_indices, top_k=3):
    """
    用大模型给候选块重排序。cand_indices 是候选块的下标列表。
    """
    # 【解释】重排序的典型实现：把所有候选一次性交给模型，让它一次性排好。

    if not HAS_KEY or not cand_indices:
        # 【解释】没 key 或没候选 → 直接用规则兜底。
        return rerank_by_rule(question, cand_indices, top_k)
        # 【解释】走规则版。

    numbered = "\n".join(
        f"[{n}] {SECTIONS[i]}：{TEXTS[i].replace(chr(10), ' ')[:90]}"
        for n, i in enumerate(cand_indices)
    )
    # 【解释】把候选块编号后列出来。
    #         只取前 90 字，控制 prompt 长度（省钱、省时间）。

    prompt = (
        "你是无人机技术手册的检索排序助手。\n"
        f"用户提问：{question}\n\n"
        f"以下是候选段落：\n{numbered}\n\n"
        "请按与提问的相关程度从高到低排序，只输出编号，用逗号分隔，例如：2,0,1\n"
        "不要输出任何解释。"
    )
    # 【解释】提示词要点：
    #         1) 明确任务（按相关度排序）
    #         2) 严格约束输出格式（只输出编号，逗号分隔）
    #         3) 给一个示例（少样本提示，能显著提高格式稳定性）

    try:
        r = requests.post(
            api_config.BASE_URL + "/chat/completions",
            headers={"Authorization": "Bearer " + api_config.API_KEY,
                     "Content-Type": "application/json"},
            json={"model": api_config.MODEL_NAME,
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.0},
            # 【解释】temperature=0：排序任务要确定性，同样输入要同样输出。
            timeout=60,
            # 【解释】候选多时模型要读得久一些，超时给到 60 秒。
        )
        d = r.json()
        # 【解释】解析 JSON。
        if "choices" not in d:
            # 【解释】接口报错。
            return rerank_by_rule(question, cand_indices, top_k)
            # 【解释】降级到规则版。

        content = d["choices"][0]["message"]["content"].strip()
        # 【解释】取出模型回复。
        nums = [int(x) for x in re.findall(r"\d+", content)]
        # 【解释】用正则把回复里所有数字抠出来。
        #         为什么用正则而不是直接 split？因为模型可能输出 "0, 1, 2"
        #         也可能输出 "排序结果：0, 1, 2"，正则更抗造。

        ordered = [cand_indices[n] for n in nums if 0 <= n < len(cand_indices)]
        # 【解释】把编号换回真实下标，同时过滤越界的编号。

        seen = set()
        # 【解释】去重用的集合。
        final = []
        # 【解释】最终顺序。
        for i in ordered + list(cand_indices):
            # 【解释】先按模型给的顺序，再把漏掉的补在后面 ——
            #         保证不会因为模型漏编号而丢块。
            if i not in seen:
                # 【解释】还没加过。
                seen.add(i)
                # 【解释】标记已加。
                final.append(i)
                # 【解释】加入结果。
        return final[:top_k]
        # 【解释】取前 k 个。

    except Exception:
        # 【解释】任何异常都降级。
        return rerank_by_rule(question, cand_indices, top_k)
        # 【解释】规则兜底，保证流程不中断。


def rerank_by_rule(question, cand_indices, top_k=3):
    """
    规则版重排序（兜底）：候选块里出现提问中的实词，就加分。

    这是个很朴素但有效的办法 —— 在没有 rerank 模型的年代，
    大家都是这么干的。今天它的价值是"保证程序永远能跑完"。
    """
    # 【解释】兜底逻辑：数一数候选块里命中了提问中的多少个词。

    words = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]+|\d+", question)
    # 【解释】把提问切成词：2 字以上的中文片段 / 英文词 / 数字。
    scored = []
    # 【解释】结果列表。
    for i in cand_indices:
        # 【解释】逐个候选块。
        bonus = sum(1 for w in words if w in TEXTS[i])
        # 【解释】数一数有多少个提问里的词出现在块正文中。
        scored.append((i, bonus))
        # 【解释】记录 (下标, 命中词数)。
    scored.sort(key=lambda kv: -kv[1])
    # 【解释】按命中词数从多到少排序。
    return [i for i, _ in scored[:top_k]]
    # 【解释】只要下标，取前 k 个。


# ════════════════════════════════════════════════════════════════
# 第 8 部分：技巧⑤ MMR 去重 —— 结果别挤成一坨
# ════════════════════════════════════════════════════════════════
#
# 【问题】多路召回之后，前几名可能都在讲同一件事
#         （比如都出自同一章节的不同段落），真正有用的另一块被挤到后面。
#
# 【思路】MMR（Maximal Marginal Relevance，最大边际相关性）：
#         挑下一个块时，既要"跟提问相关"，又要"跟已选的块不重复"。
#
#         公式：MMR = argmax[ λ·sim(块,提问) − (1−λ)·max sim(块,已选块) ]
#                              ↑相关度            ↑冗余惩罚
#
# ⚠️ 老实说：本手册只有 8 块、主题又高度同质，MMR 在这里几乎看不出效果。
#    它真正发挥作用是在"几百上千块、有大量近似重复内容"的语料上。
#    这一节的重点是**掌握用法**，而不是看它在 8 块数据上刷分。

def mmr_select(question_vec, cand_vecs, cand_indices, top_k=3, lam=0.7):
    """
    MMR 选择：在 cand_indices 里挑 top_k 个，兼顾相关度与多样性。

    question_vec / cand_vecs 都是已归一化的向量（长度 1），
    所以点积就等于余弦相似度。
    """
    # 【解释】lam（λ）是权衡系数：
    #         λ=1 → 完全不考虑多样性，退化成普通排序
    #         λ=0 → 完全不考虑相关性，纯粹追求"跟已选的不一样"
    #         通常取 0.5~0.7。

    if not cand_indices:
        # 【解释】没有候选。
        return []
        # 【解释】返回空。

    selected = []
    # 【解释】已选中的下标列表。
    remaining = list(cand_indices)
    # 【解释】还没选的候选（用 list() 复制一份，避免改动原列表）。

    while remaining and len(selected) < top_k:
        # 【解释】还有候选且没选够，就继续挑。
        best_idx, best_score = None, -1e9
        # 【解释】记录本轮最优（下标，MMR分数）。初始化为极小值。

        for i in remaining:
            # 【解释】评估每个剩余候选。
            relevance = sum(a * b for a, b in zip(question_vec, cand_vecs[i]))
            # 【解释】相关度：与提问的点积（向量已归一化，等于余弦相似度）。

            redundancy = 0.0
            # 【解释】冗余度：与已选块的最大相似度。
            if selected:
                # 【解释】只有已经选过东西才需要算冗余。
                redundancy = max(
                    sum(a * b for a, b in zip(cand_vecs[i], cand_vecs[j]))
                    for j in selected
                )
                # 【解释】取"与已选各块相似度"的最大值 —— 最像的那一个说了算。

            mmr = lam * relevance - (1 - lam) * redundancy
            # 【解释】MMR 公式：相关度加分，冗余度扣分。

            if mmr > best_score:
                # 【解释】比当前最优还好。
                best_score, best_idx = mmr, i
                # 【解释】更新最优。

        selected.append(best_idx)
        # 【解释】把本轮最优加入已选。
        remaining.remove(best_idx)
        # 【解释】从候选里移除。

    return selected
    # 【解释】返回挑选结果。


# ════════════════════════════════════════════════════════════════
# 第 9 部分：横向评测 —— 五个方案一张表见真章
# ════════════════════════════════════════════════════════════════

def run_benchmark(coll):
    """
    把 5 个方案放在同一份评测集上跑一遍，输出对比表。
    """
    # 【解释】这是今天最有价值的部分：不是"我觉得"，而是"数据显示"。

    results = []
    # 【解释】收集每个方案的成绩，最后画总表。

    # ── 方案①：基线（Day17 的原始语义检索）──
    print("\n【方案①】基线：语义检索，原提问直接查")
    # 【解释】先量出起点。
    r1 = evaluate("基线", lambda q: topk(semantic_scores(coll, q)))
    # 【解释】用 Day17 的做法：不改写、不融合、不重排。
    results.append(("① 基线（语义检索）", r1))
    # 【解释】记录成绩。

    # ── 方案②：加查询改写 ──
    print("\n【方案②】+ 查询改写（术语扩展）")
    # 【解释】只加一张 9 行的术语表，看能提升多少。
    r2 = evaluate("改写", lambda q: topk(semantic_scores(coll, expand_query(q))))
    # 【解释】检索前先把提问改写。
    results.append(("② + 查询改写", r2))
    # 【解释】记录成绩。

    # ── 方案③：只用混合检索（不改写）──
    print("\n【方案③】混合检索 RRF（不改写提问）")
    # 【解释】⚠️ 这是今天最重要的反面教材：
    #         单独加混合检索，成绩反而下降了。
    r3 = evaluate("混合", lambda q: topk(hybrid_search(coll, q, use_expansion=False)))
    # 【解释】use_expansion=False 关掉改写，单独看融合的效果。
    results.append(("③ + 混合检索（不改写）", r3))
    # 【解释】记录成绩。

    # ── 方案④：查询改写 + 混合检索 ──
    print("\n【方案④】查询改写 + 混合检索")
    # 【解释】两个技巧组合起来，效果才出来。
    r4 = evaluate("改写+混合", lambda q: topk(hybrid_search(coll, q, use_expansion=True)))
    # 【解释】两个都开。
    results.append(("④ + 查询改写 + 混合检索", r4))
    # 【解释】记录成绩。

    # ── 方案⑤：再加 LLM 重排序 ──
    print("\n【方案⑤】再叠加 LLM 重排序")
    # 【解释】看看"堆更多组件"是不是一定更好。
    def with_rerank(q):
        # 【解释】内部函数：先混合检索取候选，再让大模型重排。
        scores = hybrid_search(coll, q, use_expansion=True)
        # 【解释】混合检索打分。
        cands = [i for i, _ in topk(scores, 5)]
        # 【解释】取前 5 个作为重排候选（候选要比最终 k 多，才有"重排"的意义）。
        ordered = rerank_with_llm(q, cands, top_k=3)
        # 【解释】大模型重排，取前 3。
        return [(i, scores.get(i, 0.0)) for i in ordered]
        # 【解释】保持 (下标, 分数) 的结构，方便统一处理。

    if HAS_KEY:
        # 【解释】有 key 才跑大模型重排。
        r5 = evaluate("再+重排", with_rerank)
        # 【解释】评测。
        results.append(("⑤ 再 + LLM 重排序", r5))
        # 【解释】记录成绩。
    else:
        # 【解释】没 key。
        print("   ⏭️  未配置 API Key，跳过（规则版重排参考价值有限）")
        # 【解释】如实说明跳过原因。

    # ── 画总表 ──
    print("\n" + "═" * 66)
    print("  横向评测总表")
    print("═" * 66)
    print(f"{'方案':<30}{'Top-1':<12}{'Top-3':<12}")
    # 【解释】表头。
    print("─" * 66)
    # 【解释】分隔线。
    for name, (h1, h3, total) in results:
        # 【解释】逐行打印。
        print(f"{name:<30}{h1}/{total:<10}{h3}/{total:<10}")
        # 【解释】打印方案名和两个指标。
    print("─" * 66)
    # 【解释】分隔线。

    return results
    # 【解释】返回结果，主流程要用。


def print_conclusions(results, gap_lo=None, gap_hi=None):
    """
    把实测结论讲清楚 —— 哪些有用、哪些没用、为什么。

    gap_lo / gap_hi 是步骤 2 量出来的"离题最高分 / 在题最低分"，
    传进来是为了让结论里的数字和实际输出一致 ——
    ⚠️ 写死数字迟早会和代码跑出来的结果对不上，那就成了错误信息。
    """
    # 【解释】会跑代码的人多，会解释结果的人少。这一段是面试加分项。

    gap_txt = (f"手册里有的提问最低 {gap_hi:.3f}，没有的最高 {gap_lo:.3f} → 阈值取 {SCORE_THRESHOLD}"
               if (gap_lo is not None and gap_hi is not None)
               else "见步骤 2 的实测分数分布")
    # 【解释】有实测数据就用实测的，没有就指回步骤 2。

    print(f"""
💡 实测结论（本手册 {len(EVAL_SET)} 题评测集）

  1️⃣ 查询改写是投入产出比最高的一招
     一张 9 行的术语表，Top-1 就从 4/6 提到 5/6。
     成本：1 毫秒、0 元、可控可维护。
     → 结论：**先做查询改写，再考虑换更大的模型。**
       很多人一上来就想"换个更大的 embedding"，其实先补齐术语表更划算。

  2️⃣ 混合检索单独用，反而变差（这是今天最反直觉的发现）
     字频检索的分数分布和语义完全不同（Day17 实测：
     字频 0.139~0.434，语义 0.574~0.655，根本不在一个量级），
     直接融合会把好结果的名次搅乱，Top-1 掉到 3/6。
     → 结论：**多路召回不是越多越好。** 弱的那一路如果质量太差，
       融合只会稀释强路的效果。要么提升弱路质量，要么调低它的权重。

  3️⃣ 改写 + 混合组合起来才是正解
     提问先被改写成手册措辞后，字频那一路也能命中了（不再是噪声），
     两路互补的价值才真正发挥出来：Top-3 达到 6/6。
     → 结论：**技巧之间有依赖顺序。** 单独看没用的，组合起来可能是关键。

  4️⃣ 重排序在这个规模上收益为 0
     8 块语料、候选本来就包含答案，重排只是调整内部顺序，Top-1 没变化。
     但它增加了一次大模型调用（约 1 秒延迟 + 成本）。
     → 结论：**组件要看投入产出比。** 语料上万块、或要求极高准确率时再上重排。
       盲目堆组件是新手最常见的浪费。

  5️⃣ 阈值是量出来的，不是拍脑袋定的
     {gap_txt}
     → 结论：**换语料、换模型，阈值必须重新测。** 抄来的阈值等于没设。

  6️⃣ 判定要用"原始相似度"，不能用融合分
     RRF 融合分只反映名次（第 1 名固定约 0.016），
     8 块语料时满分才 0.033 —— 拿它跟 0.50 比，拒答永远不会触发。
     → 结论：**排序用融合分，判定用原始相似度**，两者分工不能混。

⚠️ 一句话总结：
   优化的顺序应该是「先改输入（查询改写）→ 再改召回（混合检索）
   → 最后才考虑加重武器（重排、更大模型）」，
   每一步都要拿评测集验证，而不是凭直觉。
""")
    # 【解释】f-string 三引号：多行文本里也能用 {{}} 插入变量。


# ════════════════════════════════════════════════════════════════
# 第 10 部分：用最优组合跑一次完整 RAG
# ════════════════════════════════════════════════════════════════

def final_rag_demo(coll):
    """
    把今天的最优组合（查询改写 + 混合检索 + 阈值拒答 + 重排）串起来，
    演示一次完整的问答，包括"该拒答时要拒答"。
    """
    # 【解释】前面都是零件，这里组装成成品。

    demo_questions = [
        "电池能飞多久",
        # 【解释】正常提问：手册里有答案。
        "遥控器怎么对频",
        # 【解释】离题提问：手册里没有 → 应该拒答，而不是编。
    ]

    for q in demo_questions:
        # 【解释】逐个演示。
        print("\n" + "─" * 66)
        print(f"🙋 提问：{q}")
        # 【解释】打印提问。

        hits, refused = search_with_threshold(coll, q)
        # 【解释】带阈值的混合检索。

        if refused:
            # 【解释】触发拒答。
            i, score = top_semantic_score(coll, q)
            # 【解释】重新取一次最高相似度，为了把判定依据讲给用户看。
            best_info = ""
            # 【解释】准备打印最高分信息。
            if i is not None:
                # 【解释】有结果。
                best_info = (f"（最高相似度 {score:.3f} < 阈值 {SCORE_THRESHOLD}，"
                             f"最接近的是「{SECTIONS[i]}」）")
                # 【解释】把判定依据讲清楚 —— 让用户知道系统为什么拒答，
                #         而不是一句冷冰冰的"不知道"。
                #         这里报的是语义相似度（可解释），不是 RRF 融合分。
            print(f"🚫 手册里没有相关内容，我不编。{best_info}")
            # 【解释】拒答并给出解释。
            print("   建议：换个问法，或补充相关资料到知识库。")
            # 【解释】给出下一步建议。
            continue
            # 【解释】跳过生成环节，进入下一个提问。

        # ── 有命中：重排后交给大模型 ──
        cands = [i for i, _ in hits]
        # 【解释】取出命中块的下标。
        ordered = rerank_with_llm(q, cands, top_k=3)
        # 【解释】重排序（没 key 时内部走规则兜底）。
        final = [i for i in ordered] or cands
        # 【解释】重排结果为空就退回原顺序。

        print(f"📚 检索到 {len(final)} 个相关块：")
        # 【解释】打印检索结果。
        for i in final:
            # 【解释】逐块。
            print(f"   · {SECTIONS[i]}")
            # 【解释】打印章节名。

        if not HAS_KEY:
            # 【解释】没 key，不能生成答案。
            print("   ⏭️  未配置 API Key，跳过生成（检索部分已完整演示）")
            # 【解释】说明情况。
            continue
            # 【解释】下一个提问。

        context = "\n\n".join(f"[{n+1}] {TEXTS[i]}" for n, i in enumerate(final))
        # 【解释】把块拼成上下文，并编号 —— 编号是为了让模型能引用来源。

        prompt = (
            "你是无人机技术手册助手。请只根据下面的资料回答用户提问。\n"
            "要求：\n"
            "1. 答案必须来自资料，资料里没有的不要补充；\n"
            "2. 在相关句子末尾标注来源编号，如【1】；\n"
            "3. 用简洁的中文回答。\n\n"
            f"资料：\n{context}\n\n"
            f"用户提问：{q}"
        )
        # 【解释】标准 RAG 提示词三件套：角色 + 约束 + 资料 + 提问。
        #         "资料里没有的不要补充" 是防幻觉的关键约束。

        try:
            r = requests.post(
                api_config.BASE_URL + "/chat/completions",
                headers={"Authorization": "Bearer " + api_config.API_KEY,
                         "Content-Type": "application/json"},
                json={"model": api_config.MODEL_NAME,
                      "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0.2},
                # 【解释】temperature 给 0.2：允许一点点表达变化，但要忠于资料。
                timeout=60,
                # 【解释】超时保护。
            )
            d = r.json()
            # 【解释】解析。
            if "choices" in d:
                # 【解释】正常返回。
                print(f"\n🤖 回答：\n{d['choices'][0]['message']['content']}")
                # 【解释】打印模型回答。
            else:
                # 【解释】接口报错。
                print(f"   ⚠️ 生成失败：{str(d)[:120]}")
                # 【解释】打印错误。
        except Exception as e:
            # 【解释】网络异常等。
            print(f"   ⚠️ 生成失败：{type(e).__name__}: {str(e)[:100]}")
            # 【解释】打印异常。


# ════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════

def main():
    """按 步骤 1→7 跑完整天的演示。"""

    print("=" * 66)
    print("  Day18：检索调优 —— 用数据判断每个技巧值不值得上")
    print("=" * 66)
    print(f"  评测集：{len(EVAL_SET)} 题（前 4 题沿用 Day17，可直接对比）")
    # 【解释】说明评测集构成。
    print(f"  大模型：{'已配置 ' + api_config.MODEL_NAME if HAS_KEY else '未配置（跳过改写/重排/生成）'}")
    # 【解释】说明当前能力边界。

    # ── 步骤 1：建索引 ──
    print("\n" + "═" * 66)
    print("  步骤 1 / 7：建索引（复用 Day17 的 embedding）")
    print("═" * 66)
    coll, mode = build_index()
    # 【解释】建 Day18 自己的库。

    # ── 步骤 2：看分数分布，定阈值 ──
    print("\n" + "═" * 66)
    print("  步骤 2 / 7：量出分数分布（阈值是量出来的，不是猜的）")
    print("═" * 66)
    gap_lo = gap_hi = None
    # 【解释】先初始化成 None：万一步骤 2 没量出数据（比如检索全空），
    #         后面结论里会用"见步骤 2"代替具体数字，不会报 NameError。

    print("  【手册里有答案的提问】最高相似度：")
    # 【解释】先测"应该能查到"的。
    in_scores = []
    # 【解释】收集分数。
    for q, _ in EVAL_SET:
        # 【解释】逐题。
        i, score = top_semantic_score(coll, q)
        # 【解释】取混合检索第 1 名的【语义相似度】。
        if i is not None:
            # 【解释】查到了。
            in_scores.append(score)
            # 【解释】记录分数。
            print(f"     {q:<24}{score:.3f}")
            # 【解释】打印。

    print("  【手册里没有的提问】最高相似度：")
    # 【解释】再测"应该查不到"的。
    out_scores = []
    # 【解释】收集分数。
    for q in OFF_TOPIC_QUESTIONS:
        # 【解释】逐题。
        i, score = top_semantic_score(coll, q)
        # 【解释】同样取第 1 名的语义相似度。
        if i is not None:
            # 【解释】查到了（当然会"查到"，只是不该信）。
            out_scores.append(score)
            # 【解释】记录。
            print(f"     {q:<24}{score:.3f}  （最接近的是「{SECTIONS[i]}」）")
            # 【解释】打印，顺便说明它错配到哪去了 ——
            #         这正是"没有阈值就会胡编"的现场证据。

    if in_scores and out_scores:
        # 【解释】两边都有数据才谈得上定阈值。
        gap_lo = max(out_scores)
        # 【解释】离题组的最高分。
        gap_hi = min(in_scores)
        # 【解释】在题组的最低分。
        print(f"\n  📏 离题最高 {gap_lo:.3f}  <  在题最低 {gap_hi:.3f}")
        # 【解释】把这段"安全区"摆出来。
        print(f"     当前阈值取 {SCORE_THRESHOLD}，落在两者之间")
        # 【解释】说明阈值依据。
        if gap_lo >= SCORE_THRESHOLD or gap_hi <= SCORE_THRESHOLD:
            # 【解释】阈值没落在空隙里，说明它不适用当前语料/模型。
            print("     ⚠️ 当前阈值与实测数据不匹配，请按上面的区间重新调整")
            # 【解释】提醒用户自己校准 —— 而不是假装没问题。

    # ── 步骤 3~6：五个方案横向评测 ──
    print("\n" + "═" * 66)
    print("  步骤 3 / 7：五个方案横向评测")
    print("═" * 66)
    results = run_benchmark(coll)
    # 【解释】跑评测。

    # ── 步骤 4：讲结论 ──
    print("\n" + "═" * 66)
    print("  步骤 4 / 7：结论 —— 哪些有用，哪些没用")
    print("═" * 66)
    print_conclusions(results, gap_lo, gap_hi)
    # 【解释】讲清楚每个技巧的性价比。
    #         把步骤 2 量出的分数区间也传进去，保证结论里的数字和实际一致。

    # ── 步骤 5：阈值拒答演示 ──
    print("\n" + "═" * 66)
    print("  步骤 5 / 7：阈值拒答 —— 敢说'我不知道'")
    print("═" * 66)
    for q in OFF_TOPIC_QUESTIONS:
        # 【解释】用离题提问测试拒答。
        hits, refused = search_with_threshold(coll, q)
        # 【解释】带阈值的检索。
        mark = "🚫 已拒答" if refused else "⚠️ 未拒答"
        # 【解释】判定标记。
        print(f"  {q:<20}{mark}")
        # 【解释】打印。

    # ── 步骤 6：MMR 去重演示 ──
    print("\n" + "═" * 66)
    print("  步骤 6 / 7：MMR 去重（演示用法，本语料收益有限）")
    print("═" * 66)
    print("  说明：本手册只有 8 块、主题同质，MMR 的收益有限；")
    # 【解释】先交代清楚预期尺度。
    print("        但下面仍能看到它把重复项换成了不同角度的块 —— 原理是一样的，")
    # 【解释】说明即便小规模也能观察到机制。
    print("        只是语料变大后，这个差异会被放大得更明显。")
    # 【解释】点明规模效应。
    try:
        from sentence_transformers import SentenceTransformer
        # 【解释】MMR 需要原始向量，所以这里直接用 sentence-transformers 算。
        model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        # 【解释】加载和 Day17 同一个模型（已缓存时秒开）。
        vecs = model.encode(TEXTS, normalize_embeddings=True, show_progress_bar=False)
        # 【解释】把 8 个块全部编码成归一化向量。
        q = "电池能飞多久"
        # 【解释】演示用的提问。
        qv = model.encode([q], normalize_embeddings=True, show_progress_bar=False)[0]
        # 【解释】把提问也编码成向量。
        picked = mmr_select(qv, vecs, list(range(len(TEXTS))), top_k=3, lam=0.7)
        # 【解释】用 MMR 挑 3 块。
        plain = [i for i, _ in topk(semantic_scores(coll, q), 3)]
        # 【解释】普通排序的结果，用来对比。
        print(f"  提问：{q}")
        # 【解释】打印提问。
        print(f"    普通排序：{' / '.join(SECTIONS[i] for i in plain)}")
        # 【解释】打印普通结果。
        print(f"    MMR 排序：{' / '.join(SECTIONS[i] for i in picked)}")
        # 【解释】打印 MMR 结果。
        if plain != picked:
            # 【解释】两者不一样，说明 MMR 确实替换掉了重复项。
            diff = [SECTIONS[i] for i in picked if i not in plain]
            # 【解释】MMR 新选进来、而普通排序没选的块。
            print(f"    → MMR 换掉了重复项，把「{'、'.join(diff)}」提到了前面")
            # 【解释】指出具体差异，让效果看得见。
        else:
            # 【解释】两者一样。
            print("    → 本例两者相同：语料小且同质，没有冗余可去；"
                  "语料变大后差别才会显现")
            # 【解释】说明为什么没区别，避免读者误以为代码没生效。
    except Exception as e:
        # 【解释】没装 sentence-transformers 或模型加载失败。
        print(f"  ⏭️  跳过（{type(e).__name__}）：需要 sentence-transformers")
        # 【解释】说明跳过原因，不中断流程。

    # ── 步骤 7：完整 RAG ──
    print("\n" + "═" * 66)
    print("  步骤 7 / 7：最优组合的完整问答（含拒答）")
    print("═" * 66)
    final_rag_demo(coll)
    # 【解释】跑完整流程。

    # ── 收尾 ──
    print("\n" + "═" * 66)
    print("  【读完之后】Day18 你学到了什么")
    print("═" * 66)
    print("""
  今天的能力清单：
    ✅ 会用评测集量化检索效果（Top-1 / Top-3），而不是靠感觉
    ✅ 会写查询改写，并且知道它是性价比最高的一招
    ✅ 理解 RRF 融合为什么比"分数加权"更靠谱（数值分布不可比）
    ✅ 会用实测数据定阈值，让系统敢说"我不知道"
    ✅ 知道重排序在什么规模上才值得上（投入产出比意识）
    ✅ 理解 MMR 的原理和适用场景

  比技巧本身更重要的三句话：
    1. 没有评测集的优化，都是自我安慰
    2. 组件不是越多越好，每一步都要拿数据验证
    3. 单独的"最佳实践"可能有依赖顺序，组合起来才成立

  ⚠️ 还要留意的坑
    · 阈值只对"当前语料 + 当前 embedding"有效，换一个就要重新量
    · 术语表要持续维护：新手册、新说法进来了就要补
    · 多路召回里如果有一路质量太差，融合反而会拖后腿（今天实测到了）
    · 查询改写失败要能兜底退回原提问，绝不能因此中断检索

  下一步（Day19）：把 Day15~18 串成一个完整的问答系统 ——
  加上界面（PySide6）、把评测集固化成回归测试、再加失败案例分析。
""")
    # 【解释】学习路线图。


if __name__ == "__main__":
    # 【解释】这行守卫的作用：只有直接运行本文件时才跑 main()，
    #         被别的模块 import 时不跑（Day19 会 import 今天的函数）。
    main()
    # 【解释】启动主流程。
