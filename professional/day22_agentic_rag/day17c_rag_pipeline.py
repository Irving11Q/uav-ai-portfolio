"""
Day 17-C：完整 RAG —— 检索出来的东西，怎么变成人话？

【这个文件回答一个问题】
    前两集解决了"存"和"找"。找到 3 段相关资料后，
    怎么让大模型照着它们回答，而不是自己瞎编？

【怎么读这个文件】按这个顺序看：
    1. 先看 main()               —— 字频 vs 语义对比，再看 RAG 带引用回答
    2. compare_two_methods()     —— Day15 的字频检索 vs Chroma 语义检索，差多少
    3. rag_answer()              —— RAG 的灵魂：那段 System Prompt 为什么那么写
    4. ask_model()               —— 和 Day10 一样的调用，纯复用

【运行方式】
    D:/Python-envs/chroma-env/Scripts/python.exe day17c_rag_pipeline.py
    没配 API Key 也没关系：对比实验照跑，只跳过最后的生成环节。

【依赖】
    复用 day17a_embedding.py 和 day17b_chroma_store.py，三个文件要放在同一目录。
"""

# ════════════════════════════════════════════════════════════════
# 第 1 部分：拿工具
# ════════════════════════════════════════════════════════════════


import os
# 【解释】os = 操作系统工具，拼路径、设环境变量都靠它。

import sys
# 【解释】sys = 系统工具，用来把 w3_ai 目录加进"模块搜索路径"。

import json
# 【解释】json = 读写 JSON 文件。Day16 把切好的块存在 chunks_parsed.json 里。

import math
# 【解释】math = 数学工具，算余弦相似度要开平方。

import re
# 【解释】re = 正则表达式，用来从文本里挑出"中文词 / 英文词 / 数字"。

import hashlib
# 【解释】hashlib = 哈希算法库。降级方案里把词映射成向量下标要用它。
#         为什么不用 Python 内置的 hash()？因为内置 hash() 对字符串
#         每次启动 Python 都会换一套"随机盐"，算出来的下标会变，
#         昨天存的库今天就查不到了。md5 是稳定的，任何时候算都一样。

import requests
# 【解释】发网络请求的工具，调大模型生成答案时用（和 Day10 一样）。
# ── 【关键技巧】自动挑选"当前能连通"的模型下载源 ──
# 【解释】embedding 模型要从 HuggingFace 下载。国内直连 huggingface.co 时快时慢，
#         于是大家习惯写死国内镜像 hf-mirror.com —— 但镜像本身也会抽风
#         （我实测时就遇到过：镜像 502，程序直接降级成英文模型，
#          中文效果全废，还查不出原因）。
#         所以正确的做法是：**先探测，再用**，而不是写死一个地址。
def pick_hf_endpoint(timeout=6):
    """
    依次探测候选下载源，返回第一个能连通的。

    timeout 给得比较短（6 秒）：这只是探活，不是下载，
    卡住太久会让人以为程序死机了。
    """
    # 【解释】候选源按"哪个更可能通"排序，用户自己设过的优先尊重。
    candidates = [
        "https://huggingface.co",   # 官方源：通的话速度最快、版本最新
        "https://hf-mirror.com",    # 国内镜像：官方不通时的首选备胎
    ]
    # 【解释】顺序很关键：先官方后镜像。很多人一上来就写死镜像，
    #         结果镜像挂了反而比直连还慢，这就是典型"优化变劣化"。

    for url in candidates:
        # 【解释】挨个试，谁先通就用谁。
        try:
            r = requests.get(url, timeout=timeout)
            # 【解释】只发一个最轻量的 GET 请求，验证网络可达。
            if r.status_code < 500:
                # 【解释】状态码 < 500 说明服务器活着
                #         （5xx 是服务器内部错误，等于没通）。
                print(f"   模型下载源探测：{url} 可用（HTTP {r.status_code}）")
                # 【解释】把探测结果打出来，便于排查。
                return url
                # 【解释】找到能用的就返回，不再试后面的。
        except Exception:
            # 【解释】连不上 / 超时 / DNS 失败，都静默跳过，试下一个源。
            continue
            # 【解释】continue = 直接进入下一轮循环。

    print("   模型下载源探测：全部不可用，将使用默认镜像（可能下载失败并降级）")
    # 【解释】两个都不通就如实告知，后面会自动降级到离线方案。
    return candidates[1]
    # 【解释】兜底返回镜像地址，至少给出一个尝试目标。


# 【解释】setdefault：用户如果自己设过 HF_ENDPOINT，就不用探测结果覆盖他。
os.environ.setdefault("HF_ENDPOINT", pick_hf_endpoint())
# 【解释】⚠️ 必须写在 import chromadb / sentence_transformers 之前：
#         huggingface_hub 在 import 时就把地址读进去了，之后再改环境变量无效。
#         这是很多人"设了镜像却没生效"的真正原因。
# ── 让 Python 能找到 w3_ai 里的 api_config.py ──
# 【解释】本文件在 w4_rag/ 目录，api_config.py 在 w3_ai/ 目录，
#         两者是兄弟目录，默认 import 不到，所以要手动把路径加进去。
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "w3_ai"))
# 【解释】os.path.abspath(__file__) 是本文件的完整路径；
#         dirname 一次 → w4_rag 目录；dirname 两次 → src 目录；
#         再拼上 "w3_ai" → src/w3_ai，把它插到搜索路径最前面。

# ════════════════════════════════════════════════════════════════
# 第 2 部分：三个"开关"—— 没装齐依赖也不许报错
# ════════════════════════════════════════════════════════════════

try:
    import chromadb
    # 【解释】向量数据库本体。装了才能建库，没装就走兜底方案。
    HAS_CHROMA = True
    # 【解释】标记：Chroma 可用。
except ImportError:
    HAS_CHROMA = False
    # 【解释】标记：Chroma 没装，后面所有建库操作都要跳过。
    print("⚠️  没装 chromadb，将只演示原理（pip install chromadb 后可完整运行）")
    # 【解释】明确告诉用户原因，别让他一头雾水。

try:
    import api_config
    # 【解释】复用 w3_ai 的配置（BASE_URL / MODEL_NAME / API_KEY）。
    API_KEY = api_config.API_KEY
    # 【解释】把配置里的 Key 取出来。
    BASE_URL = api_config.BASE_URL
    # 【解释】接口地址，例如智谱是 https://open.bigmodel.cn/api/paas/v4。
    MODEL_NAME = api_config.MODEL_NAME
    # 【解释】模型名，例如 glm-4-flash。
    HAS_KEY = bool(API_KEY)
    # 【解释】Key 非空才算"配置好了"，空字符串要当成没配。
except Exception:
    # 【解释】api_config 里没配 Key 时会主动 raise，这里一并接住。
    HAS_KEY = False
    # 【解释】标记：没有 Key，最后一步"生成答案"要跳过。
    BASE_URL = MODEL_NAME = ""
    # 【解释】给个空值，避免后面引用时报错。
try:
    import chromadb
    # 【解释】向量数据库本体。装了才能建库，没装就走兜底方案。
    HAS_CHROMA = True
    # 【解释】标记：Chroma 可用。
except ImportError:
    HAS_CHROMA = False
    # 【解释】标记：Chroma 没装，后面所有建库操作都要跳过。
    print("⚠️  没装 chromadb，将只演示原理（pip install chromadb 后可完整运行）")
    # 【解释】明确告诉用户原因，别让他一头雾水。
# ── 文件路径常量 ──
# 【解释】下面这几个路径都用 __file__ 推导，这样你在任何目录下运行都不会找不到文件。
HERE = os.path.dirname(os.path.abspath(__file__))
# 【解释】HERE = 本文件所在目录（w4_rag）。

CHUNKS_FILE = os.path.join(HERE, "chunks_parsed.json")
# 【解释】Day16 产出的切块文件，今天是它的"下游消费者"。

DB_DIR = os.path.join(HERE, "chroma_db")
# 【解释】Chroma 的数据库落盘目录。运行一次后这里会生成一堆文件，
#         那就是"持久化"的证据 —— 数据真的存下来了。


def _sep(title):
    """打印分节标题，让输出好看一点。"""
    # 【解释】纯辅助函数，负责在终端打印一条带标题的分隔线。
    print("\n" + "═" * 62)
    # 【解释】先打 62 个 ═，作为视觉分隔。
    print("  " + title)
    # 【解释】再打标题本身，前面留两个空格。
    print("═" * 62)
    # 【解释】结尾再来一条线，形成"夹心"效果。

# ════════════════════════════════════════════════════════════════
# 第 3 部分：加载 Day16 的产物 —— 跨天复用真实数据
# ════════════════════════════════════════════════════════════════

def _find_chunks_file():
    """
    在两个地方找 Day16 的产物：
        1) 本目录下（正常学习流程：你在 w4_rag 里依次跑 Day16 → Day17）
        2) 兄弟目录 day16_doc_parsing/ 下（发布到 GitHub 后的目录结构）
    """
    # 【解释】为什么要找两个地方？因为本文件会被单独发布到作品集仓库，
    #         那时候 Day16 的文件在兄弟目录里，不在同一层。
    candidates = [
        # 【解释】候选路径列表，按顺序试。
        os.path.join(HERE, "chunks_parsed.json"),
        # 【解释】候选 1：同目录（本地学习时的情况）。
        os.path.join(HERE, "..", "day16_doc_parsing", "chunks_parsed.json"),
        # 【解释】候选 2：兄弟目录（GitHub 仓库里的情况）。
    ]
    for p in candidates:
        # 【解释】依次检查每个候选路径。
        if os.path.exists(p):
            # 【解释】找到了就返回这个路径。
            return os.path.normpath(p)
            # 【解释】normpath 把 "a/../b" 整理成干净路径，打印时好看。
    return None
    # 【解释】都不存在，返回 None，调用方会走兜底数据。


def load_chunks(path=None):
    """
    读取 Day16 切好的块。读不到就现场造一份（保证程序永远能跑）。
    """
    # 【解释】这是今天的数据入口：Day16 负责"切"，Day17 负责"存和查"。

    if path is None:
        # 【解释】没指定路径时，自动去找（同目录 → 兄弟目录）。
        path = _find_chunks_file()
        # 【解释】调用上面的查找函数。

    if path and os.path.exists(path):
        # 【解释】找到了文件才读取。
        with open(path, "r", encoding="utf-8") as f:
            # 【解释】打开文件读文本。encoding="utf-8" 必须有，否则中文乱码。
            chunks = json.load(f)
            # 【解释】json.load 把文件内容变成 Python 的 list[dict]。
        print(f"✅ 已从 Day16 产物加载 {len(chunks)} 个块：{os.path.basename(path)}")
        # 【解释】告诉用户加载成功，并打印块的数量。
        return chunks
        # 【解释】拿到数据就返回。

    # ── 兜底：文件不在（比如别人只拷走了这一个 .py）──
    print("⚠️  没找到 chunks_parsed.json，使用内置手册兜底")
    # 【解释】明确提示，而不是静默造假数据。
    return [
        # 【解释】手写 4 块最小数据，保证后面的演示不会因为缺文件而中断。
        {"id": 0, "text": "## 2. 电压管理\n\n电压低于 9.6V 时必须尽快降落。低电压报警阈值建议设为 10.2V。", "source": "fallback", "chunk_strategy": "by_heading"},
        {"id": 1, "text": "## 3. 电流与功耗\n\n悬停状态下整机电流约 12A，全速爬升时可达 28A。", "source": "fallback", "chunk_strategy": "by_heading"},
        {"id": 2, "text": "## 5. 续航估算\n\n理论续航 = 容量 ÷ 平均电流 ≈ 25 分钟。建议预留 20% 电量返航。", "source": "fallback", "chunk_strategy": "by_heading"},
        {"id": 3, "text": "## 6. 充电与维护\n\n建议 1C 电流充电，长期存放充到单节 3.8V，每 3 个月补电一次。", "source": "fallback", "chunk_strategy": "by_heading"},
    ]
    # 【解释】兜底数据保留"## N. 章节名"的格式，好让章节提取逻辑照常工作。


def extract_section(text):
    """
    从块文本里抽出章节名，用作 Chroma 的 metadata（标签）。
    例如 "## 5. 续航估算\\n\\n理论续航..." → "5. 续航估算"
    """
    # 【解释】metadata 是向量数据库的"结构化标签"，能让你按条件过滤。
    m = re.match(r"^#+\s*(.+?)\s*$", text.strip().split("\n")[0])
    # 【解释】取文本第一行，用正则匹配开头的 # 号（1~n 个），
    #         捕获 # 后面的内容作为章节名。
    if m:
        # 【解释】匹配成功，说明这一块是以标题开头的。
        return m.group(1)
        # 【解释】group(1) 就是正则里第一个括号捕获的内容。
    return "未分章节"
    # 【解释】没匹配上就给个默认标签，保证每条数据都有 metadata。



# ════════════════════════════════════════════════════════════════
# 第 4 部分：复用 Day17-A / Day17-B 的成果
# ════════════════════════════════════════════════════════════════

sys.path.insert(0, HERE)
# 【解释】把本文件所在目录加进模块搜索路径。

from day17a_embedding import build_embedding_function
# 【解释】embedding 工厂（三档降级），一行都不用重写。

from day17b_chroma_store import build_collection, search_chroma
# 【解释】建库与检索函数，同样直接复用。
#         注意：import day17b 时不会执行它的 main()（因为有 __name__ 保护）。


# ════════════════════════════════════════════════════════════════
# 第 7 部分：和 Day15 的字频向量对比 —— 看清语义检索强在哪
# ════════════════════════════════════════════════════════════════

def day15_keyword_search(question, chunks, top_k=3):
    """
    复刻 Day15 的"字频向量"检索，作为对比基线。

    思路：数每个字/词出现几次 → 得到稀疏向量 → 算余弦相似度。
    特点：只认字面重合，不认意思。
    """
    # 【解释】这段代码就是 Day15 的核心算法，搬过来做对照组。
    #         真实项目里应该直接 from day15_rag_basics import search，
    #         这里重写一遍是为了让这个文件自包含、不依赖兄弟文件也能跑。

    def tokenize(text):
        # 【解释】分词函数：中文取 bigram，英文数字取整词。
        parts = re.findall(r"[\u4e00-\u9fff]+|[A-Za-z]+|\d+", text)
        # 【解释】和 LocalHashEmbedding 里一样的拆法，保证两边口径一致。
        out = []
        # 【解释】累积结果。
        for p in parts:
            # 【解释】逐片段处理。
            if re.match(r"^[\u4e00-\u9fff]+$", p):
                # 【解释】纯中文：做二字滑窗。
                out.extend([p[i:i + 2] for i in range(len(p) - 1)] or [p])
                # 【解释】or [p] 处理单字情况，避免空列表。
            else:
                # 【解释】英文/数字：转小写整体作为一个词。
                out.append(p.lower())
                # 【解释】加入词表。
        return out
        # 【解释】返回词列表。

    def vector(text, vocab):
        # 【解释】根据词表把文本变成向量：词表里有这个词，对应位就 +1。
        v = [0.0] * len(vocab)
        # 【解释】全 0 向量，长度 = 词表大小。
        for t in tokenize(text):
            # 【解释】遍历文本的每个词。
            if t in vocab:
                # 【解释】只统计"词表里出现过"的词 —— 这正是它的致命弱点：
                #         问题里的新词（词表里没有）会被直接忽略！
                v[vocab[t]] += 1.0
                # 【解释】对应下标 +1。
        return v
        # 【解释】返回稀疏词频向量。

    # ── 建词表（只用文档，不用问题）——──
    # 【解释】这一步模拟真实场景：你先有资料（文档），后有用户提问。
    #         用户问题里的词如果资料里没出现过，字频向量就完全抓瞎。
    vocab = {}
    # 【解释】词 → 下标 的映射表。
    for c in chunks:
        # 【解释】遍历所有文档块。
        for t in tokenize(c["text"]):
            # 【解释】遍历块里的每个词。
            if t not in vocab:
                # 【解释】新词：给它分配一个递增的下标。
                vocab[t] = len(vocab)
                # 【解释】下标 = 当前词表长度，天然递增不重复。

    doc_vecs = [vector(c["text"], vocab) for c in chunks]
    # 【解释】预先算好每个文档块的词频向量。
    q_vec = vector(question, vocab)
    # 【解释】算出问题的词频向量。

    def cosine(a, b):
        # 【解释】余弦相似度 = 点积 ÷ (两个向量的模长相乘)。
        dot = sum(x * y for x, y in zip(a, b))
        # 【解释】点积：对应位相乘再求和。
        na = math.sqrt(sum(x * x for x in a))
        # 【解释】a 的模长。
        nb = math.sqrt(sum(y * y for y in b))
        # 【解释】b 的模长。
        if na == 0 or nb == 0:
            # 【解释】有零向量（问题里全是新词）时，相似度为 0。
            return 0.0
            # 【解释】返回 0，表示完全不相似。
        return dot / (na * nb)
        # 【解释】返回余弦值，范围 [-1, 1]，越大越像。

    scored = [(cosine(q_vec, dv), i) for i, dv in enumerate(doc_vecs)]
    # 【解释】算出问题和每个块的相似度，带上原始下标。
    scored.sort(reverse=True)
    # 【解释】按相似度从高到低排序。
    return [{"index": i, "similarity": s, "section": extract_section(chunks[i]["text"])}
            for s, i in scored[:top_k]]
    # 【解释】取前 top_k 条，整理成和 search_chroma 一样的结构，方便对比。


def compare_two_methods(collection, chunks, is_chinese):
    """
    用几条"故意不说手册原话"的提问，对比两种方法。

    这些提问是精心设计的：用户不会照着手册措辞问，
    所以字频向量会漏，中文语义向量能救回来。

    is_chinese=False 时（英文模型或哈希兜底），这场对比是不公平的，
    代码会如实说明，不会硬吹语义检索。
    """
    # 【解释】这是今天最有说服力的演示 —— 眼见为实地看出差距。

    questions = [
        # 【解释】每组是 (用户提问, 期望命中的章节名)。
        #         期望章节是【照着手册内容逐条核对过的】，不是拍脑袋写的：
        #         比如"电池胀起来"对应的是第 6 章（手册原文：电池鼓包、漏液应立即停止使用），
        #         而不是名字听起来更像的第 7 章"安全预警"。
        #         ⚠️ 评测集标错了，再好的模型也会被判成错的 —— 这是 RAG 工程里
        #            非常常见、又极其容易被忽略的失误。
        ("电池能飞多久", "5. 续航估算"),
        # 【解释】手册里写的是"续航"，用户说的是"能飞多久" —— 措辞完全不同。
        ("多少伏就必须降落了", "2. 电压管理"),
        # 【解释】手册写"电压低于 9.6V"，用户说"多少伏"。
        ("冬天飞行掉电特别快是为什么", "4. 温度管理"),
        # 【解释】手册写"低温环境（低于 0℃）时电池内阻增大，续航会明显下降"。
        ("电池胀起来了还能继续用吗", "6. 充电与维护"),
        # 【解释】手册写"电池鼓包、漏液应立即停止使用"。
    ]

    print(f"{'提问':<22} {'方法':<12} {'最相似的块':<16} {'相似度':>7}  Top3含期望")
    # 【解释】打印表头。< 左对齐，> 右对齐，数字是宽度。
    print("─" * 70)
    # 【解释】表头下方的横线。

    wins = 0
    # 【解释】统计"语义对、字频错"的次数 = 语义检索挽回的次数。

    stat = {"kw1": 0, "kw3": 0, "ch1": 0, "ch3": 0}
    # 【解释】用字典累计四种命中数：字频/语义 × Top1/Top3。
    #         ⚠️ 为什么要同时看 Top-3？因为真实 RAG 会把前 3 块一起塞给大模型，
    #            答案只要在这 3 块里就能答对 —— 只盯 Top-1 会低估检索能力。
    #            这是面试常考的"召回率指标怎么选"。

    for q, expect_sec in questions:
        # 【解释】逐条提问做对比。
        kw_hits = day15_keyword_search(q, chunks, top_k=3)
        # 【解释】Day15 字频检索，取前 3 名。
        ch_hits = search_chroma(collection, q, top_k=3)
        # 【解释】Chroma 语义检索，取前 3 名。

        kw_sec = kw_hits[0]["section"] if kw_hits else "无"
        # 【解释】字频检索命中的章节名（第 1 名）。
        kw_sim = kw_hits[0]["similarity"] if kw_hits else 0.0
        # 【解释】字频检索的相似度数值。

        ch_sec = ch_hits[0]["section"] if ch_hits else "无"
        # 【解释】Chroma 命中的章节名（第 1 名）。
        ch_sim = ch_hits[0]["similarity"] if ch_hits else 0.0
        # 【解释】Chroma 的相似度数值。

        kw_ok1 = bool(kw_hits) and expect_sec in kw_hits[0]["section"]
        # 【解释】字频方法：第 1 名是不是期望章节。
        ch_ok1 = bool(ch_hits) and expect_sec in ch_hits[0]["section"]
        # 【解释】语义方法：第 1 名是不是期望章节。
        kw_ok3 = any(expect_sec in h["section"] for h in kw_hits)
        # 【解释】字频方法：期望章节在不在前 3 名里。
        #         any(...) 是"只要有一个满足就返回 True"的简洁写法。
        ch_ok3 = any(expect_sec in h["section"] for h in ch_hits)
        # 【解释】语义方法：期望章节在不在前 3 名里。

        stat["kw1"] += kw_ok1
        # 【解释】bool 可以直接当 0/1 加到整数上（True=1，False=0）。
        stat["kw3"] += kw_ok3
        # 【解释】累计字频 Top-3 命中数。
        stat["ch1"] += ch_ok1
        # 【解释】累计语义 Top-1 命中数。
        stat["ch3"] += ch_ok3
        # 【解释】累计语义 Top-3 命中数。

        print(f"{q:<22} {'字频(Day15)':<12} {kw_sec:<16} {kw_sim:>8.3f}   {'✅' if kw_ok3 else '❌'}")
        # 【解释】第一行：Day15 字频方法的结果，最后一列是 Top-3 是否含期望。
        print(f"{'':<22} {'语义(Chroma)':<12} {ch_sec:<16} {ch_sim:>8.3f}   {'✅' if ch_ok3 else '❌'}")
        # 【解释】第二行：Chroma 语义方法的结果。提问列留空，视觉上成对。

        if ch_ok1 and not kw_ok1:
            # 【解释】语义对、字频错 → 语义检索挽回了一次。
            wins += 1
            # 【解释】计数 +1。
            print(f"{'':<22} → ✅ 语义检索挽回（字频查的是「{kw_sec}」）")
            # 【解释】明确标出这一局语义检索赢了。
        elif ch_ok1 and kw_ok1:
            # 【解释】两边都对，平局。
            print(f"{'':<22} → 两者都命中")
            # 【解释】说明这题两种办法都行（通常是因为有字面重合）。
        else:
            # 【解释】语义也没命中。
            print(f"{'':<22} → ⚠️ 都未命中（期望：{expect_sec}）")
            # 【解释】诚实地标出来，不粉饰结果。
        print("─" * 70)
        # 【解释】每条提问之间画一条分隔线。

    n = len(questions)
    # 【解释】题目总数，算命中率要用。
    print(f"📊 命中统计（共 {n} 题）")
    # 【解释】先给个总览标题。
    print(f"   Top-1 命中：字频 {stat['kw1']}/{n}    语义 {stat['ch1']}/{n}")
    # 【解释】只看第 1 名的命中数。
    print(f"   Top-3 命中：字频 {stat['kw3']}/{n}    语义 {stat['ch3']}/{n}")
    # 【解释】前 3 名里包含的命中数 —— 真实 RAG 看的其实是这个。
    print()
    # 【解释】空一行，让后面的结论更醒目。

    if not is_chinese:
        # 【解释】当前是哈希兜底或英文模型，都不具备中文语义能力。
        print("⚠️  注意：当前不是中文语义模型，上面的对比不公平 ——")
        # 【解释】必须说清楚，否则用户会误以为"语义检索也不过如此"。
        print("    · 本地哈希向量：只认字面，不认意思")
        # 【解释】第一种情况。
        print("    · 英文 MiniLM 模型：训练语料全是英文，中文语义它抓不住")
        # 【解释】第二种情况，这是最容易被忽略的坑。
        print("    想看到真正的语义效果：pip install sentence-transformers 后重跑，")
        # 【解释】给出解决办法。
        print("    程序会自动加载中文 BGE 模型，那时差距会非常明显。")
        # 【解释】预告真实效果。
    elif stat["ch3"] > stat["kw3"] or stat["ch1"] > stat["kw1"]:
        # 【解释】语义在 Top-1 或 Top-3 上确实比字频强。
        print(f"🎯 语义检索 Top-1 挽回 {wins} 次；"
              f"Top-3 命中 {stat['kw3']}/{n} → {stat['ch3']}/{n}")
        # 【解释】用数据说话，前后对比一目了然。
        print("   而且注意相似度数值本身：字频的分数普遍在 0.1~0.4（几乎等于随机），")
        # 【解释】这是字频向量的致命伤：分数没有区分度。
        print("   语义的分数集中在 0.55~0.70 —— 有了区分度，")
        # 【解释】语义向量的分数可解释。
        print("   才谈得上'设阈值过滤掉不相关内容'（Day18 会做）。")
        # 【解释】为下一课埋伏笔：分数可用，阈值才有意义。
    else:
        # 【解释】真语义模式下也没赢，如实说明，绝不粉饰。
        print("ℹ️  本次对比里语义检索没有拉开差距 —— 这是真实结果，不粉饰。")
        # 【解释】诚实是教学代码的底线。
        print("   原因有三，都是 RAG 实战中的常见情况：")
        # 【解释】逐条给出原因，这才是"会分析"而不是"会跑代码"。
        print("   1) 语料太小（只有 8 块），且全部讲电池，主题高度同质，")
        # 【解释】第一条：同质语料让"语义"没多少发挥空间。
        print("      向量都挤在一起，谁跟谁都不算远。")
        # 【解释】补充第一条的后果。
        print("   2) 提问里仍有字面重合（如'多少伏'对'电压'），字频侥幸能碰对。")
        # 【解释】第二条：字面重合让字频蒙对。
        print("   3) 可以换更大的模型（bge-base-zh / bge-large-zh）再试，")
        # 【解释】第三条：给出可验证的改进方向。
        print("      工业界正是靠'换更强的 embedding'来拉开差距的。")
        # 【解释】说明这是标准做法。


# ════════════════════════════════════════════════════════════════
# 第 10 部分：闭环 —— Chroma 检索 + 大模型生成 = 完整 RAG
# ════════════════════════════════════════════════════════════════

def ask_model(messages, temperature=0.3):
    """调大模型生成答案（复用 w3_ai 的配置，和 Day10~Day15 一致）。"""
    # 【解释】这段和 Day15 的 ask_model 几乎一样，体现"跨周复用"。

    if not HAS_KEY:
        # 【解释】没配 Key 就没法生成。
        return None
        # 【解释】返回 None，调用方据此跳过。

    url = BASE_URL.rstrip("/") + "/chat/completions"
    # 【解释】拼接接口地址。rstrip("/") 防止配置里末尾带斜杠导致出现双斜杠。

    headers = {
        # 【解释】HTTP 请求头。
        "Authorization": "Bearer " + API_KEY,
        # 【解释】鉴权：Bearer + Key，这是 OpenAI 兼容格式的标准写法。
        "Content-Type": "application/json",
        # 【解释】声明请求体是 JSON。
    }

    payload = {
        # 【解释】请求体。
        "model": MODEL_NAME,
        # 【解释】用哪个模型，取自 api_config。
        "messages": messages,
        # 【解释】对话历史，格式是 [{"role": "...", "content": "..."}, ...]。
        "temperature": temperature,
        # 【解释】温度：越低越严谨。RAG 场景要"照着资料说"，所以设得低。
    }

    try:
        # 【解释】网络请求随时可能失败，必须包 try。
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        # 【解释】发 POST 请求，超时 60 秒。
        r.raise_for_status()
        # 【解释】HTTP 状态码不是 2xx 就抛异常（例如 401 Key 错、429 限流）。
        data = r.json()
        # 【解释】把响应解析成字典。
        if "choices" not in data:
            # 【解释】接口返回了错误结构（Day12 踩过的坑），提前拦住，
            #         否则下面 data["choices"][0] 会抛 KeyError。
            print("   ⚠️ 接口返回异常：", str(data)[:150])
            # 【解释】打印错误内容，方便定位。
            return None
            # 【解释】返回 None，不继续解析。
        return data["choices"][0]["message"]["content"]
        # 【解释】取出模型回答的正文。
    except Exception as e:
        # 【解释】任何异常都接住，不能让程序崩。
        print(f"   ⚠️ 调用失败：{type(e).__name__}: {str(e)[:120]}")
        # 【解释】打印异常类型和信息。
        return None
        # 【解释】返回 None。


def rag_answer(collection, question, top_k=3):
    """完整 RAG：Chroma 检索 → 拼 Prompt → 大模型生成。"""
    # 【解释】这是 Day15 五步流程的"工业版"：把自研检索换成 Chroma。

    hits = search_chroma(collection, question, top_k=top_k)
    # 【解释】第 4 步：检索。这次用的是向量数据库。

    if not hits:
        # 【解释】一条都没检索到，就没法生成。
        return None, []
        # 【解释】返回空结果。

    context = "\n\n".join(
        # 【解释】把命中的块拼成一段"参考资料"。
        f"[{i+1}] 来源：{h['section']}\n{h['text']}"
        # 【解释】每条资料带上编号和章节名，方便模型引用出处。
        for i, h in enumerate(hits)
        # 【解释】enumerate 同时给出序号和内容。
    )
    # 【解释】用两个换行分隔不同块，让模型能区分开。

    messages = [
        # 【解释】构建对话消息（System + User 两段，Day12 学过）。
        {
            "role": "system",
            # 【解释】System 消息用来设定模型的行为准则。
            "content": (
                # 【解释】下面这段 Prompt 是 RAG 的"灵魂"，三条规定缺一不可。
                "你是无人机电池与能耗方面的技术助手。\n"
                # 【解释】第 1 条：定角色，限定回答范围。
                "请严格根据【参考资料】回答用户问题。\n"
                # 【解释】第 2 条：强制它只用资料，这是消除幻觉的关键。
                "如果资料里没有相关信息，直接说「资料中没有提到」，不要编造。\n"
                # 【解释】第 3 条：给"不知道"留出口，否则模型会硬编。
                "回答时请注明你引用了哪一条资料。"
                # 【解释】第 4 条：要求标注出处，方便人工核查。
            ),
        },
        {
            "role": "user",
            # 【解释】User 消息 = 资料 + 问题。
            "content": f"【参考资料】\n{context}\n\n【用户问题】\n{question}",
            # 【解释】把检索到的资料和原始问题拼在一起发出去。
        },
    ]

    answer = ask_model(messages)
    # 【解释】第 5 步：生成。拿到模型回答（没 Key 时是 None）。
    return answer, hits
    # 【解释】同时返回答案和命中记录，方便把"引用了哪些块"也展示出来。



# ════════════════════════════════════════════════════════════════
# 第 8 部分：动手跑对比 + 完整问答
# ════════════════════════════════════════════════════════════════

def main():
    print("=" * 62)
    print("  Day 17-C：完整 RAG —— 检索到的资料怎么变成人话")
    print("=" * 62)

    _sep("第 1 步 / 3：加载数据 + 建库（复用 A、B 两个文件）")
    chunks = load_chunks()
    embed_fn, mode_name, is_semantic, is_chinese = build_embedding_function()
    print(f"   当前模式：{mode_name}（共 {len(chunks)} 块）")

    if not HAS_CHROMA:
        print("\n⚠️  没装 chromadb，无法继续。请先安装：")
        print("   D:/Python-envs/chroma-env/Scripts/python.exe -m pip install chromadb")
        return

    collection = build_collection(chunks, embed_fn, rebuild=True)

    _sep("第 2 步 / 3：Day15 字频检索 vs Chroma 语义检索，到底差多少")
    compare_two_methods(collection, chunks, is_chinese)
    # 【解释】传 is_chinese 而不是 is_semantic：英文模型虽然"是语义模型"，
    #         但对中文无效，不能让它冒充中文语义能力去对比 —— 这是诚实评测。

    _sep("第 3 步 / 3：完整 RAG —— 检索 + 生成，并标出引用来源")
    if HAS_KEY:
        ans, used = rag_answer(collection, "冬天低温飞行，电池要注意什么？")
        # 【解释】这个问题跨两个章节（温度 + 续航），能测出检索是不是真的懂意思。
        if ans:
            print("【模型回答】")
            print(ans)
            print("\n【本次引用了这些块】")
            for h in used:
                print(f"   - {h['section']}（相似度 {h['similarity']:.3f}）")
            # 【解释】把引用来源打出来，这是 RAG 相对"裸问模型"最大的价值：可追溯。
    else:
        print("ℹ️  没检测到 API Key，跳过生成环节（检索对比部分已全部演示）")
        print("   配置方法：在 w3_ai/api_config.py 里选 PROVIDER 并设置对应环境变量")

    # ── 收尾：学习路线图 ──
    print("\n" + "═" * 62)
    print("  【读完之后】Day17-C 你学到了什么")
    print("═" * 62)
    print("""
    ✅ 已掌握
       · RAG 五步闭环：加载 → 切块 → 向量化 → 检索 → 生成
       · 字频检索只能匹配字面，语义检索能命中"换个说法"的资料
       · System Prompt 四要素：定角色 / 只许用资料 / 允许说不知道 / 要求标出处
       · 可追溯：回答的同时能列出引用了哪些块，这是 RAG 的核心价值

    ⚠️  还要留意的坑
       · 「只许用资料」这句必须写，否则模型会拿自己的知识瞎编（幻觉）
       · 「允许说不知道」也要写，不然没有出口，模型反而更倾向硬编
       · 温度要调低（0.3 左右）：RAG 场景要"照着资料说"，不是要创造性

    ➡️  下一步
       Day18：检索调优 —— 相似度阈值拒答、查询改写、混合检索、重排序
    """)


if __name__ == "__main__":
    main()
