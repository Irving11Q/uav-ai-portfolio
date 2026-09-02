"""
════════════════════════════════════════════════════════════════
【教学版】Day 17：向量数据库 Chroma —— 让检索"懂意思"（第 4 周第 3 天）
════════════════════════════════════════════════════════════════

这个程序做的事（先记住，再往下读）：
    1. 加载 Day16 切好的手册块（chunks_parsed.json，8 块）
    2. 用真正的 embedding 模型把每块变成"语义向量"（384 维）
    3. 存进 Chroma 向量数据库（落盘，关掉程序也不丢）
    4. 提问 → Chroma 按"意思"找最相关的块（不是按字面相同）
    5. 对比 Day15 的"字频向量"检索：看语义检索强在哪
    6. 演示元数据过滤：只在"第 2 章 电压管理"里搜
    7. 串成完整 RAG：Chroma 检索 → 喂给 GLM → 生成回答

【为什么学这个】
    Day15 为了讲清原理，用的是"字频向量"（数词出现几次）。
    那个办法能跑，但有三个致命问题，今天就是要解决它们：

        ❌ 痛点 1：只认字面，不认意思
           手册里写"理论续航 ≈ 25 分钟"，你问"电池能飞多久"，
           一个字都不重合 → 字频向量算出来相似度 = 0 → 检索失败。
           （真实用户根本不会照着手册的措辞提问！）

        ❌ 痛点 2：每次都要全量暴力扫描
           8 块无所谓，10 万块时每次提问都要算 10 万次余弦 → 卡死。
           向量数据库内部用 HNSW 索引，毫秒级返回。

        ❌ 痛点 3：内存里算完就丢
           程序一关，向量全没了，下次启动要重新算一遍。
           向量数据库会把索引落盘，下次直接加载。

    今天上场的是 Chroma —— 目前最流行的开源向量数据库之一，
    招聘 JD 里高频出现（和 LangChain 一起），面试聊 RAG 必提。

【核心知识点】（今天讲透 4 个）
    ⭐ Embedding（嵌入）：把一个文本压缩成一串固定长度的数字（向量），
        语义越接近 → 向量在空间里越靠近（夹角越小）。
        例："电池能飞多久" 和 "理论续航 25 分钟" 字面不同，
            但向量夹角很小 → 语义检索能配上。
    ⭐ 向量数据库：专门存向量 + 做"最近邻搜索"的数据库。
        Chroma 三板斧：建集合(collection) → add 存 → query 查。
    ⭐ 相似度度量：Chroma 默认用 L2 距离（越小越像）。
        本程序把向量做 L2 归一化，于是有换算公式：
            余弦相似度 = 1 - L2距离² / 2
        （归一化的两个向量，和就是 1 - dist²/2，代码里会演示）
    �.metadata 过滤：给每块贴标签（章节名、来源文件），
        查询时可以先按标签筛，再在子集里搜 → 企业里非常常用。

【运行方式】
    cd w4_rag
    pip install chromadb sentence-transformers
    python day17_vector_db.py

    关于 embedding 模型（三档自动降级，国内网络必看）：
        A 档｜中文语义模型 BAAI/bge-small-zh-v1.5（512 维）
              → 需要 sentence-transformers，首次下载约 95MB
        B 档｜Chroma 自带英文模型 all-MiniLM-L6-v2（384 维）
              → 对中文几乎无效，仅作"反面教材"保留
        C 档｜本地哈希向量（256 维）
              → 零依赖零网络，保证任何环境都能跑通全流程

    模型默认从 huggingface.co 下载，国内连不上，
    本程序在 import 之前就把下载源换成了国内镜像 hf-mirror.com。
    三档依次尝试，哪一档成功就用哪一档，绝不会卡死或报错。

    ⚠️ 本文件最重要的一个工程教训（面试可以讲）：
       做中文 RAG 用英文 embedding 模型 = 白做。
       B 档模型就是活生生的反例，程序运行时会把它"翻车"的结果
       原样打印出来，让你亲眼看到差距。

【怎么读这个文件】—— 顺序：
    第 1 步：读文件头（现在）—— 先建立整体印象
    第 2 步：跳到最后的 main() → 看全流程怎么串
    第 3 步：读"加载 Day16 产物" → 输入数据长什么样
    第 4 步：读"Embedding 三选一" → 今天最关键的设计决策
    第 5 步：读"建库 / 检索 / 过滤 / 持久化" → Chroma 的 API
    第 6 步：读"和 Day15 对比" → 理解语义检索到底强在哪
    第 7 步：读"完整 RAG 问答" → 闭环到 Day15 的五步流程

规则：每一行代码下面，紧跟一行【解释】。
════════════════════════════════════════════════════════════════
"""

# ════════════════════════════════════════════════════════════════
# 第 1 部分：import —— 拿工具 + 处理"环境不齐"的降级
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
# 第 4 部分：Embedding 三选一 —— 今天最关键的设计决策
#
#   方案 A（最优）：Chroma 自带 ONNX 模型 all-MiniLM-L6-v2
#                   384 维，真·语义，首次下载 ~90MB
#   方案 B（次优）：云端 embedding API（需 Key，本程序留出接口）
#   方案 C（兜底）：本地哈希向量，256 维，无语义，但离线必跑通
#
#   为什么要设计三层？因为"教学代码"必须能在任何人的机器上跑起来。
#   面试官 clone 下来没网、没 Key，也应该能看到完整流程。
# ════════════════════════════════════════════════════════════════

class LocalHashEmbedding:
    """
    方案 C：本地哈希向量（兜底用）。

    原理：把文本切成词 → 每个词用 md5 映射到一个下标 →
          该下标 +1 → 最后整体归一化。
    特点：确定性（同一文本永远同一向量）、离线、极快，
          但【没有语义】——"续航"和"能飞多久"在它眼里毫无关系。
    """

    def __init__(self, dim=256):
        # 【解释】dim = 向量维度。256 维对演示够用了（真模型是 384 维）。
        self.dim = dim
        # 【解释】把维度存到实例上，后面 _embed 要用。

    def _tokenize(self, text):
        """
        把一段文本切成"词"。中文按 2 字滑窗（bigram），
        英文/数字按整词。
        """
        # 【解释】中文没有空格分词，最简单的办法就是按 2 个字一组滑窗：
        #         "续航估算" → ["续航", "续估", "估算"]，能保留一点词序信息。
        tokens = []
        # 【解释】用来累积所有切出来的词。
        parts = re.findall(r"[\u4e00-\u9fff]+|[A-Za-z]+|\d+", text)
        # 【解释】正则把文本拆成三类片段：连续中文 / 连续英文 / 连续数字。
        for p in parts:
            # 【解释】逐个片段处理。
            if re.match(r"^[\u4e00-\u9fff]+$", p):
                # 【解释】如果是纯中文片段：
                if len(p) == 1:
                    # 【解释】单字没法做 bigram，直接作为一个词。
                    tokens.append(p)
                    # 【解释】加入词表。
                else:
                    for i in range(len(p) - 1):
                        # 【解释】从第 0 个字滑到倒数第 2 个字，每次取 2 个字。
                        tokens.append(p[i:i + 2])
                        # 【解释】把这个二字词加入词表。
            else:
                # 【解释】英文或数字片段，整体当一个词即可。
                tokens.append(p.lower())
                # 【解释】统一转小写，这样 "OK" 和 "ok" 算同一个词。
        return tokens
        # 【解释】返回切好的词列表。

    def _embed_one(self, text):
        """把一段文本变成一条向量（list[float]）。"""
        vec = [0.0] * self.dim
        # 【解释】先造一条全 0 的向量，长度 = dim。
        for tok in self._tokenize(text):
            # 【解释】遍历每个词。
            h = hashlib.md5(tok.encode("utf-8")).digest()
            # 【解释】对词做 md5，得到 16 字节的二进制摘要。
            #         用 md5 而不是内置 hash()，是因为它跨进程、跨机器都稳定。
            idx = int.from_bytes(h[:2], "big") % self.dim
            # 【解释】取摘要前 2 个字节转成整数，再对 dim 取余 → 得到 0~255 的下标。
            vec[idx] += 1.0
            # 【解释】该下标计数 +1（这就是"哈希词袋"：词落在哪，哪就加一）。
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        # 【解释】算向量的模长（L2 范数）。or 1.0 是为了防止全 0 向量导致除零。
        return [x / norm for x in vec]
        # 【解释】归一化：每个分量除以模长 → 向量长度变成 1。
        #         归一化后，两个向量的点积就等于余弦相似度，后面计算更方便。

    def __call__(self, input):
        """
        Chroma 要求的接口：传入文本列表，返回向量列表。
        注意参数名必须是 input（Chroma 内部这样调用）。
        """
        # 【解释】Chroma 的 EmbeddingFunction 协议很简单：
        #         实现 __call__(self, input: list[str]) -> list[list[float]] 即可。
        return [self._embed_one(t) for t in input]
        # 【解释】对每条文本分别算向量，组成列表返回。

    def embed_query(self, input):
        """
        给"提问"算向量。默认和算文档完全一样。
        """
        # 【解释】Chroma 1.x 查询时会调 embed_query() 而不是 __call__()。
        #         为什么分两个？因为有些模型（如 BGE、E5）在编码"提问"和
        #         编码"文档"时要区别对待，检索才准。
        #         哈希向量没有这个讲究，直接复用 __call__。
        return self.__call__(input)
        # 【解释】转交给 __call__，行为一致。

    # ── Chroma 1.x 额外要求的三个方法 ──
    # 【解释】Chroma 从 1.0 起，自定义 embedding 除了 __call__ 还必须实现
    #         name() / get_config() / build_from_config()，否则一跑就报：
    #         AttributeError: 'XXX' object has no attribute 'name'
    #         原因是 Chroma 要把 embedding 的"身份"写进数据库元数据里，
    #         下次打开同一个库时，它会比对名字 —— 发现你换了模型，
    #         就知道旧向量全部作废了（不同模型的向量空间不兼容）。
    #         这个设计其实是在帮你避免一个很隐蔽的坑：
    #         "换了 embedding 模型却忘了重建库，检索结果全是乱的"。

    @staticmethod
    def name() -> str:
        """这个 embedding 的唯一标识（会存进 Chroma 的库配置里）。"""
        # 【解释】必须是 staticmethod：Chroma 有时只拿到类、没拿到实例就要问名字。
        return "local_hash_v1"
        # 【解释】名字随意，但要能表达"这是哪个实现"。

    def get_config(self) -> dict:
        """把构造参数存下来，供 build_from_config 复原。"""
        # 【解释】只存可序列化的基本类型（dict/str/int/float/bool），
        #         别把模型对象塞进去 —— 那样写不进数据库。
        return {"dim": self.dim}
        # 【解释】把维度记下来，重建时能还原成一样的配置。

    @classmethod
    def build_from_config(cls, config: dict) -> "LocalHashEmbedding":
        """根据配置复原一个实例（Chroma 重新打开旧库时会调用）。"""
        # 【解释】classmethod：Chroma 传的是类，由类自己 new 一个实例出来。
        return cls(**config)
        # 【解释】config 是 get_config() 存的那份，原样解包传回构造函数。


class SentenceTransformerEmbedding:
    """
    方案 A：真正的中文语义模型（sentence-transformers 加载）。

    默认用 BAAI/bge-small-zh-v1.5 —— 目前中文检索效果最好的开源小模型之一
    （512 维，约 95MB），在中文语义相似度任务上远胜英文模型。

    ⚠️ 这是本文件最重要的工程经验：
       做中文 RAG，**千万不能直接用英文 embedding 模型**。
       Chroma 自带的 all-MiniLM-L6-v2 是纯英文语料训练的，
       拿它算中文，"电池能飞多久"和"续航估算"在它眼里毫无关系，
       检索出来全是错的 —— 后面会实测给你看。

    【关于模型选型：我在本手册上实测过两档】
       bge-small-zh-v1.5（95MB）  → Top-1 命中 2/4，Top-3 命中 2/4
       bge-base-zh-v1.5（400MB）  → Top-1 命中 2/4，Top-3 命中 3/4
       结论：base 只在"前 3 名里含不含答案"上多救回 1 题，
       代价是体积 ×4、速度更慢。
       所以教学和大多数业务场景，small 是更划算的选择；
       真要上线前，把自己的评测集跑一遍再决定（改下面一行即可切换）。
    """

    def __init__(self, model_name="BAAI/bge-small-zh-v1.5"):
        # 【解释】model_name 是 HuggingFace 上的模型 id，下载走探测到的可用源。
        self.model_name = model_name
        # 【解释】存到实例上：get_config() 要把它写进 Chroma 的库配置，
        #         这样下次打开同一个库时，Chroma 知道这些向量是哪个模型算的。

        from sentence_transformers import SentenceTransformer
        # 【解释】延迟导入：只有确定要用这个方案时才 import，
        #         这样没装 sentence-transformers 的人也不会在启动时就崩。

        self.model = SentenceTransformer(model_name)
        # 【解释】加载模型（首次会从镜像下载约 95MB，之后走缓存）。
        self.dim = self.model.get_sentence_embedding_dimension()
        # 【解释】问模型拿到向量维度（bge-small-zh 是 512）。

    def __call__(self, input):
        """Chroma 接口：文本列表 → 向量列表。"""
        # 【解释】和 LocalHashEmbedding 保持一样的签名，Chroma 不关心你内部怎么实现。
        vecs = self.model.encode(
            # 【解释】批量编码，一次算完比逐条快得多。
            list(input),
            # 【解释】input 可能是别的序列类型，转一下更保险。
            normalize_embeddings=True,
            # 【解释】直接让模型输出归一化向量（长度 1），
            #         这样后面的余弦相似度计算最省事，语义也更准。
            show_progress_bar=False,
            # 【解释】不要进度条，免得刷屏。
        )
        return [v.tolist() for v in vecs]
        # 【解释】numpy 数组转成 Python 列表，Chroma 才认。

    # ── 关于 BGE 官方推荐的"查询指令前缀"：实测后决定不用 ──
    # 【解释】BGE 官方建议在检索时给"提问"拼一句指令：
    #         "为这个句子生成表示以用于检索相关文章："
    #         我在本手册上实测了加与不加（4 条提问，看 Top-1 命中率）：
    #             不加前缀 → 2/4 命中
    #             加前缀   → 1/4 命中（反而更差）
    #         ⚠️ 工程启示：**官方最佳实践也要在自己的数据上验证，别照抄。**
    #         所以这里默认留空；如果你换了自己的语料，可以打开试试。
    QUERY_INSTRUCTION = ""
    # 【解释】这是"类变量"：所有实例共用，想改就改成那句中文指令。

    def embed_query(self, input):
        """
        给"提问"算向量 —— Chroma 查询时走的是这个方法，不是 __call__。
        """
        # 【解释】Chroma 1.x 会区分"入库文档"和"查询提问"两条路径：
        #         add()   → __call__()
        #         query() → embed_query()
        #         这样设计就是为了兼容 BGE / E5 这类"问答要用不同编码"的模型。
        texts = list(input)
        # 【解释】转成列表，方便下面逐条处理。

        if self.QUERY_INSTRUCTION:
            # 【解释】只有当 QUERY_INSTRUCTION 非空时才拼前缀。
            texts = [self.QUERY_INSTRUCTION + t for t in texts]
            # 【解释】给每条提问加上指令前缀。

        vecs = self.model.encode(
            texts,
            # 【解释】编码这批提问。
            normalize_embeddings=True,
            # 【解释】和入库时保持一致：都归一化，余弦相似度才准。
            show_progress_bar=False,
            # 【解释】不显示进度条。
        )
        return [v.tolist() for v in vecs]
        # 【解释】同样转成 Python 列表返回。

    # ── Chroma 1.x 要求的三个配套方法（和 LocalHashEmbedding 同理）──
    # 【解释】没有这三个方法，Chroma 会直接抛
    #         AttributeError: 'SentenceTransformerEmbedding' object has no attribute 'name'
    #         这是我在 Chroma 1.5.9 上实测踩到的坑，不是书上写的。

    @staticmethod
    def name() -> str:
        """唯一标识：这里带上了模型名，方便日后排查"库是用哪个模型建的"。"""
        # 【解释】把模型名编进 name 里，好处是打开旧库时
        #         Chroma 能一眼看出 embedding 换过，提示你重建。
        return "sentence_transformers_bge_small_zh"
        # 【解释】名字要稳定：改了名字等于告诉 Chroma"换模型了"。

    def get_config(self) -> dict:
        """记录用的是哪个模型，重建时能原样恢复。"""
        # 【解释】存模型名就够了，模型权重本身不用存（在 HF 缓存里）。
        return {"model_name": self.model_name}
        # 【解释】所以 __init__ 里要把 model_name 存到 self 上（见下方改动）。

    @classmethod
    def build_from_config(cls, config: dict) -> "SentenceTransformerEmbedding":
        """根据配置复原实例（会触发模型加载）。"""
        # 【解释】注意：这一步可能联网下载模型，所以要能容忍失败。
        return cls(**config)
        # 【解释】把存下来的 model_name 传回构造函数。


def build_embedding_function():
    """
    按 A → B → C 的顺序尝试，返回 (函数, 模式名, 是否真语义, 是否支持中文)。

    A：中文语义模型 BGE-small-zh（最优，中文任务必选）
    B：Chroma 自带英文 ONNX 模型（中文效果差，仅作对比演示）
    C：本地哈希向量（离线兜底，无语义）

    返回四元组是为了让后面知道"当前是哪种模式"：
    只有中文语义模式（A）才跑得动"同义词对比"，否则要如实说明效果打折。
    """
    # 【解释】这是"优雅降级"的写法：能力强的优先，失败就退到下一档，
    #         而不是直接崩掉。工业代码里这种写法很常见。

    if not HAS_CHROMA:
        # 【解释】Chroma 都没装，方案 A/B 无从谈起，直接用兜底。
        return LocalHashEmbedding(), "本地哈希向量（Chroma 未安装）", False, False
        # 【解释】四个值：函数 / 说明 / 非真语义 / 不支持中文。

    # ── 尝试方案 A：中文语义模型 ──
    try:
        # 【解释】把"可能失败"的代码包在 try 里，失败就跳到 except。
        fn = SentenceTransformerEmbedding()
        # 【解释】实例化：会自动下载 BGE 中文模型（走 hf-mirror 镜像）。

        _ = fn(["测试中文语义向量"])
        # 【解释】先小规模试跑一次，确认模型确实能用
        #         （光实例化成功不算数，模型文件可能是坏的）。

        print(f"✅ Embedding 方案 A：中文语义模型（BAAI/bge-small-zh-v1.5，{fn.dim} 维）")
        # 【解释】打印当前用的是哪种方案和维度。
        return fn, "中文语义模型 (bge-small-zh-v1.5)", True, True
        # 【解释】真语义 + 支持中文 → 后面敢跑同义词对比。

    except Exception as e:
        # 【解释】没装 sentence-transformers / 下载失败 / 模型损坏，都会落到这里。
        print(f"⚠️  中文语义模型加载失败（{type(e).__name__}），尝试英文模型")
        # 【解释】把失败原因类型打出来，方便判断是网络问题还是没装依赖。
        print(f"    原因：{str(e)[:120]}")
        # 【解释】错误信息截断，避免刷屏。

    # ── 尝试方案 B：Chroma 自带的英文 ONNX 小模型 ──
    try:
        # 【解释】英文模型作为第二档，保证没装 sentence-transformers 也能跑。
        from chromadb.utils import embedding_functions
        # 【解释】Chroma 把各种 embedding 实现放在这个子模块里。

        fn = embedding_functions.ONNXMiniLM_L6_V2()
        # 【解释】实例化：all-MiniLM-L6-v2 的 ONNX 版，384 维。

        _ = fn(["test embedding"])
        # 【解释】同样先试跑一次，验证模型可用。

        print("⚠️  Embedding 方案 B：英文模型 all-MiniLM-L6-v2（384 维）")
        # 【解释】用 ⚠️ 而不是 ✅，因为它是降级方案。
        print("    提醒：英文模型对中文语义几乎无效，下面的对比会『翻车』——这是刻意保留的")
        # 【解释】提前预告，让用户看懂后面的对比结果为什么差。
        return fn, "英文 ONNX 模型 (all-MiniLM-L6-v2，中文效果差)", True, False
        # 【解释】是真语义模型，但不支持中文 → 第四个值 False。

    except Exception as e:
        # 【解释】下载失败 / 缺少 onnxruntime 等。
        print(f"⚠️  英文模型也加载失败（{type(e).__name__}），降级为本地哈希向量")
        # 【解释】说明状况。
        print(f"    原因：{str(e)[:120]}")
        # 【解释】打印原因。

    # ── 方案 C：兜底 ──
    return LocalHashEmbedding(), "本地哈希向量（离线兜底）", False, False
    # 【解释】走到这里说明 A、B 都没成，用离线方案保证流程完整。


# ════════════════════════════════════════════════════════════════
# 第 5 部分：建库 —— Chroma 的第一板斧 add
# ════════════════════════════════════════════════════════════════

def build_collection(chunks, embed_fn, rebuild=True):
    """
    把块写进 Chroma。返回 collection 对象（后面查询要用）。

    rebuild=True 表示"每次运行都重建"，方便教学演示看到干净的结果；
    真实项目里应该设 False，直接复用已存在的库。
    """
    # 【解释】这是向量数据库的"写入"操作，对应 SQL 里的 INSERT。

    client = chromadb.PersistentClient(path=DB_DIR)
    # 【解释】PersistentClient = 持久化客户端，数据会写到 DB_DIR 目录里。
    #         对比：chromadb.Client() 是内存版，程序一关数据就没了。

    name = "uav_manual"
    # 【解释】集合名，相当于 SQL 里的"表名"。

    if rebuild:
        # 【解释】教学模式下先删旧库，保证每次看到的都是干净结果。
        try:
            client.delete_collection(name)
            # 【解释】删掉同名集合。注意：删集合会连里面的数据一起删。
        except Exception:
            # 【解释】第一次运行时集合本来就不存在，delete 会抛异常，忽略即可。
            pass
            # 【解释】pass = 什么都不做，继续执行。

    collection = client.get_or_create_collection(
        # 【解释】get_or_create = 有就打开，没有就新建。这是最常用的写法。
        name=name,
        # 【解释】集合名，必须传。
        embedding_function=embed_fn,
        # 【解释】绑定 embedding 函数：以后 add / query 传文本时，
        #         Chroma 会自动用这个函数把文本转成向量，你不用手动转。
        metadata={"hnsw:space": "cosine"},
        # 【解释】指定相似度度量方式为余弦（默认其实是 L2）。
        #         用余弦后，Chroma 返回的 distance 直接就是"余弦距离"，
        #         相似度 = 1 - distance，不用再做平方换算，教学更直观。
    )

    ids = [str(c["id"]) for c in chunks]
    # 【解释】每条记录的唯一 ID。Chroma 要求必须是字符串，所以 str() 转一下。

    documents = [c["text"] for c in chunks]
    # 【解释】正文原文。Chroma 会拿它去算向量，并原样存下来供你取用。

    metadatas = [
        # 【解释】每条记录贴的标签，查询时可以按标签过滤。
        {
            "section": extract_section(c["text"]),
            # 【解释】章节名，从文本第一行提取（例如 "5. 续航估算"）。
            "source": c.get("source", "unknown"),
            # 【解释】来源文件名。用 get 给默认值，防止原始 JSON 里没有这个字段。
            "strategy": c.get("chunk_strategy", "unknown"),
            # 【解释】切块策略（Day16 用的是 by_heading）。
            "length": len(c["text"]),
            # 【解释】块长度。Chroma 支持数值型 metadata，可以做范围过滤。
        }
        for c in chunks
        # 【解释】遍历每个块，生成对应的 metadata 字典。
    ]

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    # 【解释】真正的写入。三个列表长度必须一致，按下标一一对应。
    #         执行完，向量索引就在 DB_DIR 里落盘了。

    print(f"✅ 已写入 {collection.count()} 条向量 → 目录：{DB_DIR}")
    # 【解释】count() 返回集合里的记录数，顺便验证写入成功。
    return collection
    # 【解释】把集合对象交出去，后面查询要用它。


# ════════════════════════════════════════════════════════════════
# 第 6 部分：检索 —— Chroma 的第二板斧 query
# ════════════════════════════════════════════════════════════════

def search_chroma(collection, question, top_k=3, where=None):
    """
    在 Chroma 里查最相关的 top_k 条。

    where 是 metadata 过滤条件，例如 {"section": "5. 续航估算"}。
    """
    # 【解释】这是向量数据库的"读"操作，是 RAG 里第 4 步"检索"的核心。

    kwargs = {"query_texts": [question], "n_results": top_k}
    # 【解释】query_texts 必须传列表（支持一次查多个问题）；
    #         n_results 控制返回几条。

    if where:
        # 【解释】只有传了过滤条件时才加 where 参数。
        kwargs["where"] = where
        # 【解释】Chroma 会先按 metadata 筛选，再在子集里做向量检索。

    res = collection.query(**kwargs)
    # 【解释】**kwargs = 把字典展开成关键字参数，等价于手写各参数。
    #         这一步 Chroma 内部会：把问题转向量 → HNSW 索引找最近邻 → 返回。

    hits = []
    # 【解释】用来装整理后的结果，方便后面统一使用。
    if not res["documents"] or not res["documents"][0]:
        # 【解释】没查到任何东西（比如过滤条件太严），直接返回空列表。
        return hits
        # 【解释】空列表 = 没命中，调用方自己处理。

    docs = res["documents"][0]
    # 【解释】第 0 个问题的命中正文列表（因为我们只查了 1 个问题）。
    metas = res["metadatas"][0]
    # 【解释】对应的 metadata 列表。
    dists = res["distances"][0]
    # 【解释】对应的距离列表。越小越相似。
    ids = res["ids"][0]
    # 【解释】对应的 ID 列表。

    for i in range(len(docs)):
        # 【解释】把四个列表按下标配对，组装成结构化的结果。
        hits.append({
            # 【解释】每条命中记成一个字典，字段见下。
            "id": ids[i],
            # 【解释】块 ID。
            "text": docs[i],
            # 【解释】块正文。
            "section": metas[i].get("section", "?"),
            # 【解释】章节名，取不到就显示 ?。
            "distance": dists[i],
            # 【解释】距离（余弦距离：0 = 完全一样，2 = 完全相反）。
            "similarity": 1.0 - dists[i],
            # 【解释】换算成相似度，越接近 1 越像。
            #         因为建库时指定了 hnsw:space = cosine，所以可以直接这么换算。
        })
    return hits
    # 【解释】返回命中列表，按相似度从高到低排（Chroma 已排好序）。


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
# 第 8 部分：metadata 过滤 —— 企业里最常用的进阶功能
# ════════════════════════════════════════════════════════════════

def demo_metadata_filter(collection):
    """演示：先按章节筛，再在子集里做语义检索。"""
    # 【解释】真实业务里，"全库搜"往往不够：
    #         用户指定"只看第二章"、或者按权限/时间范围过滤，都要用到它。

    q = "电压低了怎么办"
    # 【解释】一个和"电压"强相关的问题。

    all_hits = search_chroma(collection, q, top_k=2)
    # 【解释】先不做过滤，全库搜 2 条。

    filtered = search_chroma(collection, q, top_k=2,
                             where={"section": "2. 电压管理"})
    # 【解释】加上过滤：只在第 2 章里搜。
    #         注意 where 用的是 extract_section 抽出来的精确章节名。

    print(f"提问：{q}")
    # 【解释】先把问题打出来，方便对照。
    print("\n【全库检索】")
    # 【解释】第一组结果。
    for h in all_hits:
        # 【解释】遍历全库检索的命中。
        print(f"   [{h['section']}] 相似度 {h['similarity']:.3f}")
        # 【解释】打印章节和相似度。

    print("\n【限定第 2 章检索】")
    # 【解释】第二组结果。
    if filtered:
        # 【解释】过滤后有结果。
        for h in filtered:
            # 【解释】遍历过滤后的命中。
            print(f"   [{h['section']}] 相似度 {h['similarity']:.3f}")
            # 【解释】打印章节和相似度，此时应该全是"2. 电压管理"。
    else:
        # 【解释】过滤后没结果。
        print("   （该章节下没有命中，换个章节名试试）")
        # 【解释】提示可能的原因。

    print("\n💡 生产价值：过滤让用户能指定范围，")
    # 【解释】总结这一功能的意义。
    print("   也能做权限隔离（例如 where={'dept': '研发部'}）。")
    # 【解释】给出企业场景的例子。


# ════════════════════════════════════════════════════════════════
# 第 9 部分：持久化验证 —— 关掉程序数据还在吗？
# ════════════════════════════════════════════════════════════════

def demo_persistence():
    """重新打开数据库，证明数据确实落盘了。"""
    # 【解释】这是对 Day15 痛点 3 的直接回应：内存算完就丢 vs 落盘复用。

    if not HAS_CHROMA:
        # 【解释】没装 Chroma 就跳过这一步。
        print("（未安装 Chroma，跳过持久化演示）")
        # 【解释】说明原因。
        return
        # 【解释】直接返回。

    client = chromadb.PersistentClient(path=DB_DIR)
    # 【解释】重新连一次数据库。注意：这次没有 add，只是打开。

    try:
        # 【解释】尝试读取之前建好的集合。
        col = client.get_collection("uav_manual")
        # 【解释】get_collection 和 get_or_create 的区别：
        #         这个不会新建，找不到就报错 —— 正好用来验证"数据还在不在"。
        print(f"✅ 重新打开数据库，里面仍有 {col.count()} 条向量")
        # 【解释】能读出来就说明持久化成功。
        print(f"   数据目录：{DB_DIR}")
        # 【解释】告诉用户数据实际存在哪。
        files = os.listdir(DB_DIR)
        # 【解释】列出目录里的文件，作为"真的落盘了"的直观证据。
        print(f"   落盘文件：{files[:5]}{' ...' if len(files) > 5 else ''}")
        # 【解释】只显示前 5 个，太多了会刷屏。
    except Exception as e:
        # 【解释】读不到集合，说明持久化有问题。
        print(f"⚠️  没能重新打开集合：{type(e).__name__}")
        # 【解释】打印异常类型，便于排查。


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
# 第 11 部分：main —— 把上面的零件串成一条完整流水线
# ════════════════════════════════════════════════════════════════

def main():
    """主流程：加载 → 嵌入 → 建库 → 检索 → 对比 → 过滤 → 持久化 → 问答"""

    print("=" * 62)
    # 【解释】程序开头画条横线。
    print("  Day 17：向量数据库 Chroma —— 让检索「懂意思」")
    # 【解释】打印标题。
    print("=" * 62)
    # 【解释】再画一条，形成标题框。

    # ── 步骤 1：准备数据 ──
    _sep("步骤 1 / 7：加载 Day16 切好的手册块")
    # 【解释】打印步骤标题。
    chunks = load_chunks()
    # 【解释】读 chunks_parsed.json，拿到 8 个块。
    for c in chunks[:3]:
        # 【解释】先展示前 3 块，让用户看到输入长什么样。
        first_line = c["text"].strip().split("\n")[0]
        # 【解释】取第一行作为块的"标题"显示。
        print(f"   块{c['id']}｜{first_line}")
        # 【解释】打印块编号和标题。
    print(f"   ... 共 {len(chunks)} 块")
    # 【解释】说明总数。

    # ── 步骤 2：选 embedding 方案 ──
    _sep("步骤 2 / 7：选择 Embedding 方案（自动 A→C 降级）")
    # 【解释】打印步骤标题。
    embed_fn, mode_name, is_semantic, is_chinese = build_embedding_function()
    # 【解释】拿到四个值：embedding 函数、模式名、是否真语义、是否支持中文。
    print(f"   当前模式：{mode_name}")
    # 【解释】显示最终选了哪个方案。

    if HAS_CHROMA:
        # 【解释】只有装了 Chroma 才能往下走。
        # ── 步骤 3：建库 ──
        _sep("步骤 3 / 7：写入 Chroma（建索引 + 落盘）")
        # 【解释】打印步骤标题。
        collection = build_collection(chunks, embed_fn, rebuild=True)
        # 【解释】建库并写入向量。rebuild=True 保证每次演示结果干净。

        # ── 步骤 4：检索 ──
        _sep("步骤 4 / 7：语义检索演示")
        # 【解释】打印步骤标题。
        q = "电池能飞多久"
        # 【解释】一个"不说手册原话"的提问。
        print(f"   提问：{q}")
        # 【解释】先把问题打出来。
        hits = search_chroma(collection, q, top_k=3)
        # 【解释】检索 Top-3。
        for h in hits:
            # 【解释】逐条展示命中结果。
            print(f"   [{h['section']}] 相似度 {h['similarity']:.3f}｜{h['text'][:38]}...")
            # 【解释】打印章节、相似度、正文前 38 字（太长会乱）。

        # ── 步骤 5：对比 ──
        _sep("步骤 5 / 7：Day15 字频检索 vs Chroma 语义检索")
        # 【解释】打印步骤标题。
        compare_two_methods(collection, chunks, is_chinese)
        # 【解释】跑对比，看看中文语义检索到底强在哪。
        #         传 is_chinese 而不是 is_semantic：英文模型虽是"语义模型"，
        #         但对中文无效，不能让它冒充中文语义能力去对比。

        # ── 步骤 6：过滤 ──
        _sep("步骤 6 / 7：metadata 过滤（只在指定章节里搜）")
        # 【解释】打印步骤标题。
        demo_metadata_filter(collection)
        # 【解释】演示按章节过滤。

        # ── 步骤 7：持久化 ──
        _sep("步骤 7 / 7：持久化验证（关掉程序数据还在吗）")
        # 【解释】打印步骤标题。
        demo_persistence()
        # 【解释】重新打开数据库，验证数据还在。

        # ── 加餐：完整 RAG 问答 ──
        _sep("加餐：Chroma 检索 + 大模型生成 = 完整 RAG")
        # 【解释】打印步骤标题。
        if HAS_KEY:
            # 【解释】有 Key 才跑生成环节。
            ans, used = rag_answer(collection, "冬天低温飞行，电池要注意什么？")
            # 【解释】问一个跨章节（温度 + 续航）的问题。
            if ans:
                # 【解释】拿到答案了。
                print("【模型回答】")
                # 【解释】标注下面是模型的回答。
                print(ans)
                # 【解释】打印回答内容。
                print("\n【本次引用了这些块】")
                # 【解释】展示引用来源，体现可追溯。
                for h in used:
                    # 【解释】遍历命中的块。
                    print(f"   - {h['section']}（相似度 {h['similarity']:.3f}）")
                    # 【解释】打印每条引用的章节和相似度。
        else:
            # 【解释】没配 Key。
            print("ℹ️  没检测到 API Key，跳过生成环节（检索部分已全部演示）")
            # 【解释】说明情况，不影响前面的学习。
            print("   配置方法：在 w3_ai/api_config.py 里选 PROVIDER 并设置对应环境变量")
            # 【解释】给出配置指引。

    else:
        # 【解释】没装 Chroma，只能做最小演示。
        _sep("降级演示（未安装 Chroma）")
        # 【解释】打印步骤标题。
        fn = LocalHashEmbedding()
        # 【解释】用本地哈希向量演示"文本 → 向量"这一步。
        a = fn(["电池能飞多久"])[0]
        # 【解释】算一条向量。
        b = fn(["理论续航约 25 分钟"])[0]
        # 【解释】再算一条。
        sim = sum(x * y for x, y in zip(a, b))
        # 【解释】因为都做了归一化，点积就等于余弦相似度。
        print(f"   本地哈希向量维度：{len(a)}")
        # 【解释】显示向量长度。
        print(f"   「电池能飞多久」vs「理论续航约 25 分钟」相似度：{sim:.3f}")
        # 【解释】显示相似度数值。
        print("   ⚠️ 哈希向量只认字面，不认意思 —— 这就是要用真 embedding 的原因")
        # 【解释】点明局限，引出安装 Chroma 的必要性。
        print("\n   安装命令：pip install chromadb")
        # 【解释】给出安装命令。

    # ── 收尾：学习路线图 ──
    print("\n" + "═" * 62)
    # 【解释】分隔线。
    print("  【读完之后】Day17 你学到了什么")
    # 【解释】小结标题。
    print("═" * 62)
    # 【解释】分隔线。
    print("""
    ✅ 已掌握
       · Embedding 是什么：文本 → 固定长度向量，语义近 = 向量近
       · Chroma 三板斧：PersistentClient / add / query
       · 优雅降级：真模型下载失败也能跑通全流程（A→C）
       · metadata 过滤：先筛再搜，企业刚需
       · 持久化：数据落盘，下次直接加载，不用重算

    ⚠️  还要留意的坑
       · 【最容易被忽略的一条】中文业务必须用中文 embedding 模型！
         Chroma 自带的 all-MiniLM-L6-v2 是英文语料训练的，
         中文检索用它等于白做 —— 本文件保留 B 档就是让你亲眼看到翻车。
       · 国内下载 HF 模型要设 HF_ENDPOINT=hf-mirror.com（第 1 部分已处理）
       · 换 embedding 模型后，旧向量全部失效，必须重建库
         （不同模型的向量空间不一样，混着用等于乱码）
       · Chroma 的 distance 含义取决于 hnsw:space 设置：
         本文件设了 cosine，所以 相似度 = 1 - distance

    ➡️  下一步（Day18-19）
       Day18：检索优化 —— 相似度阈值、重排序(rerank)、查询改写
       Day19：完整问答系统 —— 把这些拼成你的作品集项目
       再往后（第 5 周）：LangGraph 把 RAG 变成会规划的 Agent
    """)
    # 【解释】三引号字符串，一次性打印整段学习总结。


if __name__ == "__main__":
    # 【解释】这行的作用：只有"直接运行这个文件"时才执行 main()；
    #         如果别的文件 import 它，main() 不会自动跑（避免副作用）。
    main()
    # 【解释】调用主流程，程序正式开始。
