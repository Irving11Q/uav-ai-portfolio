"""
Day 17-A：Embedding —— 文字到底是怎么变成数字的？

【这个文件回答一个问题】
    一段中文，被塞进一个函数，吐出来一串几百个小数。
    这串数字是什么？为什么它能代表"意思"？

【怎么读这个文件】按这个顺序看，不要跳：
    1. 先看最下面的 main()          —— 跑起来，先看到现象
    2. 回到 build_embedding_function() —— 明白"三档降级"的思路
    3. 再读 LocalHashEmbedding      —— 最笨但一定能跑的方案（离线兜底）
    4. 最后读 SentenceTransformerEmbedding —— 真正的中文语义模型

【运行方式】
    D:/Python-envs/chroma-env/Scripts/python.exe day17a_embedding.py
    （用系统 Python 也能跑，只是会降级到本地哈希方案）

【为什么单独拆出来】
    原 Day17 有 1369 行，光 embedding 就占了 320 行，
    混在一起根本读不完。单独成文件后，你只要搞懂一个概念就能往下走。
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
# 第 3 部分：动手看现象 —— 先别管原理，看它到底输出了什么
# ════════════════════════════════════════════════════════════════

def cosine(a, b):
    """两个向量的余弦相似度。

    本文件里所有 embedding 方案输出的向量都做过归一化（长度=1），
    所以"点积"就直接等于"余弦相似度"，不用再除以模长。
    """
    return sum(x * y for x, y in zip(a, b))
    # 【解释】zip 把两个向量按下标配对，x*y 逐位相乘再求和。


def main():
    print("=" * 62)
    print("  Day 17-A：Embedding —— 文字是怎么变成数字的")
    print("=" * 62)

    _sep("第 1 步：看看当前能用哪档方案")
    embed_fn, mode_name, is_semantic, is_chinese = build_embedding_function()
    # 【解释】三档自动降级：中文 BGE → 英文 ONNX → 本地哈希，谁先通就用谁。
    print(f"   当前模式：{mode_name}")
    print(f"   是真语义模型吗：{is_semantic}    支持中文吗：{is_chinese}")
    # 【解释】这两个标记很重要：只有"是语义 + 支持中文"，下面的同义词实验才成立。

    _sep("第 2 步：一句话 → 一串数字")
    s = "电池能飞多久"
    v = embed_fn([s])[0]
    # 【解释】embed_fn 接收"列表"、返回"列表的列表"，所以取 [0] 拿到这一条的向量。
    print(f"   原文：{s}")
    print(f"   向量长度（维度）：{len(v)}")
    # 【解释】维度是固定的：不管句子多长，都压成同样长度的一串数字。
    print(f"   前 5 个数字：{[round(x, 4) for x in v[:5]]}")
    print("   【解释】看不懂这 5 个数字很正常 —— 它们不是给人看的，是给计算机算相似度用的。")

    _sep("第 3 步：意思越像 → 数字越像（这是 embedding 的核心）")
    sents = [
        "电池能飞多久",
        "理论续航约 25 分钟",
        "这块电池充一次电能用多长时间",
        "今天天气不错",
    ]
    # 【解释】前两句意思相关但字面几乎不重叠；第 3 句是第 1 句的同义改写；
    #         第 4 句是完全无关的干扰项。
    vecs = [embed_fn([x])[0] for x in sents]
    # 【解释】一次性把 4 句都转成向量（批量转换比一句一句转快得多）。

    print(f"   基准句：「{sents[0]}」")
    for s2, v2 in zip(sents[1:], vecs[1:]):
        sim = cosine(vecs[0], v2)
        # 【解释】算基准句和其余各句的相似度，范围 -1 ~ 1，越接近 1 越像。
        print(f"   相似度 {sim:+.3f}   ←  {s2}")
    print("   【解释】如果是中文语义模型，第 2、3 句应该明显高，第 4 句应该明显低。")

    _sep("第 4 步：本地哈希 vs 语义模型 —— 一眼看出差距在哪")
    hash_fn = LocalHashEmbedding()
    # 【解释】本地哈希方案：把词用 md5 映射成下标，只认字面、不认意思。
    pairs = [
        ("电池能飞多久", "续航时间有多长"),     # 同义改写：字面不同、意思一样
        ("电池能飞多久", "螺旋桨怎么安装"),     # 无关句
    ]
    for a, b in pairs:
        ha, hb = hash_fn([a])[0], hash_fn([b])[0]
        sa, sb = embed_fn([a])[0], embed_fn([b])[0]
        print(f"   「{a}」 vs 「{b}」")
        print(f"      本地哈希：{cosine(ha, hb):+.3f}      当前方案：{cosine(sa, sb):+.3f}")
    print("   【解释】哈希方案下，同义改写两句的相似度会低得离谱（它只数字面重合）；")
    print("           语义模型下则明显更高 —— 这就是「懂意思」和「数字面」的区别。")

    # ── 收尾：学习路线图 ──
    print("\n" + "═" * 62)
    print("  【读完之后】Day17-A 你学到了什么")
    print("═" * 62)
    print("""
    ✅ 已掌握
       · Embedding = 把任意长度的文本压成固定长度的一串小数
       · 语义相近 → 向量相近 → 余弦相似度接近 1
       · 三档降级：中文 BGE → 英文 ONNX → 本地哈希（没网也能跑通全流程）
       · 本地哈希只认字面，认不出同义词 —— 所以它只能兜底，不能当真

    ⚠️  还要留意的坑
       · 【最容易被忽略的一条】中文业务必须用中文 embedding 模型！
         Chroma 自带的 all-MiniLM-L6-v2 是英文语料训练的，
         中文检索用它等于白做。
       · 换 embedding 模型后，旧向量全部失效，必须重建库
         （不同模型的向量空间不一样，混着用等于乱码）
       · HF_ENDPOINT 必须在 import chromadb / sentence_transformers 之前设置，
         否则不生效（huggingface_hub 在 import 时就读了）

    ➡️  下一步
       Day17-B：这些数字存到哪儿去？怎么快速找回来？（Chroma 向量数据库）
    """)


if __name__ == "__main__":
    main()
