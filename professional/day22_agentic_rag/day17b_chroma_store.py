"""
Day 17-B：Chroma 向量数据库 —— 数字存哪儿，怎么找回来？

【这个文件回答一个问题】
    上一集把文字变成了向量。几万条向量怎么存？
    下次想找"和这句话最像的 3 条"，总不能一条条比过去吧？

【怎么读这个文件】按这个顺序看：
    1. 先看 main()                —— 建库 → 查询 → 过滤 → 重开验证，一条龙
    2. build_collection()         —— 写：Chroma 的 add
    3. search_chroma()            —— 读：Chroma 的 query
    4. demo_metadata_filter()     —— 先按标签筛，再在子集里搜（企业刚需）
    5. demo_persistence()         —— 关掉程序，数据还在吗？

【运行方式】
    D:/Python-envs/chroma-env/Scripts/python.exe day17b_chroma_store.py

【依赖】
    复用 day17a_embedding.py 里的 build_embedding_function（跨天复用）。
    所以要先有 day17a 这个文件，别单独拷走 day17b。
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
# 第 4 部分：复用 Day17-A 的成果
# ════════════════════════════════════════════════════════════════

sys.path.insert(0, HERE)
# 【解释】把本文件所在目录加进模块搜索路径，这样才能 import 同目录的 day17a。

from day17a_embedding import build_embedding_function
# 【解释】直接复用上一集写好的"三档降级"embedding 工厂，
#         这里一行都不用重写 —— 这就是把代码拆成小文件的好处。


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
# 第 7 部分：动手跑一遍完整流程
# ════════════════════════════════════════════════════════════════

def main():
    print("=" * 62)
    print("  Day 17-B：Chroma —— 向量存哪儿，怎么找回来")
    print("=" * 62)

    _sep("第 1 步 / 5：加载 Day16 切好的手册块")
    chunks = load_chunks()
    # 【解释】读 chunks_parsed.json，拿到 Day16 切好的 8 个块。
    for c in chunks[:3]:
        first_line = c["text"].strip().split("\n")[0]
        print(f"   块{c['id']}｜{first_line}")
    print(f"   ... 共 {len(chunks)} 块")

    _sep("第 2 步 / 5：选 embedding 方案（复用 Day17-A）")
    embed_fn, mode_name, is_semantic, is_chinese = build_embedding_function()
    print(f"   当前模式：{mode_name}")

    if not HAS_CHROMA:
        # 【解释】没装 Chroma 就没法往下演示，如实告知并给出安装命令。
        print("\n⚠️  没装 chromadb，无法演示建库与检索。")
        print("   安装命令：D:/Python-envs/chroma-env/Scripts/python.exe -m pip install chromadb")
        return
        # 【解释】return 直接结束 main，不做无谓的挣扎。

    _sep("第 3 步 / 5：写入 Chroma（建索引 + 落盘）")
    collection = build_collection(chunks, embed_fn, rebuild=True)
    # 【解释】rebuild=True 每次重建，教学演示时结果干净可复现。

    _sep("第 4 步 / 5：语义检索 —— 故意用「不说手册原话」的提问")
    q = "电池能飞多久"
    # 【解释】手册里写的是"续航"，用户问的是"能飞多久"，字面完全不重合。
    print(f"   提问：{q}")
    hits = search_chroma(collection, q, top_k=3)
    for h in hits:
        print(f"   [{h['section']}] 相似度 {h['similarity']:.3f}｜{h['text'][:38]}...")
    print("   【解释】如果命中了「续航」那一章，说明它真的懂意思，而不是在匹配关键词。")

    _sep("第 5 步 / 5：metadata 过滤 + 持久化验证")
    demo_metadata_filter(collection)
    # 【解释】只在指定章节里搜 —— 企业里最常见的需求（比如"只搜最新版手册"）。
    demo_persistence()
    # 【解释】重新打开数据库，证明数据真的落盘了，不是存在内存里。

    # ── 收尾：学习路线图 ──
    print("\n" + "═" * 62)
    print("  【读完之后】Day17-B 你学到了什么")
    print("═" * 62)
    print("""
    ✅ 已掌握
       · Chroma 三板斧：PersistentClient（建/开）→ add（写）→ query（读）
       · 相似度换算：建库时设了 hnsw:space=cosine，所以 相似度 = 1 - distance
       · metadata 过滤：先筛再搜，比全库搜更准也更快
       · 持久化：数据真的写到磁盘，下次直接加载，不用重算向量

    ⚠️  还要留意的坑
       · Chroma 的 id 必须是字符串，Day16 的块 id 是数字，要 str() 转一下
       · collection.add() 的三个列表长度必须一致，按下标一一对应
       · 换 embedding 模型后必须重建库，旧向量不能直接复用

    ➡️  下一步
       Day17-C：检索出来的东西怎么变成人话？（拼 Prompt + 大模型生成 = 完整 RAG）
    """)


if __name__ == "__main__":
    main()
