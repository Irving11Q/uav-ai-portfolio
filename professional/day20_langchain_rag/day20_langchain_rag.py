"""
Day 20：用 LangChain 重写 Day17 的 RAG —— 框架到底帮我们做了什么？

【这个文件回答一个问题】
    Day15~Day18 我手写了整条 RAG 链路：切块、向量化、检索、拼 Prompt、生成，
    加起来一千多行。现在用 LangChain 重写一遍，回答三个问题：
      · 哪些代码它可以一句话替我搞定？
      · 哪些控制力它悄悄拿走了？
      · 生产里到底该不该用框架？

【怎么读这个文件】按这个顺序看：
    1. 先看最下面的 main()          —— 跑起来，先看现象
    2. BgeEmbeddings                —— 怎么把"已有的模型"接进 LangChain（适配器模式）
    3. build_rag_chain()            —— LCEL 的管道符 | 到底在干什么
    4. compare_with_handwriting()   —— 手写版 vs 框架版的逐项对比（面试重点）

【运行方式】
    D:/Python-envs/chroma-env/Scripts/python.exe day20_langchain_rag.py

【依赖说明】
    复用 Day16 的产物：chunks_parsed.json、uav_battery_manual.md
    复用 w3_ai/api_config.py：没配 Key 也能跑，只是跳过最后的生成环节
    需要 langchain（已装在 D:/Python-envs/chroma-env）

【为什么学这个】
    80% 的 AI 应用岗 JD 都点名 LangChain / LangGraph。
    但只会调包是不够的 —— 你手写过一遍，才知道框架在替你做什么、
    出问题时该往哪一层查。这也是我前四天坚持手写的原因。
"""

# ════════════════════════════════════════════════════════════════
# 第 1 部分：拿工具 + 处理"环境不齐"
# ════════════════════════════════════════════════════════════════

import os
import sys
import json

import requests
# 【解释】调大模型用的，和 Day10 一样。

HERE = os.path.dirname(os.path.abspath(__file__))
# 【解释】本文件所在目录（w4_rag）。

sys.path.insert(0, os.path.join(os.path.dirname(HERE), "w3_ai"))
# 【解释】把 w3_ai 目录加进模块搜索路径，这样才能 import 到 api_config。
#         dirname(HERE) 是 src 目录，再拼 w3_ai 就对了。

try:
    import api_config
    # 【解释】复用第 3 周写好的配置：BASE_URL / MODEL_NAME / API_KEY。
    API_KEY = api_config.API_KEY
    BASE_URL = api_config.BASE_URL
    MODEL_NAME = api_config.MODEL_NAME
    HAS_KEY = bool(API_KEY)
    # 【解释】Key 非空才算配置好了。
except Exception:
    # 【解释】没配 Key 时会 raise，这里接住，让离线部分照常跑。
    HAS_KEY = False
    BASE_URL = MODEL_NAME = ""

try:
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    HAS_LANGCHAIN = True
    # 【解释】这 6 个是 LangChain 最常用的核心接口，记住它们就够用一大半场景。
except ImportError as e:
    HAS_LANGCHAIN = False
    # 【解释】没装 LangChain，下面会给出安装命令并退出。
    print(f"⚠️  没装 langchain：{e}")

try:
    from langchain_text_splitters import MarkdownHeaderTextSplitter
    HAS_SPLITTER = True
    # 【解释】切分器是独立包，单独检测。
except ImportError:
    HAS_SPLITTER = False


def find_local_model(model_name="BAAI/bge-small-zh-v1.5"):
    """
    在 HF 缓存里找模型的本地路径。找到返回路径，没找到返回 None。

    【为什么需要这个函数 —— 一个真实的踩坑记录】
        我一开始只设了 HF_HUB_OFFLINE=1 就以为万事大吉，
        结果它照样去请求 huggingface.co，连不上就重试 5 次、
        每次等 20 秒超时 —— 光加载模型就白白耗掉两分钟，
        日志里那句 `Retrying in 1s [Retry 1/5]` 就是它。

        最稳的办法是：**直接把本地路径喂给 SentenceTransformer**，
        它压根不会发起任何网络请求，秒开。

        （对比 Day17：那是第一次用模型、必须下载，所以写的是"探测哪个源能通"。
          今天模型已经在本地了，直接走本地路径才对。）
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        cache = HF_HUB_CACHE
        # 【解释】直接问 huggingface_hub 自己"缓存放在哪"，比手拼路径可靠得多。
        #         手拼默认路径（~/.cache/huggingface/hub）在两种情况下会失效：
        #           ① 用户设了 HF_HOME / HF_HUB_CACHE 环境变量，把缓存挪走了
        #              （比如 C 盘空间紧张时挪到 D 盘）
        #           ② 不同操作系统默认位置不一样
        #         我第一次就是手拼路径，结果检测失败、白等了十几秒联网 —— 别踩这个坑。
    except ImportError:
        # 【解释】极端情况下没装 huggingface_hub，才退回手拼默认路径。
        cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")

    folder = "models--" + model_name.replace("/", "--")
    # 【解释】缓存目录的命名规则：把 "BAAI/bge-small-zh-v1.5"
    #         变成 "models--BAAI--bge-small-zh-v1.5"（斜杠换成双横线）。

    snap = os.path.join(cache, folder, "snapshots")
    # 【解释】真正的模型文件在 snapshots/<一串哈希>/ 下面，不在 folder 根目录。

    if not os.path.isdir(snap):
        return None
        # 【解释】没有 snapshots 目录 = 这个模型从没下载成功过。

    subs = [d for d in os.listdir(snap) if os.path.isdir(os.path.join(snap, d))]
    # 【解释】snapshots 下每个子目录是一个版本的哈希名。
    if not subs:
        return None

    return os.path.join(snap, sorted(subs)[-1])
    # 【解释】有多个版本时取排序后的最后一个（通常就是最新下载的那个）。


LOCAL_MODEL = find_local_model()
# 【解释】模块加载时就把本地路径找好，后面直接用。

if LOCAL_MODEL:
    os.environ["HF_HUB_OFFLINE"] = "1"
    # 【解释】双保险：既然本地有，就明确告诉 HF 别联网。
    print(f"   模型已在本地缓存 → 直接用本地路径加载（零网络请求，秒开）")
    print(f"   路径：{LOCAL_MODEL}")
else:
    # ── 本地没有，需要下载：这时才去探测哪个源能通 ──
    for url in ["https://huggingface.co", "https://hf-mirror.com"]:
        # 【解释】顺序很重要：先官方后镜像（镜像也会挂，别一上来就写死它）。
        try:
            if requests.get(url, timeout=6).status_code < 500:
                # 【解释】< 500 说明服务器活着（5xx 是服务端错误，等于没通）。
                os.environ["HF_ENDPOINT"] = url
                print(f"   本地无缓存 → 模型下载源：{url}")
                break
        except Exception:
            # 【解释】连不上/超时/DNS 失败，试下一个。
            continue
    else:
        print("   ⚠️ 两个源都不通，首次运行可能无法下载模型")

# 【解释】⚠️ 环境变量必须在 import sentence_transformers 之前设好！
#         huggingface_hub 在 import 时就把它们读进去了，之后再改无效
#         （这就是"设了镜像却没生效"的真正原因，Day17 踩过一次）。
#         本文件把 sentence_transformers 的 import 放在 BgeEmbeddings 里（延迟导入），
#         所以在这里设置来得及。


def _sep(title):
    """打印分节标题。"""
    print("\n" + "═" * 62)
    print("  " + title)
    print("═" * 62)


# ════════════════════════════════════════════════════════════════
# 第 2 部分：把"已有的模型"接进 LangChain —— 适配器模式
# ════════════════════════════════════════════════════════════════

class BgeEmbeddings(Embeddings):
    """
    把 Day17 用的中文 BGE 模型包装成 LangChain 的 Embeddings 接口。

    【为什么要写这个类】
        LangChain 不关心你用什么模型，它只要求你实现两个方法：
            embed_documents(texts) -> 批量转向量（建库时调用）
            embed_query(text)      -> 单条转向量（查询时调用）
        这种"定义一个中间层，让不同的东西都能接上来"的写法叫适配器模式。

    【实际价值】
        有了这一层，换模型 = 改一行。比如换成 OpenAI 的 embedding，
        只要把 self.model 换掉，上层链路一个字都不用动。
    """

    def __init__(self, model_name="BAAI/bge-small-zh-v1.5"):
        # 【解释】默认用 Day17 验证过的中文小模型（512 维，95MB）。
        from sentence_transformers import SentenceTransformer
        # 【解释】延迟导入：用到时才 import，这样没装也不影响文件被 import。
        target = LOCAL_MODEL or model_name
        # 【解释】有本地缓存就用路径（零网络请求），没有才用模型名去下载。
        self.model = SentenceTransformer(target)
        # 【解释】传路径和传模型名效果一样，但传路径不会联网。

    def embed_documents(self, texts):
        """批量转向量。LangChain 建库时会调用它。"""
        return self.model.encode(list(texts), normalize_embeddings=True).tolist()
        # 【解释】normalize_embeddings=True 做归一化，
        #         之后点积就直接等于余弦相似度，不用再除模长。

    def embed_query(self, text):
        """单条转向量。LangChain 检索时会调用它。"""
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()
        # 【解释】取 [0] 是因为 encode 返回的是二维列表。


# ════════════════════════════════════════════════════════════════
# 第 3 部分：调大模型 —— 用 RunnableLambda 把老代码接进新链路
# ════════════════════════════════════════════════════════════════

def ask_model(messages, temperature=0.3):
    """调大模型（和 Day17 完全一致，纯复用）。"""
    if not HAS_KEY:
        # 【解释】没 Key 直接返回 None，调用方据此跳过。
        return None

    url = BASE_URL.rstrip("/") + "/chat/completions"
    # 【解释】拼接接口地址，rstrip 防止出现双斜杠。

    try:
        r = requests.post(
            url,
            headers={"Authorization": "Bearer " + API_KEY,
                     "Content-Type": "application/json"},
            json={"model": MODEL_NAME, "messages": messages,
                  "temperature": temperature},
            timeout=60,
        )
        # 【解释】发 POST 请求。GLM / DeepSeek 都是 OpenAI 兼容格式，所以这样就能调。
        r.raise_for_status()
        # 【解释】非 2xx 会抛异常。
        data = r.json()
        if "choices" not in data:
            # 【解释】接口返回了错误结构，提前拦住（Day12 踩过的坑）。
            print("   ⚠️ 接口返回异常：", str(data)[:150])
            return None
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"   ⚠️ 调用失败：{type(e).__name__}: {str(e)[:120]}")
        return None


def call_llm(prompt_value):
    """
    把 LangChain 的 PromptValue 转成我们自己的 messages，再调模型。

    【这个函数是"胶水"】
        LangChain 的 Prompt 输出是一个 PromptValue 对象，
        而我们的 ask_model 要的是 [{"role": ..., "content": ...}]。
        有了它，你完全可以不用 LangChain 官方的 ChatModel，
        照样把自研/小众模型接进 LCEL 链路里。

    【踩坑记录：角色名对不上，接口直接 400】
        LangChain 的消息类型名 ≠ OpenAI 兼容接口的 role 名：
            LangChain 的 HumanMessage.type 是 "human"
            但接口要的是                    "user"
        我第一次直接把 m.type 传过去，智谱返回 400 Bad Request。
        所以这里必须做一层角色名映射 —— 这正是"框架的抽象"与
        "接口的实际要求"之间的缝隙，框架不会替你补，得自己来。
    """
    role_map = {"system": "system", "human": "user", "ai": "assistant"}
    # 【解释】LangChain 用 human / ai，OpenAI 兼容接口用 user / assistant。
    #         用 .get(...) 兜底：遇到没见过的类型就原样传，不至于直接崩。

    messages = [{"role": role_map.get(m.type, m.type), "content": m.content}
                for m in prompt_value.to_messages()]
    # 【解释】to_messages() 把 PromptValue 拆成 System + User 两条消息。
    return ask_model(messages) or ""
    # 【解释】没 Key 或调用失败时返回空字符串兜底，
    #         避免下游的 StrOutputParser 拿到 None 报错。


# ════════════════════════════════════════════════════════════════
# 第 4 部分：加载 Day16 的成品数据
# ════════════════════════════════════════════════════════════════

def load_documents():
    """把 Day16 切好的块，转成 LangChain 的 Document 对象。"""
    path = os.path.join(HERE, "chunks_parsed.json")
    # 【解释】复用 Day16 的产物，不重复劳动 —— 这就是"跨天复用"。

    with open(path, encoding="utf-8") as f:
        chunks = json.load(f)
    # 【解释】读 JSON。utf-8 必须显式指定，Windows 下默认是 gbk，中文会炸。

    docs = []
    for c in chunks:
        # 【解释】逐块转换成 Document。
        text = c["text"]
        lines = text.strip().split("\n")
        # 【解释】第一行是标题（Day16 按标题切的，所以每块开头都是 "## x. xxx"）。
        docs.append(Document(
            page_content=text,
            # 【解释】page_content = 正文，检索时匹配的就是它。
            metadata={"section": lines[0].lstrip("# ").strip(),
                      "id": str(c["id"])},
            # 【解释】metadata = 标签。LangChain 的过滤、引用标注都靠它。
        ))
    return docs


# ════════════════════════════════════════════════════════════════
# 第 5 部分：搭链路 —— LCEL 的管道符到底在干什么
# ════════════════════════════════════════════════════════════════

def format_docs(docs):
    """把检索到的 Document 列表，拼成一段"参考资料"文本。"""
    return "\n\n".join(
        f"[{i+1}] 来源：{d.metadata.get('section', '?')}\n{d.page_content}"
        for i, d in enumerate(docs)
    )
    # 【解释】这个步骤在手写版里也有，只是散落在 rag_answer 函数里。
    #         LangChain 要求你把它抽成独立函数 —— 其实这样更好测。


def build_rag_chain(embeddings, docs):
    """
    搭建完整 RAG 链路。核心就是最后那段管道。

    LCEL 的 | 是什么意思？
        它把每一步串成一条流水线，上一步的输出 = 下一步的输入。
        等价于手写版的"先检索 → 再拼 Prompt → 再调模型 → 再取文本"，
        只是不用你自己一步步传参了。
    """

    _sep("建库：把 Document 塞进向量库")
    store = InMemoryVectorStore(embedding=embeddings)
    # 【解释】内存向量库：程序一关数据就没了。
    #         对比 Day17 的 Chroma（PersistentClient）会落盘 —— 这是两者的关键差别，
    #         LangChain 里换成 langchain_chroma 即可得到持久化，接口几乎一样。
    store.add_documents(docs)
    # 【解释】这一步内部会自动调用 embeddings.embed_documents() 批量转向量。
    print(f"✅ 已写入 {len(docs)} 条向量")

    retriever = store.as_retriever(search_kwargs={"k": 3})
    # 【解释】retriever = "检索器"，是 LangChain 的核心抽象之一。
    #         它的职责只有一件事：给我一个问题，返回最相关的 k 条 Document。
    #         Day17 里这件事是 search_chroma() 干的，大约 50 行；这里 1 行。
    #         更重要的是：换成 Chroma / FAISS / Milvus，这行代码都不用改。

    prompt = ChatPromptTemplate.from_messages([
        # 【解释】Prompt 模板。用 {context} / {question} 占位，运行时自动填。
        ("system",
         "你是无人机电池与能耗方面的技术助手。\n"
         "请严格根据【参考资料】回答用户问题。\n"
         "如果资料里没有相关信息，直接说「资料中没有提到」，不要编造。\n"
         "回答时请注明你引用了哪一条资料。"),
        # 【解释】这四句和 Day17 手写的完全一致 —— 框架不会替你把 Prompt 写对。
        ("user", "【参考资料】\n{context}\n\n【用户问题】\n{question}"),
    ])

    rag_chain = (
        {
            "context": retriever | format_docs,
            # 【解释】问题先喂给 retriever 拿回 3 条，再交给 format_docs 拼成文本。
            "question": RunnablePassthrough(),
            # 【解释】RunnablePassthrough = 原样透传，问题本身也要传给 Prompt。
        }
        | prompt
        # 【解释】把两个变量填进模板，得到 System + User 两条消息。
        | RunnableLambda(call_llm)
        # 【解释】调我们的模型。用 RunnableLambda 包装普通函数，就能接进管道。
        | StrOutputParser()
        # 【解释】把返回值规整成字符串（这里已经是字符串了，但保留它是好习惯）。
    )
    return rag_chain, retriever


# ════════════════════════════════════════════════════════════════
# 第 6 部分：切块对比 —— Day16 手写 vs LangChain 切分器
# ════════════════════════════════════════════════════════════════

def compare_splitter():
    """对比 Day16 手写的按标题切块，和 LangChain 的 MarkdownHeaderTextSplitter。"""
    if not HAS_SPLITTER:
        # 【解释】没装切分器包就跳过，不影响主线。
        print("   （未安装 langchain-text-splitters，跳过本项）")
        return

    md_path = os.path.join(HERE, "uav_battery_manual.md")
    with open(md_path, encoding="utf-8") as f:
        text = f.read()
    # 【解释】读 Day16 用的原始手册（markdown 格式，用 ## 分章节）。

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "section")])
    # 【解释】告诉它：遇到 "##" 开头的行就切一刀，并把标题内容存进 metadata 的 section 字段。
    #         对比 Day16：我手写了一个状态机（约 60 行）做同样的事，这里只要 2 行。

    docs = splitter.split_text(text)
    # 【解释】切完直接就是 Document 列表，metadata 里已经带好了 section。

    n_head = sum(1 for d in docs if not d.metadata.get("section"))
    # 【解释】数一下有多少块没有 section —— 那就是第一个 ## 之前的"文档标题区"。

    print(f"   LangChain 切分器：{len(docs)} 块（{n_head} 块标题区 + {len(docs) - n_head} 块章节）")
    print(f"   Day16 手写切块  ：8 块（1 块标题区 + 7 块章节，手册共 7 个 ## 章节）")
    # 【解释】手册里有 7 个 ## 章节，加上开头的文档标题区，正好 8 块。

    if len(docs) == 8:
        print("   ✅ 两者完全一致 —— Day16 手写那个状态机是对的，不是碰巧蒙对的")
    else:
        print(f"   ⚠️ 块数不一致（{len(docs)} vs 8），值得查查边界处理差在哪")
    # 【解释】这种"用框架的独立实现反验自己手写代码"的做法很值钱：
    #         手写时的盲点，往往要靠另一个独立实现才能暴露出来。

    for d in docs[:3]:
        print(f"      section={d.metadata.get('section')!r}  {len(d.page_content)} 字")
    print("   【解释】结论：这种通用需求，框架确实比手写划算；")
    print("           但正因为手写过一遍，我才知道它内部是「按行判定标题」来避免误匹配的。")


# ════════════════════════════════════════════════════════════════
# 第 7 部分：手写版 vs 框架版 —— 逐项对比（面试重点）
# ════════════════════════════════════════════════════════════════

def compare_with_handwriting():
    """把两版实现的差异讲清楚 —— 这段是面试时最值钱的部分。"""
    _sep("手写版（Day15-18）vs 框架版（Day20）")

    rows = [
        ("切块",     "自写状态机 ~60 行",        "MarkdownHeaderTextSplitter 2 行"),
        ("向量化",   "三档降级 ~100 行",          "继承 Embeddings 实现 2 个方法"),
        ("检索",     "search_chroma() ~50 行",    "as_retriever() 1 行"),
        ("拼 Prompt","字符串拼接散在函数里",       "ChatPromptTemplate 集中管理"),
        ("生成",     "自己 requests.post",        "RunnableLambda 包装进管道"),
        ("持久化",   "Chroma 落盘，重启还在",      "InMemoryVectorStore 关掉就丢"),
        ("换向量库", "要改调用处",                 "改一行（Chroma/FAISS/Milvus 接口一致）"),
        ("调试",     "每步都能 print，所见即所得", "链路是黑盒，要靠回调/日志"),
    ]
    # 【解释】这个表不是要分出高下，而是要看清"框架到底替你做了什么"。

    print(f"   {'环节':<8} {'手写版':<26} {'LangChain 版'}")
    print("   " + "-" * 66)
    for a, b, c in rows:
        print(f"   {a:<8} {b:<26} {c}")

    print("\n   【我的判断】")
    print("   · 框架真省力的地方：切分器、retriever 抽象、可插拔（换库换模型只改一行）")
    print("   · 框架没帮上忙的：Prompt 怎么写对、阈值怎么定、评测怎么建 —— 这些还是人脑的活")
    print("   · 框架的代价：链路变黑盒，出 bug 要顺着回调往上翻；版本迭代快（1.x 与 0.x 差异大）")
    print("   · 所以我的路线是：先手写搞懂每一层，再用框架提效 —— 不是二选一")
    # 【解释】这段结论很适合在面试里说，因为它体现的是工程判断力，而不是"我会用某个包"。


# ════════════════════════════════════════════════════════════════
# 第 8 部分：main
# ════════════════════════════════════════════════════════════════

def main():
    print("=" * 62)
    print("  Day 20：用 LangChain 重写 Day17 的 RAG")
    print("=" * 62)

    if not HAS_LANGCHAIN:
        # 【解释】没装 LangChain 就没法演示，给安装命令后退出。
        print("\n⚠️  没装 langchain，无法继续。安装命令：")
        print("   D:/Python-envs/chroma-env/Scripts/python.exe -m pip install langchain langchain-text-splitters")
        return

    _sep("第 1 步 / 4：加载 Day16 切好的块")
    docs = load_documents()
    print(f"   共 {len(docs)} 块")
    for d in docs[:3]:
        print(f"   [{d.metadata['section']}] {d.page_content[:32]}...")

    _sep("第 2 步 / 4：接入中文 embedding（适配器模式）")
    try:
        embeddings = BgeEmbeddings()
        dim = len(embeddings.embed_query("测试"))
        print(f"✅ 中文语义模型已接入，向量维度 {dim}")
        # 【解释】打印维度，确认模型真的加载成功（而不是静默降级）。
    except Exception as e:
        # 【解释】模型加载失败（没装 sentence-transformers 或文件损坏）。
        print(f"⚠️  embedding 加载失败（{type(e).__name__}），无法继续")
        print(f"   原因：{str(e)[:150]}")
        return

    _sep("第 3 步 / 4：搭链路 + 提问")
    rag_chain, retriever = build_rag_chain(embeddings, docs)

    q = "电池能飞多久"
    # 【解释】沿用 Day17 的问题，方便对比两次的输出是否一致。
    print(f"\n   提问：{q}")

    hits = retriever.invoke(q)
    # 【解释】单独调一次 retriever，看看检索到了什么 —— 这就是"可观测"的价值。
    print("   检索命中：")
    for h in hits:
        first = h.page_content.strip().split("\n")[0][:30]
        # 【解释】只取首行：正文里带换行，直接截断会把排版打乱。
        print(f"      [{h.metadata['section']}] {first}...")
    print("   【解释】手册里写的是「续航」，用户问「能飞多久」—— 命中说明它真懂意思。")

    sections = [h.metadata.get("section", "") for h in hits]
    if "5. 续航估算" not in sections:
        # 【解释】下面这个观察很值钱，别跳过。
        print("   ⚠️ 但注意：最该命中的「5. 续航估算」并没进 Top-3。")
        print("      框架把链路搭好了，不代表检索就准 —— 这正是 Day18 要做检索调优的原因。")
        print("      很多人以为「上了 LangChain = RAG 就好了」，其实召回质量是另一回事，")
        print("      得靠评测集 + 调优化手段解决，框架帮不上忙。")

    if HAS_KEY:
        # 【解释】有 Key 才跑生成环节。
        print("\n   生成答案：")
        ans = rag_chain.invoke(q)
        # 【解释】一行调用跑完整条链路：检索 → 拼 Prompt → 生成 → 取文本。
        #         对比 Day17 的 rag_answer()，逻辑一样，但这里不用自己一步步传参。
        print(ans)
        print("\n   ⚠️ 对照检查：资料里其实没有续航计算公式。")
        print("      如果模型算出了一个具体数字，那就是幻觉 —— 说明")
        print("      「只许用资料」这句 Prompt 还得写得更硬")
        print("      （例如：要求逐句标注出处、明确禁止自行推算）。")
    else:
        print("\n   ℹ️  没检测到 API Key，跳过生成环节（检索链路已完整演示）")

    _sep("第 4 步 / 4：切块对比 + 与手写版逐项对比")
    compare_splitter()
    print()
    compare_with_handwriting()

    # ── 收尾：学习路线图 ──
    print("\n" + "═" * 62)
    print("  【读完之后】Day20 你学到了什么")
    print("═" * 62)
    print("""
    ✅ 已掌握
       · LCEL 管道符 |：上一步的输出 = 下一步的输入
       · 适配器模式：继承 Embeddings 实现 2 个方法，任何模型都能接进来
       · RunnableLambda：把普通函数接进链路（不依赖官方 ChatModel 也能用）
       · retriever 抽象：换向量库只改一行，上层链路不动
       · ChatPromptTemplate：Prompt 集中管理，不再散落在各个函数里

    ⚠️  还要留意的坑
       · InMemoryVectorStore 不落盘！生产要换 langchain_chroma / FAISS
       · 框架不会替你把 Prompt 写对 —— "只许用资料 / 允许说不知道"仍然要你自己写
       · LangChain 1.x 与 0.x 的 API 差异很大，搜到的老教程可能跑不通
       · 链路是黑盒：调试时先单独 invoke(retriever) 看检索结果，再查生成

    ➡️  下一步（第 5 周）
       Day21：LangGraph —— 把 RAG 从"一问一答"变成会规划、能调工具的 Agent
       面试前务必补：LangChain 重构 + 最小 LangGraph Agent，80% 的 JD 都点名
    """)


if __name__ == "__main__":
    # 【解释】只有直接运行本文件时才执行 main()，被 import 时不会自动跑。
    main()
