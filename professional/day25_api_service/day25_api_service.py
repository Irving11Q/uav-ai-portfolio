#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════
  Day 25：把 Agent 变成 HTTP 服务（FastAPI 服务化）
═══════════════════════════════════════════════════════════════

【程序做什么】
  前 5 周写的都是「脚本」：你双击、它在黑框里跑完就结束。
  但真实项目里，AI 能力是**被别人调用**的：
      · 上位机（PySide6）想问一句「这架飞机还能飞多久」
      · 微信小程序想把用户问题转给知识库
      · 前端页面想拿到一条分析结论
  它们都不会去 import 你的 .py —— 它们只会发 HTTP 请求。
  今天就是把 Day23 的数据分析能力和 Day24 的手册检索，包成 5 个 HTTP 接口：
      GET  /health          服务活着吗？模型配了没？索引建了吗？
      POST /manual/search   检索《电池手册》（不调大模型，零成本，永远可用）
      POST /data/filter     按条件过滤飞行数据（pandas 算，不是模型数）
      POST /data/summary    某一列的统计摘要
      POST /ask             完整 Agent（自然语言 → 调工具 → 出结论，需要 Key）

【怎么读这个文件（7 块）】
  1. 环境 + 依赖探测      没 Key / 没 fastapi 都不许崩，要优雅降级
  2. 资源层                ★ 重资源（bge 模型）只加载一次，全局共用
  3. Pydantic 契约         请求/响应的字段与校验（FastAPI 的灵魂）
  4. app + lifespan        ★ 启动时预热，关闭时收尾
  5. 五个端点              每个都标注了「为什么是 def 不是 async def」
  6. 全局异常处理          任何报错都返回 JSON，不许给调用方吐 HTML
  7. self_test()           ★ 不用起服务、不占端口，用 TestClient 自测全部端点

【运行方式】
  A. 自测（推荐先看这个，零成本、不占端口）：
       cd w6_service
       D:/Python-envs/chroma-env/Scripts/python.exe day25_api_service.py
  B. 真正起服务（起完浏览器打开 http://127.0.0.1:8000/docs 看自动文档）：
       D:/Python-envs/chroma-env/Scripts/python.exe -m uvicorn day25_api_service:app --reload
  C. 跳过 bge 预热（启动秒开，/manual/search 会降级成关键词匹配）：
       set DAY25_PREHEAT=0

【为什么学这个（面试能讲）】
  AI 应用岗的 JD 里，「FastAPI / 把模型封装成服务」出现频率极高，
  因为它隔着一道分水岭：
      会调 API 的人  = 能跑通 demo
      会封装服务的人 = 能交付给别人用
  今天要记住的四句话：
    ① 重资源（模型/索引）必须在**启动时**加载一次，不能每请求加载一次
    ② 同步的活（pandas / 模型推理）用 def 端点，FastAPI 会自动丢线程池，
       不会堵死事件循环；乱加 async def 反而会堵
    ③ 输入必须用 Pydantic 校验，非法参数由框架挡在门外（422），别进业务代码
    ④ 依赖没配好（没 Key）要返回 503 说清楚，不能让服务起不来

【和前几天的关系】
  Day23 数据分析 Agent → 今天 /data/* 与 /ask
  Day24 手册检索       → 今天 /manual/search
  图的骨架没变，变的是「谁来调用」：从 main() 变成 HTTP 请求。
"""

# ════════════════════════════════════════════════════════════════
# 第 1 部分：环境 + 依赖探测
# ════════════════════════════════════════════════════════════════

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_PARENT, "w5_agent"))   # Day23/24 的工具在这
sys.path.insert(0, os.path.join(_PARENT, "w3_ai"))      # api_config 在这

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.concurrency import run_in_threadpool
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except Exception as e:
    HAS_FASTAPI = False
    FASTAPI_ERR = e

try:
    import api_config
    API_KEY = api_config.API_KEY
    BASE_URL = api_config.BASE_URL
    MODEL = api_config.MODEL_NAME
    HAS_KEY = bool(API_KEY)
except Exception:
    API_KEY, BASE_URL, MODEL = "", "", "glm-4-flash"
    HAS_KEY = False

# 【解释】Day23 的数据工具（pandas 算的，不是模型猜的）直接拿来用。
# 注意：是 import 模块而不是 copy 代码 —— 服务层和能力层分开，改一处两处生效。
try:
    from day23a_data_agent import (load_df, filter_rows, describe_column,
                                   TOOLS, SYSTEM_PROMPT, build_data_agent)
    HAS_DATA = True
except Exception as e:
    HAS_DATA = False
    DATA_ERR = e

try:
    from day21b_agent_tools import SimpleChatModel
    HAS_AGENT = True
except Exception:
    HAS_AGENT = False


def find_local_model(model_name="BAAI/bge-small-zh-v1.5"):
    """在 HF 缓存里找模型的本地路径（沿用 Day20 最稳的写法）。

    【坑】只设 HF_HUB_OFFLINE=1 不够，它照样去请求 huggingface.co 重试 5 次。
    把本地快照路径直接喂给 SentenceTransformer，才能零网络秒开。
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        cache = HF_HUB_CACHE
    except ImportError:
        cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    snap = os.path.join(cache, "models--" + model_name.replace("/", "--"), "snapshots")
    if not os.path.isdir(snap):
        return None
    subs = [d for d in os.listdir(snap) if os.path.isdir(os.path.join(snap, d))]
    return os.path.join(snap, sorted(subs)[-1]) if subs else None


LOCAL_MODEL = find_local_model()

_MANUAL_CANDIDATES = [
    os.path.join(_HERE, "uav_battery_manual.md"),                      # 发布目录自带
    os.path.join(_PARENT, "w4_rag", "uav_battery_manual.md"),           # 本地学习仓库
    os.path.join(_PARENT, "w5_agent", "uav_battery_manual.md"),         # 兜底
]
MANUAL_PATH = next((p for p in _MANUAL_CANDIDATES if os.path.exists(p)), _MANUAL_CANDIDATES[0])


# ════════════════════════════════════════════════════════════════
# 第 2 部分：★ 资源层 —— 重资源只加载一次
# ════════════════════════════════════════════════════════════════
#
# 【解释】这是服务化和写脚本最大的区别。
#   脚本：main() 里加载一次 → 跑完退出，资源随进程销毁，没人觉得有问题。
#   服务：进程长时间活着，每个请求都会走一遍端点函数。
#        如果把 SentenceTransformer(...) 写在端点里面，
#        用户每问一句就重载一次模型（这里约 60 秒）→ 服务直接不可用。
#   正解：进程级单例 + 启动时预热（见第 4 部分 lifespan）。

class ManualIndex:
    """《电池手册》的内存向量索引（按 ## 切块 + 余弦相似度）。

    为什么不用 Chroma：要落盘建库，拷出去容易空；这里只有几段文本，
    内存里算余弦就够了，目录自包含、零配置。
    """

    def __init__(self):
        self.chunks = []      # [(标题, 正文), ...]
        self.vectors = []     # 每条正文一个向量
        self.ready = False
        self.backend = "none"  # bge / keyword / none

    def build(self):
        if self.ready:
            return
        if not os.path.exists(MANUAL_PATH):
            self.ready = True
            return
        text = open(MANUAL_PATH, encoding="utf-8").read()
        for i, part in enumerate(text.split("\n## ")):
            if i == 0:
                title, body = "总览", part
            else:
                title, body = (part.split("\n", 1) + [""])[:2]
            if body.strip():
                self.chunks.append((title.strip(), body.strip()))

        if LOCAL_MODEL:
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(LOCAL_MODEL)
                self.vectors = model.encode([c[1] for c in self.chunks],
                                            normalize_embeddings=True)
                self._model = model
                self.backend = "bge"
            except Exception:
                self.vectors = []
                self.backend = "keyword"
        else:
            self.backend = "keyword"
        self.ready = True

    def search(self, query, k=3):
        """返回 [(标题, 正文, 分数), ...]。没 bge 时退化为关键词命中数。"""
        self.build()
        if len(self.vectors) > 0:
            qv = self._model.encode([query], normalize_embeddings=True)[0]
            scores = [float(qv @ v) for v in self.vectors]
            idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            return [(self.chunks[i][0], self.chunks[i][1], scores[i]) for i in idx]

        ranked = []
        for title, body in self.chunks:
            hit = sum(body.count(ch) for ch in query if ch.strip())
            ranked.append((title, body, float(hit)))
        ranked.sort(key=lambda x: x[2], reverse=True)
        return ranked[:k]


# 【解释】进程级单例：整个服务共用这一个索引对象和一个 Agent 图。
_RESOURCES = {"index": None, "agent_graph": None, "boot_at": time.time()}


def get_index() -> ManualIndex:
    if _RESOURCES["index"] is None:
        _RESOURCES["index"] = ManualIndex()
    return _RESOURCES["index"]


def get_agent_graph():
    """懒加载 Agent 图：第一个 /ask 请求来时才建（建图要连模型，可能失败）。"""
    if _RESOURCES["agent_graph"] is not None:
        return _RESOURCES["agent_graph"]
    if not (HAS_KEY and HAS_AGENT and HAS_DATA):
        return None
    # 【坑】★ 必须先 bind_tools 再建图 —— day21b.build_agent 的约定是
    #        「传进来的 model 已经是绑好工具的」，它内部不会再帮你 bind。
    #        我第一次写漏了 .bind_tools(TOOLS)，结果模型不知道自己有工具，
    #        直接编了个「0 条」出来（真值 4 条）—— 恰好是 Day23 那条结论的反面：
    #        「不把工具给模型，它就会一本正经地算错」。
    #        另外 bind 的工具列表要和 build_data_agent 里的 TOOLS 是同一份，
    #        否则模型调了一个执行节点里不存在的工具（Day22-A 踩过）。
    model = SimpleChatModel(model_id=MODEL, api_key=API_KEY, base_url=BASE_URL).bind_tools(TOOLS)
    _RESOURCES["agent_graph"] = build_data_agent(model)
    return _RESOURCES["agent_graph"]


# ════════════════════════════════════════════════════════════════
# 第 3 部分：Pydantic 契约（请求/响应长什么样，写在明面上）
# ════════════════════════════════════════════════════════════════
#
# 【解释】为什么每个接口都要定义 Pydantic 模型：
#   ① 自动校验：类型错、缺字段、op 传了「大于」这种非法值 → 框架直接回 422，
#      附带哪个字段错了，根本进不到你的业务代码
#   ② 自动文档：/docs 页面的示例请求、字段说明全自动生成，前端不用问你
#   ③ 自动序列化：返回对象自动变 JSON，不用手写 json.dumps

class HealthOut(BaseModel):
    status: str = "ok"
    version: str = "1.0"
    uptime_sec: float = 0.0
    has_llm: bool = False          # 配没配 Key
    has_data: bool = False         # Day23 工具在不在
    data_rows: int = 0             # 飞行数据多少行
    manual_chunks: int = 0         # 手册切了几块
    index_backend: str = "none"    # bge / keyword / none


class ManualQueryIn(BaseModel):
    query: str = Field(..., min_length=1, description="要问手册的话，例如「温度超过多少要降落」")
    top_k: int = Field(3, ge=1, le=10, description="返回几段")


class ManualHit(BaseModel):
    title: str
    score: float
    snippet: str


class ManualOut(BaseModel):
    query: str
    backend: str
    hits: list


class DataFilterIn(BaseModel):
    col: str = Field(..., description="列名，可写模糊名如「温度」，服务会去匹配真实列名「温度C」")
    op: str = Field(..., description="比较符：> < >= <= == !=")
    value: float = Field(..., description="阈值")


class DataSummaryIn(BaseModel):
    col: str = Field(..., description="列名，可写模糊名")


class AskIn(BaseModel):
    question: str = Field(..., min_length=1, description="自然语言问题，交给 Agent 调工具解答")


class AskOut(BaseModel):
    question: str
    answer: str
    mode: str          # agent / unavailable


# ════════════════════════════════════════════════════════════════
# 第 4 部分：app + lifespan（启动预热、关闭收尾）
# ════════════════════════════════════════════════════════════════

from contextlib import asynccontextmanager   # noqa: E402

PREHEAT = os.environ.get("DAY25_PREHEAT", "1") != "0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动时做重活，关闭时做清理。

    【解释】★ await run_in_threadpool(...) 这一行很关键：
        bge 加载是同步且 CPU 密集的（约 60 秒），
        如果直接写 index.build() 会把事件循环卡死 60 秒，
        这期间连 /health 都请求不通 —— 健康检查全挂，容器会被判定不健康而重启。
        丢到线程池里，事件循环继续能响应别的请求。
    """
    print("  ▶ 服务启动中……")
    if PREHEAT:
        t0 = time.time()
        await run_in_threadpool(get_index().build)
        print("  ▶ 索引预热完成：后端=%s，块数=%d，耗时 %.1f 秒"
              % (get_index().backend, len(get_index().chunks), time.time() - t0))
    else:
        print("  ▶ 已跳过预热（DAY25_PREHEAT=0），首次检索时再建索引")
    if not HAS_KEY:
        print("  ⚠️ 没配 API Key：/ask 会返回 503，其余接口照常可用")
    print("  ▶ 就绪。文档在 http://127.0.0.1:8000/docs")
    yield
    print("  ■ 服务关闭，资源随进程释放")


app = FastAPI(
    title="无人机 AI 服务（Day25）",
    description="把 Day23 数据分析 + Day24 手册检索封装成 HTTP 接口",
    version="1.0",
    lifespan=lifespan,
)


# ════════════════════════════════════════════════════════════════
# 第 5 部分：五个端点
# ════════════════════════════════════════════════════════════════
#
# 【解释】为什么全是 def 而不是 async def：
#   FastAPI 的规则是 ——
#     async def：直接在事件循环里跑，里面只能有 await 的异步 IO
#     def      ：自动丢进线程池跑，不会阻塞事件循环
#   我们端点里的活（pandas 计算、bge 推理）全是同步阻塞的 CPU/IO 操作，
#   所以用 def 才是正确的。写成 async def 又不 await 的话，
#   一个请求就能把整个服务堵住 —— 这是 FastAPI 新手最常见的坑。

@app.get("/health", response_model=HealthOut, summary="健康检查")
def health():
    """给调用方（以及 Docker/K8s 的健康检查）看：服务活着吗、依赖齐不齐。"""
    idx = get_index()
    rows = 0
    if HAS_DATA:
        try:
            rows = len(load_df())
        except Exception:
            rows = 0
    return HealthOut(
        uptime_sec=round(time.time() - _RESOURCES["boot_at"], 1),
        has_llm=HAS_KEY,
        has_data=HAS_DATA,
        data_rows=rows,
        manual_chunks=len(idx.chunks) if idx.ready else 0,
        index_backend=idx.backend if idx.ready else "not_ready",
    )


@app.post("/manual/search", response_model=ManualOut, summary="检索《电池手册》")
def manual_search(q: ManualQueryIn):
    """纯检索，不调大模型 —— 零成本、零延迟波动，是最该服务化的一类能力。"""
    idx = get_index()
    hits = idx.search(q.query, q.top_k)
    return ManualOut(
        query=q.query,
        backend=idx.backend,
        hits=[ManualHit(title=t, score=round(s, 4), snippet=b[:200]) for t, b, s in hits],
    )


def _match_col(name: str, cols):
    """列名模糊匹配：用户说「温度」，实际列名是「温度C」。

    【解释】别指望调用方（或模型）记住精确列名，容错要在服务端做。
    策略：完全一致 → 互相包含 → 认输（返回 None，由端点报 404 并给出可用列名）。
    """
    name = (name or "").strip()
    for c in cols:
        if str(c).strip() == name:
            return c
    for c in cols:
        if name in str(c) or str(c) in name:
            return c
    return None


@app.post("/data/filter", summary="按条件过滤飞行数据")
def data_filter(q: DataFilterIn):
    """用 pandas 真算，不是让模型数 —— 这正是 Day23 那个对照实验的结论。"""
    if not HAS_DATA:
        raise HTTPException(status_code=503, detail="Day23 数据工具不可用：%s" % DATA_ERR)
    if q.op not in (">", "<", ">=", "<=", "==", "!="):
        # 【解释】Pydantic 能挡类型错误，但「业务上合法的值域」要自己挡。
        raise HTTPException(status_code=422, detail="不支持的比较符：%s" % q.op)

    cols = list(load_df().columns)
    real = _match_col(q.col, cols)
    if real is None:
        raise HTTPException(status_code=404,
                            detail="找不到列「%s」，可用列：%s" % (q.col, cols))

    out = filter_rows.invoke({"col": real, "op": q.op, "value": q.value})
    return {"col": real, "op": q.op, "value": q.value, "result": out}


@app.post("/data/summary", summary="某列统计摘要")
def data_summary(q: DataSummaryIn):
    if not HAS_DATA:
        raise HTTPException(status_code=503, detail="Day23 数据工具不可用：%s" % DATA_ERR)
    cols = list(load_df().columns)
    real = _match_col(q.col, cols)
    if real is None:
        raise HTTPException(status_code=404,
                            detail="找不到列「%s」，可用列：%s" % (q.col, cols))
    return {"col": real, "result": describe_column.invoke({"col": real})}


@app.post("/ask", response_model=AskOut, summary="完整 Agent（需 API Key）")
def ask(q: AskIn):
    """自然语言 → Agent 自己选工具 → 出结论。没配 Key 时返回 503 而不是崩掉。

    【解释】★ 依赖缺失的正确姿势：
        让 /ask 返回 503（服务不可用）并说清原因，
        而 /health、/manual/search、/data/* 全部照常工作。
        这叫「优雅降级」—— 一个能力挂了不能拖垮整个服务。
    """
    graph = get_agent_graph()
    if graph is None:
        raise HTTPException(
            status_code=503,
            detail="Agent 不可用（需要 API Key）。无 Key 时请用 /manual/search 和 /data/*。",
        )
    from langchain_core.messages import HumanMessage, SystemMessage
    # 【坑】★ SystemMessage 不能漏。第一次写这里只传了 HumanMessage，
    #        结果模型回「信息不足，请提供更多数据」—— 它根本不知道自己手上有工具。
    #        系统提示词才是「告诉模型你能调什么工具」的地方，Day23 run_question 里就有。
    init = {"messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=q.question)]}
    answer = ""
    for event in graph.stream(init, stream_mode="values"):
        for m in event["messages"]:
            if m.type == "ai" and (m.content or "").strip():
                answer = m.content.strip()
    return AskOut(question=q.question, answer=answer or "（模型没给出文字结论）", mode="agent")


# ════════════════════════════════════════════════════════════════
# 第 6 部分：全局异常处理
# ════════════════════════════════════════════════════════════════

from fastapi.responses import JSONResponse   # noqa: E402


@app.exception_handler(Exception)
def on_error(request, exc):
    """任何没接住的异常都返回 JSON。

    【解释】默认行为是抛栈给调用方（生产环境还会泄露路径和代码结构）。
    统一兜底成 JSON，既安全又让前端好处理。真实项目这里要接日志/告警。
    """
    return JSONResponse(status_code=500, content={"error": type(exc).__name__,
                                                  "detail": str(exc)[:300]})


# ════════════════════════════════════════════════════════════════
# 第 7 部分：self_test —— 不起服务也能测全部端点
# ════════════════════════════════════════════════════════════════

def self_test():
    """用 TestClient 在进程内直接调 app，不占端口、不用另开终端。

    【解释】★ 这是我最想让你记住的工程习惯：
        服务化的代码必须能「不开服务就验证」。
        TestClient 会完整跑一遍 lifespan + 路由 + 校验 + 序列化，
        等价于真起一个服务，但零端口冲突、能写进 CI、离线可跑。
    """
    if not HAS_FASTAPI:
        print("❌ 没装 fastapi：%s" % FASTAPI_ERR)
        print("   装法：D:/Python-envs/chroma-env/Scripts/python.exe -m pip install fastapi uvicorn")
        return 1

    from fastapi.testclient import TestClient

    print("═" * 62)
    print("  Day25 自测：用 TestClient 打全部端点（不占端口）")
    print("═" * 62)

    with TestClient(app) as client:        # with 会触发 lifespan（预热索引）
        ok = 0

        def check(name, resp, expect=200, show=None):
            nonlocal ok
            flag = "✅" if resp.status_code == expect else "❌"
            if resp.status_code == expect:
                ok += 1
            print("\n%s %s  → HTTP %d" % (flag, name, resp.status_code))
            try:
                data = resp.json()
            except Exception:
                data = resp.text
            if show:
                for k in show:
                    v = data.get(k) if isinstance(data, dict) else None
                    print("     %s = %s" % (k, str(v)[:160]))
            elif isinstance(data, dict) and resp.status_code >= 400:
                print("     detail = %s" % str(data.get("detail"))[:160])
            return data

        # 1) 健康检查
        check("GET /health", client.get("/health"),
              show=["status", "has_llm", "data_rows", "manual_chunks", "index_backend"])

        # 2) 手册检索
        r = client.post("/manual/search", json={"query": "温度超过多少要降落散热", "top_k": 2})
        d = check("POST /manual/search", r, show=["backend"])
        if isinstance(d, dict) and d.get("hits"):
            h = d["hits"][0]
            print("     Top1【%s】score=%.3f" % (h["title"], h["score"]))
            print("     %s" % h["snippet"][:100].replace("\n", " "))

        # 3) 数据过滤（模糊列名「温度」→ 真实列「温度C」）
        r = client.post("/data/filter", json={"col": "温度", "op": ">", "value": 55})
        d = check("POST /data/filter", r, show=["col"])
        if isinstance(d, dict) and "result" in d:
            print("     %s" % d["result"].replace("\n", " ")[:150])

        # 4) 统计摘要
        r = client.post("/data/summary", json={"col": "电压"})
        d = check("POST /data/summary", r, show=["col"])
        if isinstance(d, dict) and "result" in d:
            print("     %s" % d["result"].replace("\n", " ")[:150])

        # 5) 校验兜底：非法比较符应被挡住（422）
        check("POST /data/filter（非法 op，应为 422）",
              client.post("/data/filter", json={"col": "温度", "op": "大于", "value": 1}), 422)
        # 6) 校验兜底：乱写列名应 404 并给出可用列
        check("POST /data/filter（乱写列名，应为 404）",
              client.post("/data/filter", json={"col": "不存在的列", "op": ">", "value": 1}), 404)

        # 7) /ask：有 Key 走 Agent，没 Key 应 503（优雅降级）
        expect_ask = 200 if (HAS_KEY and HAS_AGENT and HAS_DATA) else 503
        d = check("POST /ask（%s）" % ("走 Agent" if expect_ask == 200 else "无 Key 应为 503"),
                  client.post("/ask", json={"question": "这次飞行温度超过 55 度的有几次？"}),
                  expect_ask)
        if isinstance(d, dict) and d.get("answer"):
            print("     answer = %s" % str(d["answer"])[:150])

    print()
    print("═" * 62)
    print("  自测完成：%d 项通过" % ok)
    print("  真起服务：python -m uvicorn day25_api_service:app --reload")
    print("  然后打开 http://127.0.0.1:8000/docs 看自动生成的接口文档")
    print("═" * 62)
    return 0


# ────────────────────────────────────────────────────────────────
# 【读完之后】
#   1. 先跑 `python day25_api_service.py` 看自测（索引预热约 60 秒）
#   2. 再真起服务，打开 /docs 亲手点一次 /manual/search，感受「自动文档」的价值
#   3. 试着给 /data/filter 加一个 limit 参数（练习 Pydantic 校验）
#   4. 下一步（Day26 方向）：把它接回 PySide6 上位机 ——
#      界面发 HTTP 请求 → 拿到 AI 结论显示在界面上，这就是 W6 综合项目的雏形
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.exit(self_test())
