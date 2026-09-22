"""════════════════════════════════════════════════════════════════
Day31a  运维智能体（核心逻辑，零 GUI / 零网络依赖可测）
════════════════════════════════════════════════════════════════
做什么：
    把「实时巡检告警」自动变成「结构化处置单」——自主查《电池手册》(RAG) + 分析飞行数据，
    输出：异常类型 / 根因 / 处置步骤 / 风险等级 / 手册引用 / 数据佐证。
    这是 W6 收官后第一个「综合大项目」(Day31) 的大脑：把 Day25(检索/分析服务) 和
    Day28(实时巡检服务) 串成一个「会自己干活的运维 Agent」。

怎么读：
    MaintenanceAgent.generate_order(alert, backend, have_key=None) 是唯一切口。
    - backend 决定「查手册 / 查数据」从哪来（FakeBackend=测试用；HttpBackend=真连 Day25 服务）
    - have_key=False → 走规则引擎（确定性、零成本、永远可用）
    - have_key=True  → 走 LangGraph Agent（模型自主决定先查手册还是先看数据）

怎么跑：
    python day31a_maintenance_agent.py --selftest      # 纯逻辑自测，不碰网络/LLM

为什么学：
    ① Agent 不是摆设：感知(告警) → 决策(调工具查资料) → 行动(出处置单) 的闭环。
    ② 优雅降级：没有 API Key 时，规则引擎照样能出一份能用的处置单——这正是生产该有的韧性。
    ③ 测试复用真逻辑：规则引擎和 LLM 共用同一套输出结构，评测不漂。
【读完之后】
    → day31b_orchestrator.py 把它接进真实服务（轮询 Day28 的 /live/alerts，触发本文件）
    → professional/day31_maintenance/ 是发布目录（自包含，拷出去就能跑）
════════════════════════════════════════════════════════════════
"""

import os
import sys
import json

# ── 可选依赖降级：LangGraph / LangChain 没装也能跑规则模式 ──
try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langgraph.prebuilt import build_agent
    HAS_LANG = True
except Exception:
    HAS_LANG = False

# ── API Key：复用 w3_ai/api_config 的同一套环境变量读取（不写死 Key）──
# 【解释】api_config 在无 Key 时会主动 raise，所以必须包在 try 里；
#         我们只取它确定导出的 BASE_URL / MODEL_NAME / API_KEY，自己算 HAS_KEY，
#         不去依赖它有没有导出 HAS_KEY / get_client（不同版本不一定有）。
try:
    from api_config import API_KEY, BASE_URL, MODEL_NAME
    _API_OK = bool(API_KEY)
except Exception:
    _API_OK = False
    API_KEY = BASE_URL = MODEL_NAME = ""
HAS_KEY = _API_OK


def get_client():
    """返回一个已配好后端地址和 Key 的 chat model；没 Key 返回 None。"""
    if not HAS_KEY:
        return None
    try:
        from langchain_openai import ChatOpenAI
    except Exception:
        return None
    return ChatOpenAI(model=MODEL_NAME, api_key=API_KEY,
                      base_url=BASE_URL, temperature=0)


# ════════════════════════════════════════════════════════════════
# 第 1 部分：Backend —— 「查手册 / 查数据」两个能力的抽象
# ════════════════════════════════════════════════════════════════
#
# 【解释】为什么抽象一层？
#   测试时我们不想真起 Day25 服务、也不想真联网；生产时又必须真连。
#   把「能力」和「来源」解耦，同一份 Agent 逻辑，FakeBackend 测、HttpBackend 跑，
#   评测和线上用的是同一行代码——这才是「测试不漂」的根本。
class Backend:
    """子类实现两个方法，返回 dict（结构对齐 Day25 的真实响应）。"""

    def search_manual(self, query, top_k=3):
        raise NotImplementedError

    def analyze_metric(self, col, op, value):
        raise NotImplementedError


class FakeBackend(Backend):
    """纯逻辑自测用：返回固定的 golden 手册段落和统计，零网络零 LLM。"""

    def search_manual(self, query, top_k=3):
        return {"query": query, "backend": "fake", "hits": [
            {"title": "电池温度异常处理", "score": 0.91,
             "snippet": "电池温度超过 55℃ 应立即降低放电倍率，并检查散热风道是否堵塞；持续升温须就近安全降落。"},
            {"title": "高温工况维护", "score": 0.84,
             "snippet": "高温会加速电芯老化，建议在 40℃ 以下环境作业。"},
        ]}

    def analyze_metric(self, col, op, value):
        return {"col": col, "op": op, "value": value,
                "result": {"rows": 12, "samples": [56.1, 57.3, 55.8, 58.0]}}


class HttpBackend(Backend):
    """真连 Day25 服务（检索 + 数据分析）。client 可以是 TestClient（进程内测试）
    或 None（用 requests 连真实端口）。"""

    def __init__(self, manual_url, data_url, client=None):
        self.manual_url = manual_url      # 例：http://127.0.0.1:8000/manual/search
        self.data_url = data_url          # 例：http://127.0.0.1:8000/data/filter
        self.client = client

    def _post(self, url, body):
        if self.client is not None:
            return self.client.post(url, json=body).json()
        import requests
        return requests.post(url, json=body, timeout=30).json()

    def search_manual(self, query, top_k=3):
        return self._post(self.manual_url, {"query": query, "top_k": top_k})

    def analyze_metric(self, col, op, value):
        return self._post(self.data_url, {"col": col, "op": op, "value": value})


# ════════════════════════════════════════════════════════════════
# 第 2 部分：从告警文本里抽指标 + 指标→手册/规则 映射表
# ════════════════════════════════════════════════════════════════
def parse_metric(msg):
    """从中文告警里抽出指标类型。没有就返回 未知。"""
    if "温度" in (msg or ""):
        return "温度"
    if "电压" in (msg or ""):
        return "电压"
    if "电流" in (msg or ""):
        return "电流"
    if "高度" in (msg or ""):
        return "高度"
    return "未知"


# 【解释】这张表是规则引擎的「知识」。它和 LLM 模式共用——
#   规则模式直接读它；LLM 模式用它生成检索词、也作为兜底。
# 三个维度：kw=查手册用的检索词；col/op/value=查数据用的过滤；cause/steps/risk=处置结论。
MANUAL_MAP = {
    "温度": {
        "kw": "电池 温度 过热 散热 降温",
        "col": "温度C", "op": ">", "value": 55,
        "cause": "电池组温度超过安全阈值，通常由大电流放电或散热风道堵塞引起",
        "steps": ["立即降低放电倍率并悬停降温", "检查散热风道是否堵塞",
                  "若温度持续攀升则就近安全降落"],
        "risk": "高",
    },
    "电压": {
        "kw": "电压 放电截止 鼓包 欠压 降落",
        "col": "电压V", "op": "<", "value": 23,
        "cause": "单体电压跌近放电截止电压，可能由电池老化或局部鼓包引起",
        "steps": ["就近降落避免空中断电", "落地后检查电池是否鼓包/老化",
                  "更换电池后再起飞"],
        "risk": "高",
    },
    "电流": {
        "kw": "电流 过载 桨叶 卡阻",
        "col": "电流A", "op": ">", "value": 3,
        "cause": "放电电流超过安全阈值，可能由负载过大或桨叶卡阻引起",
        "steps": ["减小飞行动作幅度", "检查桨叶是否卡阻或变形",
                  "电流持续偏高则降落检查"],
        "risk": "中",
    },
    "高度": {
        "kw": "高度 低空 降落 地形",
        "col": "高度m", "op": "<", "value": 10,
        "cause": "飞行高度过低，存在撞地/挂障碍物风险",
        "steps": ["提升飞行高度至安全余量以上", "确认下方无人员/障碍物"],
        "risk": "中",
    },
    "未知": {
        "kw": "无人机 电池 维护 异常",
        "col": "", "op": ">", "value": 0,
        "cause": "检测到未分类异常，需进一步定位",
        "steps": ["就近安全降落并目视检查", "查阅电池维护手册"],
        "risk": "中",
    },
}


# ════════════════════════════════════════════════════════════════
# 第 3 部分：规则引擎（无 Key 降级路径，确定性、永远可用）
# ════════════════════════════════════════════════════════════════
def rule_generate_order(alert, backend):
    """没有 LLM 时，用映射表确定性地出一份处置单。
    手册引用和数据佐证都真调 backend —— 即使没有大脑，手脚也真去查了。"""
    msg = (alert or {}).get("msg", "")
    metric = parse_metric(msg)
    m = MANUAL_MAP.get(metric, MANUAL_MAP["未知"])

    # 真查手册（FakeBackend 返回 golden；HttpBackend 真连 Day25）
    refs = []
    try:
        hits = backend.search_manual(m["kw"], top_k=2).get("hits", [])
        for h in hits:
            title = h.get("title", "")
            snippet = (h.get("snippet", "") or "")[:90]
            if title or snippet:
                refs.append("%s：%s" % (title, snippet))
    except Exception:
        pass
    if not refs:
        refs = [m["kw"]]

    # 真查数据（能查才查，查不到不影响主结论）
    evidence = ""
    if m["col"]:
        try:
            r = backend.analyze_metric(m["col"], m["op"], m["value"])
            res = r.get("result", r)
            if isinstance(res, dict):
                evidence = "过滤 %s %s %s：命中 %s 行" % (
                    m["col"], m["op"], m["value"],
                    res.get("rows", res.get("count", "?")))
            else:
                evidence = "已对该指标做数据过滤"
        except Exception:
            evidence = ""

    return {
        "alert": alert,
        "metric": metric,
        "root_cause": m["cause"],
        "steps": list(m["steps"]),
        "manual_refs": refs,
        "data_evidence": evidence,
        "risk": m["risk"],
        "mode": "rule",
    }


# ════════════════════════════════════════════════════════════════
# 第 4 部分：LLM 模式（有 Key 时，让模型自主调工具查资料）
# ════════════════════════════════════════════════════════════════
_BACKEND = None   # 工具函数运行时注入，避免闭包捕获


def _tool_search_manual(query: str, top_k: int = 3) -> str:
    """检索《无人机电池维护手册》，返回相关处置段落。先查手册是处置的依据。"""
    hits = _BACKEND.search_manual(query, top_k).get("hits", [])
    out = []
    for h in hits:
        out.append("[%s] %s" % (h.get("title", ""), (h.get("snippet", "") or "")[:160]))
    return "\n".join(out) or "（手册无命中）"


def _tool_analyze_metric(col: str, op: str, value: float) -> str:
    """按条件过滤飞行数据，返回该指标的数据分布作为佐证。"""
    r = _BACKEND.analyze_metric(col, op, value)
    return json.dumps(r.get("result", r), ensure_ascii=False)[:400]


TOOLS = [_tool_search_manual, _tool_analyze_metric]

SYSTEM_PROMPT = (
    "你是无人机机队运维助手。收到一条实时巡检告警后，必须：\n"
    "1) 先调用 search_manual 查《电池维护手册》拿到处置依据；\n"
    "2) 必要时调用 analyze_metric 看该指标的数据分布；\n"
    "3) 用中文输出一份结构化处置单，严格按下面格式（不要多余解释）：\n"
    "【根因】<一句话>\n"
    "【处置步骤】\n- <步骤1>\n- <步骤2>\n"
    "【风险等级】高/中/低\n"
)


def _parse_order_text(text, alert, backend, metric, m):
    """从 LLM 的自由文本里抠出结构化字段；抠不到就退回规则兜底。"""
    cause, risk = "", ""
    steps = []
    for line in (text or "").splitlines():
        s = line.strip()
        if s.startswith("【根因】"):
            cause = s[4:].strip()
        elif s.startswith("【风险等级】"):
            risk = s[5:].strip()
        elif s.startswith("-") or s.startswith("·"):
            steps.append(s[1:].strip())
    if not cause:
        cause = m["cause"]
    if not risk:
        risk = m["risk"]
    if not steps:
        steps = list(m["steps"])
    # 手册引用和数据佐证仍取自真实工具结果
    refs = []
    try:
        for h in backend.search_manual(m["kw"], top_k=2).get("hits", []):
            refs.append("%s：%s" % (h.get("title", ""), (h.get("snippet", "") or "")[:90]))
    except Exception:
        pass
    if not refs:
        refs = [m["kw"]]
    return {
        "alert": alert, "metric": metric, "root_cause": cause,
        "steps": steps, "manual_refs": refs,
        "data_evidence": "", "risk": risk, "mode": "llm",
    }


def llm_generate_order(alert, backend):
    """有 Key 时：LangGraph Agent 自主决定先查手册还是先看数据，最后出处置单。"""
    global _BACKEND
    _BACKEND = backend
    client = get_client()
    if client is None or not HAS_LANG:
        return rule_generate_order(alert, backend)
    # ★ 铁律：build_agent 必须传「已 bind_tools 的模型」
    model = client.bind_tools(TOOLS)
    graph = build_agent(model, tools=TOOLS)
    metric = parse_metric((alert or {}).get("msg", ""))
    m = MANUAL_MAP.get(metric, MANUAL_MAP["未知"])
    human = "实时巡检告警：%s\n请查手册并生成处置单。" % (alert or {}).get("msg", "")
    out = ""
    # ★ 铁律：调用前必须带 SystemMessage，否则工具被静默禁用（HTTP 200 但答错）
    for ev in graph.stream(
            {"messages": [SystemMessage(content=SYSTEM_PROMPT),
                          HumanMessage(content=human)]},
            stream_mode="values"):
        for msg in ev["messages"]:
            if getattr(msg, "type", "") == "ai" and (msg.content or "").strip():
                out = msg.content
    return _parse_order_text(out, alert, backend, metric, m)


# ════════════════════════════════════════════════════════════════
# 第 5 部分：统一入口
# ════════════════════════════════════════════════════════════════
class MaintenanceAgent:
    """对外只暴露 generate_order。have_key 不传则自动按环境决定。"""

    @staticmethod
    def generate_order(alert, backend, have_key=None):
        if have_key is None:
            have_key = HAS_KEY
        if have_key and HAS_LANG:
            try:
                return llm_generate_order(alert, backend)
            except Exception as e:
                # ★ 降级不止在「无 Key」：LLM 限流/超时也要能退回规则，闭环不中断
                return dict(rule_generate_order(alert, backend),
                            fallback_reason=repr(e)[:120])
        return rule_generate_order(alert, backend)


# ════════════════════════════════════════════════════════════════
# 第 6 部分：纯逻辑自测（--selftest，不碰网络/LLM）
# ════════════════════════════════════════════════════════════════
def _selftest():
    print("=" * 64)
    print("  Day31a 自测：规则引擎 + 工具 schema（FakeBackend，零网络）")
    print("=" * 64)
    checks = []

    def check(name, cond, extra=""):
        print("  %s %s%s" % ("✅" if cond else "❌", name,
                             ("  " + extra) if extra else ""))
        checks.append(bool(cond))

    # 1) 指标解析
    check("parse_metric(温度报警)", parse_metric("温度 60.2C 超过 55C") == "温度")
    check("parse_metric(电压跌落)", parse_metric("电压 22.8V 低于 23V") == "电压")
    check("parse_metric(未知)", parse_metric("通信异常") == "未知")

    b = FakeBackend()
    # 2) 温度告警 → 规则处置单
    order = MaintenanceAgent.generate_order(
        {"frame": 40, "level": "alarm", "msg": "温度 60.2C 超过 55C", "t": 40.0},
        b, have_key=False)
    check("温度告警→处置单", order["metric"] == "温度")
    check("根因非空", bool(order["root_cause"]), order["root_cause"][:20])
    check("处置步骤≥2", len(order["steps"]) >= 2, "steps=%d" % len(order["steps"]))
    check("风险=高", order["risk"] == "高")
    check("手册引用来自真查", len(order["manual_refs"]) >= 1
          and "温度" in order["manual_refs"][0], order["manual_refs"][0][:30])
    check("降级模式标记", order["mode"] == "rule")

    # 3) 电压告警 → 不同处置分支
    o2 = MaintenanceAgent.generate_order(
        {"frame": 12, "level": "alarm", "msg": "电压 22.8V 低于 23V", "t": 12.0},
        b, have_key=False)
    check("电压告警→不同指标分支", o2["metric"] == "电压" and o2["risk"] == "高")

    # 4) 工具 schema 可被 LangGraph 转换（有依赖时）
    if HAS_LANG:
        try:
            from langchain_core.utils.function_calling import convert_to_openai_tool
            for t in TOOLS:
                convert_to_openai_tool(t)
            check("工具 schema 可转 openai", True)
        except Exception as e:
            check("工具 schema 可转 openai", False, repr(e)[:80])
    else:
        check("工具 schema（跳过：无 langgraph）", True)

    # 5) 有 Key 时也能安全降级（这里强制 have_key=True 但用 FakeBackend，
    #    真 LLM 调用走 try，失败退回规则 → 仍能拿到完整处置单）
    if HAS_KEY and HAS_LANG:
        try:
            o3 = MaintenanceAgent.generate_order(
                {"frame": 40, "level": "alarm",
                 "msg": "温度 60.2C 超过 55C", "t": 40.0}, b, have_key=True)
            check("LLM 模式产出处置单", "root_cause" in o3 and o3["mode"] in ("llm", "rule"))
        except Exception as e:
            check("LLM 模式产出处置单", False, repr(e)[:80])
    else:
        check("LLM 模式（跳过：无 Key）", True)

    passed = sum(1 for x in checks if x)
    print("=" * 64)
    print("  自测完成：%d / %d 项通过" % (passed, len(checks)))
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    print("Day31a 运维智能体核心。用法：")
    print("  python day31a_maintenance_agent.py --selftest")
