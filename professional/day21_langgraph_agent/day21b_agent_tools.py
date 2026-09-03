"""
Day 21-B：真 Agent —— 让模型自己决定「调几次工具、什么时候停」

【这个文件回答一个问题】
    Day21-A 你看到了"循环"，但那个循环是 if/else 写的（我替模型决定）。
    真正的 Agent：**下一步去哪，由模型说了算。**

    今天要解决两个障碍：
      障碍 1：langchain_openai 没装 → 不能直接用 ChatOpenAI
              解法：自研一个适配器（第 3 部分），把 Day10 的 requests 调用
                    包装成 LangChain 认识的 ChatModel。
      障碍 2：模型怎么"指挥"图的走向？
              解法：模型返回 tool_calls → 条件边看到就拐去 tools 节点
                    → 执行完拐回 agent → 模型看完结果再决定。

【怎么读这个文件】按这个顺序看：
    1. 先看 main() 的演示一      —— 不需要 Key，先看懂图的形状
    2. 看第 2 部分的两个工具      —— 它们是"串行依赖"的（重点）
    3. 看第 3 部分的适配器        —— 这是今天最值钱的部分（面试能讲）
    4. 看第 4 部分的图            —— 只有 4 行，简单到你会怀疑
    5. 最后跑演示二               —— 真模型真的会连调两次工具

【运行方式】
    D:/Python-envs/chroma-env/Scripts/python.exe day21b_agent_tools.py

    没配 API Key 也能跑：演示一照常，演示二自动跳过并说明原因。

【依赖】
    复用 w3_ai/api_config.py（跨周复用，和 Day15/17/20 一个套路）。

【为什么这两个工具要"串行依赖"】
    这是本文件最核心的教学设计，务必看懂：
        check_battery_health  → 算出"实际可用容量"（比如 5Ah 的电池老化后只剩 4Ah）
        estimate_endurance    → 必须用上面算出的可用容量，不能用标称 5Ah

    模型必须先调第一个、拿到结果、再调第二个。**这是串行的两步。**
    回想 Day13：你的代码第 256 行直接 return 了 —— 它只能调一轮工具，
    这种"先查 A 再算 B"的需求，Day13 的写法做不到。这就是框架的价值。
"""

# ════════════════════════════════════════════════════════════════
# 第 1 部分：拿工具 + 读配置
# ════════════════════════════════════════════════════════════════

import os
import sys
import json
import requests

_HERE = os.path.dirname(os.path.abspath(__file__))
# 【解释】本文件所在目录（w5_agent）。

sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "w3_ai"))
# 【解释】w5_agent 的上级是 src，再拼 w3_ai 就是配置所在处。
#         这是 W3 以来一直用的跨周复用套路。

try:
    import api_config
    HAS_KEY = bool(api_config.API_KEY)
    API_KEY = api_config.API_KEY
    BASE_URL = api_config.BASE_URL
    MODEL = api_config.MODEL_NAME
except Exception:
    # 【解释】任何异常（没配 Key、文件不存在）都降级成"离线演示模式"。
    HAS_KEY = False
    API_KEY = ""
    BASE_URL = ""
    MODEL = ""

from typing import List
# 【解释】List 用来给工具的参数标注类型，LangChain 靠它生成工具说明书。

from langchain_core.tools import tool
# 【解释】@tool 装饰器：把一个普通 Python 函数变成"模型能调用的工具"。

from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage, ToolMessage,
)
# 【解释】LangChain 的四种消息。和 OpenAI 的 role 一一对应：
#         SystemMessage → system，HumanMessage → user，
#         AIMessage → assistant，ToolMessage → tool。

from langchain_core.language_models.chat_models import BaseChatModel
# 【解释】BaseChatModel = LangChain 给"聊天模型"定的规矩。
#         继承它、实现 _generate，你就做出了一个 LangChain 认的模型。

from langchain_core.outputs import ChatResult, ChatGeneration
# 【解释】_generate 的返回值必须包装成 ChatResult。

from langchain_core.utils.function_calling import convert_to_openai_tool
# 【解释】把 LangChain 的 @tool 转成 OpenAI 格式的工具说明书。
#         注意：这个函数在 langchain_core.utils.function_calling 里，
#         不在 langchain_core.tools 里（我踩过这个坑）。

from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
# 【解释】MessagesState = LangGraph 自带的 State，只有一个字段 messages，
#         并且配好了"追加"的合并规则（不用你手工 state["messages"] + [...]）。

from langgraph.prebuilt import ToolNode
# 【解释】ToolNode = LangGraph 自带的"执行工具"节点。
#         你给它工具列表，它负责：看 AIMessage 里的 tool_calls → 执行 → 返回 ToolMessage。


# ════════════════════════════════════════════════════════════════
# 第 2 部分：两个工具 —— 故意设计成"必须先 A 后 B"
# ════════════════════════════════════════════════════════════════

@tool
def check_battery_health(cell_voltages: List[float], nominal_capacity_ah: float):
    """检查电池健康度，返回**实际可用容量**（安时）。

    任何涉及"能飞多久 / 续航估算"的问题，都必须**先调用本工具**拿到可用容量，
    再把结果传给 estimate_endurance。不要直接用标称容量估算。

    Args:
        cell_voltages: 每片电芯的电压，例如 [3.9, 3.7, 3.85]（3S 电池 3 个值）
        nominal_capacity_ah: 电池标称容量，单位安时(Ah)，例如 5.0
    """
    # 【解释】⭐ 函数的文档字符串（docstring）不是写给你看的，是写给模型看的！
    #         LangChain 会把它原样放进工具说明书，模型靠这段话决定：
    #         什么时候调、传什么参数。所以描述写得越明确，模型调得越准。

    if not cell_voltages:
        return {"错误": "没有收到电芯电压"}

    gap = max(cell_voltages) - min(cell_voltages)
    # 【解释】压差 = 最高 - 最低。这是判断电池老化最直接的指标。

    lowest = min(cell_voltages)

    # ── 根据压差决定"可用容量打几折" ──
    if gap > 0.20:
        usable_ratio = 0.70
        health = "严重老化"
        advice = "压差过大，建议更换电池，不要再用于长航线"
    elif gap > 0.10:
        usable_ratio = 0.85
        health = "轻度老化"
        advice = "建议先做一次均衡充电，可用容量已打折"
    else:
        usable_ratio = 1.00
        health = "健康"
        advice = "状态良好，可以按标称容量使用"
    # 【解释】这是工程经验值，不是物理定律：
    #         压差大的电池，放电时最弱那片会先触底保护，整块容量就发挥不出来。

    # ── 单节电压过低 → 额外打折 ──
    if lowest < 3.5:
        usable_ratio *= 0.9
        advice += "；有电芯电压偏低，进一步打折"
    # 【解释】3.5V 是常用的"该返航"阈值，低于它放电曲线会陡降。

    usable_ah = round(nominal_capacity_ah * usable_ratio, 2)
    # 【解释】标称容量 × 折扣 = 实际可用容量。

    return {
        "压差V": round(gap, 3),
        "最低单片V": round(lowest, 2),
        "健康度": health,
        "可用容量Ah": usable_ah,
        # 【解释】⭐ 这个字段是给下一个工具用的 —— 这就是"串行依赖"的桥梁。
        "建议": advice,
    }


@tool
def estimate_endurance(usable_capacity_ah: float, avg_current_a: float):
    """根据**可用容量**和平均电流，估算无人机续航时间（分钟）。

    重要：usable_capacity_ah 必须来自 check_battery_health 的「可用容量Ah」，
    不要直接填电池标称容量（那会高估续航）。

    Args:
        usable_capacity_ah: 可用容量，单位安时(Ah)，来自 check_battery_health
        avg_current_a: 平均工作电流，单位安培(A)
    """
    if avg_current_a <= 0:
        return {"错误": "平均电流必须大于 0"}

    SAFETY_FACTOR = 0.8
    # 【解释】留 20% 余量。飞手的常识：不能把电池飞到自动降落。
    #         这个系数和 Day13 的 estimate_flight_time 保持一致。

    hours = usable_capacity_ah / avg_current_a
    minutes = hours * 60 * SAFETY_FACTOR
    # 【解释】容量 ÷ 电流 = 小时；× 60 转分钟；× 0.8 留余量。

    return {
        "理论续航分钟": round(minutes, 1),
        "可用容量Ah": usable_capacity_ah,
        "平均电流A": avg_current_a,
        "已留余量": "20%（安全系数 0.8）",
        "提醒": "实际航时受风速、温度、载重影响，请再留 10% 返航余量",
    }


TOOLS = [check_battery_health, estimate_endurance]
# 【解释】工具列表。后面要交给模型（bind_tools）和 ToolNode（执行）。


# ════════════════════════════════════════════════════════════════
# 第 3 部分：自研模型适配器（今天最值钱的一段）
# ════════════════════════════════════════════════════════════════
# 背景：官方的 ChatOpenAI 在 langchain_openai 包里，这个环境没装。
#       与其装新包，不如自己写一个 —— 反正本质就是"发 HTTP 请求 + 解析响应"。
#       而且自己写一遍，你会彻底明白 LangChain 的模型接口到底要什么。
#
# 要继承 BaseChatModel，只需要做两件事：
#     ① 实现 _generate()：收到 messages → 返回 ChatResult
#     ② 实现 _llm_type：给这个模型起个名字（LangChain 内部标识用）


class SimpleChatModel(BaseChatModel):
    """把任意 OpenAI 兼容接口包装成 LangChain 的 ChatModel。"""

    model_id: str = ""
    # 【解释】模型名，如 glm-4-flash。
    #         不叫 model_name 是因为 BaseChatModel 内部有同名 property，会冲突。

    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.3
    tool_specs: list = []
    # 【解释】OpenAI 格式的工具说明书（bind_tools 时会填进来）。

    @property
    def _llm_type(self) -> str:
        # 【解释】给模型类型起个名字，随便写，只要唯一即可。
        return "simple-openai-compatible"

    def bind_tools(self, tools, **kwargs):
        """把工具绑到模型上 —— 返回一份"带工具说明书的自己"。"""
        specs = [convert_to_openai_tool(t, strict=False) for t in tools]
        # 【解释】convert_to_openai_tool：LangChain 工具 → OpenAI 格式。
        #         strict=False：不启用严格模式（GLM 不支持 strict，开了会报错）。

        return self.model_copy(update={"tool_specs": specs})
        # 【解释】model_copy 是 pydantic 的"复制并改几个字段"。
        #         必须返回新对象：原对象保持没有工具的状态，互不影响。

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        """核心：LangChain 消息 → HTTP 请求 → OpenAI 响应 → LangChain 消息。"""

        payload = {
            "model": self.model_id,
            "messages": [_to_openai_message(m) for m in messages],
            "temperature": self.temperature,
        }
        # 【解释】拼请求体。messages 要从 LangChain 对象转成普通字典。

        if self.tool_specs:
            payload["tools"] = self.tool_specs
        # 【解释】⭐ 关键：有工具就把说明书带上，模型才知道自己能调什么。

        if stop:
            payload["stop"] = stop
        # 【解释】stop = 遇到这些字符串就停止生成（LangChain 的通用参数）。

        resp = requests.post(
            self.base_url + "/chat/completions",
            # 【解释】OpenAI 兼容接口的固定路径。GLM / DeepSeek 都一样。

            headers={
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=90,
            # 【解释】90 秒超时。模型思考时可能很慢，太短会误判失败。
        )
        resp.raise_for_status()
        # 【解释】HTTP 状态码不是 2xx 就抛异常（4xx/5xx 直接暴露，别静默失败）。

        data = resp.json()
        msg = data["choices"][0]["message"]
        # 【解释】取出模型这条消息，可能带 tool_calls。

        tool_calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            # 【解释】把 OpenAI 格式的 tool_calls 转成 LangChain 格式。
            #         OpenAI: {"id":.., "function":{"name":.., "arguments":"{json字符串}"}}
            #         LangChain: {"name":.., "args":{字典}, "id":..}

            try:
                args = json.loads(tc["function"]["arguments"])
            except Exception:
                # 【解释】模型偶尔会吐出不合法的 JSON，兜住别让整个流程崩掉。
                args = {}

            tool_calls.append({
                "name": tc["function"]["name"],
                "args": args,
                "id": tc.get("id") or ("call_%d" % i),
                # 【解释】id 必须有：模型靠它把"工具结果"对上"哪次调用"。
                "type": "tool_call",
            })

        ai = AIMessage(
            content=msg.get("content") or "",
            # 【解释】调工具时 content 常常是空的，用 "" 兜底（None 会让后面报错）。

            tool_calls=tool_calls,
        )

        return ChatResult(generations=[ChatGeneration(message=ai)])
        # 【解释】包装成 LangChain 规定的返回格式。


def _to_openai_message(m):
    """LangChain 消息对象 → OpenAI 接口的字典。"""
    if isinstance(m, SystemMessage):
        return {"role": "system", "content": m.content}

    if isinstance(m, HumanMessage):
        return {"role": "user", "content": m.content}

    if isinstance(m, ToolMessage):
        return {
            "role": "tool",
            "tool_call_id": m.tool_call_id,
            # 【解释】⭐ tool_call_id 不能漏！漏了模型会说"不知道你在回哪次调用"。
            "content": m.content,
        }

    if isinstance(m, AIMessage):
        d = {"role": "assistant", "content": m.content or ""}
        if m.tool_calls:
            d["tool_calls"] = [{
                "id": tc.get("id") or ("call_%d" % i),
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc.get("args", {}), ensure_ascii=False),
                    # 【解释】ensure_ascii=False：中文参数不要转成 \uXXXX，
                    #         否则模型可能读不懂（虽然大多数模型能处理，但没必要冒险）。
                },
            } for i, tc in enumerate(m.tool_calls)]
        return d

    raise ValueError("不认识的消息类型：%r" % type(m))
    # 【解释】遇到没处理的类型就明确报错，比默默丢掉消息好排查得多。


# ════════════════════════════════════════════════════════════════
# 第 4 部分：图 —— 只有 4 行，简单到你会怀疑
# ════════════════════════════════════════════════════════════════

def build_agent(model, checkpointer=None):
    """搭一个 ReAct Agent：agent ⇄ tools，循环由模型决定。

    Args:
        model: 绑定好工具的模型
        checkpointer: 存档器（Day21-C 用）。传 None = 不记忆，
                      传 MemorySaver() = 记住整个对话。
                      其余部分完全一样 —— 加记忆只多这一个参数。
    """

    def call_model(state: MessagesState):
        # 【解释】节点 1：问模型。
        #         MessagesState 里 messages 是"追加"语义，所以这里只返回新消息。
        return {"messages": [model.invoke(state["messages"])]}

    def should_continue(state: MessagesState) -> str:
        # 【解释】条件边：看模型最后那条消息有没有要调工具。
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
            # 【解释】有工具要调 → 去执行。
        return END
        # 【解释】没工具要调 → 说明模型给出最终答案了，结束。

    builder = StateGraph(MessagesState)

    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(TOOLS))
    # 【解释】ToolNode 是现成的：读 tool_calls → 执行 → 返回 ToolMessage 列表。

    builder.add_edge(START, "agent")
    # 【解释】入口 → 先问模型。

    builder.add_conditional_edges("agent", should_continue, ["tools", END])
    # 【解释】⭐ 分叉：要么去执行工具，要么结束。

    builder.add_edge("tools", "agent")
    # 【解释】⭐⭐ 就是这一行创造了循环！
    #         工具执行完 → 回到模型 → 模型看结果 → 可能又要调工具 → 再绕一圈。
    #         Day13 你的手写代码缺的就是这一行。

    return builder.compile(checkpointer=checkpointer)
    # 【解释】这里多传个 checkpointer 就"有记忆"了（Day21-C 会演示）。
    #         默认是 None = 无记忆。


# ════════════════════════════════════════════════════════════════
# 第 5 部分：main()
# ════════════════════════════════════════════════════════════════

def demo_shape():
    """演示一：不需要 Key，先看清图的形状和工具说明书。"""
    print()
    print("=" * 64)
    print("  演示一：图的形状 + 工具长什么样（不需要 API Key）")
    print("=" * 64)

    print()
    print("  图的走法：")
    print()
    print("      START ──→ agent ──条件边──┬──→ tools ──┐")
    print("                  ↑            │            │")
    print("                  └────────────┴────────────┘")
    print("                  （tools 执行完回到 agent = 循环）")
    print("                            └──→ END（模型不再要工具时）")
    print()
    print("  就这 4 行代码，循环次数由模型决定，你不用写 while。")

    print()
    print("  " + "-" * 60)
    print("  工具说明书（模型看到的就是这个）：")
    for t in TOOLS:
        spec = convert_to_openai_tool(t, strict=False)
        fn = spec["function"]
        print()
        print("    · %s" % fn["name"])
        print("      说明：%s" % fn["description"].strip().split("\n")[0])
        print("      参数：%s" % ", ".join(fn["parameters"]["properties"].keys()))

    print()
    print("  " + "-" * 60)
    print("  ⭐ 注意 check_battery_health 的说明里写了：")
    print("     「任何涉及续航的问题，都必须先调用本工具」")
    print("     这句话就是让模型串行调用的关键 —— 工具描述要写清依赖关系。")


def demo_real_agent():
    """演示二：真模型，真的会连调两次工具。"""
    print()
    print("=" * 64)
    print("  演示二：真模型跑起来（需要 API Key）")
    print("=" * 64)

    if not HAS_KEY:
        print()
        print("  ⚠️  没读到 API Key，跳过这个演示。")
        print("     配好 w3_ai/api_config.py 里的 API_KEY 后重跑即可。")
        print("     演示一不需要 Key，可以先看那个。")
        return

    print()
    print("  模型：%s" % MODEL)
    print("  接口：%s" % BASE_URL)

    model = SimpleChatModel(
        model_id=MODEL,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0.2,
        # 【解释】温度调低：Agent 要的是稳定地选对工具，不是创造力。
    ).bind_tools(TOOLS)
    # 【解释】⭐ bind_tools 返回"带说明书的模型"，图里的 agent 节点用它。

    graph = build_agent(model)

    question = (
        "我的无人机电池是 3S，三片电压分别是 3.92、3.68、3.88，"
        "标称容量 5.0Ah，平均工作电流 20A。这块电池还能飞多久？"
    )
    # 【解释】这个问题故意设置成"必须先查健康度、再算续航"：
    #         压差 = 3.92 - 3.68 = 0.24V > 0.20 → 可用容量打 7 折 = 3.5Ah
    #         然后 3.5 / 20 × 60 × 0.8 = 8.4 分钟
    #         如果模型直接用标称 5Ah 算 → 12 分钟，那就错了（高估 43%）。

    print()
    print("  问题：%s" % question)
    print()
    print("  " + "-" * 60)
    print("  执行过程（每一步都打出来，看清楚循环了几圈）：")
    print("  " + "-" * 60)

    step = 0
    for event in graph.stream(
        {"messages": [HumanMessage(content=question)]},
        stream_mode="values",
        # 【解释】stream_mode="values" = 每走完一个节点就把"当前完整状态"吐出来。
        #         这样你能一步步看到消息是怎么变长的 —— 正是你要的"看得见的反馈"。
    ):
        step += 1
        last = event["messages"][-1]
        # 【解释】取最新那条消息。

        kind = type(last).__name__

        if kind == "HumanMessage":
            # 【解释】stream_mode="values" 第一步吐出的是"初始状态"，
            #         里面只有用户提问。不单独判断的话，它会被当成"最终回答"打印出来。
            print()
            print("  [第 %d 步] 用户提问：%s" % (step, last.content))

        elif kind == "AIMessage" and getattr(last, "tool_calls", None):
            print()
            print("  [第 %d 步] 模型决定调工具：" % step)
            for tc in last.tool_calls:
                print("      → %s(%s)" % (tc["name"], json.dumps(
                    tc["args"], ensure_ascii=False)))

        elif kind == "ToolMessage":
            print("      ✔ 工具返回：%s" % last.content)

        else:
            print()
            print("  [第 %d 步] 最终回答：" % step)
            print("      %s" % (last.content or "").strip())

    print()
    print("  " + "-" * 60)
    print("  ⭐ 数一下：如果上面出现了两轮「模型决定调工具」，")
    print("     说明循环真的跑起来了 —— 这正是 Day13 手写版做不到的。")


def main():
    print()
    print("╔" + "═" * 62 + "╗")
    print("║" + " " * 12 + "Day 21-B：真 Agent（模型决定循环）" + " " * 12 + "║")
    print("╚" + "=" * 62 + "╝")

    demo_shape()
    demo_real_agent()

    print()
    print("=" * 64)
    print("  【读完之后】Day21-B 你学到了什么")
    print("=" * 64)
    print("""
    ✅ 已掌握
       · Agent 和 RAG 的区别：RAG 是一条直线，Agent 是一个圈
       · 模型通过"返回 tool_calls"来指挥图的走向，不需要你写 if/else
       · 循环就是一行 add_edge("tools", "agent") —— 回到模型再问一次
       · 没装 langchain_openai 也能用：继承 BaseChatModel 写个适配器就行
       · 工具描述要写清"依赖关系"，模型才会按顺序调用

    ⚠️  还要留意的坑
       · 工具函数修改列表时必须返回新列表，不能原地 append
       · tool_call_id 不能漏，否则模型对不上是哪次调用的结果
       · strict=False：GLM 不支持 OpenAI 的严格模式，开了会报错
       · 字段别叫 model_name，BaseChatModel 内部有同名 property 会冲突
       · AIMessage.content 调工具时常为 None，要 or "" 兜底

    💡 面试怎么讲（三句话）
       1. Day13 我手写过工具调用循环，但它只能跑一轮，模型没法说"再查一个"
       2. Day21 用 LangGraph 重写：agent 和 tools 两个节点，条件边决定
          继续还是结束，循环由模型决定，我不用管终止条件
       3. 环境没装 langchain_openai，我自己继承了 BaseChatModel 写适配器，
          本质就是把 requests 调用包成 LangChain 的接口

    ➡️  下一步
       Day21-C：给 Agent 加"记忆"（checkpointer）和多轮对话；
       或者回到主线，把 Day21 发布到作品集。
    """)


if __name__ == "__main__":
    main()
