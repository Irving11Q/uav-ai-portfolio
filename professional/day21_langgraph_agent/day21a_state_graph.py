"""
Day 21-A：LangGraph 的骨架 —— 图到底是什么？为什么要有它？

【这个文件回答一个问题】
    Day13 你已经写过一个"模型调工具"的循环（w3_ai/day13_function_calling.py）。
    但那个循环有个硬伤：**它是固定两轮的**。
    第 1 轮问模型要什么工具 → 执行 → 第 2 轮拿最终回答 → 直接 return。
    如果模型看完工具结果说"我还得再查一个东西"，你的代码做不到。

    LangGraph 解决的就这一件事：**让循环能绕回去，绕几次由模型决定。**

【怎么读这个文件】按这个顺序看，不要跳：
    1. 先跑起来看输出          —— 你会看到 check 节点被访问了 4 次（重点！）
    2. 回到 class State        —— Agent 的"记忆本"，节点之间就靠它传话
    3. 再看 check / report     —— 节点其实就是一个普通函数
    4. 然后看 router 那两个    —— 条件边：返回值决定"下一步去哪"
    5. 最后看 build_graph()    —— 把零件拼起来

【运行方式】
    D:/Python-envs/chroma-env/Scripts/python.exe day21a_state_graph.py

    ⚠️ 必须用 chroma-env（langgraph / langchain_core 装在这个环境里）。
       系统 Python 没装，会 ImportError。

【为什么这个文件不需要 API Key】
    前 3 周你学到的"先看现象后读代码"原则 —— 图的结构跟大模型无关。
    这个文件的"决策"用普通 if/else 写的（模拟模型），
    所以没网、没 Key 也能跑通，你能先看清骨架。
    真的模型接进来的版本在 Day21-B。

【为什么单独拆出来】
    Day21 如果一口气写完 = State + 节点 + 条件边 + 工具调用 + 模型适配器，
    那是 5 个新概念，超出你一次能消化的量（工作记忆 4±1）。
    所以拆成 A（骨架，4 个概念）/ B（接真模型，2 个新概念）。
"""

# ════════════════════════════════════════════════════════════════
# 第 1 部分：拿工具
# ════════════════════════════════════════════════════════════════

from typing import TypedDict
# 【解释】TypedDict = 带类型提示的字典。
#         LangGraph 用它定义 State，好处是写 state["xxx"] 时编辑器能提示你有哪些字段。

from langgraph.graph import StateGraph, START, END
# 【解释】三个核心零件：
#         StateGraph = 画布（你在上面画节点和箭头）
#         START      = 入口的标记（不是节点，是个占位符）
#         END        = 出口的标记（走到它就结束）


# ════════════════════════════════════════════════════════════════
# 第 2 部分：State —— Agent 的记忆本（最重要的一节）
# ════════════════════════════════════════════════════════════════
# 一句话理解 State：
#     **它是一本在节点之间传着走的记录本。**
#     每个节点都能读它、往里面写东西，然后传给下一个节点。
#
# 这就是为什么 Agent 能"记住前面发生了什么" —— 全靠这本子。

class State(TypedDict):
    # 【解释】定义一本记录本长什么样（有哪些字段、分别是什么类型）。

    battery: dict
    # 【解释】电池遥测数据：电压、温度、容量。这是"输入"，全程不变。

    checklist: list
    # 【解释】检查清单：待办的检查项。
    #         ⭐ 关键：每检查完一项就 pop 掉一个 —— 它是"会变短的"，
    #            条件边就靠"它空没空"来决定继续还是结束。

    issues: list
    # 【解释】发现的问题列表。检查过程中不断往里追加。

    logs: list
    # 【解释】每一步的日志。用来打印，让你看清执行顺序。

    verdict: str
    # 【解释】最终结论："可以飞" / "不能飞"。
    #         一开始是空的，最后由 report 节点填上。


# ════════════════════════════════════════════════════════════════
# 第 3 部分：节点 —— 其实就是一个普通函数
# ════════════════════════════════════════════════════════════════
# 节点函数的规矩只有两条：
#     1. 参数必须是 state（拿到整本记录本）
#     2. 返回值是一个字典（只写"我要改哪些字段"，不用返回整本）
#
# 这个"只返回增量"的设计很关键：LangGraph 会帮你合并回原本子，
# 所以你不用把没改的字段也抄一遍。


def check(state: State) -> dict:
    # 【解释】⭐ 这个节点会被访问多次 —— 每检查一项进一次。
    #         这就是"循环"：不是把 4 个检查写进一个函数，
    #         而是同一个节点反复执行，每次处理清单里剩下的一项。

    item = state["checklist"][0]
    # 【解释】取清单第一项（还没检查的里面最靠前的那个）。

    remaining = state["checklist"][1:]
    # 【解释】剩下的：[1:] 表示"从下标 1 开始到末尾"，也就是砍掉第 0 个。

    battery = state["battery"]
    # 【解释】取出电池数据。

    found_issue = None
    # 【解释】先假设这一项没问题。

    # ── 逐项检查（真实的判断逻辑，不是假的）──
    if item == "电芯电压一致性":
        cells = battery["cell_voltages"]
        # 【解释】3S 电池 = 3 片电芯串联，每片电压应该差不多。
        gap = max(cells) - min(cells)
        # 【解释】压差 = 最高 - 最低。压差大说明电池老化或某片坏了。
        if gap > 0.10:
            found_issue = "电芯压差 %.2fV（超过 0.10V 上限）" % gap
            # 【解释】飞手常识：压差超 0.1V 就该做均衡维护了。

    elif item == "总电压":
        total = sum(battery["cell_voltages"])
        # 【解释】总电压 = 三片相加。
        if total < battery["min_total_voltage"]:
            found_issue = "总电压 %.2fV 低于安全线 %.2fV" % (
                total, battery["min_total_voltage"])

    elif item == "温度":
        temp = battery["temperature"]
        if temp < 5 or temp > 45:
            # 【解释】锂电池怕冷也怕热：低于 5℃ 放电能力骤降，高于 45℃ 有风险。
            found_issue = "温度 %d℃ 超出 5~45℃ 安全区间" % temp

    elif item == "剩余容量":
        soc = battery["soc_percent"]
        if soc < 30:
            found_issue = "剩余电量仅 %d%%，不足以完成航线" % soc

    # ── 把这一步的结果写进记录本 ──
    new_issues = state["issues"] + ([found_issue] if found_issue else [])
    # 【解释】⭐ 注意这里：state["issues"] 是旧的，加上新的，得到新的列表。
    #         不能直接 state["issues"].append(...) —— 那叫"原地修改"，
    #         LangGraph 靠比较前后值判断有没有变化，原地改它看不见。
    #         这是新手最常踩的坑之一。

    log_line = "  [check #%d] %s → %s" % (
        4 - len(remaining), item,
        ("⚠️ " + found_issue) if found_issue else "✅ 通过")
    # 【解释】拼一行日志。4 - len(remaining) 就是"第几次检查"。

    print(log_line)
    # 【解释】直接打印 —— 让你一眼看到这个节点跑了几次。

    return {
        "checklist": remaining,
        # 【解释】清单变短了（消费掉一项）。

        "issues": new_issues,
        # 【解释】问题列表（可能没变，也可能多了）。

        "logs": state["logs"] + [log_line],
        # 【解释】日志追加同样一条。
    }


def consult_manual(state: State) -> dict:
    # 【解释】只在"发现了问题"时才走的节点。
    #         演示条件边的分叉：有问题才查手册，没问题就跳过。

    print("  [manual ] 查手册：找到 %d 条相关处置建议" % len(state["issues"]))

    log_line = "  [manual ] 已查阅手册，%d 个问题有对应处置流程" % len(state["issues"])
    return {"logs": state["logs"] + [log_line]}
    # 【解释】这个节点只改 logs，别的字段不动 —— 这就是"返回增量"的好处。


def report(state: State) -> dict:
    # 【解释】收尾节点：把记录本里攒的东西变成一句人话。

    if state["issues"]:
        verdict = "❌ 不建议起飞：" + "；".join(state["issues"])
        # 【解释】join 把问题列表拼成一句话，用分号隔开。
    else:
        verdict = "✅ 电池状态良好，可以起飞"

    print("  [report ] %s" % verdict)

    return {
        "verdict": verdict,
        "logs": state["logs"] + ["  [report ] 出结论"],
    }


# ════════════════════════════════════════════════════════════════
# 第 4 部分：条件边 —— 图的"方向盘"
# ════════════════════════════════════════════════════════════════
# 普通边：A → B，走到 A 就一定去 B（没有选择）。
# 条件边：走到 A，先问一个函数"下一步去哪"，函数返回字符串 = 目标节点名。
#
# ⭐ 循环就是这么来的：条件函数返回"回到 A"，图就绕回去了。


def route_after_check(state: State) -> str:
    # 【解释】每次 check 跑完，都来问这个函数一次。

    if state["checklist"]:
        # 【解释】清单还有剩 → 返回 "check" → 回到 check 节点 → 形成循环！
        return "check"

    # 清单空了，看有没有查出问题
    if state["issues"]:
        return "consult_manual"
        # 【解释】有问题 → 去查手册。

    return "report"
    # 【解释】没问题 → 直接出结论（跳过查手册）。


def route_after_manual(state: State) -> str:
    # 【解释】查完手册，无条件去出结论。
    #         这里其实可以用普通边，但写成函数你能看清"返回值决定去向"这件事。
    return "report"


# ════════════════════════════════════════════════════════════════
# 第 5 部分：把零件拼成图
# ════════════════════════════════════════════════════════════════

def build_graph():
    # 【解释】画布：告诉 LangGraph 你的 State 长什么样。
    builder = StateGraph(State)

    # ── 加节点：名字 → 函数 ──
    builder.add_node("check", check)
    # 【解释】节点的名字（字符串）随便起，但条件边返回的字符串必须和它一模一样。
    #         对不上就报错，这是最常见的低级错误。

    builder.add_node("consult_manual", consult_manual)
    builder.add_node("report", report)

    # ── 连边 ──
    builder.add_edge(START, "check")
    # 【解释】入口 → 第一个节点。START 是特殊的，不是真节点。

    builder.add_conditional_edges(
        "check",
        # 【解释】第一个参数：从哪个节点出发。

        route_after_check,
        # 【解释】第二个参数：问路的函数。它返回什么，就去哪。

        {
            "check": "check",
            # 【解释】映射表：函数返回值 → 目标节点名。
            #         这里的 "check" → "check" 就是**循环**（自己指回自己）。

            "consult_manual": "consult_manual",
            "report": "report",
        },
    )

    builder.add_edge("consult_manual", "report")
    # 【解释】普通边：查完手册一定去出结论。

    builder.add_edge("report", END)
    # 【解释】出结论就是终点。

    return builder.compile()
    # 【解释】compile() = 把画好的图"编译"成可执行对象。
    #         编译时会检查：有没有走不到的节点？返回的名字对不对得上？
    #         有问题这里就报错，不会等到运行时才崩。


# ════════════════════════════════════════════════════════════════
# 第 6 部分：main() —— 跑两个场景，看清"路径不一样"
# ════════════════════════════════════════════════════════════════

def make_battery(cells, temp, soc, min_total=10.5):
    # 【解释】造一份电池遥测数据。3S 电池，满电 4.2V×3=12.6V，
    #         放到 3.5V×3=10.5V 就该降落了。
    return {
        "cell_voltages": cells,
        "temperature": temp,
        "soc_percent": soc,
        "min_total_voltage": min_total,
    }


def run_case(title, battery):
    # 【解释】跑一个场景，打印输入和最终结论。

    print()
    print("=" * 64)
    print("  %s" % title)
    print("=" * 64)
    print("  电池：%.2fV/%.2fV/%.2fV · %d℃ · 剩余 %d%%" % (
        battery["cell_voltages"][0],
        battery["cell_voltages"][1],
        battery["cell_voltages"][2],
        battery["temperature"],
        battery["soc_percent"]))
    print("  " + "-" * 60)

    graph = build_graph()
    # 【解释】每次都重新建图（图是轻量的，不用复用）。

    result = graph.invoke({
        # 【解释】invoke = 启动！参数是初始 State（记录本的初始内容）。
        "battery": battery,
        "checklist": ["电芯电压一致性", "总电压", "温度", "剩余容量"],
        # 【解释】4 个检查项 → 你会看到 check 节点被访问 4 次。
        "issues": [],
        "logs": [],
        "verdict": "",
    })

    print("  " + "-" * 60)
    print("  结论：%s" % result["verdict"])
    # 【解释】result 就是跑完之后的完整记录本，能取到任何字段。
    print("  共访问节点 %d 次（含循环）" % (len(result["logs"])))
    return result


def main():
    print()
    print("╔" + "═" * 62 + "╗")
    print("║" + " " * 14 + "Day 21-A：LangGraph 的骨架" + " " * 23 + "║")
    print("╚" + "=" * 62 + "╝")
    print()
    print("  重点看两件事：")
    print("    ① check 节点会打印 4 次 —— 同一个节点反复执行 = 循环")
    print("    ② 场景二比场景一多走了 consult_manual —— 条件边在分叉")

    # ── 场景一：健康电池 → 走最短路径 ──
    run_case(
        "场景一：健康电池（4 项全过，直奔结论）",
        make_battery([3.85, 3.86, 3.85], 25, 88))
    # 【解释】三片电压几乎一样（压差 0.01V），温度正常，电量充足。

    # ── 场景二：问题电池 → 多走一步查手册 ──
    run_case(
        "场景二：老化电池（压差 + 低温，触发查手册）",
        make_battery([3.92, 3.71, 3.88], 2, 55))
    # 【解释】压差 0.21V 超标，温度 2℃ 过低 → 会查出 2 个问题 → 走查手册分支。

    # ── 场景三：临界电量 ──
    run_case(
        "场景三：电量不足（只有一项不过）",
        make_battery([3.72, 3.73, 3.72], 28, 18))
    # 【解释】前三项都过，只有电量 18% 不达标，但依然会触发查手册。

    print()
    print("=" * 64)
    print("  对照：Day13 你手写的循环 vs 今天的图")
    print("=" * 64)
    print("""
    Day13（手写）：
        messages = [问题]
        msg = 问模型(问题 + 工具说明书)
        if msg 要调工具:
            执行工具，追加结果
            return 问模型(messages)      ← 直接 return，死路一条
        else:
            return msg.content

    问题：模型看完工具结果说"我还得再查一个" → 做不到。
          要支持就得写 while True，然后自己管：
          循环几次退出？出错了怎么办？中间状态存哪？

    Day21（图）：
        check → 条件边 → check（回来）/ report（出去）
                    ↑
              返回 "check" 就绕回去，返回 "report" 就出去
              绕几次由状态决定，不用你写 while

    框架替你扛的：循环终止、状态传递、路径记录、出错回滚。
    你只管写：每个节点干什么 + 下一步去哪。
    """)

    print()
    print("  ➡️  下一步：Day21-B 把真模型接进来，")
    print("      让「下一步去哪」由模型说了算（而不是今天的 if/else）。")
    print()


if __name__ == "__main__":
    main()
