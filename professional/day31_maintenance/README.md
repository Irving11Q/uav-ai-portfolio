# Day31 — 运维智能体闭环（W7 综合大项目 · Day 1）

**把「板子报警」自动变成「处置单」：感知(实时告警) → 决策(查手册/RAG + 分析数据) → 行动(出结构化处置单) 的端到端闭环。**

这是 W6 收官之后的第一个综合大项目。前面 30 天把「板子 → 服务 → 屏幕 → 部署 → 评测」全链路做完了，Day31 把它们**串成一个会自己干活的 Agent**——实时巡检（Day28）一旦告警，运维智能体自动调 Day25 的《电池手册》检索 + 飞行数据分析，生成一份带根因 / 处置步骤 / 风险等级的处置单。

```
无人机板子
   │ 串口/合成帧
   ▼
Day28 实时飞行巡检服务  ── /live/alerts（已发生告警）
   │                      ── /live/status（续航预测→预测性预警）
   ▼
Day31 运维智能体（本目录）
   │  generate_order(alert)
   ├── 查手册  ─────────────►  Day25 /manual/search   (RAG 检索《电池维护手册》)
   ├── 看数据  ─────────────►  Day25 /data/filter      (pandas 真算)
   ▼
结构化处置单：{ 异常类型, 根因, 处置步骤[], 风险等级, 手册引用[], 数据佐证 }
```

## 文件

| 文件 | 职责 | 需要 Qt |
|---|---|---|
| `day31a_maintenance_agent.py` | 智能体核心：Backend 抽象(Fake/Http) · 指标解析 · 规则引擎(无 Key 降级) · LLM 模式(LangGraph Agent) · 统一入口 `MaintenanceAgent.generate_order` | 否 |
| `day31b_orchestrator.py` | 编排层：`MaintenanceLoop` 轮询 Day28 告警触发 Agent · `run_selftest` 进程内起 Day25+Day28 真跑闭环 · `run_demo` 起真实端口常驻演示 | 否 |

> 拆分沿用 Day29 的思路：**能脱服务/LLM 测的逻辑（a）** 和 **薄薄的接线层（b）** 分开，评测不漂。

## 怎么跑

```bash
# 纯逻辑自测（规则引擎 + 工具 schema，零网络/零 LLM，秒回）
python day31a_maintenance_agent.py --selftest

# 完整链路自测：进程内起 Day25 + Day28，真跑一遍闭环（约 30 秒，含 bge 索引预热）
python day31b_orchestrator.py --selftest

# 常驻演示：起真实端口 Day25:8000 / Day28:8001，给面试官现场看（Ctrl-C 退出）
python day31b_orchestrator.py --demo
```

## 自测结果（本目录内实跑，10/10）

```
✅ Day25 /health
✅ Day28 /health
✅ Day28 start(synth)
🔮 预测预警→处置单[电压]：预测约 71 秒后电压跌破 23.0V
🛠 告警→处置单[温度]：温度 50.6℃ 超过预警线 50℃
✅ synth 触发温度告警→处置单
✅ 处置单含根因
✅ 处置单含步骤≥2
✅ 手册引用真查 Day25   →「4. 温度管理：锂电池最佳工作温度是 20℃~4…」（真实手册段落）
✅ 风险等级非空
✅ 预测性预警分支→处置单
✅ Day28 stop
```

两条触发路径都验证：`已发生告警`（温度超预警线）→ 处置单，以及 `预测性预警`（续航模型算出即将跌破安全线）→ 处置单。手册引用是 Day25 真实 RAG 检索返回的段落，闭环不假。

## 三个能讲的设计点

1. **闭环真的落地成一个常驻进程**：不是一次性脚本。`MaintenanceLoop.poll_once()` 反复读 Day28 快照，有新告警就触发 Agent。感知 → 决策 → 行动 在这里合龙。
2. **同指标告警去重**：合成数据一秒刷几十条温度告警，每条都调一次检索既慢又刷屏。已发生告警按 frame 去重，同类指标只出一份处置单。
3. **优雅降级（生产该有的韧性）**：没有 API Key 时，规则引擎照样出一份能用的处置单（根因/步骤/风险来自映射表，手册引用和数据佐证仍真去查）。LLM 模式（有 Key）是同结构的增强，不是替代品——两套模式共用同一份输出 schema，评测不漂。

## 依赖与诚实标注

- 运行：`fastapi` `uvicorn` `requests` `pandas`；LLM 模式还需 `langchain` `langgraph` 与 API Key（OpenAI 兼容）。
- 本目录已自包含：`day25`(检索/分析服务) + `day28`(实时巡检) + `day27`(规则/预测) + `day23a`(数据分析) + `day21b` + `api_config` + 手册 + `w2_comms` 数据。
- **实测范围**：本机未注入 LLM 环境变量且 `langgraph` 未装，因此 **LLM Agent 模式（模型自主调工具）未在本环境真跑**，仅验证了「无 Key 规则闭环」10/10（手册检索为真实 RAG 结果）。LLM 模式代码就绪，需在配置 Key + 安装 langgraph 的环境跑 `--demo` 验证。这不影响闭环架构的演示价值——规则模式已完整跑通感知→查资料→出单。

> 延续 Day30 的「评测化」：Day31 的闭环同样可在 Day30 的 `day30_eval.py` 框架里加一组「告警→处置单」用例，把它变成可回归的评测。
