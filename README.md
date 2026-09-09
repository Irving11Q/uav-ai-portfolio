# uav-ai-portfolio · 无人机 AI 应用开发作品集

> 一个研究生从零自学 **AI 应用开发** 的全程记录：从 PySide6 桌面界面、串口上位机，
> 一路走到大模型 API、RAG 知识库、数据分析 Agent、多 Agent 协作。
> 每个 `day` 都是**可运行、自包含**的最小例子，配英文 `README` 讲清「为什么这样写」。

**目标岗位**：AI 应用 / AI 工程岗（非算法岗）—— 考 RAG 全链路、Agent、工程落地、效果评测。
**技术栈**：Python · PySide6 · pyserial · requests · DeepSeek / 智谱 GLM（OpenAI 兼容）·
sentence-transformers(bge) · Chroma · LangChain · LangGraph · pandas · matplotlib。

---

## 这条学习线怎么读（8 周 → 4 个阶段）

| 阶段 | 目录 | 解决什么问题 | 一句话成果 |
|---|---|---|---|
| **W1 界面** | `day1` `day4` `day5` `day6` `day8` `day9` | 桌面程序怎么搭 | PySide6 控件/布局/信号槽、QTimer、QThread 不卡 UI、实时曲线 |
| **W2 上位机** | `day7` `day8` | 怎么从硬件拿数据 | 串口读飞行参数（电压/电流/温度/高度）、解析、实时绘图 |
| **W3 大模型 API** | `day10`–`day14` | 怎么调 LLM | 首调 DeepSeek → 自封装 Client → Function Calling（Agent 基石）→ 多模型对比 |
| **W4 RAG 全链路** | `day15`–`day20` | 怎么让模型「不瞎编」 | 朴素 RAG → 切块策略 → Chroma 向量库 → 检索调优 → 带引用的 QA → LangChain 重写 |
| **W5 Agent** | `day21`–`day24` | 怎么让模型「自己干活」 | 单 Agent → 工具即检索(CRAG) → 数据分析 Agent → 多 Agent 协作 |

**贯穿线索**：W2 那 23 行真实串口飞行日志、W4 那本《电池手册》，被 W3/W4/W5 反复复用——
同一个数据，从「画出来」到「喂给模型回答」，这就是一个 AI 应用从原型到落地的全过程。

---

## 面试最该讲的 4 个项目（深度优先）

按「讲透 > 学新」原则，下面 4 个是可以展开讲 20 分钟以上的：

1. **`day23_data_agent` + `day24_multi_agent`（W5）** — 数据分析 / 多 Agent。
   核心论点：**LLM 算数不可靠、但「决定算什么」很擅长**；对照组实验证明模型数数会错(3)、工具算数对(5)；
   同一张 LangGraph 图被复用 4 次，换工具=换应用；多 Agent = 调度者 `.invoke(worker)` 合并消息。
2. **`day22_agentic_rag`（W5）** — Agentic RAG / CRAG。
   核心论点：检索从「写死」变「模型决定」，加 guard 节点强制检索堵住「模型跳过工具瞎编」的洞。
3. **`day17`–`day19`（W4）** — RAG 全链路 + 检索评测。
   核心论点：切块质量决定检索上限；守住统一检索接口，换真检索只改一行；实测 Top-1 5/6、Top-3 6/6、离题拒答 3/3。
4. **`day13_function_calling（W3）** — Function Calling 是 Agent 的基石。
   核心论点：模型决定何时调、调哪个、传什么参；你的代码执行后把结果喂回。

---

## 每个 day 在 `professional/` 下都是自包含的

每个 `dayXX_*/` 目录带着自己需要的依赖副本（如 `api_config.py`、数据文件、`dayNN_*.py`），
**拷出来 `python dayXX_xxx.py` 就能跑**，不依赖仓库其他部分。带 Key 走模型、不带 Key 走降级演示。

环境（学习用）：`langgraph 1.2.11` · `langchain-core 1.6.1` · `chromadb 1.5.9` ·
`sentence-transformers 6.0.1` · `pandas` · `matplotlib` · `PySide6 6.11` · `pyserial`。

---

## 目录速查

```
professional/
├── day1_controls/      day4_timer_thread/   day5_panel/      day6_panel/      day8_plot/      day9_monitor/   # W1 界面
├── day7_serial/        day8_plot/                                                            # W2 串口+绘图
├── day10_llm/          day11_api_client/     day12/           day13_function_calling/  day14_multi_model/  # W3 大模型
├── day15_rag/          day16_doc_parsing/    day17_vector_db/ day18_retrieval_tuning/  # W4 RAG
│   day19_rag_qa_system/  day20_langchain_rag/                                                 # W4 RAG
└── day21_langgraph_agent/  day22_agentic_rag/  day23_data_agent/  day24_multi_agent/         # W5 Agent
```

每个目录里的 `README.md` 都是独立的「这一天的来龙去脉」。

---

## 配套资料

- `面试讲解稿.md` — 每个项目能讲清什么、踩过什么坑（带数字）、面试官可能怎么追问。
- 微信小程序「小店天天有客」（第八届中国研究生人工智能创新大赛参赛作品，已提交）。
- STM32（洋桃1号）+ PySide6 上位机 + 串口 的嵌入式全链路经验。
