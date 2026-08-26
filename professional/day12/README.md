# Day 12: Prompt Engineering

通过 4 个实验学习控制大模型输出的技巧。

## 代码文件

| 文件 | 说明 |
|---|---|
| `day12_prompt_engineering.py` | 4 个 Prompt 技巧演示：system prompt、few-shot、JSON 输出、temperature |

## 运行

```bash
cd w3_ai
python day12_prompt_engineering.py
```

会依次运行 4 个对比实验，每个都输出提示词和模型回答。

## 学到的技巧

1. **System Prompt** - 给模型定人设，回答更专业、有结构
2. **Few-shot** - 给 2-3 个例子，模型会照着格式输出
3. **JSON 输出** - 让模型返回结构化数据，程序直接解析
4. **Temperature** - 控制模型的发散程度（0=稳定，1.5=发散）

## 场景

用无人机飞行数据（电压/电流/温度/高度）做演示，接续第 2 周的串口数据格式。

---

*AI 应用开发学习第 3 周内容*