"""
Day 16：文档解析与切块进阶 —— 把"假数据"换成"真文件"

演示如何把一份真实的无人机电池手册（markdown 文件）读进内存，
并用三种切块策略切成小块，最后持久化为 JSON：
- 生成/读取真实手册文件（处理编码、换行、空行）
- 策略 A：按段落切（每段一块）
- 策略 B：固定窗口切（每 N 字一块 + 重叠，缓解切断语义）
- 策略 C：按标题切（按 ## 章节，块自带标题，检索最友好）
- 用字频向量 + 余弦相似度验证：哪种切法检索命中更准
- 把切块结果存成 chunks_parsed.json（为 Day17 向量数据库入库做准备）

完全离线运行，不需要 API Key。
"""

import os
import re
import json

# ── 内置一份"真实"手册（markdown 格式，可整体替换成自己的手册）──
MANUAL_MD = """# 无人机锂电池使用手册

## 1. 电池规格

本机采用 3S 锂电池，单节额定电压 3.7V，整机额定电压 11.1V。

电池容量 5200mAh，放电倍率 20C，最大持续放电电流 104A。

电池重量约 420g，工作温度范围 -10℃~60℃。

## 2. 电压管理

电压低于 9.6V 时必须尽快降落，否则电池会因过放而损坏。

单节电压低于 3.2V 属于深度过放，会永久损伤电芯。

飞控应实时监控电压，低电压报警阈值建议设为 10.2V。

## 3. 电流与功耗

悬停状态下整机电流约 12A，全速爬升时电流可达 28A。

电流越大，电池消耗越快，续航越短。

额定功耗：悬停约 133W，最大功率约 310W。

## 4. 温度管理

锂电池最佳工作温度是 20℃~40℃。

若温度超过 55℃，应尽快降落散热；超过 60℃ 会加速老化，甚至鼓包起火。

低温环境（低于 0℃）时电池内阻增大，续航会明显下降。

## 5. 续航估算

理论续航 = 容量 ÷ 平均电流 ≈ 25 分钟。

风速大或负载重时续航明显缩短；低温也会缩短续航。

实际飞行建议预留 20% 电量返航。

## 6. 充电与维护

建议 1C 电流充电（即 5.2A），禁止过放。

长期存放时把电池充到单节 3.8V（存储电压），每 3 个月补电一次。

电池鼓包、漏液应立即停止使用并妥善处理。

## 7. 安全预警

若飞行中电压跌落过快，或电池温度骤升超过 50℃，应触发返航指令。

起飞前检查电池外观、插头接触和电量，确保安全。
"""

# 手册文件与输出文件放在本脚本同目录
HERE = os.path.dirname(os.path.abspath(__file__))
MANUAL_FILE = os.path.join(HERE, "uav_battery_manual.md")
JSON_FILE = os.path.join(HERE, "chunks_parsed.json")


# ── 文档解析：生成文件 + 读取文件 ──

def ensure_manual_file():
    """文件不存在则用内置内容生成一份，方便开箱即跑。"""
    if os.path.exists(MANUAL_FILE):
        return
    with open(MANUAL_FILE, "w", encoding="utf-8") as f:
        f.write(MANUAL_MD)
    print(f"已生成示例手册：{MANUAL_FILE}")


def read_manual(path):
    """读取手册文件，统一换行符并去掉首尾空白，返回干净纯文本。"""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    return text.replace("\r\n", "\n").strip()


# ── 三种切块策略 ──

def chunk_by_paragraph(text):
    """策略 A：按段落切（\n\n 为分隔符）。语义完整但块大小不均。"""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def chunk_by_window(text, size=60, overlap=15):
    """策略 B：固定窗口切。每 size 字符一块，块间重叠 overlap 字符。"""
    chunks = []
    step = size - overlap
    i = 0
    while i < len(text):
        piece = text[i : i + size]
        if piece.strip():
            chunks.append(piece.strip())
        i += step
    return chunks


def chunk_by_heading(text):
    """策略 C：按 markdown 标题切（## 章节）。块自带标题，检索最友好。"""
    chunks, current = [], []
    for line in text.split("\n"):
        if re.match(r"^#{1,3}\s", line):
            if current:
                chunks.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        chunks.append("\n".join(current).strip())
    return chunks


# ── 检索验证：字频向量 + 余弦相似度（Day15 的最小复刻，自包含）──

def build_vocab(chunks):
    """收集所有块中出现过的非空白字符，组成排序词典。"""
    vocab = set()
    for chunk in chunks:
        for ch in chunk:
            if ch.strip():
                vocab.add(ch)
    return sorted(vocab)


def make_vector(text, vocab):
    """把文字变成与词典对齐的字频向量。"""
    counts = {}
    for ch in text:
        if ch in vocab:
            counts[ch] = counts.get(ch, 0) + 1
    return [counts.get(ch, 0) for ch in vocab]


def cosine_sim(a, b):
    """余弦相似度：越接近 1 越相似。"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def top1(question, chunks):
    """返回问题和最相关块的 (相似度, 块文本)。"""
    vocab = build_vocab(chunks)
    qv = make_vector(question, vocab)
    best_sim, best = -1, ""
    for c in chunks:
        sim = cosine_sim(qv, make_vector(c, vocab))
        if sim > best_sim:
            best_sim, best = sim, c
    return best_sim, best


# ── 持久化：切块结果存 JSON（Day17 Chroma 入库的输入）──

def save_chunks_json(chunks, strategy, source):
    """把切块结果写成带 id/text/source/strategy 的 JSON 文件。"""
    records = [
        {"id": i, "text": c, "source": source, "chunk_strategy": strategy}
        for i, c in enumerate(chunks)
    ]
    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"已保存切块结果：{JSON_FILE}（{len(records)} 条）")


def main():
    print("=" * 60)
    print("Day 16：文档解析与切块进阶 —— 无人机电池手册")
    print("=" * 60)

    ensure_manual_file()
    text = read_manual(MANUAL_FILE)
    print(f"① 文档解析完成：{len(text)} 字符，{text.count(chr(10)) + 1} 行")

    para_chunks = chunk_by_paragraph(text)
    win_chunks = chunk_by_window(text, size=60, overlap=15)
    head_chunks = chunk_by_heading(text)

    print("\n② 三种切块策略对比：")
    for name, chunks in (("A 按段落", para_chunks), ("B 固定窗口", win_chunks), ("C 按标题", head_chunks)):
        lens = [len(c) for c in chunks]
        print(f"   {name}: {len(chunks)} 块｜平均 {sum(lens)//len(lens)} 字｜最短 {min(lens)}｜最长 {max(lens)}")

    question = "电池温度太高会怎样？"
    print(f"\n③ 检索验证（问题：{question}）：")
    for name, chunks in (("A 按段落", para_chunks), ("B 固定窗口", win_chunks), ("C 按标题", head_chunks)):
        sim, hit = top1(question, chunks)
        print(f"   {name}: 相似度 {sim:.3f}｜{hit[:45]}...")

    print("\n④ 持久化切块结果（用最推荐的标题切法）：")
    save_chunks_json(head_chunks, "by_heading", os.path.basename(MANUAL_FILE))

    print("\n" + "=" * 60)
    print("💡 结论：标题切块保留章节结构、块自带标题，检索最友好；")
    print("   真实系统常组合使用：先按标题切，超长章节再按窗口二次切。")
    print("=" * 60)


if __name__ == "__main__":
    main()
