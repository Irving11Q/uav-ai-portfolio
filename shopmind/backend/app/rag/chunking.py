"""结构感知的中文文本分块。

## 两种模式（自动判断）

先看文本是不是 Markdown（含 `#` 标题或 `|` 表格）：

**A. Markdown 文本 → 结构感知切分**
1. 先按 `#` / `##` / `###` 切节，**标题保留在块内容里**——单独检索出来时上下文完整，
   大模型也能知道这段话属于哪一节。
2. **表格视为原子块**：连续的 `|` 行整体成块；表格若长到必须拆开，**每一块都会带上表头**，
   杜绝「只有数据、没有列名」的片段。
3. 仍然超长的正文再用递归字符切分（中文分隔符优先）。
4. 过短的碎块合并进相邻块，避免低信息量片段挤占召回名额。

**B. 普通文本 → 中文递归字符切分**（与改造前行为一致）
从粗到细逐级降级：`\n\n` → `\n` → `。！？` → `；，、` → 空格 → 硬切，
相邻块保留 overlap 重叠，避免跨块的句子两边都不完整。

块大小与重叠来自 config，可在 `.env` 调。
"""
import re

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from ..config import settings

# 中文句子/段落常用分隔符优先，再退到空格与字符级
_CHINESE_SEPARATORS = ["\n\n", "\n", "。", "！？", "；", "，", "、", " ", ""]

# 小于此长度的块尝试并进相邻块（碎块 embedding 信息量低）
_MIN_CHUNK_CHARS = 100
# 合并后允许超出 chunk_size 的比例（略超换来更完整的语义，是划算的）
_MERGE_MAX_RATIO = 1.5

_MD_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]
_MD_HEADING_RE = re.compile(r"^#{1,6}\s+\S", re.M)
_MD_TABLE_RE = re.compile(r"^\s*\|.*\|", re.M)


def split_text(text: str, chunk_size: int | None = None, overlap: int | None = None):
    """把长文本切成若干块，过滤空块。

    Args:
        text: 待切分的文本（Markdown 或纯文本）。
        chunk_size: 单块目标长度（字符），默认取配置。
        overlap: 相邻块重叠长度，默认取配置；传 0 表示不重叠。
    """
    # 用 None 判断而非 `or`，否则显式传 0 会被当成假值而被默认值顶掉
    if chunk_size is None:
        chunk_size = settings.CHUNK_SIZE
    if overlap is None:
        overlap = settings.CHUNK_OVERLAP

    text = (text or "").strip()
    if not text:
        return []

    if _looks_like_markdown(text):
        chunks = _split_markdown(text, chunk_size, overlap)
    else:
        chunks = _recursive_split(text, chunk_size, overlap)

    chunks = _merge_small_chunks(chunks, chunk_size)
    return [c.strip() for c in chunks if c and c.strip()]


# ------------------------------------------------------------ 模式判断


def _looks_like_markdown(text: str) -> bool:
    """出现 Markdown 标题或表格，就按结构化文本处理。"""
    return bool(_MD_HEADING_RE.search(text) or _MD_TABLE_RE.search(text))


def _build_md_splitter() -> MarkdownHeaderTextSplitter:
    try:
        return MarkdownHeaderTextSplitter(
            headers_to_split_on=_MD_HEADERS, strip_headers=False
        )
    except TypeError:  # pragma: no cover - 兼容没有 strip_headers 参数的旧版本
        return MarkdownHeaderTextSplitter(headers_to_split_on=_MD_HEADERS)


_md_splitter = _build_md_splitter()


# ------------------------------------------------------------ A. Markdown 路径


def _split_markdown(text: str, chunk_size: int, overlap: int) -> list[str]:
    """按标题切节，节内再做表格保护与长度控制。"""
    out: list[str] = []
    for section in _md_splitter.split_text(text):
        body = (section.page_content or "").strip()
        if not body:
            continue
        if len(body) <= chunk_size:
            out.append(body)
        else:
            out.extend(_split_section(body, chunk_size, overlap))
    return out


def _split_section(body: str, chunk_size: int, overlap: int) -> list[str]:
    """节内切分：表格整体成块，正文走递归字符切分。"""
    out: list[str] = []
    for is_table, lines in _group_table_lines(body):
        content = "\n".join(lines).strip()
        if not content:
            continue
        if len(content) <= chunk_size:
            out.append(content)
        elif is_table:
            out.extend(_split_table(lines, chunk_size))
        else:
            out.extend(_recursive_split(content, chunk_size, overlap))
    return out


def _group_table_lines(text: str) -> list[tuple[bool, list[str]]]:
    """按「表格行 / 普通行」分组，保持原有顺序。"""
    groups: list[tuple[bool, list[str]]] = []
    buf: list[str] = []
    is_table: bool | None = None
    for line in text.split("\n"):
        cur = _is_table_line(line)
        if is_table is None or cur == is_table:
            buf.append(line)
            is_table = cur
        else:
            groups.append((bool(is_table), buf))
            buf, is_table = [line], cur
    if buf:
        groups.append((bool(is_table), buf))
    return groups


def _is_table_line(line: str) -> bool:
    """Markdown 表格行：以 | 开头且至少有 2 个竖线。"""
    s = line.strip()
    return s.startswith("|") and s.count("|") >= 2


def _split_table(lines: list[str], chunk_size: int) -> list[str]:
    """拆长表格：**每块都重复表头**（列名行 + --- 分隔行）。

    这是「续块无表头」问题的根治点——只带数据行、没有列名的片段，
    无论对检索还是对大模型理解都是废块。
    """
    header = lines[:2] if len(lines) >= 3 else lines[:1]
    rows = lines[len(header) :]
    if not rows:
        return ["\n".join(lines)]

    base_len = len("\n".join(header)) + 1
    out: list[str] = []
    buf: list[str] = []
    cur = base_len
    for row in rows:
        if buf and cur + len(row) + 1 > chunk_size:
            out.append("\n".join(header + buf))
            buf, cur = [], base_len
        buf.append(row)
        cur += len(row) + 1
    if buf:
        out.append("\n".join(header + buf))
    return out


def _merge_small_chunks(chunks: list[str], chunk_size: int) -> list[str]:
    """把过短的块并进相邻块，保持原有顺序；合并上限为 chunk_size * 1.5。"""
    if not chunks:
        return []
    limit = int(chunk_size * _MERGE_MAX_RATIO)
    merged: list[str] = []
    for c in chunks:
        if (
            merged
            and len(c) < _MIN_CHUNK_CHARS
            and len(merged[-1]) + len(c) + 1 <= limit
        ):
            merged[-1] = merged[-1] + "\n" + c
        else:
            merged.append(c)
    # 末块过短时也尝试并入前一块（否则会剩一个孤零零的碎片）
    if len(merged) > 1 and len(merged[-1]) < _MIN_CHUNK_CHARS:
        if len(merged[-2]) + len(merged[-1]) + 1 <= limit:
            merged[-2] = merged[-2] + "\n" + merged.pop()
    return merged


# ------------------------------------------------------------ B. 纯文本路径


def _recursive_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """中文友好的递归字符切分：能保住语义边界就保住。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=_CHINESE_SEPARATORS,
        keep_separator=True,
    )
    return [c for c in splitter.split_text(text) if c.strip()]
