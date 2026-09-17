"""对话导出：Markdown 与 PDF。

## 为什么 PDF 不用「浏览器打印」
浏览器打印确实零依赖，但导出的不是**文件** —— 用户还得在打印面板里手动另存。
这里用 reportlab + 系统中文字体直接生成真 PDF 返给前端下载，一步到位。

## 中文字体
优先注册系统 `msyh.ttc`（微软雅黑），失败退 `simhei.ttf`，再失败退 reportlab 内置的
中文 CID 字体 `STSong-Light`（不依赖任何系统字体文件，保证 PDF 里中文不会变方块）。
"""
import io
import logging
import re
from datetime import datetime
from urllib.parse import quote

logger = logging.getLogger("rag")

_FONT_CANDIDATES = [
    (r"C:\Windows\Fonts\msyh.ttc", "MSYH"),
    (r"C:\Windows\Fonts\simhei.ttf", "SIMHEI"),
    (r"C:\Windows\Fonts\simsun.ttc", "SIMSUN"),
]

# 表格每列的最小可用宽度（pt）。低于它就说明列太多、版面撑不住，
# 退化成文本行渲染 —— reportlab 在负宽度时是直接抛异常，不是自动省略。
_MIN_COL_WIDTH = 28

_font_name: str | None = None


def _ensure_font() -> str:
    """注册一个可用的中文字体，返回字体名。"""
    global _font_name
    if _font_name:
        return _font_name
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    for path, name in _FONT_CANDIDATES:
        try:
            pdfmetrics.registerFont(TTFont(name, path))
            _font_name = name
            return name
        except Exception:  # noqa: BLE001
            continue
    # 兜底：reportlab 自带中文 CID 字体（无需系统字体文件）
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    _font_name = "STSong-Light"
    return _font_name


def _fmt_time(dt) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else ""


def _refs_block(references: list[dict]) -> list[str]:
    lines = []
    for i, r in enumerate(references, 1):
        snippet = (r.get("snippet") or "").replace("\n", " ").strip()
        lines.append(f"  [{i}] 来源：{r.get('source')}（相关度 {r.get('score')}）")
        lines.append(f"      {snippet}")
    return lines


def session_to_markdown(session, messages: list[dict]) -> str:
    """把一次会话渲染成 Markdown 文本。"""
    out: list[str] = []
    title = session.title or "新会话"
    out.append(f"# {title}")
    out.append("")
    out.append(f"> 会话 ID：{session.id}　创建时间：{_fmt_time(session.created_at)}")
    out.append(f"> 导出时间：{_fmt_time(datetime.now())}")
    out.append("")
    out.append("---")
    out.append("")

    for m in messages:
        if m["role"] == "user":
            out.append(f"## 用户提问")
            out.append("")
            out.append(m["content"].strip())
            out.append("")
        elif m["role"] == "assistant":
            out.append("## 助手回答")
            out.append("")
            out.append(m["content"].strip())
            out.append("")
            refs = m.get("references") or []
            if refs:
                out.append("**引用知识库片段：**")
                out.append("")
                out.extend(_refs_block(refs))
                out.append("")
            if m.get("feedback"):
                label = "👍 满意" if m["feedback"] == "up" else "👎 不满意"
                reason = f"（{m['feedback_reason']}）" if m.get("feedback_reason") else ""
                out.append(f"*用户反馈：{label}{reason}*")
                out.append("")
        out.append("---")
        out.append("")

    return "\n".join(out)


def content_disposition(filename: str) -> str:
    """生成兼容中文文件名的 Content-Disposition（同时给 ASCII 与 RFC5987 两种写法）。"""
    ascii_fallback = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
    return (
        f'attachment; filename="{ascii_fallback}"; '
        f"filename*=UTF-8''{quote(filename)}"
    )


# ---------------- Markdown -> PDF ----------------

_INLINE_RULES = [
    (re.compile(r"\*\*(.+?)\*\*"), r"<b>\1</b>"),
    (re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)"), r"<i>\1</i>"),
    (re.compile(r"`(.+?)`"), r"<font face=\"Courier\">\1</font>"),
]


def _escape(text: str) -> str:
    """先转义 XML 特殊字符，再套用允许的内联标记（顺序不能反，否则标记会被转义掉）。"""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for pattern, repl in _INLINE_RULES:
        text = pattern.sub(repl, text)
    return text


def _is_table_row(line: str) -> bool:
    return line.strip().startswith("|") and line.strip().endswith("|")


def _parse_table_row(line: str) -> list[str]:
    cells = line.strip().strip("|").split("|")
    return [c.strip() for c in cells]


def _is_separator_row(line: str) -> bool:
    cells = _parse_table_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", c or "") for c in cells)


def markdown_to_pdf_bytes(title: str, markdown_text: str) -> bytes:
    """把 Markdown 渲染成 PDF。支持标题/段落/列表/加粗/表格 —— 覆盖本项目实际产出的内容形态。"""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    font = _ensure_font()

    body = ParagraphStyle(
        "body", fontName=font, fontSize=10, leading=15, spaceAfter=4
    )
    h1 = ParagraphStyle("h1", fontName=font, fontSize=18, leading=24, spaceAfter=8)
    h2 = ParagraphStyle("h2", fontName=font, fontSize=13, leading=19, spaceBefore=8, spaceAfter=5)
    h3 = ParagraphStyle(
        "h3", fontName=font, fontSize=11, leading=16, spaceBefore=6, spaceAfter=4
    )
    quote = ParagraphStyle(
        "quote", fontName=font, fontSize=9, leading=13, textColor=colors.HexColor("#666666")
    )
    bullet = ParagraphStyle(
        "bullet", fontName=font, fontSize=10, leading=15, leftIndent=12, bulletIndent=4
    )
    code = ParagraphStyle(
        "code", fontName=font, fontSize=9, leading=13, textColor=colors.HexColor("#444444")
    )
    style_map = {"#": h1, "##": h2, "###": h3, "####": h3, "#####": h3, "######": h3}

    # SimpleDocTemplate 会把内容写进传入的 file-like 对象，所以先给一个 BytesIO
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=title,
    )

    flow: list = []
    lines = markdown_text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # 表格块：连续 |...| 行整体成一个 reportlab Table
        if _is_table_row(line):
            block = []
            while i < len(lines) and _is_table_row(lines[i]):
                block.append(lines[i])
                i += 1
            rows = [
                _parse_table_row(r) for r in block if not _is_separator_row(r)
            ]
            if rows:
                width = max(len(r) for r in rows)
                rows = [r + [""] * (width - len(r)) for r in rows]
                available = A4[0] - 36 * mm
                # 列数太多时 reportlab 会算出负的可用宽度并直接抛异常
                # （实测：一行被解析出 77 列 → availWidth=-3.59 → ValueError）。
                # 知识库里的宽表、或误判成表格的长行都可能触发，所以这里兜底：
                # 每列窄于 MIN_COL_WIDTH 就退化成文本行，宁可不好看也不能导出失败。
                if available / width < _MIN_COL_WIDTH:
                    for row in rows:
                        flow.append(Paragraph(_escape(" | ".join(row)), body))
                    flow.append(Spacer(1, 6))
                    continue
                data = [[Paragraph(_escape(c), body) for c in r] for r in rows]
                table = Table(
                    data,
                    colWidths=[available / width] * width,
                    repeatRows=1,
                )
                table.setStyle(
                    TableStyle(
                        [
                            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f4f7")),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("TOPPADDING", (0, 0), (-1, -1), 4),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                            ("LEFTPADDING", (0, 0), (-1, -1), 5),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                        ]
                    )
                )
                flow.append(table)
                flow.append(Spacer(1, 6))
            continue

        if not stripped:
            flow.append(Spacer(1, 5))
            i += 1
            continue

        if stripped == "---":
            flow.append(Spacer(1, 8))
            i += 1
            continue

        # 标题
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            style = style_map.get(m.group(1), body)
            flow.append(Paragraph(_escape(m.group(2)), style))
            i += 1
            continue

        # 引用
        if stripped.startswith(">"):
            flow.append(Paragraph(_escape(stripped.lstrip("> ").strip()), quote))
            i += 1
            continue

        # 无序列表
        if re.match(r"^[-*+]\s+", stripped):
            flow.append(
                Paragraph(_escape(re.sub(r"^[-*+]\s+", "", stripped)), bullet, bulletText="•")
            )
            i += 1
            continue

        # 有序列表
        om = re.match(r"^(\d+)[.)]\s+(.*)$", stripped)
        if om:
            flow.append(
                Paragraph(_escape(om.group(2)), bullet, bulletText=f"{om.group(1)}.")
            )
            i += 1
            continue

        # 缩进代码/片段行（引用片段缩进两个空格）
        if line.startswith("  "):
            flow.append(Paragraph(_escape(stripped), code))
            i += 1
            continue

        flow.append(Paragraph(_escape(stripped), body))
        i += 1

    doc.build(flow)
    return buf.getvalue()
