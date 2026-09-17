"""文档解析：把上传文件转成**文本供索引使用**，供后续分块与向量化。

注意这是「索引层」：用户上传的原始文件原样保存在 `data/uploads/`，不会被改动，
用户下载/查看拿到的仍是原始格式（如 .docx）。这里只决定「机器怎么读它」。

为让分块器能按结构切分（按标题切、表格不被切断），结构化格式统一转成 Markdown：
- .docx  → Markdown（保留标题层级 / 表格 / 列表）
- .csv / .xlsx → Markdown 表格
- .md    → 原样（本身就是 Markdown）
- .txt / .json / .pdf → 纯文本

支持的格式：txt / md / json / pdf / docx / csv / xlsx / 图片（OCR）

OCR 预处理（可选增强）：
- 图片（png/jpg/...）→ 直接走视觉模型 OCR 抠字，用于扫描版商品图册。
- 扫描版 PDF（pypdf 抽不出多少文字）→ 逐页渲染成图后 OCR。
- 电子版 PDF 仍走正常文本抽取，不会被拖去 OCR 白花时间与费用。
"""
import csv
import json
import logging
from pathlib import Path

from ..config import settings
from . import ocr as ocr_mod
from .to_markdown import docx_to_markdown, rows_to_markdown

logger = logging.getLogger("rag")


def parse_file_with_meta(path: str) -> tuple[str, dict]:
    """解析文件，返回 (文本, 元信息)。

    元信息形如 `{"used_ocr": bool, "note": str, "format": str}`，
    供调用方写审计日志与给前端提示「这篇是 OCR 出来的」。
    """
    suffix = Path(path).suffix.lower().lstrip(".")
    if ocr_mod.is_image(path):
        text = _read_image(path)
        return text, {"used_ocr": True, "format": suffix, "note": "图片经 OCR 识别"}

    dispatch = {
        "txt": _read_text,
        "md": _read_text,
        "text": _read_text,
        "json": _read_text,
        "csv": _read_csv,
        "xlsx": _read_xlsx,
        "docx": _read_docx,
    }
    if suffix == "pdf":
        return _read_pdf_with_meta(path)
    if suffix not in dispatch:
        raise ValueError(f"不支持的文件类型：.{suffix}")
    return dispatch[suffix](path), {"used_ocr": False, "format": suffix, "note": ""}


def parse_file(path: str) -> str:
    """只取文本（保持向后兼容的简便入口）。"""
    return parse_file_with_meta(path)[0]


def _read_text(path: str) -> str:
    # 优先 utf-8，失败再退到 gbk（中文 Windows 常见）
    for enc in ("utf-8", "gbk", "utf-8-sig"):
        try:
            return Path(path).read_text(encoding=enc)
        except (UnicodeDecodeError, LookupError):
            continue
    # 最后用二进制兜底
    raw = Path(path).read_bytes()
    return raw.decode("utf-8", errors="ignore")


def _read_csv(path: str) -> str:
    """CSV → Markdown 表格（首行作表头），使分块时表格保持完整。"""
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="", errors="ignore") as f:
        reader = csv.reader(f)
        for r in reader:
            rows.append([str(c) for c in r])
    return rows_to_markdown(rows)


def _read_xlsx(path: str) -> str:
    """每个工作表转成「## 工作表名 + Markdown 表格」。"""
    try:
        import openpyxl
    except ImportError as e:  # pragma: no cover
        raise ValueError("缺少依赖 openpyxl，无法解析 xlsx") from e
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    parts = []
    for ws in wb.worksheets:
        rows = [
            ["" if v is None else str(v) for v in row]
            for row in ws.iter_rows(values_only=True)
        ]
        table = rows_to_markdown(rows)
        if table:
            parts.append(f"## 工作表：{ws.title}\n\n{table}")
    return "\n\n".join(parts)


def _read_image(path: str) -> str:
    """图片 → OCR 文本（扫描版商品图册、截图型说明书的兜底路径）。"""
    if not settings.OCR_ENABLED:
        raise ValueError(
            "这是图片型文档，需要 OCR 才能提取文字，但当前 OCR 未开启（OCR_ENABLED=false）"
        )
    try:
        text = ocr_mod.ocr_image_file(path)
    except ocr_mod.OCRError as e:
        raise ValueError(f"图片 OCR 失败：{e}") from e
    return f"<!-- 来源：图片经 OCR 识别 -->\n{text}"


def _read_pdf_with_meta(path: str) -> tuple[str, dict]:
    """PDF 解析：先判断是否扫描版，再决定走文本抽取还是 OCR。"""
    need, reason = ocr_mod.needs_ocr(path)
    logger.info("PDF 判定: %s（%s）", "需要 OCR" if need else "普通文本抽取", reason)

    if need:
        if not settings.OCR_ENABLED:
            raise ValueError(
                f"检测到扫描版 PDF（{reason}），需要 OCR，但当前 OCR 未开启"
                "（可在 .env 设 OCR_ENABLED=true）"
            )
        try:
            text = ocr_mod.ocr_pdf(path)
        except ocr_mod.OCRError as e:
            raise ValueError(f"扫描版 PDF 的 OCR 失败：{e}") from e
        return text, {
            "used_ocr": True,
            "format": "pdf",
            "note": f"扫描版 PDF 已 OCR（{reason}）",
        }

    return _read_pdf(path), {
        "used_ocr": False,
        "format": "pdf",
        "note": reason,
    }


def _read_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover
        raise ValueError("缺少依赖 pypdf，无法解析 pdf") from e
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        pages.append(f"--- 第 {i} 页 ---\n{text}")
    return "\n".join(pages)


def _read_docx(path: str) -> str:
    """Word → Markdown（索引层归一化，用户拿到的仍是原始 .docx）。

    保留标题层级与表格结构，使后续分块能「按标题切、表格不被切断、续块带表头」。
    """
    return docx_to_markdown(path)
