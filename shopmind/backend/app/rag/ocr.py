"""OCR 预处理（可选增强）：把扫描版商品图册 / 截图型说明书里的文字抠出来。

## 为什么走云端视觉模型而不是本地 OCR 引擎
本地方案（PaddleOCR / Tesseract / RapidOCR）要么装几百 MB 模型，要么得额外装系统二进制，
对一个「双击 bat 就能跑」的毕设项目来说摩擦太大。
通义千问的视觉模型走的是**已有的同一个 API Key**，零额外安装，且对中文排版和表格的识别效果更好。

## 两条路径
1. **图片直接识别**：`.png/.jpg` 等 → 转 base64 → 视觉模型 → Markdown 文本。
2. **扫描版 PDF**：先用 pypdf 试着抽文本，若「每页平均可提取字符数」低于阈值，
   判定为扫描件 → 用 pypdfium2 把每页渲染成图片 → 逐页 OCR。
   正常电子版 PDF 走原本的文本抽取，**不会**被拖去 OCR 白花时间与费用。

## 降级原则
没配 Key、开关关闭、页数超限、接口报错 —— 一律「明确告知 + 不影响其他格式上传」，绝不静默吞掉。
"""
import base64
import io
import logging

from ..config import settings

logger = logging.getLogger("rag")

# 给视觉模型的指令。强调「按阅读顺序」「保留表格」「不臆造」，
# 因为 OCR 结果会直接进知识库被检索，幻觉出来的内容危害比缺字更大。
_OCR_PROMPT = (
    "请提取这张图片中的所有文字，严格按从上到下、从左到右的阅读顺序输出。"
    "如果图片里有表格，请用 Markdown 表格还原（保留表头）。"
    "只输出识别到的文字内容本身，不要添加任何解释、总结或说明。"
    "如果某个字看不清，就跳过它，不要猜测或编造内容。"
)


class OCRError(RuntimeError):
    """OCR 相关的可预期错误（用于给用户看的中文提示）。"""


def is_image(filename: str) -> bool:
    lower = (filename or "").lower()
    return lower.endswith(tuple(settings.OCR_IMAGE_EXTS))


def _data_url(image_bytes: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(image_bytes).decode("ascii")


def _call_vision(image_data_url: str) -> str:
    """调用通义千问视觉模型做一次 OCR，返回纯文本。"""
    if not settings.DASHSCOPE_API_KEY:
        raise OCRError("未配置 DASHSCOPE_API_KEY，无法进行 OCR 识别")

    import requests

    try:
        resp = requests.post(
            f"{settings.DASHSCOPE_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.DASHSCOPE_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.OCR_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                            {"type": "text", "text": _OCR_PROMPT},
                        ],
                    }
                ],
            },
            timeout=180,
        )
        resp.raise_for_status()
        payload = resp.json()
    except OCRError:
        raise
    except Exception as e:  # noqa: BLE001
        raise OCRError(f"OCR 接口调用失败：{type(e).__name__}: {e}")

    choices = payload.get("choices") or []
    if not choices:
        raise OCRError(f"OCR 返回为空：{str(payload)[:200]}")

    content = (choices[0].get("message") or {}).get("content")
    # 有的模型会把 content 返回成 [{type: text, text: ...}] 结构，这里都兼容
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text") or "")
            else:
                parts.append(str(item))
        content = "".join(parts)
    return (content or "").strip()


def ocr_image_file(path: str) -> str:
    """对图片文件做 OCR，返回识别出的 Markdown 文本。"""
    if not settings.OCR_ENABLED:
        raise OCRError("OCR 功能未开启（OCR_ENABLED=false）")
    with open(path, "rb") as f:
        data = f.read()
    if not data:
        raise OCRError("图片文件为空")
    mime = "image/jpeg" if path.lower().endswith((".jpg", ".jpeg")) else "image/png"
    text = _call_vision(_data_url(data, mime))
    if not text:
        raise OCRError("OCR 未能从图片中识别出文字")
    return text


def pdf_text_density(path: str) -> tuple[int, int, float]:
    """返回 (总字符数, 页数, 每页平均字符数)，用于判断是不是扫描版 PDF。"""
    try:
        from pypdf import PdfReader

        reader = PdfReader(path)
        pages = len(reader.pages)
        total = 0
        for page in reader.pages:
            try:
                total += len((page.extract_text() or "").strip())
            except Exception:  # noqa: BLE001
                continue
        return total, pages, (total / pages if pages else 0.0)
    except Exception as e:  # noqa: BLE001
        logger.warning("统计 PDF 文本密度失败: %s", e)
        return 0, 0, 0.0


def needs_ocr(path: str) -> tuple[bool, str]:
    """判断这个 PDF 是否需要走 OCR，返回 (是否需要, 原因说明)。"""
    total, pages, per_page = pdf_text_density(path)
    if pages == 0:
        return False, "无法读取 PDF 页数，按普通 PDF 处理"
    if per_page < settings.OCR_MIN_TEXT_PER_PAGE:
        return True, (
            f"每页平均仅 {per_page:.0f} 个可提取字符（阈值 {settings.OCR_MIN_TEXT_PER_PAGE}），"
            f"判定为扫描版，改用 OCR"
        )
    return False, f"电子版 PDF（每页平均 {per_page:.0f} 字符），走正常文本抽取"


def ocr_pdf(path: str) -> str:
    """把扫描版 PDF 逐页渲染成图片后 OCR，拼成一整段文本。"""
    if not settings.OCR_ENABLED:
        raise OCRError("OCR 功能未开启（OCR_ENABLED=false）")

    try:
        import pypdfium2 as pdfium
    except Exception as e:  # noqa: BLE001
        raise OCRError(f"缺少 pypdfium2，无法渲染 PDF 页面：{e}")

    doc = pdfium.PdfDocument(path)
    try:
        pages = len(doc)
        limit = min(pages, settings.OCR_MAX_PAGES)
        chunks: list[str] = []
        for i in range(limit):
            page = doc[i]
            bitmap = page.render(scale=settings.OCR_RENDER_SCALE)
            image = bitmap.to_pil()
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            text = _call_vision(_data_url(buf.getvalue(), "image/png"))
            if text:
                chunks.append(f"<!-- 第 {i + 1} 页 -->\n{text}")
        if not chunks:
            raise OCRError("OCR 未从任何页面识别出文字")
        if pages > limit:
            chunks.append(
                f"\n> 注：本文档共 {pages} 页，按配置仅识别了前 {limit} 页"
                f"（可调整 OCR_MAX_PAGES）。"
            )
        return "\n\n".join(chunks)
    finally:
        try:
            doc.close()
        except Exception:  # noqa: BLE001
            pass
