"""知识库管理（仅管理员）：上传 / 列表 / 统计 / 删除 / 重索引 / 下载原件。"""
import json
import os
import traceback

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from .. import audit, auth, models, schemas
from ..config import settings
from ..db import SessionLocal
from ..rag import loaders, chunking, vectorstore, keyword

router = APIRouter(prefix="/api/documents", tags=["documents"])

# 文本类格式
ALLOWED_EXT = {"txt", "md", "pdf", "docx", "csv", "xlsx", "json", "text"}
# 图片类格式：走 OCR 预处理，用于扫描版商品图册 / 截图型说明书
ALLOWED_EXT |= {e.lower().lstrip(".") for e in settings.OCR_IMAGE_EXTS}


def _process_document(doc_id: int, file_path: str, db: Session) -> None:
    """后台任务：解析（必要时 OCR）-> 分块 -> 向量化 -> 更新文档状态。"""
    # 每次后台任务用独立 session
    database = SessionLocal()
    try:
        doc = database.query(models.Document).filter(models.Document.id == doc_id).first()
        if not doc:
            return
        # parse_file_with_meta 会告诉我们这篇是不是走了 OCR（图片 / 扫描版 PDF）
        text, meta = loaders.parse_file_with_meta(file_path)
        doc.meta_json = json.dumps(meta, ensure_ascii=False)
        chunks = chunking.split_text(text)
        if not chunks:
            doc.status = "failed"
            doc.error = "解析后无有效文本内容"
            database.commit()
            return
        # 重索引时先清理旧向量与旧分块记录
        try:
            vectorstore.delete_document_chunks(doc_id)
        except Exception:
            pass
        database.query(models.Chunk).filter(models.Chunk.doc_id == doc_id).delete()
        database.commit()

        # 分块落 SQLite chunks 表，拿到主键供「关键词 BM25 检索 + 混合检索融合」使用
        chunk_rows = [
            models.Chunk(
                doc_id=doc_id, chunk_index=i, source=doc.filename, content=c
            )
            for i, c in enumerate(chunks)
        ]
        database.add_all(chunk_rows)
        database.commit()
        chunk_ids = [c.id for c in chunk_rows]

        # 向量库写入（带 chunk_id，便于与关键词召回按同一主键融合去重）
        vectorstore.add_document_chunks(doc_id, doc.filename, chunk_ids, chunks)
        # 关键词索引失效，下次查询按新分块重建
        keyword.invalidate()

        doc.chunk_count = len(chunks)
        doc.char_count = len(text)
        doc.status = "ready"
        doc.error = None
        database.commit()

        if meta.get("used_ocr"):
            owner = (
                database.query(models.User).filter(models.User.id == doc.owner_id).first()
            )
            audit.log(
                "ocr",
                user=owner,
                detail=f"《{doc.filename}》经 OCR 识别后入库，共 {len(chunks)} 个分块（{meta.get('note')}）",
                target=doc.filename,
                database=database,
            )
    except Exception as e:  # noqa: BLE001
        database.rollback()
        doc = database.query(models.Document).filter(models.Document.id == doc_id).first()
        if doc:
            doc.status = "failed"
            doc.error = f"{type(e).__name__}: {e}"
            database.commit()
        traceback.print_exc()
    finally:
        database.close()


@router.post("", response_model=schemas.DocumentOut, status_code=status.HTTP_201_CREATED)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    background: BackgroundTasks = None,
    admin: models.User = Depends(auth.require_admin),
    database: SessionLocal = Depends(auth.get_db),
):
    ext = os.path.splitext(file.filename or "")[1].lower().lstrip(".")
    if ext not in ALLOWED_EXT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的文件类型 .{ext}，支持：{', '.join(sorted(ALLOWED_EXT))}",
        )

    doc = models.Document(
        owner_id=admin.id,
        filename=file.filename or f"doc.{ext}",
        file_type=ext,
        status="processing",
    )
    database.add(doc)
    database.commit()
    database.refresh(doc)

    # 保存原始文件到 uploads 目录
    save_path = os.path.join(settings.UPLOAD_DIR, f"{doc.id}_{file.filename}")
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    if background is not None:
        background.add_task(_process_document, doc.id, save_path, database)

    audit.log(
        "upload",
        user=admin,
        detail=f"上传《{doc.filename}》（{len(content)} 字节）"
        + ("，图片类将走 OCR 识别" if _doc_is_image(ext) else ""),
        target=doc.filename,
        ip=audit.client_ip(request),
        database=database,
    )
    return doc


def _doc_is_image(ext: str) -> bool:
    return ext.lower().lstrip(".") in {
        e.lower().lstrip(".") for e in settings.OCR_IMAGE_EXTS
    }


@router.get("", response_model=list[schemas.DocumentOut])
def list_documents(
    admin: models.User = Depends(auth.require_admin),
    database: SessionLocal = Depends(auth.get_db),
):
    return (
        database.query(models.Document)
        .order_by(models.Document.created_at.desc())
        .all()
    )


@router.get("/stats", response_model=schemas.DocStats)
def doc_stats(
    admin: models.User = Depends(auth.require_admin),
    database: SessionLocal = Depends(auth.get_db),
):
    docs = database.query(models.Document).all()
    total = len(docs)
    ready = sum(1 for d in docs if d.status == "ready")
    chunks = sum(d.chunk_count or 0 for d in docs)
    return schemas.DocStats(
        total_documents=total, ready_documents=ready, total_chunks=chunks
    )


@router.post("/{doc_id}/reindex", response_model=schemas.DocumentOut)
async def reindex_document(
    doc_id: int,
    request: Request,
    background: BackgroundTasks,
    admin: models.User = Depends(auth.require_admin),
    database: SessionLocal = Depends(auth.get_db),
):
    doc = database.query(models.Document).filter(models.Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="文档不存在")
    save_path = os.path.join(settings.UPLOAD_DIR, f"{doc.id}_{doc.filename}")
    if not os.path.exists(save_path):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="原始文件已丢失，无法重索引")
    doc.status = "processing"
    doc.error = None
    database.commit()
    background.add_task(_process_document, doc.id, save_path, database)
    audit.log(
        "reindex",
        user=admin,
        detail=f"触发重索引《{doc.filename}》",
        target=doc.filename,
        ip=audit.client_ip(request),
        database=database,
    )
    return doc


@router.get("/{doc_id}/download")
def download_document(
    doc_id: int,
    request: Request,
    admin: models.User = Depends(auth.require_admin),
    database: SessionLocal = Depends(auth.get_db),
):
    """下载**原始文件**（不使用内部 Markdown）。

    用户视角与机器视角在这里分界：用户上传什么格式，下载回去就是什么格式
    （.docx 仍是 .docx），Markdown 只是索引用中间表示，从不外露。
    """
    doc = database.query(models.Document).filter(models.Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="文档不存在")
    save_path = os.path.join(settings.UPLOAD_DIR, f"{doc.id}_{doc.filename}")
    if not os.path.exists(save_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="原始文件已丢失")
    audit.log(
        "download_doc",
        user=admin,
        detail=f"下载原件《{doc.filename}》",
        target=doc.filename,
        ip=audit.client_ip(request),
        database=database,
    )
    return FileResponse(
        save_path, filename=doc.filename, media_type="application/octet-stream"
    )


@router.delete("/{doc_id}", status_code=status.HTTP_200_OK)
def delete_document(
    doc_id: int,
    request: Request,
    admin: models.User = Depends(auth.require_admin),
    database: SessionLocal = Depends(auth.get_db),
):
    doc = database.query(models.Document).filter(models.Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="文档不存在")
    name = doc.filename
    # 清理向量
    try:
        vectorstore.delete_document_chunks(doc_id)
    except Exception:
        pass
    # 清理分块表（先于文档删除，避免外键残留）并失效关键词索引
    database.query(models.Chunk).filter(models.Chunk.doc_id == doc_id).delete()
    keyword.invalidate()
    # 清理原始文件
    save_path = os.path.join(settings.UPLOAD_DIR, f"{doc.id}_{doc.filename}")
    if os.path.exists(save_path):
        try:
            os.remove(save_path)
        except OSError:
            pass
    database.delete(doc)
    database.commit()
    audit.log(
        "delete_doc",
        user=admin,
        detail=f"删除文档《{name}》及其全部分块与向量",
        target=name,
        ip=audit.client_ip(request),
        database=database,
    )
    return {"msg": "已删除", "id": doc_id}
