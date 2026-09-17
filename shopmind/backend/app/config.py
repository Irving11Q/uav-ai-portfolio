import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")


class Settings:
    # 路径
    BACKEND_DIR = BACKEND_DIR
    DATA_DIR = BACKEND_DIR / "data"
    DB_PATH = DATA_DIR / "app.db"
    CHROMA_DIR = DATA_DIR / "chroma"
    UPLOAD_DIR = DATA_DIR / "uploads"
    VECTOR_COLLECTION = "kb_chunks"

    # 安全
    SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production-rag-thesis-2026")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

    # 通义千问 DashScope（OpenAI 兼容模式）
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
    DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    LLM_MODEL = os.getenv("LLM_MODEL", "qwen-plus")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
    # 向量化后端：dashscope（默认）| local（本地 bge，需自行安装 sentence-transformers）
    EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "dashscope")
    # 单次向量化批量上限：DashScope text-embedding 一次最多 10 条，
    # 超限会报 400 InvalidParameter: batch size is invalid。必须分批调用！
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))

    # 分块参数（按字符数，适配中文）
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
    # 检索召回数（返回给大模型的最终片段数）
    RETRIEVE_TOP_K = int(os.getenv("RETRIEVE_TOP_K", "5"))
    # 混合检索候选池大小（向量 + 关键词各自召回后融合/重排的候选数量）
    HYBRID_POOL_SIZE = int(os.getenv("HYBRID_POOL_SIZE", "20"))

    # 重排（Rerank）：用通义千问 gte-rerank 对融合候选二次精排
    RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"
    # ⚠️ 必须带 `-v2`：老的 `gte-rerank` 现在会返回 403 AccessDenied，
    # 而异常被 retrieval._rerank 吞掉降级成融合排序 —— 表面上「一切正常」，
    # 实则重排从未生效。实测 gte-rerank-v2 才返回 200。（2026-09-17 定位）
    RERANK_MODEL = os.getenv("RERANK_MODEL", "gte-rerank-v2")
    RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "10"))  # 送重排的候选数

    # 语义缓存：相似问题直接复用答案，降低大模型调用
    CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    CACHE_COLLECTION = "qa_cache"
    CACHE_SIMILARITY_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.96"))

    # 限流与并发（企业级稳定性）
    RATE_LIMIT_PER_USER_PER_MIN = int(os.getenv("RATE_LIMIT_PER_USER_PER_MIN", "20"))
    LLM_MAX_CONCURRENCY = int(os.getenv("LLM_MAX_CONCURRENCY", "4"))

    # OCR 预处理（可选增强）：应对扫描版商品图册、截图型说明书
    # 走通义千问视觉模型（多模态，OpenAI 兼容），无需本地安装 OCR 引擎
    OCR_ENABLED = os.getenv("OCR_ENABLED", "true").lower() == "true"
    OCR_MODEL = os.getenv("OCR_MODEL", "qwen-vl-ocr")
    # PDF 每页可提取文本少于这个字符数，就判定为「扫描版」，转走 OCR
    OCR_MIN_TEXT_PER_PAGE = int(os.getenv("OCR_MIN_TEXT_PER_PAGE", "50"))
    # 单篇文档最多 OCR 多少页（成本与耗时兜底）
    OCR_MAX_PAGES = int(os.getenv("OCR_MAX_PAGES", "20"))
    # 渲染扫描版 PDF 时的放大倍数，越大越清晰也越慢
    OCR_RENDER_SCALE = float(os.getenv("OCR_RENDER_SCALE", "2.0"))
    # 直接当图片做 OCR 的扩展名
    OCR_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")

    # 管理员引导
    ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "123456")
    ADMIN_ROLE = "admin"
    USER_ROLE = "user"

    # ── 演示模式（对外交付给他人体验时开启）─────────────────────────────
    # 为什么要有：在线体验版消耗的是真实 LLM 额度。链接一旦被转发或扫到，
    # 没有总上限就可能被刷爆账单。这里给「整个站点」设一个总次数上限。
    #
    # 为什么不做成后台可调：若登录后台就能把上限改大，这层保护等于没有。
    # 所以额度只从环境变量读，只有部署者能改。
    DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
    # 全局额度：真实调用大模型的问答总次数（命中语义缓存的提问不计数）
    DEMO_MAX_QUESTIONS = int(os.getenv("DEMO_MAX_QUESTIONS", "200"))

    # 跨域
    CORS_ORIGINS = os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")

    def ensure_dirs(self) -> None:
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
