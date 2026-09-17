import logging

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import auth, db, demo, models, stats as stats_mod
from .config import settings
from .routers import auth as auth_router, users as users_router
from .routers import documents as documents_router, chat as chat_router
from .routers import feedback as feedback_router, dashboard as dashboard_router
from .routers import settings_api as settings_router, audit_log as audit_router
from .static_site import mount_frontend

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag")

app = FastAPI(title="电商 RAG 知识库问答系统", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router.router)
app.include_router(users_router.router)
app.include_router(documents_router.router)
app.include_router(chat_router.router)
# 阶段6：反馈 / 仪表盘 / 运行时配置 / 审计日志
app.include_router(feedback_router.router)
app.include_router(dashboard_router.router)
app.include_router(settings_router.router)
app.include_router(audit_router.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/demo/quota")
def demo_quota(
    user: models.User = Depends(auth.get_current_user),
    database=Depends(auth.get_db),
):
    """公开演示环境的剩余额度。前端用它提示体验者，避免额度用完后莫名其妙报错。"""
    return demo.quota_info(database)


@app.get("/")
def root():
    return {"msg": "电商 RAG 知识库问答系统 API", "docs": "/docs"}


@app.get("/api/stats")
def query_stats(admin: models.User = Depends(auth.require_admin)):
    """检索/缓存/时延可观测指标（仅管理员）。内存统计，服务重启清零。"""
    return stats_mod.snapshot()


def seed_admin() -> None:
    database = db.SessionLocal()
    try:
        existing = (
            database.query(models.User)
            .filter(models.User.username == settings.ADMIN_USERNAME)
            .first()
        )
        if existing:
            return
        admin = models.User(
            username=settings.ADMIN_USERNAME,
            password_hash=auth.hash_password(settings.ADMIN_PASSWORD),
            role=settings.ADMIN_ROLE,
        )
        database.add(admin)
        database.commit()
        logger.info(
            "已创建管理员账号 %s / %s", settings.ADMIN_USERNAME, settings.ADMIN_PASSWORD
        )
    finally:
        database.close()


@app.on_event("startup")
def on_startup():
    db.init_db()
    seed_admin()
    try:
        from .rag.vectorstore import get_chroma_client

        get_chroma_client()
        logger.info("Chroma 向量库目录已就绪")
    except Exception as e:  # pragma: no cover
        logger.warning("Chroma 初始化跳过: %s", e)


# 前端构建产物托管 —— 必须放在最后注册，兜底路由才不会抢走 /api/* 与 /docs。
# 开发态 app/static 不存在时自动跳过，不影响 Vite 双进程流程。
mount_frontend(app)
