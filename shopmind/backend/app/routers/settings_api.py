"""运行时配置接口：管理员在后台切换模型、调整检索参数，改完立即生效（无需重启）。

白名单、类型转换与范围校验全部在 `appconfig` 里做，本层只负责鉴权、写审计、转成 schema。
"""
from fastapi import APIRouter, Depends, HTTPException, Request, status

from .. import appconfig, audit, auth, models, schemas
from ..db import SessionLocal

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("", response_model=schemas.SettingsOut)
def read_settings(
    _: models.User = Depends(auth.require_admin),
):
    """读取全部可配置项（含当前值与元信息，前端据此自动渲染表单）。"""
    return appconfig.describe()


@router.get("/public", response_model=dict)
def public_settings(_: models.User = Depends(auth.get_current_user)):
    """非敏感配置，任何登录用户可见（例如问答页顶部展示「当前模型」）。"""
    values = appconfig.all_values()
    return {
        "LLM_MODEL": values.get("LLM_MODEL"),
        "RETRIEVE_TOP_K": values.get("RETRIEVE_TOP_K"),
        "RERANK_ENABLED": values.get("RERANK_ENABLED"),
        "CACHE_ENABLED": values.get("CACHE_ENABLED"),
    }


@router.put("", response_model=schemas.SettingsOut)
def update_settings(
    body: schemas.SettingsUpdate,
    request: Request,
    admin: models.User = Depends(auth.require_admin),
    database: SessionLocal = Depends(auth.get_db),
):
    """批量更新配置。任一项非法则整批拒绝，不会出现「改了一半」。"""
    before = appconfig.all_values()
    try:
        appconfig.update(body.values or {})
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    after = appconfig.all_values()
    changed = {
        k: {"from": before.get(k), "to": v}
        for k, v in after.items()
        if before.get(k) != v
    }
    if changed:
        audit.log(
            "config_update",
            user=admin,
            detail="；".join(f"{k}: {c['from']} -> {c['to']}" for k, c in changed.items()),
            target="runtime_config",
            ip=audit.client_ip(request),
            extra={"changed": changed},
            database=database,
        )
    return appconfig.describe()


@router.post("/reset", response_model=schemas.SettingsOut)
def reset_settings(
    request: Request,
    admin: models.User = Depends(auth.require_admin),
    database: SessionLocal = Depends(auth.get_db),
):
    """清空后台覆盖，全部回到 `.env` 默认值。"""
    appconfig.reset()
    audit.log(
        "config_reset",
        user=admin,
        detail="运行时配置已恢复为 .env 默认值",
        target="runtime_config",
        ip=audit.client_ip(request),
        database=database,
    )
    return appconfig.describe()
