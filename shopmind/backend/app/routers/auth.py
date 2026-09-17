from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from .. import audit, auth, models, schemas
from ..config import settings
from ..db import SessionLocal

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=schemas.UserOut, status_code=status.HTTP_201_CREATED)
def register(
    body: schemas.UserCreate,
    request: Request,
    database: SessionLocal = Depends(auth.get_db),
):
    if (
        database.query(models.User)
        .filter(models.User.username == body.username)
        .first()
    ):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="用户名已存在")
    user = models.User(
        username=body.username,
        password_hash=auth.hash_password(body.password),
        role=settings.USER_ROLE,
    )
    database.add(user)
    database.commit()
    database.refresh(user)
    audit.log(
        "register",
        user=user,
        detail=f"新用户注册：{user.username}",
        target=f"user#{user.id}",
        ip=audit.client_ip(request),
        database=database,
    )
    return user


@router.post("/login", response_model=schemas.Token)
def login(
    body: schemas.UserLogin,
    request: Request,
    response: Response,
    database: SessionLocal = Depends(auth.get_db),
):
    user = (
        database.query(models.User)
        .filter(models.User.username == body.username)
        .first()
    )
    if not user or not auth.verify_password(body.password, user.password_hash):
        # 登录失败也要留痕（审计的基本要求：失败尝试同样重要）
        audit.log(
            "login_failed",
            user=None,
            detail=f"登录失败，用户名：{body.username}",
            target=body.username,
            ip=audit.client_ip(request),
            database=database,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误"
        )
    token = auth.create_access_token(user.id, user.role)
    # 同时写一份 cookie：浏览器会自动携带，即使前端自定义头被中间层丢掉也能鉴权。
    # SameSite=Lax 可以挡掉跨站发起的 POST/PUT/DELETE，避免 cookie 带来的 CSRF 风险。
    response.set_cookie(
        key=auth.TOKEN_COOKIE,
        value=token,
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax",
        httponly=False,
    )
    audit.log(
        "login",
        user=user,
        detail=f"{user.username} 登录成功",
        target=f"user#{user.id}",
        ip=audit.client_ip(request),
        database=database,
    )
    return {"access_token": token}


@router.post("/logout", status_code=status.HTTP_200_OK)
def logout(response: Response):
    """清掉登录 cookie。

    为什么需要：登录时会写一份 access_token cookie（用于绕过网关对
    Authorization 头的改写）。前端只清 localStorage 的话 cookie 仍在，
    服务端会认为浏览器依旧处于登录态。

    刻意不做鉴权：token 已过期时也应该能正常退出，不该再抛 401。
    """
    response.delete_cookie(auth.TOKEN_COOKIE)
    return {"msg": "已退出"}


@router.get("/me", response_model=schemas.UserOut)
def me(user: models.User = Depends(auth.get_current_user)):
    return user


@router.post("/change-password", status_code=status.HTTP_200_OK)
def change_password(
    body: schemas.ChangePassword,
    request: Request,
    user: models.User = Depends(auth.get_current_user),
    database: SessionLocal = Depends(auth.get_db),
):
    if not auth.verify_password(body.old_password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="原密码错误"
        )
    if len(body.new_password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="新密码至少 6 位"
        )
    user.password_hash = auth.hash_password(body.new_password)
    database.commit()
    audit.log(
        "change_password",
        user=user,
        detail=f"{user.username} 修改了密码",
        target=f"user#{user.id}",
        ip=audit.client_ip(request),
        database=database,
    )
    return {"msg": "密码已修改"}
