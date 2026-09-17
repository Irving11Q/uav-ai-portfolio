import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from .config import settings
from .db import SessionLocal
from .models import User

# auto_error=False：缺 token 时不要由 FastAPI 直接抛「Not authenticated」，
# 而是交给我们自己统一判断 —— 因为 token 可能从 cookie 或自定义头来。
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

# 浏览器端自动携带 token 的 cookie 名（见 routers/auth.py 登录时写入）
TOKEN_COOKIE = "access_token"

# 常量
PBKDF2_ITERATIONS = 100_000


def extract_token(request: Request, bearer: str | None) -> str | None:
    """从多个通道里取 token，按「最不容易被中间层改动」的顺序：

    1. `X-Auth-Token` 自定义头（首选）
    2. `access_token` cookie（浏览器自动携带）
    3. 标准 `Authorization: Bearer`（最后兜底）

    **为什么需要这样兜底**：部署到云平台后，边缘网关可能会注入或改写
    `Authorization` 头，导致客户端带来的 token 被顶掉 —— 实测现象是
    「登录接口正常返回 token，但拿这个 token 打任何鉴权接口都 401」。
    换成自定义头与 cookie 后即可绕开这种改写。
    """
    custom = request.headers.get("x-auth-token")
    if custom:
        return custom.strip() or None
    cookie = request.cookies.get(TOKEN_COOKIE)
    if cookie:
        return cookie.strip() or None
    return bearer


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), PBKDF2_ITERATIONS
    )
    return f"pbkdf2${salt}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, salt, hashed = stored.split("$")
        if algo != "pbkdf2":
            return False
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt), PBKDF2_ITERATIONS
        )
        return hmac.compare_digest(dk.hex(), hashed)
    except Exception:
        return False


def create_access_token(user_id: int, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    payload = {"sub": str(user_id), "role": role, "exp": expire}
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(
    request: Request,
    bearer: str | None = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    cred_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无效或过期的凭证",
        headers={"WWW-Authenticate": "Bearer"},
    )
    token = extract_token(request, bearer)
    if not token:
        raise cred_exc
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        user_id = int(payload.get("sub"))
    except Exception:
        raise cred_exc
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise cred_exc
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != settings.ADMIN_ROLE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="需要管理员权限"
        )
    return user
