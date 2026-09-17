from fastapi import APIRouter, Depends, HTTPException, status

from .. import auth, models, schemas
from ..config import settings
from ..db import SessionLocal

router = APIRouter(prefix="/api/users", tags=["users"])


@router.get("", response_model=list[schemas.UserOut])
def list_users(
    admin: models.User = Depends(auth.require_admin),
    database: SessionLocal = Depends(auth.get_db),
):
    return database.query(models.User).all()


@router.post(
    "", response_model=schemas.UserOut, status_code=status.HTTP_201_CREATED
)
def create_user(
    body: schemas.UserCreate,
    admin: models.User = Depends(auth.require_admin),
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
    return user
