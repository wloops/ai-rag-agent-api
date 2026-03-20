from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import (
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)
from app.models.user import User
from app.schemas.auth import TokenResponse, UserLogin, UserRegister, UserResponse


# 这个 router 专门处理登录、注册、获取当前用户等认证相关接口。
router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=UserResponse)
def register(data: UserRegister, db: Session = Depends(get_db)):
    # 先检查邮箱是否已注册，避免重复创建用户。
    existing_user = db.query(User).filter(User.email == data.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # 创建 ORM 对象，相当于先在 Python 里准备一条待插入的数据。
    user = User(
        email=data.email,
        password_hash=hash_password(data.password),
        nickname=data.nickname,
    )
    # add 放入会话，commit 真正提交到数据库，refresh 取回数据库最新值。
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=TokenResponse)
def login(data: UserLogin, db: Session = Depends(get_db)):
    # 登录分两步：先找用户，再校验密码。
    user = db.query(User).filter(User.email == data.email).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not verify_password(data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # sub 是 JWT 常用字段，这里存用户 id，后续可据此还原当前用户。
    access_token = create_access_token({"sub": str(user.id)})
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=UserResponse)
def me(current_user: User = Depends(get_current_user)):
    # 只要 token 合法，依赖注入会把当前用户对象塞进来。
    return current_user
