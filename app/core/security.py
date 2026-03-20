from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.models.user import User


# CryptContext 统一管理密码哈希算法。
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
# HTTPBearer 会从请求头里读取 Authorization: Bearer <token>
bearer_scheme = HTTPBearer()


def hash_password(password: str) -> str:
    # 注册时不保存明文密码，只保存哈希后的结果。
    return pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    # 登录时用用户输入的明文密码去和数据库里的哈希值比对。
    return pwd_context.verify(password, password_hash)


def create_access_token(data: dict) -> str:
    # JWT 里通常会放用户 id、过期时间等信息。
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.access_token_expire_minutes
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    # 先从 Authorization 头里拿到 Bearer Token。
    token = credentials.credentials

    try:
        # 解码 JWT，拿出登录时写进去的 sub（用户 id）。
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    # 再根据 token 里的用户 id 去数据库查询当前用户。
    user = db.query(User).filter(User.id == int(user_id)).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user

