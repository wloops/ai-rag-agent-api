from pydantic import BaseModel, EmailStr, Field


# Schema 是“接口层的数据结构”。
# 它和 ORM 模型不同，主要负责请求校验和响应格式控制。
class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)
    nickname: str = Field(min_length=1, max_length=100)


class UserLogin(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: int
    email: EmailStr
    nickname: str

    # 允许直接从 SQLAlchemy 对象读取字段并序列化成响应。
    model_config = {"from_attributes": True}
