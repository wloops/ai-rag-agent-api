from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # 这些字段会从 .env 里读取，并自动转换成对应的 Python 类型。
    database_url: str
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440

    # env_file=".env" 表示优先从项目根目录的 .env 加载配置。
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# 在应用启动时创建一个全局配置对象，其他模块直接 import 使用。
settings = Settings()
