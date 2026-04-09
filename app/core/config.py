from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str | None = None
    celery_result_backend: str | None = None
    celery_task_always_eager: bool = False
    celery_task_eager_propagates: bool = True
    celery_document_max_retries: int = 3
    celery_document_retry_base_seconds: int = 5
    embedding_api_key: str = ""
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "glm-4.5-air"
    rerank_enabled: bool = False
    rerank_api_key: str = ""
    rerank_base_url: str = ""
    rerank_model: str = ""
    rerank_top_n: int = 5
    retrieval_dense_candidate_k: int = 12
    retrieval_bm25_candidate_k: int = 12
    retrieval_fusion_candidate_k: int = 16
    retrieval_final_top_k: int = 5
    retrieval_relevance_threshold: float = 0.35
    cors_allow_origins: str = "https://rag.restflux.online"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def resolved_celery_broker_url(self) -> str:
        return self.celery_broker_url or self.redis_url

    @property
    def resolved_celery_result_backend(self) -> str:
        return self.celery_result_backend or self.redis_url

    @property
    def resolved_cors_allow_origins(self) -> list[str]:
        # 用逗号分隔，便于线上快速切换允许的前端域名，而不需要改代码。
        origins = [
            origin.strip()
            for origin in self.cors_allow_origins.split(",")
            if origin.strip()
        ]
        return origins or ["https://rag.restflux.online"]


settings = Settings()
