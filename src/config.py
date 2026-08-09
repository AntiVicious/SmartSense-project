"""Application configuration.

All environment variables are read here, once, through pydantic-settings.
Required variables (the Postgres credentials, the Groq key) have no default,
so instantiating Settings() fails loudly and immediately if one is missing —
instead of silently building a connection string like
"postgresql://None:None@postgres-db/None" that only breaks later, confusingly,
when something tries to actually connect.

Nothing in this module reads the environment at import time: get_settings()
is only ever called from the app's lifespan and from request-time FastAPI
dependencies, never at module scope.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- PostgreSQL ---
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str = "postgres-db"  # Docker Compose service name

    # --- Qdrant ---
    QDRANT_HOST: str = "qdrant-db"  # Docker Compose service name
    QDRANT_VECTOR_COLLECTION: str = "properties"

    # --- Groq / LLM ---
    GROQ_API_KEY: str
    LLM_MODEL_NAME: str = "openai/gpt-oss-120b"

    # --- Embeddings ---
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}/{self.POSTGRES_DB}?sslmode=disable"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
