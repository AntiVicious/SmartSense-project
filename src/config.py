"""Application configuration.

All environment variables are read here through pydantic-settings.
Required variables have no default, so Settings() fails at startup if one is
missing. get_settings() is called from the app's lifespan and request
dependencies, never at import time.
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
    # "disable" for local Docker Postgres; set to "require" for managed
    # services such as Neon.
    POSTGRES_SSLMODE: str = "disable"

    # --- Qdrant ---
    # A hostname (self-hosted) or a full URL (Qdrant Cloud); see
    # db.get_qdrant_client().
    QDRANT_HOST: str = "qdrant-db"
    QDRANT_API_KEY: str | None = None  # required for Qdrant Cloud, unused self-hosted
    QDRANT_VECTOR_COLLECTION: str = "properties"

    # --- Groq / LLM ---
    GROQ_API_KEY: str
    LLM_MODEL_NAME: str = "openai/gpt-oss-120b"

    # --- Embeddings ---
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

    # --- Internal API key for Airflow's calls to /internal/* ---
    INTERNAL_API_KEY: str

    # --- Directory watched by the Airflow ingestion DAG ---
    INGEST_LANDING_DIR: str = "/app/data/_incoming"

    # --- Logging ---
    LOG_LEVEL: str = "INFO"

    # --- API key for POST /ingest and POST /parse-floorplan-debug, sent by
    # the ui service ---
    API_KEY: str

    # --- Upload validation ---
    MAX_UPLOAD_SIZE_MB: int = 15

    @property
    def max_upload_size_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    @property
    def database_url(self) -> str:
        # A host starting with "/" is a Unix socket directory (e.g. Cloud
        # SQL's /cloudsql/<instance>), which psycopg2 takes as a query
        # parameter.
        if self.POSTGRES_HOST.startswith("/"):
            return (
                f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
                f"@/{self.POSTGRES_DB}?host={self.POSTGRES_HOST}"
            )
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}/{self.POSTGRES_DB}?sslmode={self.POSTGRES_SSLMODE}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
