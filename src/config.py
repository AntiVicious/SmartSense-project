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
    # Free-tier managed Postgres (Neon, Supabase, ...) rejects unencrypted
    # connections -- local Docker Postgres has no TLS cert to offer, so the
    # default has to stay "disable" or local dev breaks. Override to
    # "require" for those.
    POSTGRES_SSLMODE: str = "disable"

    # --- Qdrant ---
    # QDRANT_HOST is either a bare hostname (self-hosted, e.g. the Docker
    # Compose service name) or a full URL including scheme (Qdrant Cloud,
    # e.g. https://xyz.cloud.qdrant.io) -- see db.py's get_qdrant_client().
    QDRANT_HOST: str = "qdrant-db"
    QDRANT_API_KEY: str | None = None  # required for Qdrant Cloud, unused self-hosted
    QDRANT_VECTOR_COLLECTION: str = "properties"

    # --- Groq / LLM ---
    GROQ_API_KEY: str
    LLM_MODEL_NAME: str = "openai/gpt-oss-120b"

    # --- Embeddings ---
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

    # --- Internal API (Airflow -> app calls for floorplan parsing / embedding) ---
    # Required, no default: these endpoints run real inference on request,
    # so an unset key should fail startup loudly rather than leave them
    # reachable with a guessable default.
    INTERNAL_API_KEY: str

    # --- Ingest landing directory (file-drop trigger for the Airflow DAGs) ---
    INGEST_LANDING_DIR: str = "/app/data/_incoming"

    # --- Logging (fix-list 2.3) ---
    LOG_LEVEL: str = "INFO"

    # --- API key for public write/debug endpoints: POST /ingest and
    # POST /parse-floorplan-debug (fix-list 2.6). Distinct from
    # INTERNAL_API_KEY -- that one is Airflow-only; this one is held by the
    # ui service (see src/main.py) and, if you're calling these endpoints
    # directly, whoever else you hand it to. Required, no default, same
    # fail-loudly reasoning as INTERNAL_API_KEY.
    API_KEY: str

    # --- Upload validation (fix-list 2.5) ---
    MAX_UPLOAD_SIZE_MB: int = 15

    @property
    def max_upload_size_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    @property
    def database_url(self) -> str:
        # Cloud SQL's Auth Proxy (or Cloud Run's built-in Cloud SQL
        # connection) exposes the database over a Unix socket at
        # /cloudsql/<INSTANCE_CONNECTION_NAME> instead of a host:port --
        # detected here by POSTGRES_HOST starting with "/", since that's
        # never a valid hostname. psycopg2's URL form for a socket
        # directory takes the host as a query param, not before the "/".
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
