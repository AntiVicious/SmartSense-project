"""Shared helpers for the Airflow DAGs.

Everything here reads Airflow Connections (BaseHook.get_connection), never
raw os.environ -- Connections are seeded via AIRFLOW_CONN_* env vars on the
Airflow containers (see docker-compose.yml), which is the framework's own
sanctioned way to provide them in a 12-factor deployment; that's a
different thing from a DAG reading os.getenv() directly, which is what
we're avoiding.

Deliberately independent of src/db.py and src/config.py: those import
langchain_community (and transitively torch via sentence-transformers),
which Airflow's image does not carry. This module only needs sqlalchemy
and qdrant-client, both in requirements-airflow.txt.
"""

import os
import sys

from airflow.hooks.base import BaseHook
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

POSTGRES_CONN_ID = "smartsense_postgres"
QDRANT_CONN_ID = "smartsense_qdrant"
INTERNAL_API_CONN_ID = "smartsense_internal_api"

# Shared with the app's own data volume -- both containers mount ./data.
INGEST_LANDING_DIR = "/app/data/_incoming"
STAGING_ROOT = "/app/data/_staging"
IMAGES_DIR = "/app/data/images"
CERTIFICATES_DIR = "/app/data/certificates"
QDRANT_COLLECTION = "properties"


def get_postgres_engine():
    conn = BaseHook.get_connection(POSTGRES_CONN_ID)
    # Built from discrete fields rather than conn.get_uri() -- Airflow's
    # URI reconstruction is provider/conn_type-dependent (e.g. some
    # produce "postgres://", which SQLAlchemy 1.4+/2.0 rejects outright;
    # it wants "postgresql://"), so this is more predictable.
    url = f"postgresql://{conn.login}:{conn.password}@{conn.host}:{conn.port}/{conn.schema}"
    sslmode = conn.extra_dejson.get("sslmode")
    if sslmode:
        url += f"?sslmode={sslmode}"
    return create_engine(url)


def get_postgres_session_factory():
    return sessionmaker(bind=get_postgres_engine())


def get_qdrant_client():
    from qdrant_client import QdrantClient

    conn = BaseHook.get_connection(QDRANT_CONN_ID)
    api_key = conn.extra_dejson.get("api_key")
    if conn.conn_type in ("http", "https"):
        # Qdrant Cloud (or any URL-addressed instance) -- the connection's
        # scheme was parsed out of the URI into conn_type, e.g.
        # "https://xyz.cloud.qdrant.io" gives conn_type="https", host="xyz...".
        # No port appended: Qdrant Cloud serves HTTPS on the standard port,
        # not 6333 -- matches src/db.py's get_qdrant_client() for the app.
        return QdrantClient(url=f"{conn.conn_type}://{conn.host}", api_key=api_key)
    return QdrantClient(host=conn.host, port=conn.port or 6333, api_key=api_key)


def get_internal_api_config() -> tuple[str, str]:
    """Returns (base_url, api_key) for the app's /internal/* endpoints."""
    conn = BaseHook.get_connection(INTERNAL_API_CONN_ID)
    base_url = f"http://{conn.host}:{conn.port}" if conn.port else f"http://{conn.host}"
    api_key = conn.extra_dejson.get("api_key")
    if not api_key:
        raise RuntimeError(f"Airflow connection {INTERNAL_API_CONN_ID!r} has no 'api_key' in its extra field")
    return base_url, api_key
