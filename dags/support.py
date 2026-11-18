"""Shared helpers for the Airflow DAGs.

Configuration comes from Airflow Connections (BaseHook.get_connection),
which docker-compose.yml seeds through AIRFLOW_CONN_* variables.

This module doesn't use src/db.py or src/config.py, because they depend on
packages the Airflow image doesn't include.
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

# Both the API and Airflow mount ./data.
INGEST_LANDING_DIR = "/app/data/_incoming"
STAGING_ROOT = "/app/data/_staging"
IMAGES_DIR = "/app/data/images"
CERTIFICATES_DIR = "/app/data/certificates"
QDRANT_COLLECTION = "properties"


def get_postgres_engine():
    conn = BaseHook.get_connection(POSTGRES_CONN_ID)
    # Built from individual fields because conn.get_uri() can produce a
    # "postgres://" scheme, which SQLAlchemy rejects.
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
        # URL-addressed instance such as Qdrant Cloud: Airflow parses the
        # scheme into conn_type. No port is added, since Qdrant Cloud uses
        # the standard HTTPS port.
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
