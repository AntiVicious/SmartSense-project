"""Lazy factories for the app's external clients.

Nothing connects at import time; each client is built the first time its
getter is called (from lifespan in api.py), so the app can be imported
without a network, models or a database. Each getter is cached, so every
caller shares one instance.
"""

from functools import lru_cache

from qdrant_client import QdrantClient
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import get_settings

# LangChain's wrapper provides the embed_query()/embed_documents() methods its
# Qdrant vector store expects.
from langchain_community.embeddings import HuggingFaceEmbeddings

Base = declarative_base()


@lru_cache
def get_engine():
    settings = get_settings()
    return create_engine(settings.database_url)


@lru_cache
def get_session_factory():
    return sessionmaker(autocommit=False, autoflush=False, bind=get_engine())


@lru_cache
def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    if settings.QDRANT_HOST.startswith("http://") or settings.QDRANT_HOST.startswith("https://"):
        # Full URL, e.g. Qdrant Cloud
        return QdrantClient(url=settings.QDRANT_HOST, api_key=settings.QDRANT_API_KEY)
    return QdrantClient(host=settings.QDRANT_HOST, port=6333, api_key=settings.QDRANT_API_KEY)


@lru_cache
def get_embedder() -> HuggingFaceEmbeddings:
    settings = get_settings()
    return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)
