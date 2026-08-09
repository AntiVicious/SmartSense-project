"""Lazy factories for the app's external clients.

Nothing here connects to anything at import time. The engine, Qdrant client,
and embedding model are only constructed the first time their getter is
called — which, in the running app, is from lifespan (see api.py). This is
what makes `import src.api` safe without a network connection, a model
download, or a live database: importing this module just defines functions.

Each getter is lru_cache'd (no-arg cache = singleton), so the app builds
each client exactly once and every caller shares the same instance — this
matches the original module-level-singleton behavior, just built lazily.
"""

from functools import lru_cache

from qdrant_client import QdrantClient
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import get_settings

# Use HuggingFaceEmbeddings (LangChain wrapper around sentence-transformers)
# so it exposes .embed_query() / .embed_documents() that LangChain's Qdrant
# vector store and RetrievalQA expect. A raw SentenceTransformer only has
# .encode() and breaks RAG search.
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
        # Qdrant Cloud (or any URL-addressed instance) -- QDRANT_HOST carries
        # the scheme, e.g. https://xyz.cloud.qdrant.io, not just a hostname.
        return QdrantClient(url=settings.QDRANT_HOST, api_key=settings.QDRANT_API_KEY)
    return QdrantClient(host=settings.QDRANT_HOST, port=6333, api_key=settings.QDRANT_API_KEY)


@lru_cache
def get_embedder() -> HuggingFaceEmbeddings:
    settings = get_settings()
    return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)
