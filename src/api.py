"""FastAPI app: routes and startup wiring only.

All client construction is lazy and lives in db.py; all business logic
lives in floorplan.py, documents.py, agents.py, and ingest.py. lifespan
builds each client exactly once at startup and stores it on app.state;
routes pull what they need via Depends, which just reads app.state.
"""

import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, HumanMessage
from qdrant_client.http.models import Distance, VectorParams
from sqlalchemy import text

from .agents import build_agent_executor
from .config import Settings, get_settings
from .db import Base, get_embedder, get_engine, get_qdrant_client, get_session_factory
from .floorplan import parse_floorplan
from .ingest import ingest_properties_sync
from .models import ChatRequest


# -----------------------------------------------------------------
# --- THE LIFESPAN FIX ---
# We initialize all DB-dependent agents inside this startup event
# -----------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs ONCE, when the app starts up
    print("FastAPI is starting up, waiting for databases...")

    settings = get_settings()
    engine = get_engine()
    qdrant_client = get_qdrant_client()
    embedder = get_embedder()
    session_factory = get_session_factory()

    # 1. Create Qdrant Collection (idempotent — do NOT wipe data on restart)
    existing = {c.name for c in qdrant_client.get_collections().collections}
    if settings.QDRANT_VECTOR_COLLECTION in existing:
        print(f"Qdrant collection '{settings.QDRANT_VECTOR_COLLECTION}' already exists. Reusing.")
    else:
        embedding_dim = len(embedder.embed_query("dimension probe"))
        qdrant_client.create_collection(
            collection_name=settings.QDRANT_VECTOR_COLLECTION,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        )
        print(f"Qdrant collection '{settings.QDRANT_VECTOR_COLLECTION}' created.")

    # 2. Create PostgreSQL Tables
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully.")

    # 3. Initialize ALL database-dependent agents
    print("Initializing agents...")
    agent_executor = build_agent_executor(
        engine=engine,
        qdrant_client=qdrant_client,
        embedder=embedder,
        session_factory=session_factory,
        settings=settings,
    )

    app.state.settings = settings
    app.state.engine = engine
    app.state.qdrant_client = qdrant_client
    app.state.embedder = embedder
    app.state.session_factory = session_factory
    app.state.agent_executor = agent_executor

    print("--- FastAPI is ready and agents are initialized! ---")

    yield

    print("FastAPI is shutting down.")


# --- FastAPI App Definition ---
app = FastAPI(title="Real-Estate API", lifespan=lifespan)


# --- Dependencies: just read what lifespan stored on app.state ---
def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings

def get_app_engine(request: Request):
    return request.app.state.engine

def get_app_qdrant_client(request: Request):
    return request.app.state.qdrant_client

def get_app_embedder(request: Request):
    return request.app.state.embedder

def get_app_session_factory(request: Request):
    return request.app.state.session_factory

def get_app_agent_executor(request: Request):
    return request.app.state.agent_executor


# --- API Endpoints ---

@app.get("/")
def read_root():
    """Root endpoint for health checks."""
    return {"status": "ok", "message": "Backend is running!"}

@app.get("/health")
def health_check(
    engine=Depends(get_app_engine),
    qdrant_client=Depends(get_app_qdrant_client),
    agent_executor=Depends(get_app_agent_executor),
):
    checks = {}
    healthy = True

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["postgres"] = True
    except Exception as e:
        print(f"Health check: Postgres connectivity failed: {e}")
        checks["postgres"] = False
        healthy = False

    try:
        qdrant_client.get_collections()
        checks["qdrant"] = True
    except Exception as e:
        print(f"Health check: Qdrant connectivity failed: {e}")
        checks["qdrant"] = False
        healthy = False

    checks["agent_executor"] = agent_executor is not None
    if not checks["agent_executor"]:
        healthy = False

    status_code = 200 if healthy else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": "ok" if healthy else "unhealthy", "checks": checks},
    )

@app.post("/chat")
async def chat_endpoint(request: ChatRequest, agent_executor=Depends(get_app_agent_executor)):
    if not agent_executor:
        raise HTTPException(status_code=500, detail="Agent not initialized.")

    chat_history = []
    for user_msg, ai_msg in request.history:
        chat_history.append(HumanMessage(content=user_msg))
        chat_history.append(AIMessage(content=ai_msg))

    try:
        # Run agent
        response = await agent_executor.ainvoke({"input": request.query, "chat_history": chat_history})
        return {"status": "success", "response": response['output']}
    except Exception as e:
        print(f"Agent execution error: {e}")
        raise HTTPException(status_code=500, detail=f"Agent Error: {e}")

@app.post("/ingest")
async def ingest_properties(
    file: UploadFile = File(...),
    session_factory=Depends(get_app_session_factory),
    qdrant_client=Depends(get_app_qdrant_client),
    embedder=Depends(get_app_embedder),
    settings: Settings = Depends(get_app_settings),
):
    # Read the file into an in-memory bytes buffer (async I/O), then hand the
    # CPU-heavy parsing/ingest work (pandas, YOLO, EasyOCR, DB writes) off to a
    # worker thread so it doesn't block the event loop for the whole request.
    file_contents = await file.read()
    return await run_in_threadpool(
        ingest_properties_sync,
        file_contents,
        session_factory=session_factory,
        qdrant_client=qdrant_client,
        embedder=embedder,
        qdrant_collection=settings.QDRANT_VECTOR_COLLECTION,
    )

@app.post("/parse-floorplan-debug")
async def parse_floorplan_debug(file: UploadFile = File(...)):
    contents = await file.read()

    # Never trust the client-supplied filename as a path (path traversal) —
    # write to a generated temp path instead, and always clean it up.
    fd, file_path = tempfile.mkstemp(dir="/tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(contents)
        data = await run_in_threadpool(parse_floorplan, file_path)
    finally:
        os.remove(file_path)

    return data
