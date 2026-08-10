"""FastAPI app: routes and startup wiring only.

All client construction is lazy and lives in db.py; all business logic
lives in floorplan.py, documents.py, agents.py. lifespan builds each
client exactly once at startup and stores it on app.state; routes pull
what they need via Depends, which just reads app.state.

Property ingestion itself runs as an Airflow DAG (dags/), not inline in a
request handler -- POST /ingest only lands the uploaded file and creates
an ingest_jobs row; GET /ingest/{job_id} reads that row back. The
/internal/* endpoints are how the DAG reaches the already-loaded
YOLO/EasyOCR/embedding models without Airflow's own image needing torch.
"""

import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from groq import BadRequestError as GroqBadRequestError
from langchain_core.messages import AIMessage, HumanMessage
from qdrant_client.http.models import Distance, VectorParams
from sqlalchemy import text

from .agents import build_agent_executor
from .config import Settings, get_settings
from .db import Base, get_embedder, get_engine, get_qdrant_client, get_session_factory
from .floorplan import parse_floorplan
from .ingest_landing import UnsupportedFileType, land_ingest_file
from .logging_config import configure_logging
from .models import ChatRequest, InternalEmbedRequest, InternalParseFloorplanRequest, IngestJob

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------
# --- THE LIFESPAN FIX ---
# We initialize all DB-dependent agents inside this startup event
# -----------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs ONCE, when the app starts up
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL)
    logger.info("FastAPI is starting up, waiting for databases...")

    engine = get_engine()
    qdrant_client = get_qdrant_client()
    embedder = get_embedder()
    session_factory = get_session_factory()

    # 1. Create Qdrant Collection (idempotent — do NOT wipe data on restart)
    existing = {c.name for c in qdrant_client.get_collections().collections}
    if settings.QDRANT_VECTOR_COLLECTION in existing:
        logger.info("Qdrant collection '%s' already exists. Reusing.", settings.QDRANT_VECTOR_COLLECTION)
    else:
        embedding_dim = len(embedder.embed_query("dimension probe"))
        qdrant_client.create_collection(
            collection_name=settings.QDRANT_VECTOR_COLLECTION,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        )
        logger.info("Qdrant collection '%s' created.", settings.QDRANT_VECTOR_COLLECTION)

    # 2. Create PostgreSQL Tables
    Base.metadata.create_all(bind=engine)
    logger.info("Tables created successfully.")

    # 3. Initialize ALL database-dependent agents
    logger.info("Initializing agents...")
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

    logger.info("FastAPI is ready and agents are initialized.")

    yield

    logger.info("FastAPI is shutting down.")


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


def verify_internal_api_key(
    x_internal_api_key: str = Header(default=""),
    settings: Settings = Depends(get_app_settings),
):
    """Gate for /internal/*. These run real inference on request and are
    reachable by anything on the Docker network (the port isn't published
    to the host, but that's not the same as authenticated) -- this is the
    shared secret Airflow's Connection for the internal API carries."""
    if not x_internal_api_key or x_internal_api_key != settings.INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing internal API key")


def verify_api_key(
    x_api_key: str = Header(default=""),
    settings: Settings = Depends(get_app_settings),
):
    """Gate for the public write/debug endpoints: POST /ingest and
    POST /parse-floorplan-debug (fix-list 2.6). A static shared-secret
    header is a deliberately minimal answer -- enough that this isn't an
    open write path into the database or a free-to-abuse inference
    endpoint once this is on a billed Cloud Run service, not a full auth
    system this project doesn't otherwise have. The ui service holds this
    key server-side (see src/main.py) and attaches it on the user's behalf."""
    if not x_api_key or x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# --- Upload validation (fix-list 2.5) ---
# Both upload endpoints used to call `await file.read()` with no size
# limit and no content-type check -- one large upload could OOM the
# container, and nothing stopped an arbitrary file from being handed to
# pandas/PIL/YOLO.

_ALLOWED_SPREADSHEET_CONTENT_TYPES = {
    "",  # some browsers/clients omit it for less-common extensions
    "text/csv",
    "application/csv",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/octet-stream",  # generic fallback some clients send
}


async def _read_upload_or_413(file: UploadFile, settings: Settings) -> bytes:
    """Reads at most max_upload_size_bytes + 1 bytes -- enough to detect
    an oversized upload without ever buffering more than that much in
    memory, regardless of what (or whether) Content-Length claims."""
    max_bytes = settings.max_upload_size_bytes
    contents = await file.read(max_bytes + 1)
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {settings.MAX_UPLOAD_SIZE_MB}MB)",
        )
    return contents


def _check_spreadsheet_content_type(file: UploadFile) -> None:
    content_type = (file.content_type or "").lower()
    if content_type not in _ALLOWED_SPREADSHEET_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported content type: {content_type or '(none)'}")


def _check_image_content_type(file: UploadFile) -> None:
    content_type = (file.content_type or "").lower()
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail=f"Expected an image upload, got: {content_type or '(none)'}"
        )


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
    except Exception:
        logger.warning("Health check: Postgres connectivity failed", exc_info=True)
        checks["postgres"] = False
        healthy = False

    try:
        qdrant_client.get_collections()
        checks["qdrant"] = True
    except Exception:
        logger.warning("Health check: Qdrant connectivity failed", exc_info=True)
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

    agent_input = request.query
    try:
        # Run agent
        response = await agent_executor.ainvoke({"input": agent_input, "chat_history": chat_history})
        return {"status": "success", "response": response["output"]}
    except GroqBadRequestError as e:
        # Groq's tool-call validator rejects `null` argument values outright
        # (schema says e.g. "type": "string", and langchain's generated
        # tool schema never marks it nullable even for Optional[...]
        # params) -- but the model reaches for null specifically on vague
        # queries ("show me anything"), which is a normal, expected input,
        # not a real error. One retry with an explicit nudge resolves it
        # in practice; anything past that falls through to the generic
        # handler below rather than retrying forever.
        if "tool_use_failed" not in str(e):
            raise
        logger.warning("Retrying chat request after a tool_use_failed response from Groq")
        try:
            nudged_input = (
                f"{agent_input}\n\n"
                "(Tool-calling note: when calling generate_property_report, omit any "
                "argument you don't have a value for -- never pass null.)"
            )
            response = await agent_executor.ainvoke({"input": nudged_input, "chat_history": chat_history})
            return {"status": "success", "response": response["output"]}
        except Exception:
            correlation_id = uuid.uuid4().hex[:12]
            logger.exception("Agent execution failed after retry [correlation_id=%s]", correlation_id)
            raise HTTPException(
                status_code=500,
                detail=f"Internal error. Reference ID: {correlation_id}",
            )
    except Exception:
        # Never return raw exception text to the client (fix-list 2.4) --
        # it can leak internals (stack frames, connection strings, prompt
        # content). Log the real error server-side keyed by a correlation
        # ID and hand the client only that ID to reference when reporting
        # the issue.
        correlation_id = uuid.uuid4().hex[:12]
        logger.exception("Agent execution failed [correlation_id=%s]", correlation_id)
        raise HTTPException(
            status_code=500,
            detail=f"Internal error. Reference ID: {correlation_id}",
        )


@app.post("/ingest", status_code=202, dependencies=[Depends(verify_api_key)])
async def ingest_properties(
    file: UploadFile = File(...),
    session_factory=Depends(get_app_session_factory),
    settings: Settings = Depends(get_app_settings),
):
    # Ingestion itself doesn't happen here anymore -- this just lands the
    # file where the Airflow poller DAG will find it and records a job row.
    # See dags/watch_ingest_landing_dag.py and dags/ingest_properties_dag.py.
    _check_spreadsheet_content_type(file)
    file_contents = await _read_upload_or_413(file, settings)
    try:
        job_id = await run_in_threadpool(
            land_ingest_file,
            session_factory=session_factory,
            landing_dir=settings.INGEST_LANDING_DIR,
            file_bytes=file_contents,
            original_filename=file.filename,
        )
    except UnsupportedFileType as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"status": "queued", "job_id": job_id, "status_url": f"/ingest/{job_id}"}


@app.get("/ingest/{job_id}")
async def get_ingest_job(job_id: str, session_factory=Depends(get_app_session_factory)):
    def _query():
        db = session_factory()
        try:
            return db.query(IngestJob).filter(IngestJob.job_id == job_id).one_or_none()
        finally:
            db.close()

    job = await run_in_threadpool(_query)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": job.job_id,
        "status": job.status,
        "original_filename": job.original_filename,
        "dag_run_id": job.dag_run_id,
        "rows_ingested": job.rows_ingested,
        "error_message": job.error_message,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
    }


@app.post("/internal/parse-floorplan", dependencies=[Depends(verify_internal_api_key)])
async def internal_parse_floorplan(request: InternalParseFloorplanRequest):
    return await run_in_threadpool(parse_floorplan, request.image_path)


@app.post("/internal/embed", dependencies=[Depends(verify_internal_api_key)])
async def internal_embed(request: InternalEmbedRequest, embedder=Depends(get_app_embedder)):
    vector = await run_in_threadpool(embedder.embed_query, request.text)
    return {"embedding": vector}


@app.post("/parse-floorplan-debug", dependencies=[Depends(verify_api_key)])
async def parse_floorplan_debug(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_app_settings),
):
    _check_image_content_type(file)
    contents = await _read_upload_or_413(file, settings)

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
