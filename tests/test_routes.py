"""
Route-level tests for src/api.py via FastAPI's TestClient. No real
Postgres, no real Qdrant, no real LLM, no real YOLO/EasyOCR:

- The real lifespan (which builds all of those) is swapped for a no-op
  before any client is created, so importing/using `app` here never
  touches app.state via the real startup path.
- Every Depends accessor (get_app_engine, get_app_qdrant_client, ...)
  is overridden per test via app.dependency_overrides with a fake, which
  is how routes get their fakes instead of the real app.state values the
  (now no-op'd) lifespan would have set.
- /parse-floorplan-debug has no Depends indirection (parse_floorplan is
  called directly), so its test replaces src.api.parse_floorplan on the
  module directly and restores it afterward.
"""

import io
import os
import shutil
import sys
import tempfile
from contextlib import asynccontextmanager

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import src.api as api_module  # noqa: E402
from src.api import (  # noqa: E402
    app,
    get_app_agent_executor,
    get_app_embedder,
    get_app_engine,
    get_app_qdrant_client,
    get_app_session_factory,
    get_app_settings,
)
from src.db import Base  # noqa: E402
from src.models import IngestJob  # noqa: E402

INTERNAL_API_KEY = "test-internal-key"
API_KEY = "test-api-key"


@asynccontextmanager
async def _noop_lifespan(_app):
    # Real lifespan builds the engine/Qdrant client/embedder/agent and
    # would try to reach real infrastructure. Route tests supply their
    # own fakes via dependency_overrides instead, so startup does nothing.
    yield


app.router.lifespan_context = _noop_lifespan


class FakeConnection:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def execute(self, stmt):
        return None


class FakeEngine:
    def __init__(self, healthy=True):
        self.healthy = healthy

    def connect(self):
        if not self.healthy:
            raise RuntimeError("connection refused")
        return FakeConnection()


class FakeCollectionsResult:
    collections = []


class FakeQdrantClient:
    def __init__(self, healthy=True):
        self.healthy = healthy
        self.upsert_calls = []

    def get_collections(self):
        if not self.healthy:
            raise RuntimeError("qdrant unreachable")
        return FakeCollectionsResult()

    def upsert(self, collection_name, points, wait=True):
        self.upsert_calls.append({"collection_name": collection_name, "points": points, "wait": wait})


class FakeEmbedder:
    def embed_query(self, text):
        return [0.1, 0.2, 0.3]


class FakeSettings:
    def __init__(self, landing_dir=None, max_upload_size_mb=15):
        self.QDRANT_VECTOR_COLLECTION = "properties"
        self.INTERNAL_API_KEY = INTERNAL_API_KEY
        self.API_KEY = API_KEY
        self.INGEST_LANDING_DIR = landing_dir or tempfile.mkdtemp()
        self.MAX_UPLOAD_SIZE_MB = max_upload_size_mb

    @property
    def max_upload_size_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024


class FakeAgentExecutor:
    def __init__(self, response_text=None, raise_exc=None):
        self.response_text = response_text
        self.raise_exc = raise_exc
        self.last_call = None

    async def ainvoke(self, payload):
        self.last_call = payload
        if self.raise_exc is not None:
            raise self.raise_exc
        return {"output": self.response_text}


def _make_session_factory():
    # StaticPool + check_same_thread=False: routes that use run_in_threadpool
    # (POST /ingest, GET /ingest/{job_id}) run their DB work in a worker
    # thread, not the thread that called create_all() below. SQLAlchemy's
    # default pool for sqlite:///:memory: is thread-scoped (a fresh,
    # separate in-memory DB per thread), so without this the worker thread
    # would see an empty database with no tables at all. StaticPool shares
    # one real connection across every thread instead.
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def _make_excel_bytes(rows: list) -> bytes:
    buf = io.BytesIO()
    pd.DataFrame(rows).to_excel(buf, index=False)
    return buf.getvalue()


def test_root_is_a_trivial_liveness_ping():
    with TestClient(app) as client:
        resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "message": "Backend is running!"}


def test_health_all_dependencies_healthy():
    app.dependency_overrides[get_app_engine] = lambda: FakeEngine(healthy=True)
    app.dependency_overrides[get_app_qdrant_client] = lambda: FakeQdrantClient(healthy=True)
    app.dependency_overrides[get_app_agent_executor] = lambda: FakeAgentExecutor()
    try:
        with TestClient(app) as client:
            resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {
            "status": "ok",
            "checks": {"postgres": True, "qdrant": True, "agent_executor": True},
        }
    finally:
        app.dependency_overrides.clear()


def test_health_postgres_down_returns_503():
    app.dependency_overrides[get_app_engine] = lambda: FakeEngine(healthy=False)
    app.dependency_overrides[get_app_qdrant_client] = lambda: FakeQdrantClient(healthy=True)
    app.dependency_overrides[get_app_agent_executor] = lambda: FakeAgentExecutor()
    try:
        with TestClient(app) as client:
            resp = client.get("/health")
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "unhealthy"
        assert body["checks"] == {"postgres": False, "qdrant": True, "agent_executor": True}
    finally:
        app.dependency_overrides.clear()


def test_health_qdrant_down_returns_503():
    app.dependency_overrides[get_app_engine] = lambda: FakeEngine(healthy=True)
    app.dependency_overrides[get_app_qdrant_client] = lambda: FakeQdrantClient(healthy=False)
    app.dependency_overrides[get_app_agent_executor] = lambda: FakeAgentExecutor()
    try:
        with TestClient(app) as client:
            resp = client.get("/health")
        assert resp.status_code == 503
        assert resp.json()["checks"] == {"postgres": True, "qdrant": False, "agent_executor": True}
    finally:
        app.dependency_overrides.clear()


def test_health_agent_not_initialized_returns_503():
    app.dependency_overrides[get_app_engine] = lambda: FakeEngine(healthy=True)
    app.dependency_overrides[get_app_qdrant_client] = lambda: FakeQdrantClient(healthy=True)
    app.dependency_overrides[get_app_agent_executor] = lambda: None
    try:
        with TestClient(app) as client:
            resp = client.get("/health")
        assert resp.status_code == 503
        assert resp.json()["checks"] == {"postgres": True, "qdrant": True, "agent_executor": False}
    finally:
        app.dependency_overrides.clear()


def test_chat_success_returns_agent_output():
    fake_agent = FakeAgentExecutor(response_text="Hello there!")
    app.dependency_overrides[get_app_agent_executor] = lambda: fake_agent
    try:
        with TestClient(app) as client:
            resp = client.post("/chat", json={"query": "hi", "history": []})
        assert resp.status_code == 200
        assert resp.json() == {"status": "success", "response": "Hello there!"}
        assert fake_agent.last_call["input"] == "hi"
    finally:
        app.dependency_overrides.clear()


def test_chat_builds_history_as_message_pairs():
    fake_agent = FakeAgentExecutor(response_text="ok")
    app.dependency_overrides[get_app_agent_executor] = lambda: fake_agent
    try:
        with TestClient(app) as client:
            resp = client.post("/chat", json={"query": "and now?", "history": [["hi", "hello!"]]})
        assert resp.status_code == 200
        history = fake_agent.last_call["chat_history"]
        assert len(history) == 2
        assert history[0].content == "hi"
        assert history[1].content == "hello!"
    finally:
        app.dependency_overrides.clear()


def test_chat_agent_not_initialized_returns_500():
    app.dependency_overrides[get_app_agent_executor] = lambda: None
    try:
        with TestClient(app) as client:
            resp = client.post("/chat", json={"query": "hi", "history": []})
        assert resp.status_code == 500
        assert resp.json()["detail"] == "Agent not initialized."
    finally:
        app.dependency_overrides.clear()


def test_chat_agent_exception_does_not_leak_raw_exception_text():
    # fix-list 2.4: the client gets a correlation ID, never the exception
    # itself (which can carry stack frames, connection strings, prompt
    # content, etc.) -- the real error is only ever logged server-side.
    fake_agent = FakeAgentExecutor(raise_exc=RuntimeError("groq exploded"))
    app.dependency_overrides[get_app_agent_executor] = lambda: fake_agent
    try:
        with TestClient(app) as client:
            resp = client.post("/chat", json={"query": "hi", "history": []})
        assert resp.status_code == 500
        detail = resp.json()["detail"]
        assert "groq exploded" not in detail
        assert detail.startswith("Internal error. Reference ID: ")
        correlation_id = detail.removeprefix("Internal error. Reference ID: ")
        assert len(correlation_id) == 12  # uuid4().hex[:12]
    finally:
        app.dependency_overrides.clear()


def test_ingest_route_lands_file_and_creates_queued_job():
    # /ingest no longer ingests anything itself -- it lands the upload for
    # the Airflow poller DAG and records a job row. No image_file/rooms/etc
    # in the payload matters here; that's the DAG's job now, tested in
    # dags/tests, not this route.
    session_factory = _make_session_factory()
    landing_dir = tempfile.mkdtemp()
    try:
        app.dependency_overrides[get_app_session_factory] = lambda: session_factory
        app.dependency_overrides[get_app_settings] = lambda: FakeSettings(landing_dir=landing_dir)
        try:
            excel_bytes = _make_excel_bytes([{"title": "Anything", "location": "X"}])
            with TestClient(app) as client:
                resp = client.post(
                    "/ingest",
                    files={
                        "file": (
                            "props.xlsx",
                            excel_bytes,
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
                    },
                    headers={"X-API-Key": API_KEY},
                )
            assert resp.status_code == 202
            body = resp.json()
            assert body["status"] == "queued"
            job_id = body["job_id"]
            assert body["status_url"] == f"/ingest/{job_id}"

            landed_files = os.listdir(landing_dir)
            assert landed_files == [f"{job_id}.xlsx"]

            db = session_factory()
            try:
                job = db.query(IngestJob).filter(IngestJob.job_id == job_id).one()
                assert job.status == "queued"
                assert job.original_filename == "props.xlsx"
            finally:
                db.close()
        finally:
            app.dependency_overrides.clear()
    finally:
        shutil.rmtree(landing_dir, ignore_errors=True)


def test_ingest_route_rejects_unsupported_file_type():
    # text/plain isn't in the allowed content-type set, so this is now
    # rejected by the content-type check before it ever reaches
    # land_ingest_file's extension check -- same 400, different reason.
    session_factory = _make_session_factory()
    landing_dir = tempfile.mkdtemp()
    try:
        app.dependency_overrides[get_app_session_factory] = lambda: session_factory
        app.dependency_overrides[get_app_settings] = lambda: FakeSettings(landing_dir=landing_dir)
        try:
            with TestClient(app) as client:
                resp = client.post(
                    "/ingest",
                    files={"file": ("notes.txt", b"not a spreadsheet", "text/plain")},
                    headers={"X-API-Key": API_KEY},
                )
            assert resp.status_code == 400
            assert os.listdir(landing_dir) == []
        finally:
            app.dependency_overrides.clear()
    finally:
        shutil.rmtree(landing_dir, ignore_errors=True)


def test_ingest_route_rejects_missing_api_key():
    app.dependency_overrides[get_app_settings] = lambda: FakeSettings()
    try:
        with TestClient(app) as client:
            resp = client.post(
                "/ingest",
                files={
                    "file": (
                        "props.xlsx",
                        b"irrelevant",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                },
            )
        assert resp.status_code == 401

        resp = client.post(
            "/ingest",
            files={
                "file": (
                    "props.xlsx",
                    b"irrelevant",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401
    finally:
        app.dependency_overrides.clear()


def test_ingest_route_rejects_oversized_upload():
    session_factory = _make_session_factory()
    landing_dir = tempfile.mkdtemp()
    try:
        app.dependency_overrides[get_app_session_factory] = lambda: session_factory
        app.dependency_overrides[get_app_settings] = lambda: FakeSettings(
            landing_dir=landing_dir, max_upload_size_mb=1
        )
        try:
            oversized = b"x" * (2 * 1024 * 1024)  # 2MB against a 1MB limit
            with TestClient(app) as client:
                resp = client.post(
                    "/ingest",
                    files={
                        "file": (
                            "props.xlsx",
                            oversized,
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
                    },
                    headers={"X-API-Key": API_KEY},
                )
            assert resp.status_code == 413
            assert os.listdir(landing_dir) == []
        finally:
            app.dependency_overrides.clear()
    finally:
        shutil.rmtree(landing_dir, ignore_errors=True)


def test_get_ingest_job_returns_status():
    session_factory = _make_session_factory()
    db = session_factory()
    try:
        db.add(IngestJob(job_id="job-1", original_filename="a.xlsx", status="succeeded", rows_ingested=12))
        db.commit()
    finally:
        db.close()

    app.dependency_overrides[get_app_session_factory] = lambda: session_factory
    try:
        with TestClient(app) as client:
            resp = client.get("/ingest/job-1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["job_id"] == "job-1"
        assert body["status"] == "succeeded"
        assert body["rows_ingested"] == 12
    finally:
        app.dependency_overrides.clear()


def test_get_ingest_job_404_for_unknown_id():
    session_factory = _make_session_factory()
    app.dependency_overrides[get_app_session_factory] = lambda: session_factory
    try:
        with TestClient(app) as client:
            resp = client.get("/ingest/does-not-exist")
        assert resp.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_internal_parse_floorplan_rejects_missing_or_wrong_key():
    app.dependency_overrides[get_app_settings] = lambda: FakeSettings()
    try:
        with TestClient(app) as client:
            resp = client.post("/internal/parse-floorplan", json={"image_path": "/x.jpg"})
            assert resp.status_code == 401

            resp = client.post(
                "/internal/parse-floorplan",
                json={"image_path": "/x.jpg"},
                headers={"X-Internal-Api-Key": "wrong-key"},
            )
            assert resp.status_code == 401
    finally:
        app.dependency_overrides.clear()


def test_internal_parse_floorplan_with_valid_key_calls_parser():
    original_parse_floorplan = api_module.parse_floorplan
    captured_paths = []

    def fake_parse_floorplan(path):
        captured_paths.append(path)
        return {"rooms": 2, "halls": 0, "kitchens": 1, "bathrooms": 1, "other rooms": 0}

    api_module.parse_floorplan = fake_parse_floorplan
    app.dependency_overrides[get_app_settings] = lambda: FakeSettings()
    try:
        with TestClient(app) as client:
            resp = client.post(
                "/internal/parse-floorplan",
                json={"image_path": "/app/data/images/house1.jpg"},
                headers={"X-Internal-Api-Key": INTERNAL_API_KEY},
            )
        assert resp.status_code == 200
        assert resp.json() == {"rooms": 2, "halls": 0, "kitchens": 1, "bathrooms": 1, "other rooms": 0}
        assert captured_paths == ["/app/data/images/house1.jpg"]
    finally:
        api_module.parse_floorplan = original_parse_floorplan
        app.dependency_overrides.clear()


def test_internal_embed_rejects_missing_key():
    app.dependency_overrides[get_app_settings] = lambda: FakeSettings()
    try:
        with TestClient(app) as client:
            resp = client.post("/internal/embed", json={"text": "hello"})
        assert resp.status_code == 401
    finally:
        app.dependency_overrides.clear()


def test_internal_embed_with_valid_key_returns_embedding():
    fake_embedder = FakeEmbedder()
    app.dependency_overrides[get_app_settings] = lambda: FakeSettings()
    app.dependency_overrides[get_app_embedder] = lambda: fake_embedder
    try:
        with TestClient(app) as client:
            resp = client.post(
                "/internal/embed",
                json={"text": "a cozy two bedroom house"},
                headers={"X-Internal-Api-Key": INTERNAL_API_KEY},
            )
        assert resp.status_code == 200
        assert resp.json() == {"embedding": [0.1, 0.2, 0.3]}
    finally:
        app.dependency_overrides.clear()


def test_parse_floorplan_debug_route_returns_parser_output():
    original_parse_floorplan = api_module.parse_floorplan
    captured_paths = []

    def fake_parse_floorplan(path):
        captured_paths.append(path)
        return {"rooms": 1, "halls": 0, "kitchens": 0, "bathrooms": 0, "other rooms": 0}

    api_module.parse_floorplan = fake_parse_floorplan
    app.dependency_overrides[get_app_settings] = lambda: FakeSettings()
    try:
        with TestClient(app) as client:
            resp = client.post(
                "/parse-floorplan-debug",
                files={"file": ("plan.jpg", b"fake-image-bytes", "image/jpeg")},
                headers={"X-API-Key": API_KEY},
            )
        assert resp.status_code == 200
        assert resp.json() == {"rooms": 1, "halls": 0, "kitchens": 0, "bathrooms": 0, "other rooms": 0}
        assert len(captured_paths) == 1
    finally:
        api_module.parse_floorplan = original_parse_floorplan
        app.dependency_overrides.clear()


CASES = [
    test_root_is_a_trivial_liveness_ping,
    test_health_all_dependencies_healthy,
    test_health_postgres_down_returns_503,
    test_health_qdrant_down_returns_503,
    test_health_agent_not_initialized_returns_503,
    test_chat_success_returns_agent_output,
    test_chat_builds_history_as_message_pairs,
    test_chat_agent_not_initialized_returns_500,
    test_chat_agent_exception_does_not_leak_raw_exception_text,
    test_ingest_route_lands_file_and_creates_queued_job,
    test_ingest_route_rejects_unsupported_file_type,
    test_ingest_route_rejects_missing_api_key,
    test_ingest_route_rejects_oversized_upload,
    test_get_ingest_job_returns_status,
    test_get_ingest_job_404_for_unknown_id,
    test_internal_parse_floorplan_rejects_missing_or_wrong_key,
    test_internal_parse_floorplan_with_valid_key_calls_parser,
    test_internal_embed_rejects_missing_key,
    test_internal_embed_with_valid_key_returns_embedding,
    test_parse_floorplan_debug_route_returns_parser_output,
]


def main() -> int:
    failures = 0
    for case in CASES:
        try:
            case()
        except AssertionError as e:
            failures += 1
            print(f"FAIL {case.__name__}: {e}")
        else:
            print(f"PASS {case.__name__}")
    if failures:
        print(f"\n{failures}/{len(CASES)} tests failed")
        return 1
    print(f"\nAll {len(CASES)} tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
