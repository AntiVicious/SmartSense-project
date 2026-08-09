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
import sys
from contextlib import asynccontextmanager

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

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
    QDRANT_VECTOR_COLLECTION = "properties"


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
    engine = create_engine("sqlite:///:memory:")
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


def test_chat_agent_exception_returns_500_with_detail():
    fake_agent = FakeAgentExecutor(raise_exc=RuntimeError("groq exploded"))
    app.dependency_overrides[get_app_agent_executor] = lambda: fake_agent
    try:
        with TestClient(app) as client:
            resp = client.post("/chat", json={"query": "hi", "history": []})
        assert resp.status_code == 500
        assert "groq exploded" in resp.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_ingest_route_wires_dependency_overrides():
    session_factory = _make_session_factory()
    qdrant_client = FakeQdrantClient()
    embedder = FakeEmbedder()

    app.dependency_overrides[get_app_session_factory] = lambda: session_factory
    app.dependency_overrides[get_app_qdrant_client] = lambda: qdrant_client
    app.dependency_overrides[get_app_embedder] = lambda: embedder
    app.dependency_overrides[get_app_settings] = lambda: FakeSettings()
    try:
        # No image_file on the row -> the row is skipped before
        # ingest_properties_sync would ever call the *real* floorplan
        # parser, so this exercises the route's DI wiring and response
        # shape without needing a fake parser injected (that's covered
        # in tests/test_ingest.py, at the function level).
        rows = [{"title": "No Image Row", "location": "X", "price": 100, "image_file": ""}]
        excel_bytes = _make_excel_bytes(rows)
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
            )
        assert resp.status_code == 200
        assert resp.json() == {"status": "success", "message": "Successfully ingested 0 properties."}
        assert qdrant_client.upsert_calls == []  # nothing to upsert, no points built
    finally:
        app.dependency_overrides.clear()


def test_parse_floorplan_debug_route_returns_parser_output():
    original_parse_floorplan = api_module.parse_floorplan
    captured_paths = []

    def fake_parse_floorplan(path):
        captured_paths.append(path)
        return {"rooms": 1, "halls": 0, "kitchens": 0, "bathrooms": 0, "other rooms": 0}

    api_module.parse_floorplan = fake_parse_floorplan
    try:
        with TestClient(app) as client:
            resp = client.post(
                "/parse-floorplan-debug",
                files={"file": ("plan.jpg", b"fake-image-bytes", "image/jpeg")},
            )
        assert resp.status_code == 200
        assert resp.json() == {"rooms": 1, "halls": 0, "kitchens": 0, "bathrooms": 0, "other rooms": 0}
        assert len(captured_paths) == 1
    finally:
        api_module.parse_floorplan = original_parse_floorplan


CASES = [
    test_root_is_a_trivial_liveness_ping,
    test_health_all_dependencies_healthy,
    test_health_postgres_down_returns_503,
    test_health_qdrant_down_returns_503,
    test_health_agent_not_initialized_returns_503,
    test_chat_success_returns_agent_output,
    test_chat_builds_history_as_message_pairs,
    test_chat_agent_not_initialized_returns_500,
    test_chat_agent_exception_returns_500_with_detail,
    test_ingest_route_wires_dependency_overrides,
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
