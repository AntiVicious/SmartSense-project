"""
Path-traversal tests for /parse-floorplan-debug.

The client controls file.filename, so a name like "../../app/api.py" must
never be used as a path. The endpoint writes to a tempfile.mkstemp() path
instead; these tests fail if the filename ever reaches the write path.
"""

import os
import sys
from contextlib import asynccontextmanager

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from fastapi.testclient import TestClient  # noqa: E402

import src.api as api_module  # noqa: E402
from src.api import app, get_app_settings  # noqa: E402

API_KEY = "test-api-key"
AUTH_HEADER = {"X-API-Key": API_KEY}


@asynccontextmanager
async def _noop_lifespan(_app):
    yield


app.router.lifespan_context = _noop_lifespan


class FakeSettings:
    def __init__(self, max_upload_size_mb=15):
        self.API_KEY = API_KEY
        self.MAX_UPLOAD_SIZE_MB = max_upload_size_mb

    @property
    def max_upload_size_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024


# /parse-floorplan-debug needs settings for the API key and size limit;
# override them once for the whole module.
app.dependency_overrides[get_app_settings] = lambda: FakeSettings()


def test_malicious_filename_never_reaches_the_filesystem_path():
    original_parse_floorplan = api_module.parse_floorplan
    captured = {}

    def fake_parse_floorplan(path):
        captured["path"] = path
        # The file exists during parsing (written first, removed after).
        captured["existed_during_parse"] = os.path.exists(path)
        return {"rooms": 0, "halls": 0, "kitchens": 0, "bathrooms": 0, "other rooms": 0}

    api_module.parse_floorplan = fake_parse_floorplan
    try:
        malicious_filename = "../../../../tmp/evil_traversal_marker.jpg"
        with TestClient(app) as client:
            resp = client.post(
                "/parse-floorplan-debug",
                files={"file": (malicious_filename, b"not a real image", "image/jpeg")},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        actual_path = captured["path"]

        # The path must not contain the client-supplied "../" segments.
        assert ".." not in actual_path
        assert not actual_path.endswith("evil_traversal_marker.jpg")
        assert os.path.dirname(actual_path) == "/tmp"
        assert captured["existed_during_parse"] is True

        # Removed afterwards
        assert not os.path.exists(actual_path)
    finally:
        api_module.parse_floorplan = original_parse_floorplan


def test_filename_with_null_byte_and_path_separators_never_reaches_the_path():
    # An absolute path with unusual characters must not reach the write
    # path either.
    original_parse_floorplan = api_module.parse_floorplan
    captured = {}

    def fake_parse_floorplan(path):
        captured["path"] = path
        return {"rooms": 0, "halls": 0, "kitchens": 0, "bathrooms": 0, "other rooms": 0}

    api_module.parse_floorplan = fake_parse_floorplan
    try:
        malicious_filename = "/etc/passwd"
        with TestClient(app) as client:
            resp = client.post(
                "/parse-floorplan-debug",
                files={"file": (malicious_filename, b"not a real image", "image/jpeg")},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        actual_path = captured["path"]
        assert actual_path != "/etc/passwd"
        assert os.path.dirname(actual_path) == "/tmp"
        assert not os.path.exists(actual_path)
    finally:
        api_module.parse_floorplan = original_parse_floorplan


def test_temp_file_is_cleaned_up_even_if_parsing_fails():
    # The temp file must be removed even when parsing fails.
    original_parse_floorplan = api_module.parse_floorplan
    captured = {}

    def failing_parse_floorplan(path):
        captured["path"] = path
        raise RuntimeError("simulated parser crash")

    api_module.parse_floorplan = failing_parse_floorplan
    try:
        # Return the 500 response instead of raising the parser error in
        # the test.
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/parse-floorplan-debug",
                files={"file": ("plan.jpg", b"not a real image", "image/jpeg")},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 500
        assert "path" in captured
        assert not os.path.exists(captured["path"])
    finally:
        api_module.parse_floorplan = original_parse_floorplan


def test_missing_api_key_is_rejected():
    with TestClient(app) as client:
        resp = client.post(
            "/parse-floorplan-debug",
            files={"file": ("plan.jpg", b"not a real image", "image/jpeg")},
        )
    assert resp.status_code == 401


def test_non_image_content_type_is_rejected():
    with TestClient(app) as client:
        resp = client.post(
            "/parse-floorplan-debug",
            files={"file": ("plan.pdf", b"%PDF-1.4 not really", "application/pdf")},
            headers=AUTH_HEADER,
        )
    assert resp.status_code == 400


def test_oversized_upload_is_rejected():
    app.dependency_overrides[get_app_settings] = lambda: FakeSettings(max_upload_size_mb=1)
    try:
        oversized = b"x" * (2 * 1024 * 1024)  # 2MB against a 1MB limit
        with TestClient(app) as client:
            resp = client.post(
                "/parse-floorplan-debug",
                files={"file": ("plan.jpg", oversized, "image/jpeg")},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 413
    finally:
        app.dependency_overrides[get_app_settings] = lambda: FakeSettings()


CASES = [
    test_malicious_filename_never_reaches_the_filesystem_path,
    test_filename_with_null_byte_and_path_separators_never_reaches_the_path,
    test_temp_file_is_cleaned_up_even_if_parsing_fails,
    test_missing_api_key_is_rejected,
    test_non_image_content_type_is_rejected,
    test_oversized_upload_is_rejected,
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
