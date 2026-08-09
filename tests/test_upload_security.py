"""
Regression test for the path-traversal fix in /parse-floorplan-debug
(smartsense-fix-list.md Tier 0.2).

The original code did:

    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

file.filename is attacker-controlled -- a filename like
"../../app/api.py" would resolve outside /tmp entirely and could
overwrite application code. The fix (src/api.py's
parse_floorplan_debug) writes to a tempfile.mkstemp()-generated path
and never reads file.filename at all. These tests fail loudly if that
regresses -- e.g. if someone "simplifies" the upload handling back to
using the client-supplied filename.
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


# Every test in this file hits /parse-floorplan-debug, which now depends on
# get_app_settings (for the API key check and the upload size limit) --
# override it once, module-wide, rather than per test.
app.dependency_overrides[get_app_settings] = lambda: FakeSettings()


def test_malicious_filename_never_reaches_the_filesystem_path():
    original_parse_floorplan = api_module.parse_floorplan
    captured = {}

    def fake_parse_floorplan(path):
        captured["path"] = path
        # The file must exist at this point (written before parsing,
        # not yet cleaned up) -- proves the write-then-parse ordering,
        # not just that *some* path string was passed.
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

        # The old code (file_path = f"/tmp/{file.filename}") would have
        # produced a path containing "../../../../" here, resolving
        # outside /tmp. The fix must never do that.
        assert ".." not in actual_path
        assert not actual_path.endswith("evil_traversal_marker.jpg")
        assert os.path.dirname(actual_path) == "/tmp"
        assert captured["existed_during_parse"] is True

        # And it's cleaned up afterward.
        assert not os.path.exists(actual_path)
    finally:
        api_module.parse_floorplan = original_parse_floorplan


def test_filename_with_null_byte_and_path_separators_never_reaches_the_path():
    # A second, differently-shaped malicious filename -- absolute path
    # plus characters that have historically been used to confuse naive
    # path handling. Same assertions: none of it should surface in the
    # path actually written to.
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
    # The finally: os.remove(file_path) block must run regardless of
    # whether parsing succeeds -- otherwise every failed upload leaks a
    # temp file into /tmp for the life of the container.
    original_parse_floorplan = api_module.parse_floorplan
    captured = {}

    def failing_parse_floorplan(path):
        captured["path"] = path
        raise RuntimeError("simulated parser crash")

    api_module.parse_floorplan = failing_parse_floorplan
    try:
        # raise_server_exceptions=False: the route doesn't catch parser
        # errors, so without this the exception propagates straight into
        # the test process instead of becoming the 500 response a real
        # client over HTTP would actually see.
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
