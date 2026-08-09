"""
Tests for src/ingest_support.py -- pure stdlib functions shared between
the app and the Airflow DAGs. No fakes needed; nothing here touches a
model, a database, or the network.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.ingest_support import compute_external_id, qdrant_point_id  # noqa: E402


def test_external_id_is_deterministic():
    a = compute_external_id("Cozy Cottage", "Springfield", "house1.jpg")
    b = compute_external_id("Cozy Cottage", "Springfield", "house1.jpg")
    assert a == b


def test_external_id_matches_known_value():
    # Pinned against the exact value migrations/versions/0001_...  backfills
    # for the same inputs -- if this ever drifts from the migration's
    # (deliberately duplicated) copy of the algorithm, old rows and newly
    # ingested rows would stop matching.
    assert compute_external_id("Old Cottage", "Springfield", "house1.jpg") == "a7a1bb912903124c132a61b9e231eca6"


def test_external_id_differs_for_different_inputs():
    a = compute_external_id("Cozy Cottage", "Springfield", "house1.jpg")
    b = compute_external_id("Cozy Cottage", "Springfield", "house2.jpg")
    c = compute_external_id("Cozy Cottage", "Shelbyville", "house1.jpg")
    d = compute_external_id("Modern Loft", "Springfield", "house1.jpg")
    assert len({a, b, c, d}) == 4


def test_external_id_handles_none_fields():
    # A row missing location or image_file shouldn't raise -- it should
    # just fold into the hash as an empty segment, same as any other row.
    result = compute_external_id("Title Only", None, None)
    assert isinstance(result, str) and len(result) == 32


def test_external_id_is_32_hex_chars():
    result = compute_external_id("Anything", "Anywhere", "any.jpg")
    assert len(result) == 32
    int(result, 16)  # raises ValueError if not valid hex


def test_qdrant_point_id_is_deterministic():
    eid = compute_external_id("Cozy Cottage", "Springfield", "house1.jpg")
    a = qdrant_point_id(eid)
    b = qdrant_point_id(eid)
    assert a == b


def test_qdrant_point_id_is_a_valid_uuid():
    import uuid

    eid = compute_external_id("Cozy Cottage", "Springfield", "house1.jpg")
    point_id = qdrant_point_id(eid)
    parsed = uuid.UUID(point_id)  # raises ValueError if not a valid UUID
    assert str(parsed) == point_id


def test_qdrant_point_id_differs_for_different_external_ids():
    a = qdrant_point_id(compute_external_id("A", "X", "a.jpg"))
    b = qdrant_point_id(compute_external_id("B", "Y", "b.jpg"))
    assert a != b


def test_qdrant_point_id_never_derived_from_a_sql_autoincrement_id():
    # Regression guard for the fix-list bug: point IDs must come from
    # content identity, not from a Postgres id that a volume reset could
    # hand out to a completely different row. "1" and "2" are what an
    # autoincrement id would look like -- qdrant_point_id must never
    # produce these for arbitrary external_ids.
    for eid in ("1", "2", "123"):
        point_id = qdrant_point_id(eid)
        assert point_id not in ("1", "2", "123")
        import uuid

        uuid.UUID(point_id)  # still a real UUID, not a passthrough of the input


CASES = [
    test_external_id_is_deterministic,
    test_external_id_matches_known_value,
    test_external_id_differs_for_different_inputs,
    test_external_id_handles_none_fields,
    test_external_id_is_32_hex_chars,
    test_qdrant_point_id_is_deterministic,
    test_qdrant_point_id_is_a_valid_uuid,
    test_qdrant_point_id_differs_for_different_external_ids,
    test_qdrant_point_id_never_derived_from_a_sql_autoincrement_id,
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
