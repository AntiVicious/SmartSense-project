"""
Tests for src/ingest.py's ingest_properties_sync() -- the Excel -> Postgres
+ Qdrant row-mapping pipeline. Uses a fake floorplan parser (injected via
parse_floorplan_fn) and a fake PDF parser (parse_local_pdf_fn) instead of
loading real YOLO/EasyOCR/PyMuPDF, an in-memory SQLite database instead of
Postgres, and hand-written fakes for the Qdrant client and the embedder --
ingest_properties_sync takes all of these as explicit arguments specifically
so this is possible.
"""

import io
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from fastapi import HTTPException  # noqa: E402

from src.db import Base  # noqa: E402
from src.ingest import ingest_properties_sync  # noqa: E402
from src.models import Property  # noqa: E402


class FakeQdrantClient:
    def __init__(self):
        self.upsert_calls = []

    def upsert(self, collection_name, points, wait=True):
        self.upsert_calls.append({"collection_name": collection_name, "points": points, "wait": wait})


class FakeEmbedder:
    """Records every text it's asked to embed, so tests can check what
    ended up in the string that gets vectorized -- the vector's actual
    values don't matter to ingest_properties_sync, only that one comes
    back per call."""

    def __init__(self):
        self.embedded_texts = []

    def embed_query(self, text):
        self.embedded_texts.append(text)
        return [0.1, 0.2, 0.3]


def _make_session_factory():
    # Fresh in-memory SQLite DB per test -- SQLAlchemy keeps a single
    # connection alive for a ":memory:" URL (SingletonThreadPool), so a
    # session_factory() opened for the write in ingest_properties_sync and
    # a second one opened afterward to read back the result share the same
    # data.
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def _make_excel_bytes(rows: list) -> bytes:
    buf = io.BytesIO()
    pd.DataFrame(rows).to_excel(buf, index=False)
    return buf.getvalue()


def _fake_parse_floorplan_fixed(result):
    def _parser(path):
        return dict(result)
    return _parser


def _fake_parse_floorplan_error_for(bad_filenames, ok_result):
    def _parser(path):
        if any(bad in path for bad in bad_filenames):
            return {"error": f"no floorplan model output for {path}"}
        return dict(ok_result)
    return _parser


FAKE_ROOM_COUNTS = {"rooms": 3, "halls": 1, "kitchens": 1, "bathrooms": 2, "other rooms": 0}


def test_ingest_maps_row_fields_and_upserts_one_point():
    session_factory = _make_session_factory()
    qdrant_client = FakeQdrantClient()
    embedder = FakeEmbedder()

    rows = [{
        "title": "Cozy Cottage",
        "location": "Springfield",
        "price": 250000,
        "listing_date": "listed-2024",
        "certificates": "fire-safety.pdf",
        "image_file": "house1.jpg",
        "long_description": "A lovely home.",
    }]

    result = ingest_properties_sync(
        _make_excel_bytes(rows),
        session_factory=session_factory,
        qdrant_client=qdrant_client,
        embedder=embedder,
        qdrant_collection="properties",
        parse_floorplan_fn=_fake_parse_floorplan_fixed(FAKE_ROOM_COUNTS),
        parse_local_pdf_fn=lambda path: "FIRE SAFETY CERTIFIED",
    )

    assert result == {"status": "success", "message": "Successfully ingested 1 properties."}

    db = session_factory()
    try:
        props = db.query(Property).all()
        assert len(props) == 1
        prop = props[0]
        assert prop.title == "Cozy Cottage"
        assert prop.location == "Springfield"
        assert prop.price == 250000.0
        assert prop.certifications_link == "fire-safety.pdf"
        assert prop.floorplan_image_url == "house1.jpg"
        assert prop.description == "A lovely home."
        # Mapped straight off the (fake) floorplan parser's output.
        assert prop.rooms == 3
        assert prop.halls == 1
        assert prop.kitchens == 1
        assert prop.bathrooms == 2
        prop_id = prop.id
    finally:
        db.close()

    assert len(qdrant_client.upsert_calls) == 1
    call = qdrant_client.upsert_calls[0]
    assert call["collection_name"] == "properties"
    assert len(call["points"]) == 1
    point = call["points"][0]
    assert point.id == prop_id
    assert point.payload["property_id"] == prop_id

    # The PDF report text (from the fake PDF parser) made it into the text
    # that got embedded, alongside the title/description/location.
    assert len(embedder.embedded_texts) == 1
    embedded = embedder.embedded_texts[0]
    assert "Cozy Cottage" in embedded
    assert "FIRE SAFETY CERTIFIED" in embedded


def test_ingest_skips_row_with_blank_image_file():
    session_factory = _make_session_factory()

    rows = [
        {"title": "No Image", "location": "X", "price": 100, "image_file": "", "certificates": "", "long_description": ""},
        {"title": "Has Image", "location": "Y", "price": 200, "image_file": "house2.jpg", "certificates": "", "long_description": ""},
    ]

    result = ingest_properties_sync(
        _make_excel_bytes(rows),
        session_factory=session_factory,
        qdrant_client=FakeQdrantClient(),
        embedder=FakeEmbedder(),
        qdrant_collection="properties",
        parse_floorplan_fn=_fake_parse_floorplan_fixed(FAKE_ROOM_COUNTS),
    )

    assert result["message"] == "Successfully ingested 1 properties."

    db = session_factory()
    try:
        titles = [p.title for p in db.query(Property).all()]
        assert titles == ["Has Image"]
    finally:
        db.close()


def test_ingest_skips_row_when_parser_reports_error():
    session_factory = _make_session_factory()

    rows = [
        {"title": "Bad Floorplan", "location": "X", "price": 100, "image_file": "missing.jpg", "certificates": "", "long_description": ""},
        {"title": "Good Floorplan", "location": "Y", "price": 200, "image_file": "house2.jpg", "certificates": "", "long_description": ""},
    ]

    result = ingest_properties_sync(
        _make_excel_bytes(rows),
        session_factory=session_factory,
        qdrant_client=FakeQdrantClient(),
        embedder=FakeEmbedder(),
        qdrant_collection="properties",
        parse_floorplan_fn=_fake_parse_floorplan_error_for(["missing.jpg"], FAKE_ROOM_COUNTS),
    )

    assert result["message"] == "Successfully ingested 1 properties."

    db = session_factory()
    try:
        titles = [p.title for p in db.query(Property).all()]
        assert titles == ["Good Floorplan"]
    finally:
        db.close()


def test_ingest_cleans_non_numeric_price_to_none():
    session_factory = _make_session_factory()

    rows = [{"title": "Weird Price", "location": "X", "price": "N/A", "image_file": "house.jpg", "certificates": "", "long_description": ""}]

    ingest_properties_sync(
        _make_excel_bytes(rows),
        session_factory=session_factory,
        qdrant_client=FakeQdrantClient(),
        embedder=FakeEmbedder(),
        qdrant_collection="properties",
        parse_floorplan_fn=_fake_parse_floorplan_fixed(FAKE_ROOM_COUNTS),
    )

    db = session_factory()
    try:
        prop = db.query(Property).one()
        assert prop.price is None
    finally:
        db.close()


def test_ingest_raises_400_on_missing_required_column():
    session_factory = _make_session_factory()

    # No "price" column at all -- df['price'] raises KeyError, which
    # ingest_properties_sync is supposed to turn into a 400, not a 500.
    rows = [{"title": "No Price Column", "location": "X", "image_file": "house.jpg"}]

    try:
        ingest_properties_sync(
            _make_excel_bytes(rows),
            session_factory=session_factory,
            qdrant_client=FakeQdrantClient(),
            embedder=FakeEmbedder(),
            qdrant_collection="properties",
            parse_floorplan_fn=_fake_parse_floorplan_fixed(FAKE_ROOM_COUNTS),
        )
    except HTTPException as e:
        assert e.status_code == 400
        assert "price" in e.detail
    else:
        raise AssertionError("expected HTTPException(400) for a missing required column")

    # And nothing got committed.
    db = session_factory()
    try:
        assert db.query(Property).count() == 0
    finally:
        db.close()


CASES = [
    test_ingest_maps_row_fields_and_upserts_one_point,
    test_ingest_skips_row_with_blank_image_file,
    test_ingest_skips_row_when_parser_reports_error,
    test_ingest_cleans_non_numeric_price_to_none,
    test_ingest_raises_400_on_missing_required_column,
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
