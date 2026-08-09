"""Pure, dependency-light helpers shared between the app and the Airflow
DAGs (dags/). Deliberately has zero imports from src.db/src.models/
src.floorplan/src.config -- Airflow's image doesn't carry torch, and this
module needs to stay importable without it. Stdlib only.
"""

import hashlib
import uuid

# Fixed namespace so qdrant_point_id(external_id) is stable across
# processes and re-runs -- any fixed UUID works, this one has no special
# meaning beyond being constant.
_QDRANT_POINT_NAMESPACE = uuid.UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479")


def compute_external_id(title, location, image_file) -> str:
    """Deterministic identity for a property row: the same (title,
    location, image_file) always produces the same external_id, so
    re-ingesting the same spreadsheet -- including via backfill, including
    after a partial failure and retry -- upserts instead of duplicating.

    Only used when the source spreadsheet has no business key of its own
    (see dags/ingest_properties_dag.py, which prefers a listing_id column
    when present). Mirrored in migrations/versions/0001_... for backfilling
    pre-existing rows -- deliberately duplicated rather than imported,
    since a migration should not depend on application code that might
    change shape later.
    """
    key = f"{title or ''}|{location or ''}|{image_file or ''}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def qdrant_point_id(external_id: str) -> str:
    """Deterministic Qdrant point ID derived from external_id -- never the
    Postgres autoincrement id. Fixes the coupling flagged in the fix list:
    if the Postgres volume is ever reset while Qdrant persists, a
    point ID derived from external_id still resolves to the same logical
    property, instead of colliding with whatever new row gets the same
    autoincrement id next.
    """
    return str(uuid.uuid5(_QDRANT_POINT_NAMESPACE, external_id))
