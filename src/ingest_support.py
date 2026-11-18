"""Standard-library helpers shared by the API and the Airflow DAGs.

The Airflow image doesn't include the API's dependencies, so this module
must not import from the rest of src/.
"""

import hashlib
import os
import uuid

# Fixed namespace so qdrant_point_id() is stable across runs.
_QDRANT_POINT_NAMESPACE = uuid.UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479")


def ensure_shared_dir(path: str) -> None:
    """Creates path if missing and makes it world-writable.

    The API (root) and Airflow (uid 50000) share this directory through a
    bind mount, so whichever creates it must leave it writable for the
    other. It is internal scratch space, not exposed outside Docker.
    """
    os.makedirs(path, exist_ok=True)
    try:
        os.chmod(path, 0o777)
    except PermissionError:
        # Owned by another UID; keep its existing mode.
        pass


def compute_external_id(title, location, image_file) -> str:
    """Deterministic ID for a property row built from title, location and
    image file, so re-ingesting a spreadsheet updates rows instead of
    duplicating them.

    Used when the spreadsheet has no listing_id column. Migration 0001 has a
    copy of this logic, since migrations shouldn't import application code.
    """
    key = f"{title or ''}|{location or ''}|{image_file or ''}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def qdrant_point_id(external_id: str) -> str:
    """Deterministic Qdrant point ID derived from external_id.

    Using external_id rather than the Postgres row id keeps points matched to
    the right property even if one store is reset.
    """
    return str(uuid.uuid5(_QDRANT_POINT_NAMESPACE, external_id))
