"""File-landing + job-row creation, shared between POST /ingest and the
backfill script (scripts/backfill_ingest.py). Needs SQLAlchemy and the
IngestJob model, so unlike ingest_support.py this is only ever run from
the app's own environment -- never imported by the Airflow DAGs.
"""

import os
import uuid
from pathlib import Path
from typing import Optional

from .ingest_support import ensure_shared_dir
from .models import IngestJob

ALLOWED_SUFFIXES = {".xlsx", ".xls", ".csv"}


class UnsupportedFileType(ValueError):
    pass


def land_ingest_file(
    *, session_factory, landing_dir: str, file_bytes: bytes, original_filename: Optional[str]
) -> str:
    """Writes file_bytes to <landing_dir>/<job_id><suffix> and inserts a
    'queued' IngestJob row. Returns the new job_id.

    The Airflow poller DAG (dags/watch_ingest_landing_dag.py) watches
    landing_dir and picks up files by job_id, not by original filename --
    job_id is server-generated (uuid4), so the destination path never
    incorporates client-supplied input (same reasoning as the
    /parse-floorplan-debug path-traversal fix).
    """
    suffix = Path(original_filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise UnsupportedFileType(f"Unsupported file type: {suffix or '(none)'}")

    job_id = str(uuid.uuid4())

    # World-writable: the Airflow poller (a different container, a
    # different UID) has to be able to rename files out of this directory.
    ensure_shared_dir(landing_dir)
    dest_path = os.path.join(landing_dir, f"{job_id}{suffix}")
    with open(dest_path, "wb") as f:
        f.write(file_bytes)

    db = session_factory()
    try:
        db.add(IngestJob(job_id=job_id, original_filename=original_filename, status="queued"))
        db.commit()
    finally:
        db.close()

    return job_id
