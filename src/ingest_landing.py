"""Saves uploaded spreadsheets for ingestion and creates their job rows.

Used by POST /ingest and scripts/backfill_ingest.py; not imported by the
Airflow DAGs.
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

    The file is named by the generated job_id, never the client's filename,
    and is picked up by dags/watch_ingest_landing_dag.py.
    """
    suffix = Path(original_filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise UnsupportedFileType(f"Unsupported file type: {suffix or '(none)'}")

    job_id = str(uuid.uuid4())

    # Airflow runs as a different user and must be able to move files out.
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
