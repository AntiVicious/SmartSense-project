#!/usr/bin/env python
"""Backfill: land a directory of historical spreadsheets for the Airflow
poller DAG to pick up, the same way POST /ingest does.

There's no separate "backfill mode" in the DAG itself -- this script just
calls the exact same src.ingest_landing.land_ingest_file() the API uses,
once per file, so every backfilled file gets a real ingest_jobs row and
goes through dags/watch_ingest_landing_dag.py -> dags/ingest_properties_dag.py
like any other upload. Re-running this script (or dropping the same file
again by any other means) is safe: external_id is derived from row
content, not upload time, so ingestion is idempotent regardless of how
many times a given spreadsheet gets backfilled.

Run from the app's environment (needs src.db/src.models, which pull in
the same dependencies the API does) -- not from Airflow's lean image:

    docker compose run --rm app python scripts/backfill_ingest.py data/backfill/

Usage:
    python scripts/backfill_ingest.py <directory-of-spreadsheets>
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.config import get_settings  # noqa: E402
from src.db import get_session_factory  # noqa: E402
from src.ingest_landing import ALLOWED_SUFFIXES, UnsupportedFileType, land_ingest_file  # noqa: E402


def backfill_directory(directory: str) -> list[str]:
    settings = get_settings()
    session_factory = get_session_factory()

    job_ids = []
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        suffix = os.path.splitext(name)[1].lower()
        if suffix not in ALLOWED_SUFFIXES:
            print(f"Skipping {name}: unsupported file type {suffix or '(none)'}")
            continue

        with open(path, "rb") as f:
            file_bytes = f.read()

        try:
            job_id = land_ingest_file(
                session_factory=session_factory,
                landing_dir=settings.INGEST_LANDING_DIR,
                file_bytes=file_bytes,
                original_filename=name,
            )
        except UnsupportedFileType as e:
            print(f"Skipping {name}: {e}")
            continue

        print(f"{name} -> job_id={job_id}")
        job_ids.append(job_id)

    return job_ids


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <directory-of-spreadsheets>", file=sys.stderr)
        return 1

    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"Not a directory: {directory}", file=sys.stderr)
        return 1

    job_ids = backfill_directory(directory)
    print(
        f"\nLanded {len(job_ids)} file(s). watch_ingest_landing picks them up "
        "on its next run (every 1 minute)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
