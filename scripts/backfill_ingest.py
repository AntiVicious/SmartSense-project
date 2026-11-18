#!/usr/bin/env python
"""Lands a directory of spreadsheets for ingestion, the same way POST /ingest
does.

Each file gets an ingest_jobs row and goes through the normal Airflow DAGs.
Re-running is safe, because rows are keyed by content-derived IDs.

Run it in the api container, which has the required dependencies:

    docker compose run --rm api python scripts/backfill_ingest.py data/backfill/

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
