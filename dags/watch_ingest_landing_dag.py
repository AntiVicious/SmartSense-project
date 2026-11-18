"""Watches INGEST_LANDING_DIR and triggers dags/ingest_properties_dag.py
once per landed file.

Runs every minute instead of using a sensor, which would need Airflow's
triggerer. Each file is claimed by an atomic rename into _claimed/ before
triggering, so overlapping runs never process a file twice.

Files landed by scripts/backfill_ingest.py are processed the same way.
"""

import os
import sys
import uuid
from datetime import timedelta

import pendulum

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import sqlalchemy as sa  # noqa: E402
from airflow.decorators import dag, task  # noqa: E402

from dags.support import INGEST_LANDING_DIR, get_postgres_engine  # noqa: E402
from src.ingest_support import ensure_shared_dir  # noqa: E402

CLAIMED_DIR = os.path.join(INGEST_LANDING_DIR, "_claimed")


@dag(
    dag_id="watch_ingest_landing",
    description="Polls INGEST_LANDING_DIR and triggers ingest_properties once per file",
    schedule=timedelta(minutes=1),
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(seconds=30)},
    tags=["ingest"],
)
def watch_ingest_landing():
    @task
    def discover_landed_files() -> list:
        if not os.path.isdir(INGEST_LANDING_DIR):
            return []
        entries = []
        for name in sorted(os.listdir(INGEST_LANDING_DIR)):
            full_path = os.path.join(INGEST_LANDING_DIR, name)
            if not os.path.isfile(full_path):
                continue  # skip the _claimed/ directory
            job_id, _ext = os.path.splitext(name)
            entries.append({"filename": name, "job_id": job_id, "path": full_path})
        return entries

    @task
    def claim_and_trigger(file_info: dict):
        from airflow.api.common.trigger_dag import trigger_dag

        job_id = file_info["job_id"]
        engine = get_postgres_engine()

        with engine.connect() as conn:
            exists = conn.execute(
                sa.text("SELECT 1 FROM ingest_jobs WHERE job_id = :job_id"), {"job_id": job_id}
            ).first()
        if not exists:
            # No job row: not landed by POST /ingest or the backfill
            # script, so skip it.
            print(f"No ingest_jobs row for job_id={job_id!r}; leaving {file_info['path']} untouched")
            return

        ensure_shared_dir(CLAIMED_DIR)
        claimed_path = os.path.join(CLAIMED_DIR, file_info["filename"])
        try:
            # Atomic rename: fails if another run already claimed the file.
            os.rename(file_info["path"], claimed_path)
        except FileNotFoundError:
            print(f"{file_info['path']} already claimed by another run; skipping")
            return

        run_id = f"triggered__{job_id}__{uuid.uuid4().hex[:8]}"
        dag_run = trigger_dag(
            dag_id="ingest_properties",
            run_id=run_id,
            conf={"spreadsheet_path": claimed_path, "job_id": job_id},
        )

        with engine.begin() as conn:
            conn.execute(
                sa.text(
                    "UPDATE ingest_jobs SET status = 'running', dag_run_id = :dag_run_id, "
                    "updated_at = now() WHERE job_id = :job_id"
                ),
                {"dag_run_id": dag_run.run_id if dag_run else run_id, "job_id": job_id},
            )
        print(f"Triggered ingest_properties run {run_id} for job {job_id}")

    claim_and_trigger.expand(file_info=discover_landed_files())


watch_ingest_landing()
