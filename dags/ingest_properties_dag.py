"""Property ingestion pipeline: Excel -> Postgres + Qdrant.

Triggered by dags/watch_ingest_landing_dag.py when a file lands in
INGEST_LANDING_DIR -- not on a schedule (schedule=None). dag_run.conf
carries {"spreadsheet_path": ..., "job_id": ...}.

Idempotent by construction: every row gets a stable external_id (a
listing_id column from the spreadsheet if present, otherwise a
deterministic hash of title/location/image_file -- see
src.ingest_support), and both the Postgres and Qdrant writes are upserts
keyed by it, never a blind insert. Re-running this DAG on the same file
(including via backfill, including after a partial failure) never
duplicates a row -- fix-list item 2.2.

Ordering fixes the other half of 2.2: load_qdrant runs before
load_postgres. If the Qdrant upsert fails and exhausts its retries, no
Postgres row is written for this run at all -- no more rows with no
vector. verify_load is the hard backstop regardless: it checks that
every staged external_id actually exists in *both* stores by identity
(not just that the counts match) and fails the DAG if not.

No DataFrames through XCom: each task writes its output to a Parquet
file under a per-job staging directory and passes only that path (plus
small scalars) through XCom.

Floorplan parsing and embedding computation call this app's
/internal/parse-floorplan and /internal/embed instead of running
in-process -- this image carries no torch, deliberately (see
requirements-airflow.txt).
"""

import os
import sys
from datetime import timedelta

import pendulum

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import sqlalchemy as sa  # noqa: E402
from airflow.decorators import dag, task  # noqa: E402
from airflow.exceptions import AirflowFailException  # noqa: E402
from airflow.models import Variable  # noqa: E402
from airflow.operators.python import get_current_context  # noqa: E402

from dags.support import (  # noqa: E402
    CERTIFICATES_DIR,
    IMAGES_DIR,
    QDRANT_COLLECTION,
    STAGING_ROOT,
    get_internal_api_config,
    get_postgres_engine,
    get_qdrant_client,
)
from src.ingest_support import compute_external_id, qdrant_point_id  # noqa: E402

REQUIRED_COLUMNS = ["title", "location", "price", "image_file", "listing_date", "certificates", "long_description"]


def _update_job_status(job_id, *, status, rows_ingested=None, error_message=None):
    if not job_id:
        print("No job_id in dag_run.conf; skipping ingest_jobs status update.")
        return
    engine = get_postgres_engine()
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                "UPDATE ingest_jobs SET status = :status, rows_ingested = :rows_ingested, "
                "error_message = :error_message, updated_at = now() WHERE job_id = :job_id"
            ),
            {"status": status, "rows_ingested": rows_ingested, "error_message": error_message, "job_id": job_id},
        )


def _on_dag_success(context):
    job_id = (context["dag_run"].conf or {}).get("job_id")
    rows_ingested = context["ti"].xcom_pull(task_ids="verify_load")
    _update_job_status(job_id, status="succeeded", rows_ingested=rows_ingested)


def _on_dag_failure(context):
    job_id = (context["dag_run"].conf or {}).get("job_id")
    exception = context.get("exception")
    _update_job_status(job_id, status="failed", error_message=str(exception)[:2000] if exception else "unknown error")


@dag(
    dag_id="ingest_properties",
    description="Excel -> Postgres + Qdrant, idempotent by external_id",
    schedule=None,
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    default_args={
        "retries": 3,
        "retry_delay": timedelta(minutes=2),
        "retry_exponential_backoff": True,
        "max_retry_delay": timedelta(minutes=15),
    },
    on_success_callback=_on_dag_success,
    on_failure_callback=_on_dag_failure,
    tags=["ingest"],
)
def ingest_properties():
    @task
    def extract_spreadsheet() -> dict:
        import pandas as pd

        context = get_current_context()
        conf = context["dag_run"].conf or {}
        spreadsheet_path = conf["spreadsheet_path"]
        job_id = conf["job_id"]

        if spreadsheet_path.lower().endswith(".csv"):
            df = pd.read_csv(spreadsheet_path)
        else:
            df = pd.read_excel(spreadsheet_path)

        staging_dir = os.path.join(STAGING_ROOT, job_id)
        os.makedirs(staging_dir, exist_ok=True)
        staging_path = os.path.join(staging_dir, "01_extracted.parquet")
        df.to_parquet(staging_path)

        print(f"Extracted {len(df)} rows from {spreadsheet_path}")
        return {"staging_path": staging_path, "job_id": job_id}

    @task(retries=0)  # a data-quality failure isn't fixed by retrying
    def validate_data_quality(extract_result: dict) -> dict:
        import pandas as pd

        df = pd.read_parquet(extract_result["staging_path"])

        min_rows = int(Variable.get("ingest_dq_min_rows", default_var=1))
        null_rate_threshold = float(Variable.get("ingest_dq_null_rate_threshold", default_var=0.2))

        if len(df) < min_rows:
            raise AirflowFailException(f"Data quality: {len(df)} rows, minimum is {min_rows}")

        missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_columns:
            raise AirflowFailException(f"Data quality: missing required column(s) {missing_columns}")

        for col in ("title", "location", "price"):
            null_rate = df[col].isna().mean()
            if null_rate > null_rate_threshold:
                raise AirflowFailException(
                    f"Data quality: column {col!r} is {null_rate:.0%} null (threshold {null_rate_threshold:.0%})"
                )

        if df["image_file"].isna().all():
            raise AirflowFailException("Data quality: image_file is null for every row -- nothing to parse")

        print(f"Data quality checks passed for {len(df)} rows")
        return extract_result

    @task
    def extract_certificate_text(prev: dict) -> dict:
        import fitz
        import pandas as pd

        df = pd.read_parquet(prev["staging_path"])

        def _report_text_for(certs_value):
            if not certs_value or pd.isna(certs_value):
                return ""
            text = ""
            for link in str(certs_value).split("|"):
                link = link.strip()
                if not link.lower().endswith(".pdf"):
                    continue
                pdf_path = os.path.join(CERTIFICATES_DIR, link)
                if not os.path.exists(pdf_path):
                    print(f"Warning: certificate not found: {pdf_path}")
                    continue
                try:
                    with fitz.open(pdf_path) as doc:
                        text += "".join(page.get_text() for page in doc) + "\n\n"
                except Exception as e:
                    print(f"Warning: could not parse {pdf_path}: {e}")
            return text

        certs_col = df["certificates"] if "certificates" in df.columns else pd.Series([None] * len(df))
        df["report_text"] = certs_col.apply(_report_text_for)

        staging_path = os.path.join(os.path.dirname(prev["staging_path"]), "02_with_certs.parquet")
        df.to_parquet(staging_path)
        return {**prev, "staging_path": staging_path}

    @task
    def transform_floorplans(prev: dict) -> dict:
        import pandas as pd
        import requests

        df = pd.read_parquet(prev["staging_path"])
        base_url, api_key = get_internal_api_config()

        kept_rows = []
        for _, row in df.iterrows():
            image_file = row.get("image_file")
            if not image_file or pd.isna(image_file):
                print(f"Skipping row {row.get('title')!r}: no image_file")
                continue

            image_path = os.path.join(IMAGES_DIR, str(image_file))
            resp = requests.post(
                f"{base_url}/internal/parse-floorplan",
                json={"image_path": image_path},
                headers={"X-Internal-Api-Key": api_key},
                timeout=120,
            )
            resp.raise_for_status()
            floorplan_data = resp.json()
            if floorplan_data.get("error"):
                print(f"Skipping row {row.get('title')!r}: {floorplan_data['error']}")
                continue

            row = row.to_dict()
            row["rooms"] = floorplan_data.get("rooms")
            row["halls"] = floorplan_data.get("halls")
            row["kitchens"] = floorplan_data.get("kitchens")
            row["bathrooms"] = floorplan_data.get("bathrooms")

            listing_id = row.get("listing_id")
            if listing_id and not pd.isna(listing_id):
                row["external_id"] = str(listing_id)
            else:
                row["external_id"] = compute_external_id(row.get("title"), row.get("location"), image_file)

            kept_rows.append(row)

        result_df = pd.DataFrame(kept_rows)
        staging_path = os.path.join(os.path.dirname(prev["staging_path"]), "03_transformed.parquet")
        result_df.to_parquet(staging_path)

        print(f"Transformed {len(result_df)}/{len(df)} rows (rest skipped: no image or parser error)")
        return {**prev, "staging_path": staging_path}

    @task
    def compute_embeddings(prev: dict) -> dict:
        import pandas as pd
        import requests

        df = pd.read_parquet(prev["staging_path"])
        base_url, api_key = get_internal_api_config()

        texts, embeddings = [], []
        for _, row in df.iterrows():
            text_to_embed = (
                f"Title: {row.get('title')}. Description: {row.get('long_description')}. "
                f"Location: {row.get('location')}. Reports: {row.get('report_text', '')}"
            )
            resp = requests.post(
                f"{base_url}/internal/embed",
                json={"text": text_to_embed},
                headers={"X-Internal-Api-Key": api_key},
                timeout=60,
            )
            resp.raise_for_status()
            texts.append(text_to_embed)
            embeddings.append(resp.json()["embedding"])

        df["text_to_embed"] = texts
        df["embedding"] = embeddings

        staging_path = os.path.join(os.path.dirname(prev["staging_path"]), "04_embedded.parquet")
        df.to_parquet(staging_path)
        return {**prev, "staging_path": staging_path}

    @task
    def load_qdrant(prev: dict) -> dict:
        import pandas as pd
        from qdrant_client.http.models import PointStruct

        df = pd.read_parquet(prev["staging_path"])
        client = get_qdrant_client()

        points = [
            PointStruct(
                id=qdrant_point_id(row["external_id"]),
                vector=list(row["embedding"]),
                payload={"text": row["text_to_embed"], "external_id": row["external_id"]},
            )
            for _, row in df.iterrows()
        ]
        if points:
            client.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=True)

        print(f"Upserted {len(points)} points to Qdrant")
        return {**prev, "external_ids": df["external_id"].tolist()}

    @task
    def load_postgres(prev: dict) -> dict:
        import pandas as pd
        from sqlalchemy import MetaData, Table
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        df = pd.read_parquet(prev["staging_path"])
        engine = get_postgres_engine()
        metadata = MetaData()
        properties = Table("properties", metadata, autoload_with=engine)

        upserted = 0
        with engine.begin() as conn:
            for _, row in df.iterrows():
                values = {
                    "external_id": row["external_id"],
                    "title": row.get("title"),
                    "description": row.get("long_description"),
                    "location": row.get("location"),
                    "price": None if pd.isna(row.get("price")) else float(row.get("price")),
                    "listing_date": None if pd.isna(row.get("listing_date")) else str(row.get("listing_date")),
                    "certifications_link": row.get("certificates"),
                    "floorplan_image_url": row.get("image_file"),
                    "rooms": row.get("rooms"),
                    "halls": row.get("halls"),
                    "kitchens": row.get("kitchens"),
                    "bathrooms": row.get("bathrooms"),
                }
                stmt = pg_insert(properties).values(**values)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["external_id"],
                    set_={k: v for k, v in values.items() if k != "external_id"},
                )
                conn.execute(stmt)
                upserted += 1

        print(f"Upserted {upserted} rows to Postgres")
        return {**prev, "rows_upserted": upserted}

    @task(retries=2, retry_delay=timedelta(seconds=30))
    def verify_load(prev: dict) -> int:
        external_ids = prev["external_ids"]
        engine = get_postgres_engine()
        client = get_qdrant_client()

        with engine.connect() as conn:
            result = conn.execute(
                sa.text("SELECT external_id FROM properties WHERE external_id = ANY(:ids)"),
                {"ids": external_ids},
            )
            found_in_postgres = {row[0] for row in result}

        point_ids = [qdrant_point_id(eid) for eid in external_ids]
        retrieved = client.retrieve(collection_name=QDRANT_COLLECTION, ids=point_ids, with_payload=True) if point_ids else []
        found_in_qdrant = {p.payload.get("external_id") for p in retrieved}

        staged = set(external_ids)
        missing_in_postgres = staged - found_in_postgres
        missing_in_qdrant = staged - found_in_qdrant

        if missing_in_postgres or missing_in_qdrant:
            raise AirflowFailException(
                f"verify_load: {len(missing_in_postgres)} row(s) missing from Postgres "
                f"{sorted(missing_in_postgres)[:10]}, {len(missing_in_qdrant)} missing from Qdrant "
                f"{sorted(missing_in_qdrant)[:10]}"
            )

        print(f"verify_load: all {len(staged)} rows confirmed present in both Postgres and Qdrant")
        return len(staged)

    extracted = extract_spreadsheet()
    validated = validate_data_quality(extracted)
    with_certs = extract_certificate_text(validated)
    transformed = transform_floorplans(with_certs)
    embedded = compute_embeddings(transformed)
    qdrant_loaded = load_qdrant(embedded)
    postgres_loaded = load_postgres(qdrant_loaded)
    verify_load(postgres_loaded)


ingest_properties()
