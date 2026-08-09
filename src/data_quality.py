"""Data-quality checks for the ingest pipeline's raw spreadsheet.

Runs as its own Airflow task (dags/ingest_properties_dag.py's
validate_data_quality) between extract and transform, so bad data fails
loudly before any per-row processing -- floorplan parsing, embeddings --
is wasted on it.

Needs pandas, so unlike src.ingest_support this isn't stdlib-only -- but
pandas is already a dependency on both sides (requirements-heavy.txt for
the app, requirements-airflow.txt for Airflow), so this doesn't add
anything new to either.
"""

import pandas as pd


class DataQualityError(ValueError):
    """Raised on the first check that fails. The DAG task wraps this as
    an AirflowFailException so it isn't retried -- a data problem isn't
    fixed by retrying."""


def check_data_quality(
    df: pd.DataFrame,
    *,
    required_columns: list,
    min_rows: int = 1,
    null_rate_threshold: float = 0.2,
) -> None:
    """Raises DataQualityError with a specific reason on the first
    violation found. Returns None if every check passes."""
    if len(df) < min_rows:
        raise DataQualityError(f"{len(df)} rows, minimum is {min_rows}")

    missing_columns = [c for c in required_columns if c not in df.columns]
    if missing_columns:
        raise DataQualityError(f"missing required column(s) {missing_columns}")

    for col in ("title", "location", "price"):
        if col not in df.columns:
            continue  # already reported by the missing_columns check above
        null_rate = df[col].isna().mean()
        if null_rate > null_rate_threshold:
            raise DataQualityError(
                f"column {col!r} is {null_rate:.0%} null (threshold {null_rate_threshold:.0%})"
            )

    if "image_file" in df.columns and df["image_file"].isna().all():
        raise DataQualityError("image_file is null for every row -- nothing to parse")
