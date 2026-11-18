"""Data-quality checks for the ingest pipeline's raw spreadsheet.

Runs as the validate_data_quality task in dags/ingest_properties_dag.py,
before floorplan parsing and embedding, so bad input fails early.
"""

import pandas as pd


class DataQualityError(ValueError):
    """Raised on the first failed check. The DAG converts it to an
    AirflowFailException so the task isn't retried."""


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
