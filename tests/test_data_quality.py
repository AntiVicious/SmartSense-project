"""
Tests for src/data_quality.py's check_data_quality() -- extracted from
dags/ingest_properties_dag.py's validate_data_quality task specifically so
it's testable without a running Airflow. Needs pandas (already a project
dependency), nothing else.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd  # noqa: E402

from src.data_quality import DataQualityError, check_data_quality  # noqa: E402

REQUIRED_COLUMNS = [
    "title",
    "location",
    "price",
    "image_file",
    "listing_date",
    "certificates",
    "long_description",
]


def _good_df(n=3):
    return pd.DataFrame(
        [
            {
                "title": f"Property {i}",
                "location": f"City {i}",
                "price": 100000 + i,
                "image_file": f"house{i}.jpg",
                "listing_date": "2024-01-01",
                "certificates": "",
                "long_description": "A nice place.",
            }
            for i in range(n)
        ]
    )


def test_passes_on_well_formed_data():
    check_data_quality(_good_df(), required_columns=REQUIRED_COLUMNS)  # no exception


def test_fails_on_empty_dataframe():
    df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    try:
        check_data_quality(df, required_columns=REQUIRED_COLUMNS, min_rows=1)
    except DataQualityError as e:
        assert "minimum is 1" in str(e)
    else:
        raise AssertionError("expected DataQualityError for an empty dataframe")


def test_fails_below_configured_min_rows():
    df = _good_df(2)
    try:
        check_data_quality(df, required_columns=REQUIRED_COLUMNS, min_rows=5)
    except DataQualityError as e:
        assert "2 rows, minimum is 5" in str(e)
    else:
        raise AssertionError("expected DataQualityError for too few rows")


def test_fails_on_missing_required_column():
    df = _good_df().drop(columns=["price"])
    try:
        check_data_quality(df, required_columns=REQUIRED_COLUMNS)
    except DataQualityError as e:
        assert "price" in str(e)
    else:
        raise AssertionError("expected DataQualityError for a missing column")


def test_fails_when_null_rate_exceeds_threshold():
    df = _good_df(10)
    df.loc[0:6, "title"] = None  # 7/10 = 70% null
    try:
        check_data_quality(df, required_columns=REQUIRED_COLUMNS, null_rate_threshold=0.2)
    except DataQualityError as e:
        assert "title" in str(e) and "70%" in str(e)
    else:
        raise AssertionError("expected DataQualityError for excessive null rate")


def test_passes_when_null_rate_is_within_threshold():
    df = _good_df(10)
    df.loc[0, "title"] = None  # 1/10 = 10% null
    check_data_quality(df, required_columns=REQUIRED_COLUMNS, null_rate_threshold=0.2)  # no exception


def test_fails_when_image_file_is_null_for_every_row():
    df = _good_df(3)
    df["image_file"] = None
    try:
        check_data_quality(df, required_columns=REQUIRED_COLUMNS)
    except DataQualityError as e:
        assert "image_file" in str(e)
    else:
        raise AssertionError("expected DataQualityError when every image_file is null")


def test_passes_when_only_some_image_files_are_null():
    # Individual rows missing an image are a normal, tolerated case
    # (that row gets skipped downstream) -- only *all* rows missing it
    # is a data-quality failure.
    df = _good_df(3)
    df.loc[0, "image_file"] = None
    check_data_quality(df, required_columns=REQUIRED_COLUMNS)  # no exception


CASES = [
    test_passes_on_well_formed_data,
    test_fails_on_empty_dataframe,
    test_fails_below_configured_min_rows,
    test_fails_on_missing_required_column,
    test_fails_when_null_rate_exceeds_threshold,
    test_passes_when_null_rate_is_within_threshold,
    test_fails_when_image_file_is_null_for_every_row,
    test_passes_when_only_some_image_files_are_null,
]


def main() -> int:
    failures = 0
    for case in CASES:
        try:
            case()
        except AssertionError as e:
            failures += 1
            print(f"FAIL {case.__name__}: {e}")
        else:
            print(f"PASS {case.__name__}")
    if failures:
        print(f"\n{failures}/{len(CASES)} tests failed")
        return 1
    print(f"\nAll {len(CASES)} tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
