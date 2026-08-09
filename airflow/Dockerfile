# Deliberately based on the plain Airflow image, not the app's own image
# -- see requirements-airflow.txt for why (no torch, no langchain).
FROM apache/airflow:2.10.5-python3.10

COPY --chown=airflow:root requirements-airflow.txt /requirements-airflow.txt
RUN pip install --no-cache-dir -r /requirements-airflow.txt

# DAGs import src.ingest_support (pure stdlib -- compute_external_id,
# qdrant_point_id) and src.data_quality (needs only pandas, already in
# requirements-airflow.txt) so both the app and Airflow compute the same
# identity/checks for the same row. Only these two modules, not the rest
# of src/: this image should never need src/db.py's langchain_community
# import.
COPY --chown=airflow:root src/__init__.py /opt/airflow/src/__init__.py
COPY --chown=airflow:root src/ingest_support.py /opt/airflow/src/ingest_support.py
COPY --chown=airflow:root src/data_quality.py /opt/airflow/src/data_quality.py

COPY --chown=airflow:root dags/ /opt/airflow/dags/
