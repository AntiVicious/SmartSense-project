# Based on the plain Airflow image; it doesn't need PyTorch or LangChain.
FROM apache/airflow:2.10.5-python3.10

COPY --chown=airflow:root requirements-airflow.txt /requirements-airflow.txt
RUN pip install --no-cache-dir -r /requirements-airflow.txt

# The DAGs share these two modules with the API so both compute the same
# row IDs and data-quality checks.
COPY --chown=airflow:root src/__init__.py /opt/airflow/src/__init__.py
COPY --chown=airflow:root src/ingest_support.py /opt/airflow/src/ingest_support.py
COPY --chown=airflow:root src/data_quality.py /opt/airflow/src/data_quality.py

COPY --chown=airflow:root dags/ /opt/airflow/dags/
