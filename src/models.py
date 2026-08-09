"""SQLAlchemy ORM models and Pydantic request/response schemas."""

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, func

from pydantic import BaseModel

from .db import Base


class Property(Base):
    __tablename__ = "properties"
    id = Column(Integer, primary_key=True, index=True)
    # Stable identity for upserts: either a business key from the source
    # spreadsheet or a deterministic hash of (title, location, image_file)
    # -- see src/ingest_support.py:compute_external_id. Never the
    # autoincrement id, and never derived from upload time, so re-running
    # the same input (including via backfill) is idempotent by construction.
    external_id = Column(String, unique=True, index=True, nullable=False)
    title = Column(String, index=True)
    description = Column(Text)
    location = Column(String)
    price = Column(Float)
    listing_date = Column(String)
    certifications_link = Column(String)
    floorplan_image_url = Column(String)
    rooms = Column(Integer)
    halls = Column(Integer)
    kitchens = Column(Integer)
    bathrooms = Column(Integer)
    other_rooms = Column(Integer)


class IngestJob(Base):
    """Tracks one /ingest upload end to end. Written by the Airflow DAGs
    (poller sets 'running', the pipeline DAG's success/failure callbacks
    set the terminal state) and read by GET /ingest/{job_id} -- the API
    never queries Airflow directly for status, only this table."""

    __tablename__ = "ingest_jobs"
    job_id = Column(String, primary_key=True)
    original_filename = Column(String, nullable=True)
    status = Column(String, nullable=False, default="queued")  # queued|running|succeeded|failed
    dag_run_id = Column(String, nullable=True)
    rows_ingested = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class ChatRequest(BaseModel):
    query: str
    history: list[tuple[str, str]] = []


class InternalParseFloorplanRequest(BaseModel):
    """Body for POST /internal/parse-floorplan. image_path is a path on the
    shared data volume (Airflow and the app both mount it) -- the Airflow
    task never uploads image bytes over HTTP, just points at the file."""

    image_path: str


class InternalEmbedRequest(BaseModel):
    text: str
