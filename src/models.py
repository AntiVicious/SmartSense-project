"""SQLAlchemy ORM models and Pydantic request/response schemas."""

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, func

from pydantic import BaseModel

from .db import Base


class Property(Base):
    __tablename__ = "properties"
    id = Column(Integer, primary_key=True, index=True)
    # Stable key for upserts: the spreadsheet's listing_id, or a hash of
    # title, location and image file (ingest_support.compute_external_id).
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
    """Status of one /ingest upload.

    The Airflow DAGs update it (the poller sets 'running'; the pipeline's
    callbacks set the final state) and GET /ingest/{job_id} reads it."""

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
    """Body for POST /internal/parse-floorplan. image_path is on the data
    volume shared by Airflow and the API."""

    image_path: str


class InternalEmbedRequest(BaseModel):
    text: str
