"""Add properties.external_id and the ingest_jobs table

This is the migration for an instance that predates the Airflow ingest
pipeline -- a fresh install can rely on Base.metadata.create_all() in
api.py's lifespan instead (it builds the current schema, external_id
included, from nothing). This migration exists for anyone upgrading an
already-running instance with existing properties rows.

external_id is added nullable first, backfilled using the exact same
sha256(title|location|floorplan_image_url) scheme as
src.ingest_support.compute_external_id (duplicated here deliberately --
migrations should not import application code that might change shape
out from under an old migration), then tightened to NOT NULL + UNIQUE.

Revision ID: 0001
Revises:
Create Date: 2026-08-09

"""

import hashlib

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def _compute_external_id(title, location, image_file) -> str:
    key = f"{title or ''}|{location or ''}|{image_file or ''}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def upgrade() -> None:
    bind = op.get_bind()

    op.add_column("properties", sa.Column("external_id", sa.String(), nullable=True))

    # Backfill any pre-existing rows so the NOT NULL constraint below can
    # be applied. A fresh install has no rows at this point -- this only
    # does real work when upgrading a live instance.
    rows = bind.execute(
        sa.text("SELECT id, title, location, floorplan_image_url FROM properties WHERE external_id IS NULL")
    ).fetchall()
    for row in rows:
        external_id = _compute_external_id(row.title, row.location, row.floorplan_image_url)
        bind.execute(
            sa.text("UPDATE properties SET external_id = :external_id WHERE id = :id"),
            {"external_id": external_id, "id": row.id},
        )

    op.alter_column("properties", "external_id", nullable=False)
    op.create_index("ix_properties_external_id", "properties", ["external_id"], unique=True)

    op.create_table(
        "ingest_jobs",
        sa.Column("job_id", sa.String(), primary_key=True),
        sa.Column("original_filename", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False, server_default="queued"),
        sa.Column("dag_run_id", sa.String(), nullable=True),
        sa.Column("rows_ingested", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
    )


def downgrade() -> None:
    op.drop_table("ingest_jobs")
    op.drop_index("ix_properties_external_id", table_name="properties")
    op.drop_column("properties", "external_id")
