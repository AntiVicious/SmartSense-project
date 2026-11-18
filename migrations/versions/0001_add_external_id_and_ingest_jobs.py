"""Add properties.external_id and the ingest_jobs table

Upgrades databases created before the Airflow pipeline. Fresh installs get
the current schema from Base.metadata.create_all() in api.py's lifespan.

external_id is added as nullable, filled using the same
sha256(title|location|floorplan_image_url) scheme as
src.ingest_support.compute_external_id (copied, since migrations shouldn't
import application code), then made NOT NULL and UNIQUE.

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

    # Fill external_id for existing rows before adding NOT NULL.
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
