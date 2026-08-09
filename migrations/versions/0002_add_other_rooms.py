"""Add properties.other_rooms

parse_floorplan() has always returned an "other rooms" count (labels the
substring cascade can't place in kitchens/halls/rooms/bathrooms), but
nothing downstream ever had a column to put it in -- dags/ingest_properties_dag.py's
transform_floorplans never read that key out of the response, so it was
silently dropped on every ingest. This adds the column and (in the same
change, outside this migration) the DAG read/write path.

Nullable, no backfill: existing rows' original floorplan_data.get("other
rooms") value isn't recoverable from what's stored -- only a future
re-ingest of those rows (idempotent upsert, same external_id) will
populate it.

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-09

"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("properties", sa.Column("other_rooms", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("properties", "other_rooms")
