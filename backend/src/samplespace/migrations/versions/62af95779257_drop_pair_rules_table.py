"""drop pair_rules table

Revision ID: 62af95779257
Revises: 09f754840071
Create Date: 2026-04-03 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "62af95779257"
down_revision = "09f754840071"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index("ix_pair_rules_active", table_name="pair_rules")
    op.drop_table("pair_rules")


def downgrade() -> None:
    op.create_table(
        "pair_rules",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("type_pair", sa.String(length=100), nullable=False),
        sa.Column("feature_name", sa.String(length=100), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("direction", sa.String(length=10), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("sample_count", sa.Integer(), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_pair_rules_active", "pair_rules", ["is_active"])
