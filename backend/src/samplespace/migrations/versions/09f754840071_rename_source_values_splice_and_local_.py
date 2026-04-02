"""rename source default from local to library

Revision ID: 09f754840071
Revises: 90f6bf27aca0
Create Date: 2026-04-02 11:30:34.468531

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "09f754840071"
down_revision = "90f6bf27aca0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("samples", "source", server_default="library")


def downgrade() -> None:
    op.alter_column("samples", "source", server_default="local")
