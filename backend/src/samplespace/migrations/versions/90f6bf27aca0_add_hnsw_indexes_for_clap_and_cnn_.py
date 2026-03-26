"""add hnsw indexes for clap and cnn embeddings

Revision ID: 90f6bf27aca0
Revises: 2ffdfff5bb86
Create Date: 2026-03-26 15:13:49.391431

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "90f6bf27aca0"
down_revision: Union[str, None] = "2ffdfff5bb86"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # HNSW indexes for pgvector cosine similarity search.
    # HNSW is preferred over IVFFlat for datasets under ~1M rows: no training
    # step required, handles inserts without reindexing, better recall.
    op.execute(
        """
        CREATE INDEX ix_samples_clap_embedding_hnsw
        ON samples USING hnsw (clap_embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """
    )
    op.execute(
        """
        CREATE INDEX ix_samples_cnn_embedding_hnsw
        ON samples USING hnsw (cnn_embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_samples_cnn_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS ix_samples_clap_embedding_hnsw")
