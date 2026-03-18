"""Sample model with pgvector embedding columns."""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from samplespace.models.base import Base


class Sample(Base):
    __tablename__ = "samples"

    id: Mapped[str] = mapped_column(primary_key=True, index=True)
    filename: Mapped[str] = mapped_column(String(255))
    key: Mapped[str | None] = mapped_column(String(10))
    bpm: Mapped[float | None]
    duration: Mapped[float | None]
    sample_type: Mapped[str | None] = mapped_column(String(50))
    clap_embedding: Mapped[list[float] | None] = mapped_column(Vector(512))
    cnn_embedding: Mapped[list[float] | None] = mapped_column(Vector(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
