"""Sample model with pgvector embedding columns."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from fastapi import HTTPException
from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, Float, String, cast, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from samplespace.models.base import Base


class SampleNotFound(HTTPException):
    def __init__(self) -> None:
        super().__init__(status_code=404, detail="Sample not found")


class AudioFileNotFound(HTTPException):
    def __init__(self) -> None:
        super().__init__(status_code=404, detail="Audio file not found")


class Sample(Base):
    __tablename__ = "samples"

    id: Mapped[str] = mapped_column(primary_key=True, index=True)
    filename: Mapped[str] = mapped_column(String(255))
    key: Mapped[str | None] = mapped_column(String(10))
    bpm: Mapped[int | None]
    duration: Mapped[float | None]
    sample_type: Mapped[str | None] = mapped_column(String(50))
    is_loop: Mapped[bool] = mapped_column(Boolean, server_default=text("false"))
    clap_embedding: Mapped[list[float] | None] = mapped_column(Vector(512))
    cnn_embedding: Mapped[list[float] | None] = mapped_column(Vector(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    @classmethod
    async def get(cls, db: AsyncSession, sample_id: str) -> Sample | None:
        """Get a single sample by ID."""
        result = await db.execute(select(cls).where(cls.id == sample_id))
        return result.scalar_one_or_none()

    @classmethod
    async def get_all(
        cls,
        db: AsyncSession,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[Sequence[Sample], int]:
        """List samples with pagination. Returns (samples, total_count)."""
        total_result = await db.execute(select(func.count()).select_from(cls))
        total = total_result.scalar_one()

        result = await db.execute(
            select(cls).order_by(cls.created_at.desc()).limit(limit).offset(offset),
        )
        samples = result.scalars().all()

        return samples, total

    @classmethod
    async def search_by_clap(
        cls,
        db: AsyncSession,
        query_embedding: list[float],
        *,
        key: str | None = None,
        bpm_min: int | None = None,
        bpm_max: int | None = None,
        sample_type: str | None = None,
        is_loop: bool | None = None,
        limit: int = 20,
    ) -> Sequence[Sample]:
        """Search samples by CLAP embedding using pgvector cosine distance."""
        distance = cls.clap_embedding.cosine_distance(cast(query_embedding, Vector(512)))

        stmt = select(cls, cast(distance, Float).label("distance")).where(cls.clap_embedding.is_not(None))

        if key is not None:
            stmt = stmt.where(cls.key == key)
        if bpm_min is not None:
            stmt = stmt.where(cls.bpm >= bpm_min)
        if bpm_max is not None:
            stmt = stmt.where(cls.bpm <= bpm_max)
        if sample_type is not None:
            stmt = stmt.where(cls.sample_type == sample_type)
        if is_loop is not None:
            stmt = stmt.where(cls.is_loop == is_loop)

        stmt = stmt.order_by(distance).limit(limit)

        result = await db.execute(stmt)
        rows = result.all()

        return [row.Sample for row in rows]

    @classmethod
    async def find_similar_by_cnn(
        cls,
        db: AsyncSession,
        cnn_embedding: list[float],
        *,
        exclude_id: str | None = None,
        limit: int = 10,
    ) -> Sequence[Sample]:
        """Find similar samples using CNN embedding nearest neighbors."""
        distance = cls.cnn_embedding.cosine_distance(cast(cnn_embedding, Vector(128)))

        stmt = (
            select(cls, cast(distance, Float).label("distance"))
            .where(cls.cnn_embedding.is_not(None))
            .order_by(distance)
            .limit(limit)
        )

        if exclude_id is not None:
            stmt = stmt.where(cls.id != exclude_id)

        result = await db.execute(stmt)
        rows = result.all()

        return [row.Sample for row in rows]
