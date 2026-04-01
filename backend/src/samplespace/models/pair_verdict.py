from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, String, func, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

# Note: thread_id is stored as a plain string (no FK) because the threads
# table has a composite PK (thread_id, agent_type).
from samplespace.models.base import Base


class PairVerdict(Base):
    __tablename__ = "pair_verdicts"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    thread_id: Mapped[str] = mapped_column(String(255))
    sample_a_id: Mapped[str] = mapped_column(String(255), ForeignKey("samples.id"))
    sample_b_id: Mapped[str] = mapped_column(String(255), ForeignKey("samples.id"))
    verdict: Mapped[bool]
    pair_score: Mapped[float]
    pair_score_detail: Mapped[dict[str, object] | None] = mapped_column(JSONB, nullable=True)
    pair_features: Mapped[dict[str, object] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_pair_verdicts_pair", "sample_a_id", "sample_b_id"),
        Index("ix_pair_verdicts_thread", "thread_id"),
    )

    @classmethod
    async def create(
        cls,
        db: AsyncSession,
        *,
        thread_id: str,
        sample_a_id: str,
        sample_b_id: str,
        verdict: bool,
        pair_score: float,
        pair_score_detail: dict[str, object] | None = None,
    ) -> PairVerdict:
        """Create a new pair verdict with canonical sample ordering (a < b)."""
        # Canonical ordering: ensure sample_a_id < sample_b_id
        if sample_a_id > sample_b_id:
            sample_a_id, sample_b_id = sample_b_id, sample_a_id

        pair_verdict = cls(
            thread_id=thread_id,
            sample_a_id=sample_a_id,
            sample_b_id=sample_b_id,
            verdict=verdict,
            pair_score=pair_score,
            pair_score_detail=pair_score_detail,
        )
        db.add(pair_verdict)
        await db.flush()
        await db.refresh(pair_verdict)
        return pair_verdict

    @classmethod
    async def get(cls, db: AsyncSession, verdict_id: int) -> PairVerdict | None:
        result = await db.execute(select(cls).where(cls.id == verdict_id))
        return result.scalar_one_or_none()

    @classmethod
    async def get_by_pair(
        cls,
        db: AsyncSession,
        sample_a_id: str,
        sample_b_id: str,
    ) -> Sequence[PairVerdict]:
        if sample_a_id > sample_b_id:
            sample_a_id, sample_b_id = sample_b_id, sample_a_id

        result = await db.execute(
            select(cls)
            .where(cls.sample_a_id == sample_a_id, cls.sample_b_id == sample_b_id)
            .order_by(cls.created_at.desc())
        )
        return result.scalars().all()

    @classmethod
    async def get_by_thread(cls, db: AsyncSession, thread_id: str) -> Sequence[PairVerdict]:
        result = await db.execute(select(cls).where(cls.thread_id == thread_id).order_by(cls.created_at.desc()))
        return result.scalars().all()

    @classmethod
    async def count_all(cls, db: AsyncSession) -> int:
        result = await db.execute(select(func.count()).select_from(cls))
        return result.scalar_one()

    @classmethod
    async def update_features(
        cls,
        db: AsyncSession,
        verdict_id: int,
        features: dict[str, float],
    ) -> None:
        verdict = await cls.get(db, verdict_id)
        if verdict:
            verdict.pair_features = features  # type: ignore[assignment]
            await db.flush()
