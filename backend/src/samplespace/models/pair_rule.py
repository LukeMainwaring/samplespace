"""PairRule model for storing learned pairing preferences.

Schema defined now; analysis/extraction logic deferred until sufficient
verdicts (~20+) are collected.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import DateTime, Index, String, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from samplespace.models.base import Base


class PairRule(Base):
    """Stores learned rules about which sample pairs work well together."""

    __tablename__ = "pair_rules"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    version: Mapped[int]
    type_pair: Mapped[str] = mapped_column(String(100))
    feature_name: Mapped[str] = mapped_column(String(100))
    threshold: Mapped[float]
    direction: Mapped[str] = mapped_column(String(10))
    confidence: Mapped[float]
    sample_count: Mapped[int]
    is_active: Mapped[bool] = mapped_column(server_default="true")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_pair_rules_active", "is_active"),)

    @classmethod
    async def get_active(cls, db: AsyncSession) -> Sequence[PairRule]:
        """Get all active rules, ordered by confidence."""
        result = await db.execute(select(cls).where(cls.is_active.is_(True)).order_by(cls.confidence.desc()))
        return result.scalars().all()
