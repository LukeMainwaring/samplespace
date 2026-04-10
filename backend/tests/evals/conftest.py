"""Fixtures for sample_agent evals.

Provides ``make_fake_deps`` — a constructor for ``AgentDeps`` instances
that don't hit the real database, CLAP processor, or CNN model. Used by
both the deterministic ``prepare_tools`` tests (TestModel-backed) and
the real-model ``@pytest.mark.eval`` tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from sqlalchemy.ext.asyncio import AsyncSession
from transformers import ClapModel, ClapProcessor

from samplespace.agents.deps import AgentDeps
from samplespace.ml.model import SampleCNN
from samplespace.schemas.thread import SongContext


def fake_clap_model() -> ClapModel:
    return MagicMock(spec=ClapModel)


def fake_clap_processor() -> ClapProcessor:
    return MagicMock(spec=ClapProcessor)


def fake_cnn_model() -> SampleCNN:
    return MagicMock(spec=SampleCNN)


def make_fake_deps(
    *,
    cnn_model: SampleCNN | None = None,
    thread_id: str = "test-thread",
    song_context: SongContext | None = None,
) -> AgentDeps:
    fake_db = MagicMock(spec=AsyncSession)
    return AgentDeps(
        db=fake_db,
        clap_model=fake_clap_model(),
        clap_processor=fake_clap_processor(),
        cnn_model=cnn_model,
        thread_id=thread_id,
        song_context=song_context,
    )
