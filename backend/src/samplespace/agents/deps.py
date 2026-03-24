"""Shared dependencies for the sample assistant agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession
from transformers import ClapModel, ClapProcessor

from samplespace.ml.model import SampleCNN

if TYPE_CHECKING:
    from samplespace.schemas.thread import SongContext


@dataclass
class AgentDeps:
    """Dependencies injected into agent runs."""

    db: AsyncSession
    clap_model: ClapModel
    clap_processor: ClapProcessor
    cnn_model: SampleCNN | None = None
    thread_id: str | None = None
    song_context: SongContext | None = None
