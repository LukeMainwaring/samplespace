"""Shared dependencies for the sample assistant agent."""

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession
from transformers import ClapModel, ClapProcessor

from samplespace.ml.model import SampleCNN


@dataclass
class AgentDeps:
    """Dependencies injected into agent runs."""

    db: AsyncSession
    clap_model: ClapModel
    clap_processor: ClapProcessor
    cnn_model: SampleCNN | None = None
