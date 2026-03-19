"""CLAP model dependency injection."""

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request
from transformers import ClapModel, ClapProcessor


@dataclass
class ClapModels:
    """Container for CLAP model and processor loaded at startup."""

    model: ClapModel
    processor: ClapProcessor


def get_clap_models(request: Request) -> ClapModels:
    """Get CLAP model and processor from app state (loaded in lifespan)."""
    return ClapModels(
        model=request.app.state.clap_model,
        processor=request.app.state.clap_processor,
    )


ClapModelsDep = Annotated[ClapModels, Depends(get_clap_models)]
