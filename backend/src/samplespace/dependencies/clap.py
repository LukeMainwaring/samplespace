from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request
from transformers import ClapModel, ClapProcessor


@dataclass
class ClapModels:
    model: ClapModel
    processor: ClapProcessor


def get_clap_models(request: Request) -> ClapModels:
    return ClapModels(
        model=request.app.state.clap_model,
        processor=request.app.state.clap_processor,
    )


ClapModelsDep = Annotated[ClapModels, Depends(get_clap_models)]
