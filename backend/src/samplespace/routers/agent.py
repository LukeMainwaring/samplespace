"""Agent streaming endpoint.

Provides POST /agent/chat for the sample assistant agent,
streaming responses in Vercel AI SDK protocol format.
"""

import logging

from fastapi import APIRouter
from pydantic_ai.ui.vercel_ai import VercelAIAdapter
from starlette.requests import Request
from starlette.responses import Response

from samplespace.agents.deps import AgentDeps
from samplespace.agents.sample_agent import sample_agent
from samplespace.dependencies.clap import get_clap_models
from samplespace.dependencies.db import AsyncPostgresSessionDep

logger = logging.getLogger(__name__)

agent_router = APIRouter(prefix="/agent", tags=["agent"])


@agent_router.post("/chat")
async def stream_chat(
    request: Request,
    db: AsyncPostgresSessionDep,
) -> Response:
    """Sample assistant streaming endpoint.

    Uses VercelAIAdapter to handle parsing, agent execution, and streaming
    in Vercel AI SDK protocol format.
    """
    clap = get_clap_models(request)

    deps = AgentDeps(
        db=db,
        clap_model=clap.model,
        clap_processor=clap.processor,
    )

    return await VercelAIAdapter.dispatch_request(
        request,
        agent=sample_agent,
        deps=deps,
    )
