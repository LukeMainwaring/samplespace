"""Agent streaming endpoint.

Provides POST /agent/chat for the sample assistant agent,
streaming responses in Vercel AI SDK protocol format.
"""

import asyncio
import logging
from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic_ai.ui.vercel_ai import VercelAIAdapter
from starlette.requests import Request
from starlette.responses import Response

from samplespace.agents.deps import AgentDeps
from samplespace.agents.sample_agent import sample_agent
from samplespace.dependencies.clap import get_clap_models
from samplespace.dependencies.cnn import get_cnn_model
from samplespace.dependencies.db import AsyncPostgresSessionDep
from samplespace.models.message import Message
from samplespace.models.thread import Thread
from samplespace.schemas.agent_type import AgentType
from samplespace.schemas.thread import SongContext
from samplespace.services.title_generator import generate_thread_title
from samplespace.utils.message_serialization import extract_latest_user_text, prepare_messages_for_storage

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
    body = await request.body()
    run_input = VercelAIAdapter.build_run_input(body)
    thread_id = run_input.id

    clap = get_clap_models(request)

    thread = await Thread.get(db, thread_id, AgentType.CHAT.value)
    existing_context = SongContext.model_validate(thread.song_context) if thread and thread.song_context else None

    deps = AgentDeps(
        db=db,
        clap_model=clap.model,
        clap_processor=clap.processor,
        cnn_model=get_cnn_model(request),
        thread_id=thread_id,
        song_context=existing_context,
    )

    user_query = extract_latest_user_text(run_input.messages)

    async def on_complete(result):  # type: ignore[no-untyped-def]
        # Save the full conversation: all_messages() includes the adapter's
        # converted history + new response. save_history is append-only.
        all_msgs = prepare_messages_for_storage(result.all_messages())
        await Thread.get_or_create(db, thread_id, AgentType.CHAT.value)
        await Message.save_history(db, thread_id, AgentType.CHAT.value, all_msgs)

        thread = await Thread.get(db, thread_id, AgentType.CHAT.value)
        if thread:
            thread.updated_at = datetime.now(timezone.utc)
            await db.flush()

        if thread and thread.title is None and result.output:
            asyncio.create_task(
                generate_thread_title(
                    thread_id=thread_id,
                    agent_type=AgentType.CHAT.value,
                    user_message=user_query,
                    assistant_response=str(result.output),
                )
            )

    return await VercelAIAdapter.dispatch_request(
        request,
        agent=sample_agent,
        deps=deps,
        on_complete=on_complete,
        sdk_version=6,
    )
