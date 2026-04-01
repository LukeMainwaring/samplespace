from datetime import datetime
from typing import Any

from pydantic import Field

from samplespace.schemas.agent_type import AgentType

from .base import BaseSchema


class SongContext(BaseSchema):
    key: str | None = None
    bpm: int | None = None
    genre: str | None = None
    vibe: str | None = None


class ThreadSchema(BaseSchema):
    thread_id: str
    agent_type: AgentType
    title: str | None = None
    song_context: SongContext | None = None
    created_at: datetime
    updated_at: datetime


class ThreadCreateSchema(BaseSchema):
    thread_id: str
    agent_type: AgentType


class ThreadSummary(BaseSchema):
    id: str
    thread_id: str
    title: str | None
    song_context: SongContext | None = None
    created_at: datetime
    updated_at: datetime


class ThreadListResponse(BaseSchema):
    threads: list[ThreadSummary]


class ThreadMessagesResponse(BaseSchema):
    thread_id: str
    messages: list[dict[str, Any]]
    song_context: SongContext | None = None


class ThreadDeleteResponse(BaseSchema):
    message: str


class ThreadRenameRequest(BaseSchema):
    title: str = Field(min_length=1, max_length=255)


class ThreadRenameResponse(BaseSchema):
    thread_id: str
    title: str
