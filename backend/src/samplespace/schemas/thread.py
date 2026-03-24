"""Thread schemas for conversation persistence."""

from datetime import datetime
from typing import Any

from pydantic import Field

from samplespace.schemas.agent_type import AgentType

from .base import BaseSchema


class ThreadSchema(BaseSchema):
    """Thread schema representing a conversation thread."""

    thread_id: str
    agent_type: AgentType
    title: str | None = None
    created_at: datetime
    updated_at: datetime


class ThreadCreateSchema(BaseSchema):
    """Schema for creating a new thread."""

    thread_id: str
    agent_type: AgentType


class ThreadSummary(BaseSchema):
    """Summary of a thread for list view."""

    id: str
    thread_id: str
    title: str | None
    created_at: datetime
    updated_at: datetime


class ThreadListResponse(BaseSchema):
    """Response containing list of threads."""

    threads: list[ThreadSummary]


class ThreadMessagesResponse(BaseSchema):
    """Response containing thread messages."""

    thread_id: str
    messages: list[dict[str, Any]]


class ThreadDeleteResponse(BaseSchema):
    """Response for successful thread delete operations."""

    message: str


class ThreadRenameRequest(BaseSchema):
    """Request to rename a thread."""

    title: str = Field(min_length=1, max_length=255)


class ThreadRenameResponse(BaseSchema):
    """Response for successful thread rename operations."""

    thread_id: str
    title: str
