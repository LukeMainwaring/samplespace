"""Hooks for the sample assistant agent.

Most tool bodies already handle anticipated failures (missing sample,
empty search result, CLAP/CNN inference errors) by returning a
plain-string error message. This module is the safety net for
*unanticipated* exceptions — anything that bubbles out of a tool body
would otherwise crash the Vercel AI SDK stream mid-response. The
``tool_execute_error`` hook intercepts those, logs the traceback, and
returns a plain-string recovery payload so the agent can explain the
failure to the user conversationally.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic_ai import ToolDefinition
from pydantic_ai.capabilities import Hooks
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import RunContext

from samplespace.agents.deps import AgentDeps

logger = logging.getLogger(__name__)


def _recovery_message(tool_name: str, error: Exception) -> str:
    return (
        f"The {tool_name} tool failed unexpectedly ({type(error).__name__}). "
        "Apologize to the user briefly, explain what you were trying to do, "
        "and suggest they retry or rephrase."
    )


async def _recover_tool_error(
    ctx: RunContext[AgentDeps],
    *,
    call: ToolCallPart,
    tool_def: ToolDefinition,
    args: dict[str, Any],
    error: Exception,
) -> str:
    logger.error(f"Unhandled exception in tool {tool_def.name}: {error!r}", exc_info=error)
    return _recovery_message(tool_def.name, error)


def build_sample_agent_hooks() -> Hooks[AgentDeps]:
    return Hooks[AgentDeps](tool_execute_error=_recover_tool_error)
