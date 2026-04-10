"""Unit tests for the sample_agent hooks module.

Exercises the recovery payload helper and the ``_recover_tool_error``
handler directly against real ``ToolCallPart`` / ``ToolDefinition``
instances. End-to-end validation (fake failing tool → agent responds
conversationally) lives in ``tests/evals/``.
"""

from __future__ import annotations

import asyncio

from pydantic_ai import ToolDefinition
from pydantic_ai.capabilities import Hooks
from pydantic_ai.messages import ToolCallPart

from samplespace.agents.hooks import (
    _recover_tool_error,
    _recovery_message,
    build_sample_agent_hooks,
)


class TestRecoveryMessage:
    def test_contains_tool_and_exception_type(self) -> None:
        msg = _recovery_message("find_similar_samples", RuntimeError("cnn inference boom"))
        assert "find_similar_samples" in msg
        assert "RuntimeError" in msg

    def test_message_shape_is_stable_across_exception_types(self) -> None:
        a = _recovery_message("search_by_description", ValueError("bad"))
        b = _recovery_message("search_by_description", ConnectionError("net down"))
        assert "ValueError" in a
        assert "ConnectionError" in b
        assert "search_by_description" in a
        assert "search_by_description" in b


class TestRecoverToolErrorHandler:
    def test_returns_recovery_message_for_unexpected_exception(self) -> None:
        call = ToolCallPart(tool_name="build_kit", tool_call_id="call-1")
        tool_def = ToolDefinition(name="build_kit")

        async def _invoke() -> str:
            return await _recover_tool_error(
                None,  # type: ignore[arg-type]  # handler doesn't touch ctx
                call=call,
                tool_def=tool_def,
                args={},
                error=RuntimeError("db gone"),
            )

        result = asyncio.run(_invoke())
        assert "build_kit" in result
        assert "RuntimeError" in result


class TestBuildSampleAgentHooks:
    def test_returns_hooks_instance(self) -> None:
        hooks = build_sample_agent_hooks()
        assert isinstance(hooks, Hooks)
