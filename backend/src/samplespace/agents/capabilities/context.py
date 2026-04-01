from dataclasses import dataclass

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset

from samplespace.agents.deps import AgentDeps
from samplespace.agents.tools.context_tools import set_song_context


async def _inject_song_context(ctx: RunContext[AgentDeps]) -> str:
    if not ctx.deps.song_context:
        return ""
    sc = ctx.deps.song_context
    parts = []
    if sc.key:
        parts.append(f"Key: {sc.key}")
    if sc.bpm:
        parts.append(f"BPM: {sc.bpm}")
    if sc.genre:
        parts.append(f"Genre: {sc.genre}")
    if sc.vibe:
        parts.append(f"Vibe: {sc.vibe}")
    if not parts:
        return ""
    return (
        "\n\n## Active Song Context\n"
        + "\n".join(f"- {p}" for p in parts)
        + "\n\nUse this context to inform your searches and recommendations. "
        "The vibe is automatically appended to CLAP searches."
    )


@dataclass
class ContextCapability(AbstractCapability[AgentDeps]):
    def get_toolset(self) -> FunctionToolset[AgentDeps]:
        ts: FunctionToolset[AgentDeps] = FunctionToolset()
        ts.tool(set_song_context)
        return ts

    def get_instructions(self) -> object:
        return _inject_song_context
