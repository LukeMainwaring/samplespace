from dataclasses import dataclass

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset

from samplespace.agents.deps import AgentDeps
from samplespace.agents.tools.pair_tools import rate_pair
from samplespace.agents.tools.preference_tools import show_preferences
from samplespace.agents.tools.verdict_tools import present_pair, record_verdict
from samplespace.services import preference as preference_service


async def _inject_preferences(ctx: RunContext[AgentDeps]) -> str:
    explanation = preference_service.explain()
    if explanation is None:
        return ""
    return f"\n\n## Learned Pairing Preferences\n\n{explanation.summary}"


@dataclass
class PairingCapability(AbstractCapability[AgentDeps]):
    def get_toolset(self) -> FunctionToolset[AgentDeps]:
        ts: FunctionToolset[AgentDeps] = FunctionToolset()
        ts.tool(rate_pair)
        ts.tool(present_pair)
        ts.tool(record_verdict)
        ts.tool(show_preferences)
        return ts

    def get_instructions(self) -> object:
        return _inject_preferences
