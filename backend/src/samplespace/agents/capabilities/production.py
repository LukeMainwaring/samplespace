"""Production capability: audio transformation and kit building."""

from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset

from samplespace.agents.deps import AgentDeps
from samplespace.agents.tools.kit_tools import build_kit
from samplespace.agents.tools.transform_tools import match_to_context


@dataclass
class ProductionCapability(AbstractCapability[AgentDeps]):
    """Audio transformation and kit building tools."""

    def get_toolset(self) -> FunctionToolset[AgentDeps]:
        ts: FunctionToolset[AgentDeps] = FunctionToolset()
        ts.tool(match_to_context)
        ts.tool(build_kit)
        return ts
