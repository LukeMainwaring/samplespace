from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset

from samplespace.agents.deps import AgentDeps
from samplespace.agents.tools.kit_tools import build_kit, preview_kit, transform_kit
from samplespace.agents.tools.transform_tools import match_to_context


@dataclass
class ProductionCapability(AbstractCapability[AgentDeps]):
    def get_toolset(self) -> FunctionToolset[AgentDeps]:
        ts: FunctionToolset[AgentDeps] = FunctionToolset()
        ts.tool(match_to_context)
        ts.tool(build_kit)
        ts.tool(transform_kit)
        ts.tool(preview_kit)
        return ts
