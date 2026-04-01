from dataclasses import dataclass

from pydantic_ai import RunContext, ToolDefinition
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset

from samplespace.agents.deps import AgentDeps
from samplespace.agents.tools.clap_tools import search_by_description
from samplespace.agents.tools.cnn_tools import find_similar_samples
from samplespace.agents.tools.upload_tools import find_similar_to_upload

_CNN_TOOLS = frozenset({find_similar_samples.__name__})


@dataclass
class SearchCapability(AbstractCapability[AgentDeps]):
    def get_toolset(self) -> FunctionToolset[AgentDeps]:
        ts: FunctionToolset[AgentDeps] = FunctionToolset()
        ts.tool(search_by_description)
        ts.tool(find_similar_samples)
        ts.tool(find_similar_to_upload)
        return ts

    async def prepare_tools(
        self,
        ctx: RunContext[AgentDeps],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        """Hide CNN tools when no CNN model is loaded."""
        if ctx.deps.cnn_model is None:
            return [td for td in tool_defs if td.name not in _CNN_TOOLS]
        return tool_defs
