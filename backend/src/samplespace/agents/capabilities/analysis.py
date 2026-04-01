from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset

from samplespace.agents.deps import AgentDeps
from samplespace.agents.tools.analysis_tools import (
    analyze_sample,
    check_key_compatibility,
    suggest_complement,
)


@dataclass
class AnalysisCapability(AbstractCapability[AgentDeps]):
    def get_toolset(self) -> FunctionToolset[AgentDeps]:
        ts: FunctionToolset[AgentDeps] = FunctionToolset()
        ts.tool(analyze_sample)
        ts.tool(check_key_compatibility)
        ts.tool(suggest_complement)
        return ts
