"""Pairing capability: pair scoring, verdicts, and learned preferences."""

from dataclasses import dataclass

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset

from samplespace.agents.deps import AgentDeps
from samplespace.agents.tools.pair_tools import rate_pair
from samplespace.agents.tools.verdict_tools import present_pair, record_verdict
from samplespace.models.pair_rule import PairRule


async def _inject_pair_rules(ctx: RunContext[AgentDeps]) -> str:
    """Inject learned pair rules into the system prompt."""
    rules = await PairRule.get_active(ctx.deps.db)
    if not rules:
        return ""
    lines = ["\n\n## Learned Pairing Preferences"]
    for rule in rules:
        lines.append(
            f"- For {rule.type_pair} pairs: prefer {rule.feature_name} "
            f"{rule.direction} {rule.threshold:.2f} "
            f"(confidence: {rule.confidence:.0%})"
        )
    return "\n".join(lines)


@dataclass
class PairingCapability(AbstractCapability[AgentDeps]):
    """Pair scoring, verdict collection, and learned pairing preferences."""

    def get_toolset(self) -> FunctionToolset[AgentDeps]:
        ts: FunctionToolset[AgentDeps] = FunctionToolset()
        ts.tool(rate_pair)
        ts.tool(present_pair)
        ts.tool(record_verdict)
        return ts

    def get_instructions(self) -> object:
        return _inject_pair_rules
