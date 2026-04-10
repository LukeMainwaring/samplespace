"""Real-model evaluation suite for sample_agent tool routing.

Marked ``@pytest.mark.eval`` so it doesn't run in the default pytest
invocation (the root ``pyproject.toml`` adds ``-m 'not eval'`` to
``addopts``). Run explicitly with::

    uv run --directory backend pytest -m eval tests/evals/

This hits the real OpenAI API via ``sample_agent``'s configured
``AGENT_MODEL`` and costs money per run. Use it as a nightly safety net
on ``main``, not on every PR. The ``prepare_tools`` filter behavior is
already covered deterministically in ``test_prepare_tools.py`` — this
suite's job is to catch regressions in:

- ``SYSTEM_PROMPT`` wording (``backend/src/samplespace/agents/sample_agent.py``)
- ``AGENT_MODEL`` version bumps (``backend/src/samplespace/core/config.py``)
- New tools being added without corresponding prompt guidance

The evaluator checks that the expected tool was *among* the tool calls
in the agent's message history, not that it was the only one — the
agent is allowed to chain tools as long as the critical one fires.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from samplespace.agents.sample_agent import sample_agent
from tests.evals.conftest import fake_cnn_model, make_fake_deps


@dataclass
class SampleAgentInput:
    prompt: str
    with_cnn: bool = False


@dataclass
class SampleAgentOutput:
    text: str
    tool_calls: list[str] = field(default_factory=list)


async def _run_sample_agent(inputs: SampleAgentInput) -> SampleAgentOutput:
    """Execute sample_agent against its real configured model.

    Uses fake deps so tool bodies that try to hit a DB / real CLAP
    inference will fail through the hooks recovery path — tool-routing
    evals only care *which* tools the agent tried to call, not whether
    they succeeded. ``with_cnn=True`` threads a ``MagicMock`` CNN model
    so ``SearchCapability.prepare_tools`` stops filtering the
    CNN-gated tool set, which is required for any case that asserts
    behavior around ``find_similar_samples``.
    """
    deps = make_fake_deps(cnn_model=fake_cnn_model() if inputs.with_cnn else None)
    with sample_agent.override(deps=deps):
        result = await sample_agent.run(inputs.prompt)

    tool_calls: list[str] = []
    for msg in result.all_messages():
        for part in getattr(msg, "parts", []):
            if hasattr(part, "tool_name") and hasattr(part, "args"):
                tool_calls.append(part.tool_name)

    return SampleAgentOutput(text=str(result.output), tool_calls=tool_calls)


@dataclass
class ExpectedToolCalled(Evaluator[SampleAgentInput, SampleAgentOutput, None]):
    """Asserts that ``expected_tool`` was among the agent's tool calls."""

    expected_tool: str = ""

    def evaluate(self, ctx: EvaluatorContext[SampleAgentInput, SampleAgentOutput, None]) -> bool:
        return self.expected_tool in ctx.output.tool_calls


@dataclass
class AnyOfToolsCalled(Evaluator[SampleAgentInput, SampleAgentOutput, None]):
    """Asserts that at least one of ``candidates`` was called."""

    candidates: tuple[str, ...] = ()

    def evaluate(self, ctx: EvaluatorContext[SampleAgentInput, SampleAgentOutput, None]) -> bool:
        return any(c in ctx.output.tool_calls for c in self.candidates)


@dataclass
class NoToolCalled(Evaluator[SampleAgentInput, SampleAgentOutput, None]):
    """Asserts a forbidden tool was *not* called — used for prepare_tools gating."""

    forbidden: str = ""

    def evaluate(self, ctx: EvaluatorContext[SampleAgentInput, SampleAgentOutput, None]) -> bool:
        return self.forbidden not in ctx.output.tool_calls


_CASES: list[Case[SampleAgentInput, SampleAgentOutput, None]] = [
    Case(
        name="search_by_description_routes_to_clap",
        inputs=SampleAgentInput(prompt="Find me a warm analog pad with a slow attack."),
        evaluators=(ExpectedToolCalled(expected_tool="search_by_description"),),
    ),
    Case(
        # Musical metadata in the prompt should trigger proactive
        # set_song_context per the SYSTEM_PROMPT workflow rule.
        name="set_song_context_on_musical_mention",
        inputs=SampleAgentInput(prompt="I'm making a house track in A minor at 128 BPM."),
        evaluators=(ExpectedToolCalled(expected_tool="set_song_context"),),
    ),
    Case(
        # CNN model must be "loaded" for find_similar_samples to be
        # offered — otherwise SearchCapability.prepare_tools filters
        # it out and a passing result would mean nothing.
        name="find_similar_samples_when_cnn_available",
        inputs=SampleAgentInput(
            prompt="Find samples that sound like sample 5.",
            with_cnn=True,
        ),
        evaluators=(ExpectedToolCalled(expected_tool="find_similar_samples"),),
    ),
    Case(
        name="build_kit_on_kit_request",
        inputs=SampleAgentInput(prompt="Build me a techno kit."),
        evaluators=(ExpectedToolCalled(expected_tool="build_kit"),),
    ),
    Case(
        name="pairing_session_starts_with_present_pair",
        inputs=SampleAgentInput(prompt="Start a pairing session with kick loops and bass loops."),
        evaluators=(ExpectedToolCalled(expected_tool="present_pair"),),
    ),
    Case(
        # The agent can reach for either upload tool depending on
        # whether it interprets the prompt as a lookup or a similarity
        # search. Either is correct.
        name="upload_reference_routes_to_upload_tool",
        inputs=SampleAgentInput(prompt="Find the track I just uploaded called warm-pad."),
        evaluators=(AnyOfToolsCalled(candidates=("find_upload", "find_similar_to_upload")),),
    ),
    Case(
        # With cnn_model=None, find_similar_samples is filtered out by
        # SearchCapability.prepare_tools. The agent should not call it
        # even when the prompt sounds like similarity search.
        name="cnn_tool_hidden_when_model_missing",
        inputs=SampleAgentInput(prompt="Find samples that sound like sample 5."),
        evaluators=(NoToolCalled(forbidden="find_similar_samples"),),
    ),
]


_dataset: Dataset[SampleAgentInput, SampleAgentOutput, None] = Dataset(
    name="sample_agent_tool_routing",
    cases=_CASES,
)


@pytest.mark.eval
def test_sample_agent_tool_routing() -> None:
    report = asyncio.run(_dataset.evaluate(_run_sample_agent))

    execution_failures = [f.name for f in report.failures]
    assert not execution_failures, f"Eval execution errors: {execution_failures}"

    assertion_failures = [
        case.name for case in report.cases if not all(a.value is True for a in case.assertions.values())
    ]
    assert not assertion_failures, f"Eval assertion failures: {assertion_failures}"
