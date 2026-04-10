"""Deterministic tests for capability ``prepare_tools`` filtering.

These tests use pydantic-ai's ``TestModel`` so they don't call the real
OpenAI API. They verify that when the agent is invoked, the tool list
it receives from the model-request-preparation path correctly reflects
the current ``AgentDeps`` state (CNN model loaded/missing).

This covers the ``prepare_tools`` override in ``SearchCapability``
end-to-end through the agent — the filter function itself is simple,
but the wiring it depends on is validated here.
"""

from __future__ import annotations

import asyncio

from pydantic_ai.models.test import TestModel

from samplespace.agents.sample_agent import sample_agent
from samplespace.ml.model import SampleCNN
from tests.evals.conftest import fake_cnn_model, make_fake_deps


def _offered_tool_names(model: TestModel) -> set[str]:
    params = model.last_model_request_parameters
    assert params is not None, "TestModel has no recorded request parameters"
    return {t.name for t in params.function_tools}


def _run_agent_with_test_model(*, cnn_model: SampleCNN | None) -> TestModel:
    test_model = TestModel(call_tools=[], custom_output_text="ok")
    deps = make_fake_deps(cnn_model=cnn_model)

    async def _run() -> None:
        with sample_agent.override(model=test_model, deps=deps):
            await sample_agent.run("hello")

    asyncio.run(_run())
    return test_model


class TestSearchCapabilityPrepareTools:
    def test_hides_cnn_tools_when_cnn_model_missing(self) -> None:
        model = _run_agent_with_test_model(cnn_model=None)
        offered = _offered_tool_names(model)
        assert "find_similar_samples" not in offered

    def test_shows_cnn_tools_when_cnn_model_loaded(self) -> None:
        model = _run_agent_with_test_model(cnn_model=fake_cnn_model())
        offered = _offered_tool_names(model)
        assert "find_similar_samples" in offered

    def test_clap_and_upload_tools_always_available(self) -> None:
        model = _run_agent_with_test_model(cnn_model=None)
        offered = _offered_tool_names(model)
        for always in (
            "search_by_description",
            "find_similar_to_upload",
            "find_upload",
            "set_context_from_upload",
        ):
            assert always in offered, f"{always} should be offered regardless of cnn_model"


class TestNonSearchToolsAlwaysAvailable:
    def test_analysis_context_pairing_production_tools_always_present(self) -> None:
        model = _run_agent_with_test_model(cnn_model=None)
        offered = _offered_tool_names(model)

        expected = {
            # AnalysisCapability
            "analyze_sample",
            "check_key_compatibility",
            "suggest_complement",
            # ContextCapability
            "set_song_context",
            # PairingCapability
            "rate_pair",
            "present_pair",
            "record_verdict",
            "show_preferences",
            # ProductionCapability
            "match_to_context",
            "build_kit",
            "transform_kit",
            "preview_kit",
        }
        missing = expected - offered
        assert not missing, f"Expected tools missing from offered set: {missing}"
