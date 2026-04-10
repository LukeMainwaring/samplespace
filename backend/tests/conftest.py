"""Test configuration and fixtures."""

import os

# ``OpenAIResponsesModel`` in ``sample_agent.py`` constructs its provider
# client eagerly at module import time, which requires ``OPENAI_API_KEY``.
# Any test that imports ``sample_agent`` — directly or transitively — would
# otherwise fail at collection. Tests that actually invoke the model use
# ``agent.override`` with ``TestModel``; this dummy value never reaches a
# real API call.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-deterministic-no-real-calls")
