"""Agent identity on the span, the way the conventions name it.

The first attempt put agent identity in infrastructure: one AgentCore endpoint
per agent. AWS's own quota said no at twelve, because endpoints are a release
mechanism -- production on a stable version while staging tests a newer one --
and ten is a budget for channels, not tenants.

The OpenTelemetry GenAI semantic conventions put identity where it belongs:
`invoke_agent` spans carry `gen_ai.agent.id` and `gen_ai.agent.name`, an
orchestrator coordinating several agents reports `invoke_workflow` around
them, and `gen_ai.conversation.id` identifies the session. TeamWeave has
recorded all of it since before the conventions settled, under names only
TeamWeave knows.

So these pin two things: the conventional names are emitted, and the original
names survive alongside them -- the dashboards and the GSIs query the old
keys, and renaming them would blind every dashboard to make a point about
naming.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

from src.orchestrator.mcp_observatory import genai_attributes  # noqa: E402


def test_the_operation_is_named_the_conventional_way():
    attrs = genai_attributes("invoke_agent", {})
    assert attrs["gen_ai.operation.name"] == "invoke_agent"
    assert attrs["gen_ai.system"] == "aws.bedrock"


def test_teamweaves_own_operation_name_maps_onto_the_convention():
    # invoke_agent_with_metrics is a TeamWeave distinction, not a GenAI one.
    assert genai_attributes("invoke_agent_with_metrics", {})["gen_ai.operation.name"] == "invoke_agent"


def test_an_unmapped_operation_is_passed_through_not_dropped():
    # Losing the operation is worse than reporting a non-standard one.
    assert genai_attributes("something_new", {})["gen_ai.operation.name"] == "something_new"


def test_agent_identity_uses_the_conventional_attribute_names():
    attrs = genai_attributes("invoke_agent", {
        "agent_id": "strategist", "agent_name": "Brand Strategist",
        "session_id": "run-42", "model_id": "us.amazon.nova-micro-v1:0",
    })
    assert attrs["gen_ai.agent.id"] == "strategist"
    assert attrs["gen_ai.agent.name"] == "Brand Strategist"
    assert attrs["gen_ai.conversation.id"] == "run-42"
    assert attrs["gen_ai.request.model"] == "us.amazon.nova-micro-v1:0"


@pytest.mark.parametrize("value", ["", "   ", None])
def test_an_absent_identity_is_omitted_not_written_blank(value):
    # The conventions mark these "conditionally required (when available)",
    # and an empty string is not an identity. On AgentCore every agent's
    # alias_id is blank, so writing the key regardless would fill dashboards
    # with rows claiming an identity they do not have.
    assert "gen_ai.agent.id" not in genai_attributes("invoke_agent", {"agent_id": value})


def test_the_original_names_are_not_replaced():
    # The Observatory GSIs and every dashboard query agent_id/operation.
    # Renaming them to make a point about naming would blind all of it.
    import inspect
    from src.orchestrator import mcp_observatory

    source = inspect.getsource(mcp_observatory)
    assert '"agent_id"' in source or "'agent_id'" in source


def test_identity_is_not_taken_from_an_endpoint():
    # The whole correction: a qualifier names a release channel, so it must
    # never be read as which agent this is.
    attrs = genai_attributes("invoke_agent", {"qualifier": "shadow", "agent_id": "writer"})
    assert attrs["gen_ai.agent.id"] == "writer"
    assert "shadow" not in str(attrs.values())
