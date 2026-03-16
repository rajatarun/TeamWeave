"""Smoke tests for worker_handler.py — mocks all AWS calls."""
import dataclasses
import importlib
import os
import sys
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")

# Inject boto3 stub so worker_handler can be imported without a real AWS SDK.
for mod_name in ("boto3", "botocore", "botocore.config", "botocore.exceptions"):
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# Stub mcp_observatory dependencies
for mod_name in (
    "mcp_observatory",
    "mcp_observatory.instrument",
    "src.orchestrator.mcp_observatory",
    "src.orchestrator.amp_metrics",
    "src.orchestrator.bedrock_wrappers",
    "src.orchestrator.gemini",
    "src.orchestrator.rag",
    "src.orchestrator.enrich",
    "src.orchestrator.profile_context",
    "src.orchestrator.storage",
    "src.orchestrator.schema_validate",
    "src.orchestrator.structured_transform",
    "src.orchestrator.prompt_builder",
    "src.orchestrator.json_utils",
    "src.orchestrator.db",
    "src.orchestrator.config_loader",
):
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

MOCK_TEAM_CONFIG = {
    "teamId": "test-team",
    "agents": [
        {
            "agentId": "sup-id",
            "agentAliasId": "alias-1",
            "agentRole": "supervisor",
            "model": "amazon.nova-premier-v1:0",
            "description": "Supervisor",
        },
        {
            "agentId": "wrtr-id",
            "agentAliasId": "alias-2",
            "agentRole": "writer",
            "model": "anthropic.claude-haiku-4-5-20251001-v1:0",
            "description": "Writer",
        },
    ],
    "guardrail": {"guardrailId": "grd-1", "guardrailVersion": "DRAFT"},
    "outputBucket": "test-runs-bucket",
}

# Expected refusal phrases (must match worker_handler.REFUSAL_PHRASES)
REFUSAL_PHRASES = [
    "I'm not able to",
    "I cannot",
    "I don't think I should",
    "I must decline",
    "I'm unable to",
    "This request",
]


def test_supervisor_is_resolved_by_role():
    agents = MOCK_TEAM_CONFIG["agents"]
    supervisor = next(a for a in agents if a.get("agentRole") == "supervisor")
    assert supervisor["agentId"] == "sup-id"


def test_workers_exclude_supervisor():
    agents = MOCK_TEAM_CONFIG["agents"]
    workers = [a for a in agents if a.get("agentRole") != "supervisor"]
    assert len(workers) == 1
    assert workers[0]["agentRole"] == "writer"


def test_refusal_detection():
    refusal_text = "I'm not able to write this content."
    assert any(phrase in refusal_text for phrase in REFUSAL_PHRASES)


def test_non_refusal_passes():
    good_text = "Here is the LinkedIn post you requested..."
    assert not any(phrase in good_text for phrase in REFUSAL_PHRASES)


def test_agent_config_has_role_field():
    models = importlib.import_module("src.orchestrator.models")
    fields = {f.name for f in dataclasses.fields(models.AgentConfig)}
    assert "role" in fields, "AgentConfig missing 'role' field"


def test_worker_handler_has_refusal_phrases():
    wh = importlib.import_module("src.orchestrator.worker_handler")
    assert hasattr(wh, "REFUSAL_PHRASES"), "REFUSAL_PHRASES not defined in worker_handler"
    assert wh.REFUSAL_PHRASES == REFUSAL_PHRASES, (
        "REFUSAL_PHRASES in worker_handler does not match expected list"
    )
