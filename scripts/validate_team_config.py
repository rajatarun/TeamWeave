#!/usr/bin/env python3
"""Validate team.json against required schema."""
import json
import sys

REQUIRED_AGENT_FIELDS = {"agentId", "agentAliasId", "agentRole", "model"}


def validate(path="config/examples/team_simple.json"):
    with open(path) as f:
        config = json.load(f)
    assert "teamId" in config, "Missing teamId"
    assert "agents" in config and len(config["agents"]) > 0, "No agents defined"
    supervisors = [a for a in config["agents"] if a.get("agentRole") == "supervisor"]
    assert len(supervisors) == 1, f"Expected 1 supervisor, found {len(supervisors)}"
    for agent in config["agents"]:
        missing = REQUIRED_AGENT_FIELDS - set(agent.keys())
        assert not missing, f"Agent {agent.get('agentId', '?')} missing fields: {missing}"
    print(f"✅ {path} valid — {len(config['agents'])} agents, 1 supervisor")


if __name__ == "__main__":
    validate(sys.argv[1] if len(sys.argv) > 1 else "config/examples/team_simple.json")
