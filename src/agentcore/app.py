"""The entrypoint AgentCore Runtime boots.

`BedrockAgentCoreApp` extends Starlette, so this module must expose the
application at import time -- the platform imports the entrypoint file and
serves what it finds. Building it inside a function would leave nothing to
serve.

The logic lives in `agent.py` so it can be tested without the SDK. In the
deployed zip both files sit at the root, which is why the import is flat.
"""
from __future__ import annotations

from bedrock_agentcore.runtime import BedrockAgentCoreApp

try:  # deployed layout: both modules at the zip root
    from agent import run_turn
except ImportError:  # repository layout
    from .agent import run_turn

app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload):
    """One turn. The payload reaches here unchanged, so agent.py validates it."""
    return run_turn(payload)


if __name__ == "__main__":  # pragma: no cover - local run only
    app.run()
