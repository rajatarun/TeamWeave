from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# globals.rag.mode values understood by rag.get_rag_context().
#   kb / none     — no retrieval
#   explicit      — pgvector similarity search (VECTOR_DB_TABLE)
#   history       — DynamoDB completed-task history
#   contextweave  — ContextWeave knowledge layer (CONTEXTWEAVE_URL)
CONTEXTWEAVE_MODE = "contextweave"
RAG_MODES = frozenset({"kb", "none", "explicit", "history", CONTEXTWEAVE_MODE})

@dataclass
class BedrockRef:
    """Where an agent lives, on whichever substrate runs it.

    agentId/aliasId address Bedrock Agents Classic; runtimeArn/qualifier
    address an AgentCore runtime. Both live here so one team.json can describe
    either, and a team can be moved a step at a time rather than all at once.
    """

    agentId: str
    aliasId: str
    model_id: str = "us.amazon.nova-micro-v1:0"
    shadow_model_id: str = ""
    model_aliases: Dict[str, str] = field(default_factory=dict)
    runtimeArn: str = ""
    qualifier: str = ""

@dataclass
class AgentConfig:
    id: str
    name: str
    bedrock: BedrockRef
    goal_template: str
    schema_ref: str

@dataclass
class TeamGlobals:
    """Team-wide settings.

    ``rag`` accepts: ``mode`` (see RAG_MODES), ``top_k``, ``rag_env_key``
    (explicit mode) and ``min_confidence`` (contextweave mode — answers below
    it are dropped rather than injected into the prompt).
    """

    north_star: str
    default_channel: str
    hard_constraints: List[str]
    features: Dict[str, Any]
    rag: Dict[str, Any]
    artifact_store: Dict[str, Any]
    revision: Dict[str, Any]

@dataclass
class TeamConfig:
    team: Dict[str, Any]
    globals: TeamGlobals
    agents: List[AgentConfig]
    workflow: List[Dict[str, Any]]
    schemas: Dict[str, Dict[str, Any]]

class StepFailed(Exception):
    def __init__(self, step_id: str, message: str, raw_output: Optional[str] = None):
        super().__init__(message)
        self.step_id = step_id
        self.raw_output = raw_output
