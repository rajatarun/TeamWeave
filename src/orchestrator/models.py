from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class BedrockRef:
    agentId: str
    aliasId: str
    model_id: str = "us.amazon.nova-micro-v1:0"
    shadow_model_id: str = ""
    model_aliases: Dict[str, str] = field(default_factory=dict)

@dataclass
class AgentConfig:
    id: str
    name: str
    bedrock: BedrockRef
    goal_template: str
    schema_ref: str

@dataclass
class TeamGlobals:
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
