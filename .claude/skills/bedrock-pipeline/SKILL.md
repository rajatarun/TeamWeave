---
name: bedrock-pipeline
description: >
  Use when working with Bedrock agent invocations, team.json config, worker_handler.py,
  or bedrock_invoke.py. Provides TeamWeave-specific patterns for agent pipeline development.
---

# Bedrock Pipeline Skill

## Invoking Bedrock Agents (boto3)

Always use `bedrock-agent-runtime` (not `bedrock-runtime`) for agent invocations:

```python
import boto3
client = boto3.client("bedrock-agent-runtime", region_name="us-east-1")

response = client.invoke_agent(
    agentId=agent_id,
    agentAliasId=agent_alias_id,
    sessionId=session_id,
    inputText=input_text,
    guardrailConfiguration={
        "guardrailId": guardrail_id,
        "guardrailVersion": guardrail_version
    } if guardrail_id else {}
)

# Stream the response
full_output = ""
for event in response["completion"]:
    if "chunk" in event:
        full_output += event["chunk"]["bytes"].decode("utf-8")
```

## Supervisor Pattern

Always resolve supervisor dynamically from team.json:
```python
supervisor = next(a for a in agents if a.role == "supervisor")
workers = [a for a in agents if a.role != "supervisor"]
```

In config_loader.py, `role` is populated from `agentRole` or `role` JSON key:
```python
role=a.get("agentRole", a.get("role", ""))
```

## Artifact Naming Convention
```
s3://<ARTIFACT_BUCKET>/runs/<run_id>/<step_id>.json
```

## Guardrail Application
Apply guardrail at invocation time via `guardrailConfiguration` parameter.
Only apply `PROMPT_ATTACK` to input; it is not valid on output filters.
PII entity names are enums — use exact AWS values (e.g., NAME, EMAIL, PHONE).

## Refusal Detection
Check `REFUSAL_PHRASES` in `worker_handler.py` after every `invoke_agent` call.
A `StepFailed` exception is raised on refusal — check CloudWatch for `agent_refusal_detected` logs.

## mcp-observatory Dual-Invoke
Set `shadow_model_id` and `model_aliases` in the agent's `bedrock` config to enable A/B comparison.
Primary and shadow are invoked in parallel; disagreement score is logged to DynamoDB.
