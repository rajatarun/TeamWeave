# Design overview

TeamWeave separates **control plane** (start/poll runs, provision configs) from
**execution plane** (Worker runs the workflow).

```
Client ──► API Gateway ──► Trigger Lambda ──► Step Functions ──► Worker Lambda
                │                                    │
                └── Status Lambda ◄── DescribeExecution
```

## Core decisions

1. **JSON workflows** — Agents, steps, and schemas live in `team.json`. Adding a team is primarily config + optional infra for a dedicated AgentCore runtime.
2. **One worker invocation per run (today)** — All steps execute in one Lambda until the 900s ceiling; large teams may later move to one Step Functions state per step.
3. **Structured outputs** — JSON Schema validation per agent; `structured_transform` repairs once via Converse.
4. **Substrate seam** — `agent_runtime.py` selects Classic vs AgentCore; prompts are composed in `prompt_builder`, not baked into Bedrock agent resources.
5. **Deterministic tools** — `pre_tools` / `post_tools` in config; no LLM tool loop in AgentCore entrypoint yet.
6. **A2A as discovery** — One agent card; skills mirror `{team}.{agent}`.

## Deeper reading

- [Architecture](../architecture.md) — C4 diagrams, data flow, API
- [EXTENDING.md](../EXTENDING.md) — how to add teams, tools, RAG modes
- [INTEGRATIONS.md](../INTEGRATIONS.md) — external ecosystem fit
