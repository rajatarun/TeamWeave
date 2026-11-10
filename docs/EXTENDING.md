# Extending TeamWeave

TeamWeave is designed so **most workflow changes are JSON**, not Lambda redeploys.
Code changes are needed when you add a new class of tool, runtime substrate, or
RAG backend.

## 1. Team configs (`team.json`)

Primary extension surface. A team defines:

- `agents` — roles, prompts (`goal_template`), optional `schema_ref`, Bedrock/AgentCore settings
- `workflow` — ordered steps, `inputs` (dot paths into prior outputs or `request`)
- `globals` — RAG mode, features, revision hooks
- `schemas` — JSON Schema for structured outputs
- optional `request_schema` — UI / API form fields

Shipped examples: [`config/examples/teams`](../config/examples/teams/).

Validate locally:

```bash
python examples/validate_team.py --all-examples
```

After deploy, upload configs with [`scripts/sync_team_configs.py`](../scripts/sync_team_configs.py) (merges runtime ids from S3; do not use plain `aws s3 sync --delete`).

## 2. Tools (`tool_registry`)

Steps may declare **`pre_tools`** and **`post_tools`**. These run as
deterministic Python — the LLM does not choose to call them.

1. Implement the callable (often under [`src/orchestrator/tools/`](../src/orchestrator/tools/)).
2. Register it in [`TOOL_REGISTRY`](../src/orchestrator/tool_registry.py).
3. Declare rules in [`tool_rules.py`](../src/orchestrator/tool_rules.py) (`effect`, `only_teams`, etc.).
4. Reference by name in `team.json`.

Commit-effect tools (`commit_api_call`, `commit_data_element`) are **refused** in
pipelines by design.

## 3. RAG modes

[`src/orchestrator/rag.py`](../src/orchestrator/rag.py) selects by
`globals.rag.mode`. Add a branch and env vars; degrade to empty context on
failure unless the mode is mandatory (e.g. `contextweave` when URL is unset).

## 4. Agent runtime substrate

[`src/orchestrator/agent_runtime.py`](../src/orchestrator/agent_runtime.py)
seams Classic Bedrock Agents vs Bedrock AgentCore. New substrates implement the
same `invoke` contract as `bedrock_invoke.py` expects.

## 5. JSON schemas

Add files under `config/examples/schemas/` (or your team's schema bundle) and
reference them from `team.json`. Worker validation uses
[`schema_validate.py`](../src/orchestrator/schema_validate.py).

## 6. Infrastructure

New teams on AgentCore need a runtime resource and map entry in
`infra/template.yaml` (see `tests/test_team_runtimes.py`). This coupling is
intentional until `Fn::ForEach` generates runtimes from a parameter.

## 7. A2A skills

Each agent becomes a skill `{team}.{agent_id}` when the card is built from live
configs — no extra registration step beyond `team.json`.
