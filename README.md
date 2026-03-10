# TeamWeave

Config-driven multi-agent orchestration platform on AWS. Define AI pipelines as JSON — no code changes needed to add new teams or workflows.

Two production patterns ship out of the box:

| Pattern | What it does |
|---------|-------------|
| **Visibility Team** | Content marketing assembly line (strategy → draft → edit → distribute) |
| **Improvement Team** | Personal learning system (coach → plan → daily tasks) |

---

## Architecture

> Full detail: [docs/architecture.md](docs/architecture.md)

TeamWeave is split into two planes:

- **Control Plane** — API Gateway → Trigger Lambda starts an async Step Functions execution and returns a `run_id`. Status Lambda polls execution state via `DescribeExecution`.
- **Execution Plane** — Worker Lambda loads a `team.json` from S3 and runs each workflow step: fetch RAG context → build prompt → invoke Bedrock Agent → validate + persist output.

```
Client → API Gateway → Trigger Lambda → Step Functions → Worker Lambda → Bedrock Agents
                                              │
                                    Status Lambda (polls)
```

### C4 Diagrams

#### Level 1 — System Context
![C4 Level 1: System Context](docs/c4/TeamWeave%20-%20C4%20Level%201:%20System%20Context.png)

> [Source PUML](docs/c4/context.puml) · [Architecture: System Overview](docs/architecture.md#system-overview)

#### Level 2 — Container Diagram
![C4 Level 2: Container Diagram](docs/c4/TeamWeave%20-%20C4%20Level%202:%20Container%20Diagram.png)

> [Source PUML](docs/c4/container.puml) · [Architecture: System Components](docs/architecture.md#system-components)

#### Level 3 — Worker Lambda Components
![C4 Level 3: Component Diagram (Worker Lambda)](docs/c4/TeamWeave%20-%20C4%20Level%203:%20Component%20Diagram%20(Worker%20Lambda).png)

> [Source PUML](docs/c4/component.puml) · [Architecture: Data Flow](docs/architecture.md#data-flow)

#### AWS Infrastructure
![AWS Infrastructure Diagram](docs/TeamWeave%20-%20AWS%20Infrastructure%20Diagram.png)

> [Source PUML](docs/aws-infrastructure.puml) · [Architecture: Infrastructure](docs/architecture.md#infrastructure)

---

## Repository Layout

```
src/orchestrator/       Runtime Lambda handlers and orchestration modules
  trigger_handler.py    API entry point — validates requests, starts executions
  worker_handler.py     Pipeline engine — runs workflow steps via Bedrock Agents
  status_handler.py     Status polling — wraps DescribeExecution
  config_loader.py      Loads team.json from S3
  rag.py                RAG retrieval (pgvector explicit mode + history mode)
  db.py                 DynamoDB DAO for runs, steps, tasks
  gemini.py             Gemini API key retrieval from Secrets Manager
config/examples/        Example team configs, schemas, roles, departments
infra/                  SAM/CloudFormation templates — see infra/README.md
tests/                  Unit tests
docs/                   Architecture doc, C4 diagrams, AWS infrastructure diagram
```

---

## Quick Start

```bash
sam build -t infra/template.yaml
sam deploy --guided -t infra/template.yaml
```

See [infra/README.md](infra/README.md) for parameters and Bedrock agent provisioning.

---

## API

### Workflow

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/team/task` | Start a pipeline run → returns `{ run_id, execution_arn }` |
| `GET` | `/team/task/{run_id}` | Poll status → `RUNNING \| SUCCEEDED \| FAILED` + result |
| `GET` | `/improve/tasks` | List improvement tasks for an owner |
| `POST` | `/improve/task/done` | Mark improvement task complete |

### Management (async via Step Functions)

`/agents`, `/teams`, `/roles`, `/departments` — full CRUD, dispatched to the Provision Lambda.

### Observability

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/observability/metrics` | Bedrock telemetry from Amazon Managed Prometheus |

Full request/response shapes: [Architecture: API Reference](docs/architecture.md#api-reference)

---

## Team Configuration

A team lives at `teams/{team}/{version}/team.json` in S3:

```json
{
  "team":     { "name": "...", "version": "v1", "owner": "..." },
  "globals":  { "north_star": "...", "features": { "explicit_rag": true } },
  "agents":   [{ "id": "...", "bedrock": { "agentId": "...", "aliasId": "..." }, "goal_template": "..." }],
  "workflow": [{ "step": "agent_id", "inputs": ["field1"] }]
}
```

Full schema: [Architecture: Data Models](docs/architecture.md#data-models)

---

## Key Environment Variables

| Variable | Purpose |
|----------|---------|
| `CONFIG_BUCKET` | S3 bucket for team configs |
| `ARTIFACT_BUCKET` | S3 bucket for run artifacts |
| `DDB_TABLE` | DynamoDB table for runs and tasks |
| `STATE_MACHINE_ARN` | Step Functions state machine |
| `VECTOR_DB_SECRET_ARN` | Secrets Manager ARN for pgvector credentials |
| `GEMINI_SECRET_ARN` | Secrets Manager ARN for Gemini API key |

All wired automatically by `infra/template.yaml`.

---

## Local Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```

---

## Example

```bash
# Start a run
curl -X POST "$API_URL/team/task" \
  -H 'content-type: application/json' \
  -d '{"team":"tarun_visibility_team","version":"v1","request":{"topic":"AI orchestration","objective":"LinkedIn post","audience":"engineering leaders"}}'

# Poll for result
curl "$API_URL/teams/task/<run_id>"
```

---

## License

MIT
