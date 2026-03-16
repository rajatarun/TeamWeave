# TeamWeave

Config-driven multi-agent orchestration platform. Runs Bedrock agents in pipelines defined by
team.json (S3-backed). Part of a 4-tier AI engineering stack:
TaskWeave → ContextWeave → **TeamWeave** → ai-content-orchestrator.

---

## Stack

| Layer | Technology |
|-------|-----------|
| Runtime | Python 3.12 (AWS Lambda) |
| AI/ML | Amazon Bedrock (Nova, Claude Haiku), Google Gemini 2.0 Flash |
| Orchestration | AWS Step Functions (Standard Workflows) |
| API | AWS API Gateway (REST) with SIWE Lambda Authorizer |
| State | DynamoDB (run/task metadata + Observatory metrics) |
| Vector DB | RDS PostgreSQL 15 + pgvector (RAG) |
| Storage | S3 (team configs, artifacts) |
| Secrets | AWS Secrets Manager |
| Observability | Amazon Managed Prometheus (AMP), CloudWatch, X-Ray, mcp-observatory |
| IaC | AWS SAM + CloudFormation |
| CI/CD | GitHub Actions (OIDC) |

---

## Repository Layout

```
src/orchestrator/     # Core Lambda handlers and business logic
  trigger_handler.py  # API entry point → starts Step Functions
  worker_handler.py   # Pipeline engine → executes workflow steps
  status_handler.py   # Polls Step Functions DescribeExecution
  config_loader.py    # Loads team.json from S3
  bedrock_invoke.py   # Bedrock agent invocation wrapper
  bedrock_wrappers.py # Low-level Bedrock API wrappers
  mcp_observatory.py  # mcp-observatory telemetry and dual-invoke
  rag.py              # pgvector retrieval (explicit + history modes)
  db.py               # DynamoDB DAO
  gemini.py           # Gemini API integration
  amp_metrics.py      # Amazon Managed Prometheus telemetry
  enrich.py           # Voice correction + schema enforcement
  prompt_builder.py   # Prompt assembly from team config
  schema_validate.py  # JSON Schema validation
  structured_transform.py  # Schema-driven output transformation
  models.py           # Dataclass models (AgentConfig, TeamConfig, etc.)
  logger.py           # Structured JSON logger

config/examples/      # Example team configurations and JSON schemas
  teams/              # Per-team workflow definitions (team.json)
  schemas/            # JSON Schema files for structured output validation

infra/                # Infrastructure as Code
  template.yaml       # Main SAM CloudFormation template
  bedrock-agents.yaml # Bedrock agent provisioning (11 agents)
  samconfig.toml      # SAM deployment parameters

scripts/              # Validation and utility scripts
  validate_team_config.py  # Validates team.json against required schema

tests/                # pytest unit tests
docs/                 # Architecture docs and C4 diagrams
.github/workflows/    # GitHub Actions CI/CD pipeline
Makefile              # Lambda packaging targets
```

---

## Common Commands

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r src/orchestrator/requirements.txt
```

### Run Tests

```bash
pytest
pytest tests/test_worker_handler.py   # single file
pytest -v                              # verbose
```

### Validate Team Config

```bash
python scripts/validate_team_config.py config/examples/team_simple.json
```

### Package Lambdas

```bash
make build-TriggerFunction
make build-WorkerFunction
make build-StatusFunction
# or build all:
make package-lambda
```

### Deploy

```bash
sam build -t infra/template.yaml
sam deploy --guided -t infra/template.yaml     # first deploy (interactive)
sam deploy -t infra/template.yaml              # subsequent deploys
```

### Live Operations

```bash
sam logs -n WorkerFunction --tail              # stream worker logs
aws s3 cp s3://<bucket>/teams/<team>/v1/team.json .   # pull live config
aws s3 cp team.json s3://<bucket>/teams/<team>/v1/team.json  # push config
aws cloudformation describe-stacks --stack-name tarun-content-team \
  --query "Stacks[0].Outputs"                  # get stack outputs
```

---

## Architecture

```
Client
  └─► API Gateway (REST + SIWE authorizer)
        ├─► Trigger Lambda  ──► Step Functions ──► Worker Lambda ──► Bedrock Agents
        ├─► Status Lambda   (polls Step Functions DescribeExecution)       │
        ├─► Provision Lambda (CRUD for agents/teams)                      ▼
        └─► Gemini Lambda   (external research)              DynamoDB / S3 / pgvector
                                                             mcp-observatory (AMP + DDB)
```

**Async pattern**: Trigger Lambda starts Step Functions execution, returns `run_id` immediately.
Client polls `GET /team/task/{run_id}` until `SUCCEEDED` or `FAILED`.
Artifacts land in S3 at `runs/<run_id>/<step_id>.json`.

**Two production workflow patterns:**
- **Visibility Team** — content marketing pipeline: director → strategist → writer → editor → distribution → approval
- **Improvement Team** — personal learning loop: coach → plan → daily tasks

---

## Agent Role System

`team.json` drives everything. Each agent entry has: `id`, `name`, `bedrock`, `goal_template`, `schema_ref`.

- **Supervisor detection** — agents with `agentRole: "supervisor"` (or `role: "supervisor"`) run first and provide the brief to all downstream workers. Falls back to first step in workflow if no supervisor is declared.
- **Models** — `us.amazon.nova-micro-v1:0` for writers, `us.amazon.nova-pro-v1:0` for reasoning, `us.amazon.nova-premier-v1:0` for long-context planning
- **Shadow model** — `shadow_model_id` + `model_aliases` enable dual-invoke via mcp-observatory for A/B comparison

### Simple (Bootstrap) Format

```json
{
  "teamId": "content-team",
  "agents": [
    { "agentId": "...", "agentAliasId": "...", "agentRole": "supervisor", "model": "..." },
    { "agentId": "...", "agentAliasId": "...", "agentRole": "linkedin_writer", "model": "..." }
  ],
  "guardrail": { "guardrailId": "...", "guardrailVersion": "DRAFT" },
  "outputBucket": "..."
}
```

### Rich (Production) Format

See `config/examples/teams/tarun_visibility_team/v1/team.json` — agents have nested `bedrock` object
with `agentId`, `aliasId`, `model_id`, `shadow_model_id`, `model_aliases`.

---

## Key Files

| File | Purpose |
|------|---------|
| `infra/template.yaml` | SAM/CloudFormation stack definition |
| `src/orchestrator/worker_handler.py` | Pipeline engine: reads team config, routes to agents |
| `src/orchestrator/bedrock_invoke.py` | Bedrock Agent invocation with retry and observability |
| `src/orchestrator/mcp_observatory.py` | Observatory telemetry wrapper (dual-invoke, AMP, DDB) |
| `config/examples/teams/*/team.json` | Team workflow configs (also stored in S3 at runtime) |
| `infra/samconfig.toml` | SAM deploy parameters |

---

## Environment Variables (auto-wired by SAM)

| Variable | Purpose |
|----------|---------|
| `CONFIG_BUCKET` | S3 bucket for team configs |
| `CONFIG_PREFIX` | S3 prefix for team configs (default: `teams`) |
| `ARTIFACT_BUCKET` | S3 bucket for run artifacts |
| `DDB_TABLE` | DynamoDB table for runs/tasks |
| `STATE_MACHINE_ARN` | Step Functions state machine ARN |
| `VECTOR_DB_SECRET_ARN` | Secrets Manager ARN for pgvector credentials |
| `GEMINI_SECRET_ARN` | Secrets Manager ARN for Gemini API key (optional) |
| `VECTOR_DB_TABLE` | pgvector table name (default: `rag_chunks`) |
| `VPC_ID` | VPC for Lambda networking |
| `LAMBDA_SUBNET_IDS` | Comma-separated private subnet IDs |
| `RDS_SECURITY_GROUP_ID` | RDS security group for Lambda ingress |
| `OBSERVATORY_METRICS_TABLE` | DynamoDB table for mcp-observatory telemetry |
| `PROVISION_FUNCTION_NAME` | ARN/name of Provision Lambda (for agent management) |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/team/task` | Start a pipeline run → returns `run_id` |
| `GET` | `/team/task/{run_id}` | Poll run status (RUNNING / SUCCEEDED / FAILED) |
| `GET` | `/improve/tasks` | List improvement tasks |
| `POST` | `/improve/task/done` | Mark a task complete |
| `GET` | `/observability/metrics` | Bedrock telemetry from AMP |

---

## AWS Context

- **Region**: `us-east-1`
- **Account**: `239571291755`
- **Stack name**: `tarun-content-team`
- **GitHub Actions Role**: `teamweave-github-actions-sam-deployer` (OIDC)
- **Bedrock Service Role**: `arn:aws:iam::239571291755:role/BedrockAgentServiceRole-Tarun`

---

## Deploy / CI

GitHub Actions workflow (`.github/workflows/deploy.yml`) uses OIDC assume-role.
Always include `--capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM` for deploys.

First deploy (one time):
```bash
cd infra && sam build && sam deploy --guided --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM
```

---

## Known Gotchas

- **StateMachineName must be set explicitly** in `template.yaml` or ARN goes stale on redeploy — currently `!Sub "${AWS::StackName}-state-machine"` ✅
- **`bedrock:InvokeAgent` and `bedrock:InvokeModel` are separate IAM actions** — grant both explicitly
- **SNS, APS/Prometheus, Marketplace** each need explicit IAM grants; they are NOT covered by `AmazonBedrockFullAccess`
- **Guardrail `PROMPT_ATTACK` strength**: set on input only — not valid on output filters
- **PII entity names in guardrails are enums** — use exact AWS values (e.g., `NAME`, `EMAIL`, `PHONE`)
- **Topic definition length** in guardrails has a hard limit — keep descriptions under 200 chars
- **LinkedIn writer agent failures**: check prompt ordering in team.json and `REFUSAL_PHRASES` list in `worker_handler.py`
- **`agentRole: "supervisor"` pattern** — `AgentConfig.role` field populated by `config_loader.py` from `agentRole` or `role` JSON key; falls back to first workflow step if none declared
- **CloudWatch log group name** must derive from `StateMachineName` explicitly — auto-generated ARN is stale after redeploys
- **VPC Lambda + pgvector**: local integration tests require VPN or RDS proxy to reach RDS
- **`CAPABILITY_NAMED_IAM`** must be in every SAM deploy command (IAM roles use explicit names)
- **First deploy chicken-and-egg**: IAM role must exist before Lambda that references it — SAM handles ordering automatically
- **mcp-observatory dual-invoke**: set `shadow_model_id` + `model_aliases` in team.json to enable A/B shadow comparison; omit for single-model runs

---

## Development Notes

- Python 3.12 — match this version locally to avoid dependency drift
- All secrets go through AWS Secrets Manager — never hardcode credentials
- Lambda functions run inside a VPC for RDS access
- Test coverage lives in `tests/` — run `pytest` before any PR
- CI/CD triggers on push to `main` — GitHub Actions runs `sam build` + `sam deploy`
- Team configs in `config/examples/` are examples only; live configs are fetched from S3 at runtime
