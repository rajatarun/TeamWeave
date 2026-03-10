# TeamWeave — Claude Code Guide

## Project Overview

TeamWeave is a **config-driven, serverless multi-agent orchestration platform** on AWS. It lets teams define AI-powered workflows as JSON configs (no code redeploy needed), orchestrating Amazon Bedrock agents through Step Functions pipelines.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Runtime | Python 3.12 (AWS Lambda) |
| AI/ML | Amazon Bedrock (Nova, Claude Haiku), Google Gemini 2.0 Flash |
| Orchestration | AWS Step Functions (Standard Workflows) |
| API | AWS API Gateway (REST) |
| State | DynamoDB (run/task metadata) |
| Vector DB | RDS PostgreSQL 15 + pgvector (RAG) |
| Storage | S3 (team configs, artifacts) |
| Secrets | AWS Secrets Manager |
| Observability | Amazon Managed Prometheus (AMP), CloudWatch, X-Ray |
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
  bedrock_invoke.py   # Bedrock agent invocation
  rag.py              # pgvector retrieval (explicit + history modes)
  db.py               # DynamoDB DAO
  gemini.py           # Gemini API integration
  amp_metrics.py      # Amazon Managed Prometheus telemetry
  requirements.txt    # Python dependencies

config/examples/      # Example team configurations and JSON schemas
  teams/              # Per-team workflow definitions (team.json)
  schemas/            # JSON Schema files for structured output validation

infra/                # Infrastructure as Code
  template.yaml       # Main SAM CloudFormation template
  bedrock-agents.yaml # Bedrock agent provisioning (11 agents)
  samconfig.toml      # SAM deployment parameters

tests/                # pytest unit tests (17 test files)
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
pip install -r requirements.txt
```

### Run Tests

```bash
pytest
pytest tests/test_worker_handler.py   # single file
pytest -v                              # verbose
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
sam deploy --guided -t infra/template.yaml
```

---

## Architecture

```
Client
  └─► API Gateway
        ├─► Trigger Lambda  ──► Step Functions ──► Worker Lambda ──► Bedrock Agents
        ├─► Status Lambda   (polls Step Functions)                       │
        ├─► Provision Lambda (CRUD for agents/teams)                    ▼
        └─► Gemini Lambda   (external research)              DynamoDB / S3 / pgvector
```

**Two production workflow patterns:**
- **Visibility Team** — content marketing pipeline: strategy → draft → edit → distribute
- **Improvement Team** — personal learning loop: coach → plan → daily tasks

---

## Key Concepts

### Team Config (`team.json`)
All workflow logic lives in JSON stored in S3. Agents, pipeline steps, RAG settings, and output schemas are all defined there. No Lambda code changes needed for new workflows.

### RAG Modes
- **Explicit mode** — pgvector similarity search on `rag_chunks` table
- **History mode** — DynamoDB-based execution history retrieval

### Structured Output
Worker validates agent outputs against JSON Schema before passing them downstream (`schema_validate.py`, `structured_transform.py`).

### Async Execution
Every run is async: `POST /team/task` returns a `run_id`, then poll `GET /team/task/{run_id}` until `SUCCEEDED` or `FAILED`.

---

## Environment Variables (auto-wired by SAM)

| Variable | Purpose |
|----------|---------|
| `CONFIG_BUCKET` | S3 bucket for team configs |
| `ARTIFACT_BUCKET` | S3 bucket for run artifacts |
| `DDB_TABLE` | DynamoDB table for runs/tasks |
| `STATE_MACHINE_ARN` | Step Functions state machine ARN |
| `VECTOR_DB_SECRET_ARN` | Secrets Manager ARN for pgvector credentials |
| `GEMINI_SECRET_ARN` | Secrets Manager ARN for Gemini API key (optional) |
| `TEAM_CONFIG_PREFIX` | S3 prefix for team configs (default: `teams`) |
| `VECTOR_DB_TABLE` | pgvector table name (default: `rag_chunks`) |
| `VPC_ID` | VPC for Lambda networking |
| `LAMBDA_SUBNET_IDS` | Comma-separated private subnet IDs |
| `RDS_SECURITY_GROUP_ID` | RDS security group for Lambda ingress |

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
- **GitHub Actions Role**: `teamweave-github-actions-sam-deployer` (OIDC)
- **Bedrock Service Role**: `arn:aws:iam::239571291755:role/BedrockAgentServiceRole-Tarun`

---

## Development Notes

- Python 3.12 — match this version locally to avoid dependency drift
- All secrets go through AWS Secrets Manager — never hardcode credentials
- Lambda functions run inside a VPC for RDS access; local integration tests require VPN or RDS proxy
- Test coverage lives in `tests/` — run `pytest` before any PR
- CI/CD triggers on push to `main` — the GitHub Actions workflow runs `sam build` + `sam deploy`
- Team configs in `config/examples/` are examples only; live configs are fetched from S3 at runtime
