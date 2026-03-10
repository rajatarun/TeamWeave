# TeamWeave Architecture

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture Principles](#architecture-principles)
3. [Technology Stack](#technology-stack)
4. [System Components](#system-components)
5. [Data Flow](#data-flow)
6. [API Reference](#api-reference)
7. [Data Models](#data-models)
8. [Infrastructure](#infrastructure)
9. [Observability](#observability)
10. [Security](#security)
11. [CI/CD Pipeline](#cicd-pipeline)
12. [Diagram Index](#diagram-index)

---

## System Overview

**TeamWeave** is a config-driven, multi-agent orchestration platform built entirely on AWS serverless infrastructure. It enables teams to define AI-powered workflows as JSON configurations — no code changes required to introduce new team compositions or step sequences.

The system orchestrates fleets of Amazon Bedrock agents through declarative team definitions, supporting complex multi-step pipelines (e.g., content marketing, personal learning) with retrieval-augmented generation (RAG), external research via Google Gemini, and comprehensive observability.

```
User Request → API Gateway → Trigger Lambda → Step Functions → Worker Lambda → Bedrock Agents
```

Two production team patterns are deployed:

| Team | Purpose | Agents |
|------|---------|--------|
| **Visibility Team** | Content marketing pipeline | Director, Strategist, Writer, Editor, Designer, Distribution, Approver |
| **Improvement Team** | Personal learning system | Coach, Strategist, Advisor |

---

## Architecture Principles

- **Config-Driven Orchestration** — Team compositions, workflow steps, agent assignments, and schemas are defined in JSON stored in S3. No deployments needed for new workflows.
- **Stateless Compute** — All Lambda functions are stateless; all state is persisted in DynamoDB, S3, or Step Functions.
- **Async-First** — Requests are asynchronous by default. Clients receive a `run_id` and poll for results.
- **Least-Privilege IAM** — Each Lambda function has a dedicated IAM role with only the permissions it needs.
- **Observability by Default** — All Bedrock agent invocations are instrumented via MCP Observatory, with metrics pushed to Amazon Managed Prometheus.
- **RAG-Augmented Intelligence** — Agents are grounded with relevant context via pgvector similarity search before each invocation.

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| **Runtime** | Python 3.12 on AWS Lambda |
| **AI/ML** | Amazon Bedrock (Nova micro/lite/pro/premier, Claude 3/3.5 Haiku) |
| **AI Research** | Google Gemini 2.0 Flash |
| **Orchestration** | AWS Step Functions (Standard Workflows) |
| **API** | Amazon API Gateway (REST) |
| **Primary DB** | Amazon DynamoDB |
| **Vector DB** | PostgreSQL 15 + pgvector on Amazon RDS |
| **Object Storage** | Amazon S3 |
| **Secrets** | AWS Secrets Manager |
| **Metrics** | Amazon Managed Prometheus (AMP) |
| **Tracing** | AWS X-Ray |
| **Logging** | Amazon CloudWatch Logs |
| **Networking** | Amazon VPC (private subnets for RDS) |
| **IaC** | AWS SAM + AWS CloudFormation |
| **CI/CD** | GitHub Actions + AWS OIDC |

---

## System Components

### Control Plane

#### Trigger Lambda (`src/orchestrator/trigger_handler.py`)
- Entry point for all API requests via API Gateway
- Starts asynchronous Step Functions executions for team workflow runs
- Handles CRUD operations (agents, teams, roles, departments) by routing to Step Functions
- Returns `run_id` and execution ARN to the caller immediately

#### Status Lambda (`src/orchestrator/status_handler.py`)
- Polls Step Functions for execution state: `RUNNING`, `SUCCEEDED`, `FAILED`
- Returns final output payload on completion
- Used by clients to implement the async polling pattern

#### Provision Lambda (`src/orchestrator/provision_team.py`)
- Creates and updates Amazon Bedrock agents and their aliases
- Manages team configuration sync to S3
- Called by Step Functions for agent/team CRUD operations

### Execution Plane

#### Worker Lambda (`src/orchestrator/worker_handler.py`)
The core pipeline execution engine. For each workflow run:

1. Loads the team definition JSON from S3
2. For each step in the workflow:
   - Resolves the assigned Bedrock agent
   - Fetches RAG context from pgvector (if `explicit_rag` enabled)
   - Builds a prompt from the goal template + request data + RAG context
   - Invokes the Bedrock agent (streaming response)
   - Extracts and validates JSON output against the step's JSON Schema
   - Transforms the output to the target schema
   - Persists step output to S3 and DynamoDB

#### Gemini Research Lambda (`config/examples/gemini_lambda.py`)
- Invoked by Bedrock agents as a **tool action** during agent execution
- Performs web research and content augmentation using Google Gemini 2.0 Flash
- API key retrieved from AWS Secrets Manager at runtime

#### Observatory Metrics Lambda (`src/orchestrator/observatory_handler.py`)
- Exposes a `/observability/metrics` REST endpoint
- Queries Amazon Managed Prometheus for Bedrock telemetry
- Returns token usage, cost, latency, and shadow disagreement metrics

### AWS Step Functions (State Machine)

The state machine is the async workflow coordinator. It:
- Accepts execution input from the Trigger Lambda
- Invokes the Worker Lambda synchronously for pipeline runs
- Routes CRUD actions to the Provision Lambda
- Manages retries and error handling

---

## Data Flow

### Workflow Execution (Happy Path)

```
1. Client POST /team/task
        │
        ▼
2. API Gateway → Trigger Lambda
   - Validate request
   - Write RUN#<id> META record to DynamoDB (status=RUNNING)
   - Start Step Functions execution (async)
   - Return { run_id, execution_arn }
        │
        ▼
3. Step Functions → Worker Lambda (sync invoke)
   - Load team.json from S3
   - For each workflow step:
       a. Resolve Bedrock agent config
       b. Query pgvector for RAG context (if enabled)
       c. Build goal prompt
       d. Invoke Bedrock Agent (streaming)
            │
            ├── Bedrock Agent may invoke Gemini Lambda as tool
            │
       e. Extract + validate JSON output
       f. Write STEP#<id> record to DynamoDB
       g. Write artifact JSON to S3
   - Write final RUN#<id> META record (status=SUCCEEDED)
        │
        ▼
4. Client GET /team/task/{run_id}
   - Status Lambda polls Step Functions
   - Returns RUNNING or { status: SUCCEEDED, result: {...} }
```

### RAG Flow (Explicit Mode)

```
Worker Lambda
   │
   ├── Generate embedding via Bedrock Titan Embed Text v1
   │
   ├── Query pgvector table (cosine similarity, top-k)
   │     └── RDS PostgreSQL over VPC private subnet (SSL)
   │
   └── Inject retrieved chunks into agent goal prompt
```

### Observability Flow

```
Worker Lambda
   │
   ├── MCP Observatory wraps every Bedrock call
   │   - Captures: tokens, latency, cost, trace_id
   │   - Shadow mode: dual-invokes with shadow model, scores disagreement
   │
   ├── Write telemetry span to DynamoDB ObservatoryMetricsTable (TTL=90d)
   │
   └── Prometheus remote_write → AMP Workspace (SigV4 signed)
```

---

## API Reference

### Workflow Execution

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/team/task` | Start a workflow run |
| `GET` | `/team/task/{run_id}` | Poll run status and result |

**POST /team/task request body:**
```json
{
  "team": "visibility_team",
  "version": "v1",
  "owner": "user@example.com",
  "request": {
    "topic": "...",
    "context": "..."
  }
}
```

**GET /team/task/{run_id} response (completed):**
```json
{
  "status": "SUCCEEDED",
  "run_id": "uuid",
  "result": { "...step outputs..." }
}
```

### Improvement Tasks

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/improve/tasks` | List owner's improvement tasks |
| `POST` | `/improve/task/done` | Mark task as complete |

### Agent Management

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/agents` | List all agents |
| `POST` | `/agents` | Create agent |
| `GET` | `/agents/{name}` | Get agent |
| `PUT` | `/agents/{name}` | Update agent |
| `DELETE` | `/agents/{name}` | Delete agent |

### Team Management

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/teams` | List all teams |
| `POST` | `/teams` | Create team |
| `GET` | `/teams/{team_name}` | Get team |
| `PUT` | `/teams/{team_name}` | Update team |
| `DELETE` | `/teams/{team_name}` | Delete team |

### Observability

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/observability/metrics` | Query Bedrock telemetry metrics from AMP |

---

## Data Models

### DynamoDB: ContentPipelineRunsTable

Single-table design using `pk` (partition key) and `sk` (sort key).

#### Run Metadata Record
```
pk: RUN#{uuid}   sk: META
─────────────────────────────────────────
status:      RUNNING | SUCCEEDED | FAILED
updatedAt:   ISO 8601 timestamp
data:        { original request payload }
publishedAt: ISO 8601 timestamp (optional)
```

#### Step Record
```
pk: RUN#{uuid}   sk: STEP#{step_id}
─────────────────────────────────────────
status:       RUNNING | SUCCEEDED | FAILED
inputs:       { resolved step inputs }
output:       { agent JSON output }
artifact_uri: s3://bucket/runs/{uuid}/{step}.json
error:        string (if FAILED)
```

#### Improvement Task Record
```
pk: TASK#{task_id}   sk: TASK#{task_id}
─────────────────────────────────────────
owner:       user identifier
status:      PENDING | DONE
task:        { task definition from Improvement Team }
createdAt:   ISO 8601 timestamp
```

### DynamoDB: ObservatoryMetricsTable

```
pk: OBSERVATORY#{operation}   sk: {iso_timestamp}#{trace_id}
─────────────────────────────────────────────────────────────
agent_id:                    Bedrock agent ID
alias_id:                    Bedrock alias ID
session_id:                  Bedrock session ID
input_length:                chars in prompt
prompt_tokens:               LLM input tokens
completion_tokens:           LLM output tokens
latency_ms:                  end-to-end latency
cost_usd:                    estimated cost
shadow_disagreement_score:   0.0–1.0 (shadow mode)
ttl:                         epoch seconds (90-day expiry)
```

### S3: Team Configuration
**Path:** `teams/{team_name}/{version}/team.json`

```json
{
  "team": { "name": "string", "version": "v1", "owner": "string" },
  "globals": {
    "north_star": "string",
    "default_channel": "linkedin",
    "hard_constraints": ["string"],
    "features": { "gemini_research": false, "explicit_rag": true },
    "rag": { "mode": "explicit", "rag_env_key": "VECTOR_DB_TABLE", "top_k": 5 },
    "artifact_store": {
      "artifact_bucket_env": "ARTIFACT_BUCKET",
      "dynamo_table_env": "DYNAMO_TABLE"
    }
  },
  "agents": [
    {
      "id": "agent_id",
      "name": "Agent Name",
      "bedrock": {
        "agentId": "string",
        "aliasId": "string",
        "model_id": "us.amazon.nova-micro-v1:0"
      },
      "goal_template": "prompt template string",
      "schema_ref": "output_schema_name"
    }
  ],
  "workflow": [
    { "step": "agent_id", "inputs": ["field1", "field2"] }
  ]
}
```

### S3: Step Artifacts
**Path:** `runs/{run_id}/{step_id}.json`

Contains the raw JSON output from the Bedrock agent for that step, validated against the step's JSON Schema.

---

## Infrastructure

### AWS Region & Account
- **Region:** `us-east-1`
- **Account:** `239571291755`

### Lambda Functions

| Function | Memory | Timeout | VPC | Purpose |
|----------|--------|---------|-----|---------|
| TriggerFunction | Default | 30s | No | API entry, Step Functions start |
| WorkerFunction | Default | 15m | Yes | Pipeline execution |
| StatusFunction | Default | 30s | No | Status polling |
| GeminiResearchFunction | Default | 5m | No | Gemini tool action |
| ProvisionTeamFunction | Default | 10m | No | Agent provisioning |
| ObservatoryMetricsFunction | Default | 30s | No | Metrics API |

### Bedrock Agents (bedrock-agents.yaml)

**Visibility Team (7 agents):**
- `director` — Supervisor/Router for content strategy
- `strategist` — Creative brief and content strategy
- `writer-linkedin` — LinkedIn post drafting
- `editor` — Copy editing and refinement
- `designer` — Visual direction recommendations
- `distribution` — Channel distribution planning
- `approve` — Final approval gate

**Improvement Team (4 agents):**
- `coach` — Supervisor/Router for learning journeys
- `strategist` — Learning strategy and goals
- `advisor` — Daily task generation

**Model Aliases per Agent:**

| Alias | Model |
|-------|-------|
| `nova-micro` | `us.amazon.nova-micro-v1:0` |
| `nova-lite` | `us.amazon.nova-lite-v1:0` |
| `nova-pro` | `us.amazon.nova-pro-v1:0` |
| `claude-haiku` | `anthropic.claude-3-haiku-20240307-v1:0` |

### Networking

```
VPC: vpc-f3c92a8a (us-east-1)
│
├── Private Subnets
│   └── WorkerFunction ENIs (for RDS access)
│
└── OrchestratorSecurityGroup
    └── Inbound: TCP 5432 from Lambda ENIs
```

The Worker Lambda is deployed in VPC private subnets to access the RDS PostgreSQL instance (pgvector) over the private network with SSL.

### Bedrock Agent Service Role

`BedrockAgentServiceRole-Tarun` — allows Bedrock agents to invoke Lambda action groups (Gemini research tool).

---

## Observability

### Metrics (Amazon Managed Prometheus)

| Metric | Description |
|--------|-------------|
| `teamweave_bedrock_prompt_tokens_total` | Total input tokens consumed |
| `teamweave_bedrock_completion_tokens_total` | Total output tokens generated |
| `teamweave_bedrock_cost_usd_total` | Estimated USD cost |
| `teamweave_bedrock_requests_total` | Total Bedrock invocations |
| `teamweave_bedrock_input_length_chars` | Input prompt size in characters |
| `teamweave_bedrock_shadow_disagreement_score` | 0–1 disagreement between primary/shadow models |
| `teamweave_bedrock_shadow_numeric_variance` | Numeric variance between model outputs |

Metrics are pushed via Prometheus remote_write (protobuf + Snappy compression) signed with SigV4.

### Shadow Mode

When `shadow_model_id` is configured on an agent, MCP Observatory dual-invokes both the primary and shadow models. The outputs are compared and scored:
- **Disagreement score:** 0.0 (identical) to 1.0 (completely different)
- **Numeric variance:** Statistical variance for numeric outputs

This enables safe model evaluation without impacting production output.

### Tracing

AWS X-Ray tracing is enabled on Lambda functions and Step Functions for end-to-end request tracing.

---

## Security

### IAM Roles (Least Privilege)

| Role | Key Permissions |
|------|----------------|
| `WorkerRole` | Bedrock InvokeAgent, S3 GetObject/PutObject, DynamoDB CRUD, Lambda Invoke (Gemini), EC2 VPC (ENI), APS RemoteWrite |
| `TriggerRole` | Step Functions StartExecution, S3 GetObject, DynamoDB CRUD, APS QueryMetrics |
| `StatusRole` | Step Functions DescribeExecution |
| `ProvisionTeamRole` | Bedrock Agent CRUD, S3 PutObject/GetObject (team configs) |
| `GeminiResearchRole` | Secrets Manager GetSecretValue, CloudWatch Logs |
| `ObservatoryRole` | APS QueryMetrics |
| `StepFunctionsRole` | Lambda InvokeFunction, X-Ray, CloudWatch Logs Delivery |

### Secrets Management
- Google Gemini API key stored in AWS Secrets Manager (`gemini/api_key`)
- Retrieved at Lambda cold start; cached in memory for warm invocations
- RDS credentials managed via RDS IAM authentication (SSL verify-full)

### Network Security
- Worker Lambda deployed in VPC private subnets
- RDS accessible only within VPC (no public endpoint)
- `OrchestratorSecurityGroup` restricts inbound to TCP 5432 from Lambda ENIs only

### API Gateway
- REST API with CORS enabled
- Assumes external authentication (API Gateway authorizer, API key, or upstream auth proxy)

### Bedrock Guardrails
- Shared guardrail applied to all Bedrock agents (`explicit v1 baseline`)
- Prevents harmful or policy-violating content generation

---

## CI/CD Pipeline

### GitHub Actions Workflow (`.github/workflows/deploy.yml`)

```
Trigger: push to main | manual workflow_dispatch
         │
         ▼
1. Checkout repository
         │
         ▼
2. Configure AWS credentials
   - OIDC role assumption (no long-lived keys)
   - Role: teamweave-github-actions-sam-deployer
         │
         ▼
3. SAM Build
   - Packages Lambda code + dependencies
   - Copies source, config, and RDS CA bundle
         │
         ▼
4. SAM Validate
   - Validates CloudFormation template
         │
         ▼
5. SAM Deploy
   - Stack: TeamWeaveStack
   - Region: us-east-1
   - Parameters:
       TeamConfigPrefix=teams
       VectorDbTable=resume-rag-db
       GeminiSecretArn=<ARN>
         │
         ▼
6. On Failure: Dump CloudFormation events
   - Main stack events
   - Bedrock agents stack events
```

### Makefile Build Targets

| Target | Description |
|--------|-------------|
| `build-TriggerFunction` | Package trigger Lambda |
| `build-WorkerFunction` | Package worker Lambda (includes RDS CA bundle) |
| `build-StatusFunction` | Package status Lambda |
| `build-GeminiResearchFunction` | Package Gemini research Lambda |
| `build-ProvisionTeamFunction` | Package provision Lambda |
| `build-ObservatoryMetricsFunction` | Package observatory Lambda |

---

## Diagram Index

The following C4 and infrastructure diagrams are available in `docs/c4/`:

| File | Description |
|------|-------------|
| `docs/c4/context.puml` | C4 Level 1 — System Context diagram |
| `docs/c4/container.puml` | C4 Level 2 — Container diagram |
| `docs/c4/component.puml` | C4 Level 3 — Component diagram (Worker Lambda) |
| `docs/aws-infrastructure.puml` | AWS Infrastructure diagram (Lucidchart style) |

To render diagrams, use [PlantUML](https://plantuml.com/) or the PlantUML VS Code extension.

```bash
# Render all diagrams to PNG
plantuml docs/c4/*.puml docs/aws-infrastructure.puml
```
