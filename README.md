# TeamWeave

**Define multi-agent AI pipelines in JSON and run them on AWS** — no Lambda
redeploy to add teams, steps, or schemas.

**Who it is for:** Platform and product engineers building async agent workflows
on Bedrock (AgentCore), with structured outputs, optional RAG, and HTTP/A2A
clients.

> **Demo recording:** A walkthrough GIF or short video will be linked here
> after the first public demo is recorded. Until then, use the offline examples
> below.

---

## What you get

| Area | Description |
|------|-------------|
| **Workflows** | Step Functions + Worker Lambda execute `team.json` pipelines |
| **Configs** | Agents, prompts, JSON Schema, RAG mode, pre/post tools |
| **Example teams** | Personal productivity flows under `config/examples/teams/` (e.g. `daily_operator`, `linkedin_quick_post`, `job_hunter`) plus `tarun_visibility_team` (content pipeline) |
| **Integrations** | A2A agent card, optional MCP sibling calls, OpenTelemetry-friendly spans — see [docs/INTEGRATIONS.md](docs/INTEGRATIONS.md) |
| **API** | REST per [openapi/teamweave.yaml](openapi/teamweave.yaml); SIWE JWT on protected routes |

**Not local-first:** A full production run needs **AWS** (Lambda, Step Functions,
DynamoDB, S3, Bedrock/AgentCore). Without cloud resources you can still develop
with **unit tests** and **offline config tools** in [`examples/`](examples/).

---

## 5-minute quickstart (no AWS account)

Verified on Python 3.12:

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -q tests/test_workflow_integrity.py tests/test_example_team_lint.py
python examples/validate_team.py --all-examples
python examples/generate_a2a_card.py --base-url https://api.example.com | head -30
```

Install virtualenv support on Debian if needed: `sudo apt install python3.12-venv`.

---

## Deploy (AWS required)

```bash
sam build -t infra/template.yaml
sam deploy --guided -t infra/template.yaml
```

Parameters, Bedrock setup, and stack layout: [infra/README.md](infra/README.md).

After deploy, resolve stack outputs for scripts and curl:

```bash
eval "$(python scripts/stack_env.py --stack YOUR_STACK_NAME --format sh)"
```

---

## Copy-paste examples

### 1. Lint a team config (offline)

```bash
python examples/validate_team.py config/examples/teams/daily_operator/v1/team.json
```

### 2. Preview the A2A Agent Card from example teams (offline)

```bash
python examples/generate_a2a_card.py --base-url "https://YOUR_API_GATEWAY_URL"
```

### 3. Start a pipeline run (after deploy)

Replace the base URL and use a valid SIWE bearer token if your stage enforces auth:

```bash
export API_URL="https://YOUR_API_GATEWAY_URL"
curl -sS -X POST "$API_URL/team/task" \
  -H 'content-type: application/json' \
  -H "authorization: Bearer YOUR_JWT_IF_REQUIRED" \
  -d '{
    "team": "daily_operator",
    "version": "v1",
    "request": {
      "brain_dump": "Inbox full, need to ship blog post and fix CI flake."
    }
  }'
# Poll: curl -sS "$API_URL/team/task/RUN_ID"
```

---

## Architecture

```
Client → API Gateway → Trigger Lambda → Step Functions → Worker Lambda → AgentCore / tools
                              │
                    Status Lambda (poll)
```

Diagrams and detail: [docs/architecture.md](docs/architecture.md) · [docs/design/overview.md](docs/design/overview.md)

---

## Repository layout

```
src/orchestrator/     Lambda handlers and pipeline logic
src/agentcore/        AgentCore runtime entrypoint (ARM64 artifact)
config/examples/      Example teams, schemas, roles
infra/                SAM / CloudFormation
tests/                pytest suite
examples/             Offline lint and A2A card preview
docs/                 Architecture, integrations, extending guide
```

---

## Configuration sketch

Teams live at `teams/{name}/{version}/team.json` in the config bucket (examples
in-repo under `config/examples/teams/`):

```json
{
  "team": { "name": "daily_operator", "version": "v1" },
  "agents": [{ "id": "triage", "goal_template": "...", "schema_ref": "day_triage_v1" }],
  "workflow": [{ "step": "triage", "inputs": ["request.brain_dump"] }],
  "schemas": { "day_triage_v1": { "type": "object" } }
}
```

Extend teams: [docs/EXTENDING.md](docs/EXTENDING.md)

---

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) · [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) ·
[SECURITY.md](SECURITY.md)

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
