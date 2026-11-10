# Good first issues (draft)

These are **scoped starter tasks** inferred from the codebase. A maintainer can
turn each block into a GitHub issue. Acceptance criteria are explicit.

---

### 1. Align README and architecture with removed teams

**Files:** `README.md`, `docs/architecture.md`, `infra/README.md`  
**Problem:** Docs still describe an "Improvement Team" and legacy Bedrock agent
names while `config/examples/teams` no longer includes `tarun_improvement_team`.  
**Acceptance:** Docs list only teams present under `config/examples/teams/`; note
`/improve/tasks` as a DynamoDB API that may be empty unless populated elsewhere.

---

### 2. Add `examples/__main__.py` or Make target for lint-all

**Files:** `Makefile`, `examples/validate_team.py`  
**Acceptance:** `make lint-teams` runs `python examples/validate_team.py --all-examples` and exits non-zero on failure.

---

### 3. Document local pytest without venv on Debian

**Files:** `CONTRIBUTING.md`  
**Acceptance:** Mentions `python3.12-venv` or `pip install --user` path when `ensurepip` is missing; no secrets in doc.

---

### 4. OpenAPI ↔ template drift guard

**Files:** `tests/test_openapi_contract.py`, `openapi/teamweave.yaml`  
**Acceptance:** If a new route is added to `infra/template.yaml`, failing test
message names the missing OpenAPI entry (extend test if gap found).

---

### 5. Gateway resolution loop test harness

**Files:** `tests/test_gateway_targets.py`, `.github/workflows/deploy.yml`  
**Problem:** Several `TestTheResolutionLoopRuns` tests fail when extracted loop
output is empty (environment/bash `set -eu` mismatch).  
**Acceptance:** All tests in that class pass on a clean checkout after fix.

---

### 6. Offline benchmark: team lint + workflow integrity

**Files:** `examples/team_lint.py`, `tests/test_example_team_lint.py`  
**Acceptance:** CI documents running example linter; optional `scripts/bench_teams.sh` times lint over all example teams (synthetic; report wall time only).

---

### 7. `datetime.utcnow()` deprecation in `db.py`

**Files:** `src/orchestrator/db.py`, `tests/test_dynamo_number_types.py`  
**Acceptance:** Use timezone-aware UTC; no new DeprecationWarning in tests touching DAO.

---

### 8. LangGraph spike (research only)

**Files:** `docs/INTEGRATIONS.md`, new `docs/design/langgraph-spike.md` (optional)  
**Acceptance:** One-page design for compiling `workflow` → LangGraph nodes; marked **planned**, no false "implemented" claims.

---

### 9. MCP server facade sketch

**Files:** `docs/INTEGRATIONS.md`, `examples/`  
**Acceptance:** Issue describes MCP tools `list_teams`, `start_run`, `get_run_status` mapping to existing HTTP routes; fit rated high, effort medium.

---

### 10. Portfolio / health team request_schema coverage

**Files:** `tests/test_request_schema.py`, `config/examples/teams/health_prep/v1/team.json`  
**Acceptance:** Test fails if any `goal_template` references `request.*` fields not declared in `request_schema`.
