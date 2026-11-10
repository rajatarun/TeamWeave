# Contributing to TeamWeave

Thanks for helping improve TeamWeave. This project is **AWS-first**; most
contributions are Python, CloudFormation/SAM, JSON team configs, and tests.

## Development setup

**Requirements:** Python 3.12 (match Lambda runtime), `pip`, optional AWS SAM CLI for deploy dry-runs.

```bash
git clone <your-fork-url>
cd TeamWeave

# Prefer a virtual environment:
python3 -m venv .venv && source .venv/bin/activate
# On Debian/Ubuntu without python3-venv: sudo apt install python3.12-venv
# Or: pip install --user -r requirements.txt -r requirements-dev.txt

pip install -r requirements.txt -r requirements-dev.txt
```

## Run tests

```bash
pytest
# Single module:
pytest tests/test_workflow_integrity.py -q
```

The suite is large (~1500 tests). CI runs `pytest` before `sam build` on push to
`main` (see `.github/workflows/deploy.yml`).

**Note:** A handful of tests exercise bash fragments extracted from the deploy
workflow; if `tests/test_gateway_targets.py` fails locally, compare with CI on
`main` before chasing regressions.

## Offline team lint (no AWS)

```bash
python examples/validate_team.py --all-examples
```

## Style and conventions

- Match existing module layout under `src/orchestrator/`.
- Team changes: edit under `config/examples/teams/` and run workflow integrity tests.
- Infrastructure: `infra/template.yaml` — run `pytest tests/test_template_lint.py` when touching IAM or env vars.
- Do not use `aws s3 sync --delete` for team configs; use `scripts/sync_team_configs.py`.

## Pull requests

1. Branch from `main` (fork workflow) or use a descriptive branch name.
2. Keep commits focused; explain *why* in the message.
3. Ensure `pytest` passes for areas you touched.
4. Do not commit secrets, account IDs, or personal data in docs or configs.
5. Open a PR against `main`; fill in the template checklist.

## Code of conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Extending the platform

Read [docs/EXTENDING.md](docs/EXTENDING.md) before adding tools, RAG modes, or teams.
