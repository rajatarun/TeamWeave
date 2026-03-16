Validate TeamWeave configuration and code before deploying.

Steps:
1. Run `python scripts/validate_team_config.py` against team_simple.json (bootstrap format)
2. Run `python -m py_compile src/orchestrator/worker_handler.py src/orchestrator/bedrock_invoke.py` to check syntax
3. Run `python -m pytest tests/ -v` if tests/ directory exists
4. Check template.yaml has StateMachineName set explicitly (grep for StateMachineName)
5. Confirm CAPABILITY_NAMED_IAM is present in samconfig.toml or .github/workflows/*.yml
6. Report any issues found
