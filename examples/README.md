# TeamWeave examples

These scripts run **without AWS credentials**. They help you explore configs and
integration surfaces before deploying the SAM stack.

| Script | What it does |
|--------|----------------|
| [`validate_team.py`](validate_team.py) | Lint `team.json` workflow wiring, schemas, and agents |
| [`generate_a2a_card.py`](generate_a2a_card.py) | Print an [A2A](https://github.com/a2aproject/A2A) Agent Card built from `config/examples/teams` |

## Quick run

```bash
pip install -r requirements.txt
python examples/validate_team.py --all-examples
python examples/generate_a2a_card.py --base-url https://api.example.com | head
```

Example team definitions live under [`config/examples/teams`](../config/examples/teams/).
After deploy, the platform reads team configs from S3 (see [README](../README.md)).
