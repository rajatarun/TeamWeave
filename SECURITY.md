# Security policy

## Supported versions

Security fixes are applied on the default branch (`main`) and released through
the project's normal deploy process. Older deployments may not receive backports
unless a maintainer explicitly tags a release — check the repository for tags.

## Reporting a vulnerability

**Please do not open a public GitHub issue for security problems.**

1. Email the repository maintainer(s) listed on the GitHub repository **About**
   section, or use GitHub **Private vulnerability reporting** if enabled for this
   repo.
2. Include: affected component (e.g. Worker Lambda, API route), steps to
   reproduce, and impact assessment.
3. Allow reasonable time for a fix before public disclosure.

We will acknowledge receipt and coordinate on disclosure timing.

## Secure development

- Do not commit AWS keys, wallet keys, or third-party API keys.
- Rotate Secrets Manager values if they appear in logs or tickets.
- Run `pytest` before submitting changes; workflow and permission tests guard
  common IAM mistakes.

See [docs/security-notes.md](docs/security-notes.md) for a threat-model sketch.
