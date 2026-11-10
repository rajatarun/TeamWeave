# Deploy configuration (GitHub Actions)

TeamWeave’s deploy workflow (`.github/workflows/deploy.yml`) reads **repository secrets** and **repository variables** instead of hardcoded account identifiers in git. Configure these in GitHub **before** the next deploy on `main`; missing values cause the workflow or CloudFormation to fail.

**Git history may still contain older literals.** After merging this cleanup, rotate or review any credentials that were ever committed. This document lists only **placeholder** examples.

## Repository secrets

| Name | Purpose | Example placeholder |
|------|---------|---------------------|
| `AWS_DEPLOY_ROLE_ARN` | IAM role ARN for GitHub OIDC (`aws-actions/configure-aws-credentials`) | `arn:aws:iam::123456789012:role/example-github-actions-sam-deployer` |
| `GEMINI_SECRET_ARN` | Secrets Manager ARN for the Gemini API key (`GeminiSecretArn` stack parameter) | `arn:aws:secretsmanager:us-east-1:123456789012:secret:gemini/api_key-AbCdEf` |
| `BEDROCK_AGENT_SERVICE_ROLE_ARN` | IAM role Bedrock Classic agents use when `ENABLE_AGENTCORE` is false (`BedrockAgentServiceRoleArn`) | `arn:aws:iam::123456789012:role/example-bedrock-agent-service-role` |

## Repository variables

| Name | Purpose | Example placeholder |
|------|---------|---------------------|
| `AWS_REGION` | AWS region for all deploy steps | `us-east-1` |
| `TEAMWEAVE_STACK_NAME` | Main SAM stack (`template.yaml`) | `teamweave` |
| `TEAMWEAVE_SHARED_STACK_NAME` | Shared networking / observability stack (`shared.yaml`) | `teamweave-shared` |
| `BEDROCK_AGENTS_STACK_NAME` | Classic Bedrock agents stack (failure dump only) | `teamweave-bedrock-agents` |
| `CONTEXTWEAVE_STACK_NAME` | ContextWeave stack whose outputs supply RAG URL and health bucket | `contextweave-rag-prod` |
| `DEVICEWEAVE_STACK_NAME` | DeviceWeave stack for `DeviceWeaveApiBaseUrl` | `deviceweave-prod` |
| `SIWE_STACK_NAME` | SIWE authorizer stack | `siwe-infra` |
| `CIPHERWEAVE_STACK_NAME` | CipherWeave stack for token endpoint and MCP target | `cipherweave-prod` |
| `SAM_DEPLOYER_ROLE_NAME` | Short IAM role name (no path) for template `SamDeployerRoleName` | `example-github-actions-sam-deployer` |
| `VPC_ID` | VPC for Lambda and shared stack (`VpcId`) | `vpc-0123456789abcdef0` |
| `RDS_SECURITY_GROUP_ID` | Security group allowing Lambda → RDS (`RdsSecurityGroupId`) | `sg-0123456789abcdef0` |
| `VECTOR_DB_TABLE` | pgvector table name (`VectorDbTable`) | `rag_chunks` |
| `VECTOR_DB_HOST` | RDS hostname for explicit RAG mode (`VectorDbHost`) | `vector-db.cluster-abcdefghijkl.us-east-1.rds.amazonaws.com` |
| `EXPERTISE_RAG_ARTIFACTS_BUCKET` | Optional S3 bucket for CI doc upload; leave empty to skip the step | `expertise-rag-artifacts-123456789012-prod` |

## Local / manual deploy

For ad-hoc `sam deploy` from a workstation, pass the same values as `--parameter-overrides` (and export role ARNs in the shell). See `infra/README.md` and `infra/samconfig.toml` for layout; do **not** put real ARNs or account IDs in `samconfig.toml` in git.

Example (placeholders only):

```bash
export AWS_REGION=us-east-1
export AWS_DEPLOY_ROLE_ARN=arn:aws:iam::123456789012:role/example-github-actions-sam-deployer
export GEMINI_SECRET_ARN=arn:aws:secretsmanager:us-east-1:123456789012:secret:gemini/api_key-AbCdEf

cd infra
sam build -t template.yaml
sam deploy \
  --stack-name teamweave \
  --resolve-s3 \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    VpcId=vpc-0123456789abcdef0 \
    RdsSecurityGroupId=sg-0123456789abcdef0 \
    GeminiSecretArn="${GEMINI_SECRET_ARN}" \
    BedrockAgentServiceRoleArn=arn:aws:iam::123456789012:role/example-bedrock-agent-service-role
```

## `.env.example`

Copy to `.env` for local scripts (never commit `.env`):

```bash
# See docs/DEPLOY_CONFIG.md — placeholder values only
AWS_REGION=us-east-1
TEAMWEAVE_STACK_NAME=teamweave
AWS_DEPLOY_ROLE_ARN=arn:aws:iam::123456789012:role/example-github-actions-sam-deployer
GEMINI_SECRET_ARN=arn:aws:secretsmanager:us-east-1:123456789012:secret:gemini/api_key-AbCdEf
VPC_ID=vpc-0123456789abcdef0
```

## Sibling MCP stack names

The deploy workflow still resolves ScreenWeave, CipherWeave, DataDictionary, and ToolWeave MCP endpoints from **fixed stack name strings** in the SAM Deploy step (see comments in `.github/workflows/deploy.yml`). Those names are environment-specific but not account IDs; change them in the workflow only when your deployed sibling stack names differ.
