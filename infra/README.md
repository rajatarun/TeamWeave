# Bedrock Multi-Agent Stack

This directory contains a CloudFormation template that provisions:
- 1 shared Bedrock guardrail + explicit version (`v1 baseline`)
- 11 Bedrock agents across two teams
- 11 Bedrock agent aliases (`production` -> `DRAFT`)
- Multi-agent collaboration on the two supervisors:
  - `tarun_visibility_team-director`
  - `tarun_improvement_team-coach`

## Prerequisites
- AWS account: `239571291755`
- Region: `us-east-1`
- Existing IAM role for all agents:
  `arn:aws:iam::239571291755:role/BedrockAgentServiceRole-Tarun` (default parameter `AgentServiceRoleArn`)

## Deploy
Set model via env var (optional):

```bash
export FOUNDATION_MODEL_ID=amazon.nova-micro-v1:0
```

Deploy:

```bash
sam deploy \
  --template-file infra/bedrock-agents.yaml \
  --stack-name tarun-bedrock-agents \
  --capabilities CAPABILITY_NAMED_IAM \
  --region us-east-1 \
  --resolve-s3 \
  --no-fail-on-empty-changeset \
  --parameter-overrides \
    FoundationModelId=${FOUNDATION_MODEL_ID:-amazon.nova-micro-v1:0} \
    AgentServiceRoleArn=arn:aws:iam::239571291755:role/BedrockAgentServiceRole-Tarun
```


Supervisor collaboration is enabled by default via parameters:
- `VisibilitySupervisorCollaborationMode=SUPERVISOR_ROUTER` (director)
- `ImprovementSupervisorCollaborationMode=SUPERVISOR_ROUTER` (coach)
- Collaborators remain `DISABLED` and are invoked through supervisor `AgentCollaborators`.

## Outputs
The stack exports these key values:
- `SharedGuardrailId` and `SharedGuardrailVersion`
- `VisibilityDirectorAgentId`
- `ImprovementCoachAgentId`
- Collaborator alias ARNs for debugging:
  - `VisibilityStrategistAliasArn`
  - `VisibilityWriterLinkedinAliasArn`
  - `VisibilitySeoAliasArn`
  - `VisibilityEditorAliasArn`
  - `VisibilityDesignerAliasArn`
  - `VisibilityDistributionAliasArn`
  - `VisibilityDirectorApproveAliasArn`
  - `ImprovementStrategistAliasArn`
  - `ImprovementAdvisorAliasArn`

## Inspect stack outputs
```bash
aws cloudformation describe-stacks \
  --stack-name tarun-bedrock-agents \
  --region us-east-1 \
  --query 'Stacks[0].Outputs'
```
