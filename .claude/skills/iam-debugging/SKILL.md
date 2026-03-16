---
name: iam-debugging
description: >
  Use when debugging IAM permission errors in TeamWeave SAM deployments.
  Covers Bedrock, Step Functions, Lambda, S3, DynamoDB, CloudWatch.
---

# IAM Debugging Skill for TeamWeave

## Common Missing Permissions

| Error | Missing IAM Action |
|---|---|
| AccessDenied on InvokeAgent | bedrock:InvokeAgent |
| AccessDenied on InvokeModel | bedrock:InvokeModel |
| SNS publish fails | sns:Publish |
| Step Functions won't start | states:StartExecution |
| CloudWatch logs missing | logs:CreateLogDelivery, logs:DescribeLogGroups |
| S3 artifact write fails | s3:PutObject on runs bucket |
| AMP remote_write fails | aps:RemoteWrite |
| Marketplace model access denied | aws-marketplace:Subscribe |

## First Deploy Checklist
1. `--capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM` must be in deploy command
2. IAM role must exist BEFORE the Lambda that references it (SAM handles ordering)
3. Step Functions IAM role needs `lambda:InvokeFunction` to call WorkerFunction
4. TriggerFunction role needs `states:StartExecution` to start the state machine
5. WorkerFunction role needs `bedrock:InvokeAgent` AND `bedrock:InvokeModel` (separate actions)

## Diagnosing from CloudFormation Events
```bash
aws cloudformation describe-stack-events --stack-name tarun-content-team \
  --query "StackEvents[?ResourceStatus=='CREATE_FAILED']"
```

## Diagnosing from CloudWatch Logs
```bash
sam logs -n WorkerFunction --stack-name tarun-content-team --tail
```

Look for: `AccessDeniedException`, `UnrecognizedClientException`, `ExpiredTokenException`

## Bedrock-Specific IAM
- `bedrock:InvokeAgent` — required for calling Bedrock Agents
- `bedrock:InvokeModel` — required for direct model calls (enrich.py, schema enforcement)
- Both must be granted; `AmazonBedrockFullAccess` covers both but is too broad for production

## GitHub Actions OIDC
Role: `arn:aws:iam::239571291755:role/teamweave-github-actions-sam-deployer`
The role must have `sts:AssumeRoleWithWebIdentity` from the GitHub OIDC provider.
