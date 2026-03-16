Run a full SAM build and deploy for TeamWeave.

Steps:
1. Run `sam build -t infra/template.yaml` and check for errors
2. Run `sam deploy --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM`
3. Tail CloudWatch logs for WorkerFunction to confirm clean startup: `sam logs -n WorkerFunction --tail`
4. Print the Outputs section from the CloudFormation stack:
   `aws cloudformation describe-stacks --stack-name tarun-content-team --query "Stacks[0].Outputs"`
