Debug a failing Bedrock agent invocation for role: $ARGUMENTS

Steps:
1. Search CloudWatch logs for the agent role name: `$ARGUMENTS`
2. Look for refusal phrases, throttling errors, or timeout patterns
3. Check the prompt ordering in team.json for this agent
4. Check src/orchestrator/worker_handler.py REFUSAL_PHRASES list
5. Suggest fixes based on the error pattern found
