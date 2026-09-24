.PHONY: package-lambda install-bedrock-kb-sdk \
	build-TriggerFunction \
	build-WorkerFunction \
	build-StatusFunction \
	build-GeminiResearchFunction \
	build-ProvisionTeamFunction \
	build-OrchestratorFunction \
	build-ObservatoryMetricsFunction \
	build-AgentMetricsDashboardFunction \
	build-UnifiedObservabilityFunction \
	build-ConversationFunction

package-lambda:
	python -m pip install -r src/requirements.txt -t "$(ARTIFACTS_DIR)" \
		--python-version 3.12 \
		--platform manylinux2014_aarch64 \
		--implementation cp \
		--only-binary=:all:
	cp -r src "$(ARTIFACTS_DIR)/"
	cp -r config "$(ARTIFACTS_DIR)/"
	mkdir -p "$(ARTIFACTS_DIR)/certs"
	curl -fsSL "https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem" -o "$(ARTIFACTS_DIR)/certs/rds-ca-bundle.pem"

# The Lambda runtime's botocore predates Managed Knowledge Bases. Its
# CreateKnowledgeBase client rejects managedKnowledgeBaseConfiguration
# before the request is signed, and its Retrieve client does the same to
# managedSearchConfiguration. 1.43.32 added the managed type; 1.43.92 is
# the first model that also describes Marengo's modelConfiguration document
# and the supplemental storage location. Installed only into the two
# functions that call those APIs. Every other function keeps the runtime
# copy — src/requirements.txt does not ship a second one.
install-bedrock-kb-sdk:
	python -m pip install -r src/requirements-bedrock-kb.txt -t "$(ARTIFACTS_DIR)" \
		--python-version 3.12 \
		--platform manylinux2014_aarch64 \
		--implementation cp \
		--only-binary=:all:

build-TriggerFunction: package-lambda

build-WorkerFunction: package-lambda install-bedrock-kb-sdk

build-StatusFunction: package-lambda

build-GeminiResearchFunction: package-lambda

build-ProvisionTeamFunction: package-lambda

# Backward-compatible alias for older templates.
build-OrchestratorFunction: package-lambda

build-ObservatoryMetricsFunction: package-lambda

build-AgentMetricsDashboardFunction: package-lambda

build-UnifiedObservabilityFunction: package-lambda

build-ConversationFunction: package-lambda

build-A2AFunction: package-lambda

build-HealthKbSyncFunction: package-lambda

build-HealthKbProvisionFunction: package-lambda install-bedrock-kb-sdk
