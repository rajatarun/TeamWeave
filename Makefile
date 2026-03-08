.PHONY: package-lambda \
	build-TriggerFunction \
	build-WorkerFunction \
	build-StatusFunction \
	build-GeminiResearchFunction \
	build-ProvisionTeamFunction \
	build-OrchestratorFunction

package-lambda:
	python -m pip install -r src/requirements.txt -t "$(ARTIFACTS_DIR)"
	cp -r src "$(ARTIFACTS_DIR)/"
	cp -r config "$(ARTIFACTS_DIR)/"
	mkdir -p "$(ARTIFACTS_DIR)/certs"
	curl -fsSL "https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem" -o "$(ARTIFACTS_DIR)/certs/rds-ca-bundle.pem"

build-TriggerFunction: package-lambda

build-WorkerFunction: package-lambda

build-StatusFunction: package-lambda

build-GeminiResearchFunction: package-lambda

build-ProvisionTeamFunction: package-lambda

# Backward-compatible alias for older templates.
build-OrchestratorFunction: package-lambda
