.PHONY: build-OrchestratorFunction

build-OrchestratorFunction:
	python -m pip install -r src/requirements.txt -t "$(ARTIFACTS_DIR)"
	cp -r src "$(ARTIFACTS_DIR)/"
	cp -r config "$(ARTIFACTS_DIR)/"
	mkdir -p "$(ARTIFACTS_DIR)/certs"
	curl -fsSL "https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem" -o "$(ARTIFACTS_DIR)/certs/rds-ca-bundle.pem"
