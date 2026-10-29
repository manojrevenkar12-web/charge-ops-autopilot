#!/usr/bin/env bash
# Deploy to an Azure subscription you own. Not run from this repository's CI (no credentials there);
# CI only compiles the Bicep and checks the policies are current.
set -euo pipefail
RG=${RG:-rg-charge-ops-dev}
LOC=${LOC:-westeurope}
uv run cpa apim
az group create -n "$RG" -l "$LOC"
az deployment group create -g "$RG" -f azure/infra/main.bicep -p env=dev postgresAdminPassword="$PG_PASSWORD"
az ml environment create -f azure/ml/environment.yml -g "$RG" -w cpa-dev-ml
az ml online-endpoint create -f azure/ml/endpoint.yml -g "$RG" -w cpa-dev-ml
az ml online-deployment create -f azure/ml/deployment.yml -g "$RG" -w cpa-dev-ml --all-traffic
