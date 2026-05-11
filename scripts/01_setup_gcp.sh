#!/usr/bin/env bash
# Enable APIs, create service account, IAM bindings, Artifact Registry repo.
# Idempotent: safe to re-run.
set -euo pipefail
source "$(dirname "$0")/common.sh"

log "Project: ${GCP_PROJECT_ID} | Region: ${GCP_REGION}"

log "== Enable required APIs =="
maybe_run gcloud services enable \
  compute.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  iam.googleapis.com \
  iamcredentials.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com \
  --project="${GCP_PROJECT_ID}"

log "== Service account =="
if gcloud iam service-accounts describe "${SERVICE_ACCOUNT_EMAIL}" --project="${GCP_PROJECT_ID}" >/dev/null 2>&1; then
  log "Service account already exists: ${SERVICE_ACCOUNT_EMAIL}"
else
  maybe_run gcloud iam service-accounts create "${SERVICE_ACCOUNT_NAME}" \
    --display-name="META-CXR training" \
    --project="${GCP_PROJECT_ID}"
fi

log "== IAM bindings =="
for role in roles/storage.objectAdmin roles/artifactregistry.reader roles/logging.logWriter roles/monitoring.metricWriter; do
  maybe_run gcloud projects add-iam-policy-binding "${GCP_PROJECT_ID}" \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="${role}" \
    --condition=None \
    --quiet
done

log "== Artifact Registry repo =="
if gcloud artifacts repositories describe "${ARTIFACT_REPO}" --location="${GCP_REGION}" --project="${GCP_PROJECT_ID}" >/dev/null 2>&1; then
  log "Repo already exists: ${ARTIFACT_REPO}"
else
  maybe_run gcloud artifacts repositories create "${ARTIFACT_REPO}" \
    --repository-format=docker \
    --location="${GCP_REGION}" \
    --description="META-CXR Docker images" \
    --project="${GCP_PROJECT_ID}"
fi

log "Setup complete. Next: ./scripts/02_build_docker.sh"
