#!/usr/bin/env bash
# Build Docker image locally and push to Artifact Registry.
# WARNING: image base ~10GB; first build pulls a lot.
set -euo pipefail
source "$(dirname "$0")/common.sh"

log "Image: ${IMAGE_URI}"

log "== Configure docker auth for Artifact Registry =="
maybe_run gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev" --quiet

log "== Build =="
maybe_run docker build -t "${IMAGE_URI}" "${REPO_DIR}"

log "== Push =="
maybe_run docker push "${IMAGE_URI}"

log "Pushed. Verify: gcloud artifacts docker images list ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${ARTIFACT_REPO}"
