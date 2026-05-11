#!/usr/bin/env bash
# Show current billing for the project this month + running compute resources.
# Free Trial credit detail: console only (no public API). This script gives an
# estimate from billing export OR falls back to listing live billable resources.
set -euo pipefail
source "$(dirname "$0")/../common.sh"

log "Active GCE instances:"
gcloud compute instances list --project="${GCP_PROJECT_ID}" \
  --format="table(name,zone.basename(),status,machineType.basename(),scheduling.provisioningModel)"

log "GCS bucket size (gs://${GCS_BUCKET}):"
gsutil du -sh "gs://${GCS_BUCKET}" || true

log "For dollar amount, open: https://console.cloud.google.com/billing?project=${GCP_PROJECT_ID}"
