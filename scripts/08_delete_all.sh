#!/usr/bin/env bash
# Tear-down: delete VM. Does NOT delete the GCS bucket or Artifact Registry repo
# (checkpoints are valuable; user can delete those manually).
set -euo pipefail
source "$(dirname "$0")/common.sh"

ZONE="$(cat "${REPO_DIR}/.vm_zone" 2>/dev/null || echo "${GCP_ZONE}")"

read -r -p "Delete VM ${VM_NAME} in ${ZONE}? GCS bucket + Artifact Registry kept. [y/N] " ans
[[ "${ans,,}" == "y" ]] || { log "Aborted."; exit 0; }

maybe_run gcloud compute instances delete "${VM_NAME}" --zone="${ZONE}" --quiet
rm -f "${REPO_DIR}/.vm_zone"
log "VM deleted. Bucket gs://${GCS_BUCKET} and repo ${ARTIFACT_REPO} still exist."
log "To delete bucket:  gsutil -m rm -r gs://${GCS_BUCKET}"
log "To delete repo:    gcloud artifacts repositories delete ${ARTIFACT_REPO} --location=${GCP_REGION}"
