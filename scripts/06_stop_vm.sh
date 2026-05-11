#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
ZONE="$(cat "${REPO_DIR}/.vm_zone" 2>/dev/null || echo "${GCP_ZONE}")"
maybe_run gcloud compute instances stop "${VM_NAME}" --zone="${ZONE}" --quiet
log "VM stopped (disk persists; resume with 07_start_vm.sh)."
