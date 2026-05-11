#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
ZONE="$(cat "${REPO_DIR}/.vm_zone" 2>/dev/null || echo "${GCP_ZONE}")"
maybe_run gcloud compute instances start "${VM_NAME}" --zone="${ZONE}"
