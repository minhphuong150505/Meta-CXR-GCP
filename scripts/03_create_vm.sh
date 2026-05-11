#!/usr/bin/env bash
# Create 2x T4 spot VM. Falls back across zones if capacity unavailable.
# Idempotent: skips if VM already exists.
set -euo pipefail
source "$(dirname "$0")/common.sh"

# If VM already exists, just print its zone and exit.
EXISTING_ZONE="$(gcloud compute instances list --filter="name=${VM_NAME}" --format="value(zone.basename())" 2>/dev/null || true)"
if [[ -n "${EXISTING_ZONE}" ]]; then
  log "VM ${VM_NAME} already exists in zone ${EXISTING_ZONE} — skipping create."
  echo "${EXISTING_ZONE}" > "${REPO_DIR}/.vm_zone"
  exit 0
fi

STARTUP_SCRIPT="${REPO_DIR}/scripts/vm_startup.sh"
[[ -f "${STARTUP_SCRIPT}" ]] || die "Missing ${STARTUP_SCRIPT}"

for ZONE in "${ZONE_FALLBACK[@]}"; do
  log "== Trying zone ${ZONE} =="
  set +e
  gcloud compute instances create "${VM_NAME}" \
    --zone="${ZONE}" \
    --project="${GCP_PROJECT_ID}" \
    --machine-type="${VM_MACHINE_TYPE}" \
    --accelerator="type=${VM_GPU_TYPE},count=${VM_GPU_COUNT}" \
    --image-family="${VM_IMAGE_FAMILY}" \
    --image-project="${VM_IMAGE_PROJECT}" \
    --boot-disk-size="${VM_DISK_SIZE_GB}GB" \
    --boot-disk-type=pd-balanced \
    --maintenance-policy=TERMINATE \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --service-account="${SERVICE_ACCOUNT_EMAIL}" \
    --scopes=cloud-platform \
    --metadata="install-nvidia-driver=True,enable-oslogin=TRUE,GCS_BUCKET=${GCS_BUCKET},DOCKER_IMAGE=${IMAGE_URI}" \
    --metadata-from-file="startup-script=${STARTUP_SCRIPT}" \
    --quiet
  rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then
    log "VM ${VM_NAME} created in ${ZONE}"
    echo "${ZONE}" > "${REPO_DIR}/.vm_zone"
    exit 0
  fi
  log "Zone ${ZONE} failed (likely no spot capacity) — trying next."
done

die "All zones exhausted (${ZONE_FALLBACK[*]}). Try again later or request quota."
