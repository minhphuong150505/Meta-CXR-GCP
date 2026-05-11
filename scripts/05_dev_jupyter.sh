#!/usr/bin/env bash
# Dev mode: open JupyterLab on VM via SSH tunnel to localhost:8888.
# Reuses the same Docker image. Ctrl+C to stop tunnel; then run ./scripts/06_stop_vm.sh.
set -euo pipefail
source "$(dirname "$0")/common.sh"

ZONE="$(cat "${REPO_DIR}/.vm_zone" 2>/dev/null || echo "${GCP_ZONE}")"
PORT="${JUPYTER_PORT:-8888}"
TOKEN="${JUPYTER_TOKEN:-meta-cxr-dev}"

# Ensure VM is running.
STATE="$(gcloud compute instances describe "${VM_NAME}" --zone="${ZONE}" --format='value(status)' 2>/dev/null || echo MISSING)"
[[ "${STATE}" != "MISSING" ]] || die "VM not found. Run 03_create_vm.sh first."
[[ "${STATE}" == "RUNNING" ]] || gcloud compute instances start "${VM_NAME}" --zone="${ZONE}"

log "Starting JupyterLab in Docker on VM (background)..."
gcloud compute ssh "${VM_NAME}" --zone="${ZONE}" --command="\
  docker ps --filter name=meta-cxr-jupyter -q | xargs -r docker rm -f; \
  docker run -d --rm --name meta-cxr-jupyter --gpus all -p ${PORT}:${PORT} \
    -v /mnt/gcs-data:/mnt/gcs-data \
    -e WANDB_API_KEY='${WANDB_API_KEY:-}' \
    -e GCS_BUCKET='${GCS_BUCKET}' \
    ${IMAGE_URI} \
    jupyter lab --ip=0.0.0.0 --port=${PORT} --no-browser --allow-root \
      --ServerApp.token='${TOKEN}' --ServerApp.root_dir=/workspace/Meta-CXR-GCP"

log "Open: http://localhost:${PORT}/lab?token=${TOKEN}"
log "Tunneling SSH -L ${PORT}:localhost:${PORT}  (Ctrl+C to stop)"
exec gcloud compute ssh "${VM_NAME}" --zone="${ZONE}" -- -N -L "${PORT}:localhost:${PORT}"
