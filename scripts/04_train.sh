#!/usr/bin/env bash
# MAIN: start VM if stopped → wait SSH → run Docker → papermill notebook → upload exec'd → stop VM.
#
# Usage:
#   ./scripts/04_train.sh                 # full run
#   SMOKE=1 ./scripts/04_train.sh         # smoke test (max_epoch=1, batch=2)
#   RUN_ID=my-run ./scripts/04_train.sh   # custom run id
set -euo pipefail
source "$(dirname "$0")/common.sh"

: "${WANDB_API_KEY:?WANDB_API_KEY must be in .env}"
RUN_ID="${RUN_ID:-mimic_cxr_$(date +%Y%m%d_%H%M%S)}"
SMOKE="${SMOKE:-0}"
RESUME_FROM="${RESUME_FROM:-}"

# Resolve VM zone (created by 03_create_vm.sh).
ZONE="$(cat "${REPO_DIR}/.vm_zone" 2>/dev/null || echo "${GCP_ZONE}")"
log "Run ID: ${RUN_ID} | Zone: ${ZONE} | Smoke: ${SMOKE}"

# Make sure VM is up.
STATE="$(gcloud compute instances describe "${VM_NAME}" --zone="${ZONE}" --format='value(status)' 2>/dev/null || echo MISSING)"
if [[ "${STATE}" == "MISSING" ]]; then
  die "VM ${VM_NAME} not found. Run ./scripts/03_create_vm.sh first."
fi
if [[ "${STATE}" != "RUNNING" ]]; then
  maybe_run gcloud compute instances start "${VM_NAME}" --zone="${ZONE}"
fi

# Wait SSH ready (DLVM startup-script can take ~3 min on first boot).
log "Waiting for SSH..."
for _ in {1..60}; do
  if gcloud compute ssh "${VM_NAME}" --zone="${ZONE}" --command="echo ok" --quiet 2>/dev/null; then
    break
  fi
  sleep 5
done

# Build the in-VM command. Heredoc writes a script onto VM and runs inside Docker.
REMOTE_SCRIPT="/tmp/run_papermill_${RUN_ID}.sh"
NB_OUT="output/${RUN_ID}.ipynb"
GS_LOG="gs://${GCS_BUCKET}/logs/${RUN_ID}.ipynb"

REMOTE_BODY=$(cat <<'EOF'
set -euo pipefail
cd /workspace/Meta-CXR-GCP
mkdir -p output
papermill notebooks/META_CXR_gcp.ipynb "$NB_OUT" \
  -p RUN_ID "$RUN_ID" \
  -p RESUME_FROM "$RESUME_FROM" \
  -p SMOKE_TEST "$SMOKE_BOOL" \
  --log-output --log-level INFO
gsutil cp "$NB_OUT" "$GS_LOG" || true
EOF
)

SMOKE_BOOL=$([[ "${SMOKE}" == "1" ]] && echo True || echo False)

log "Copy run script to VM..."
gcloud compute ssh "${VM_NAME}" --zone="${ZONE}" --command="cat > ${REMOTE_SCRIPT}" <<EOF
RUN_ID="${RUN_ID}"
RESUME_FROM="${RESUME_FROM}"
SMOKE_BOOL="${SMOKE_BOOL}"
NB_OUT="${NB_OUT}"
GS_LOG="${GS_LOG}"
GCS_BUCKET="${GCS_BUCKET}"
${REMOTE_BODY}
EOF

log "Run inside Docker on VM..."
set +e
gcloud compute ssh "${VM_NAME}" --zone="${ZONE}" --command="\
  docker run --rm --gpus all \
    -v /mnt/gcs-data:/mnt/gcs-data \
    -v ${REMOTE_SCRIPT}:/run.sh:ro \
    -e WANDB_API_KEY='${WANDB_API_KEY}' \
    -e WANDB_ENTITY='${WANDB_ENTITY:-}' \
    -e WANDB_PROJECT='${WANDB_PROJECT:-meta-cxr}' \
    -e GCS_BUCKET='${GCS_BUCKET}' \
    -e RUN_ID='${RUN_ID}' \
    ${IMAGE_URI} bash /run.sh"
RC=$?
set -e

log "Docker exited with ${RC}. Stopping VM (auto_stop)."
maybe_run gcloud compute instances stop "${VM_NAME}" --zone="${ZONE}" --quiet

if [[ $RC -ne 0 ]]; then
  die "Training failed (exit ${RC}). Executed notebook at ${GS_LOG} (if uploaded)."
fi
log "Done. Notebook: ${GS_LOG}"
log "Checkpoints: gs://${GCS_BUCKET}/checkpoints/${RUN_ID}/"
