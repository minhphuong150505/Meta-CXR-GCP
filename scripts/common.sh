#!/usr/bin/env bash
# Common helpers sourced by every script in scripts/.
# Strict mode + .env loading + tiny logging.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load .env if present (POSIX export of every line; ignore comments/blank).
if [[ -f "${REPO_DIR}/.env" ]]; then
  set -o allexport
  # shellcheck disable=SC1090
  source "${REPO_DIR}/.env"
  set +o allexport
fi

# Defaults (mirror .env.example).
: "${GCP_PROJECT_ID:?GCP_PROJECT_ID must be set in .env}"
: "${GCP_REGION:=us-central1}"
: "${GCP_ZONE:=us-central1-a}"
: "${GCS_BUCKET:?GCS_BUCKET must be set in .env}"
: "${VM_NAME:=meta-cxr-train}"
: "${VM_MACHINE_TYPE:=n1-standard-8}"
: "${VM_GPU_TYPE:=nvidia-tesla-t4}"
: "${VM_GPU_COUNT:=2}"
: "${VM_DISK_SIZE_GB:=200}"
: "${VM_IMAGE_FAMILY:=common-cu121-debian-11}"
: "${VM_IMAGE_PROJECT:=deeplearning-platform-release}"
: "${ARTIFACT_REPO:=meta-cxr}"
: "${DOCKER_IMAGE:=meta-cxr-gcp}"
: "${DOCKER_TAG:=latest}"
: "${SERVICE_ACCOUNT_NAME:=meta-cxr-training}"
: "${SERVICE_ACCOUNT_EMAIL:=${SERVICE_ACCOUNT_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com}"

ZONE_FALLBACK=("${GCP_ZONE}" "us-central1-b" "us-central1-c" "us-central1-f")
IMAGE_URI="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${ARTIFACT_REPO}/${DOCKER_IMAGE}:${DOCKER_TAG}"

log() {
  printf '[%s] %s\n' "$(date +'%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

# Print intent for --dry-run mode: prefix with "WOULD RUN: ".
maybe_run() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    log "WOULD RUN: $*"
  else
    log "RUN: $*"
    "$@"
  fi
}
