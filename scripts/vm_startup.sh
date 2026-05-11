#!/usr/bin/env bash
# Runs on VM at boot (via instances create --metadata-from-file=startup-script=...).
# Mounts GCS bucket and pulls Docker image. Idempotent.
set -euo pipefail
exec > >(tee -a /var/log/meta-cxr-startup.log) 2>&1

echo "[startup] $(date) — begin"

# Metadata from instance.
META="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
fetch() { curl -sf -H "Metadata-Flavor: Google" "${META}/$1"; }

GCS_BUCKET="$(fetch GCS_BUCKET || true)"
DOCKER_IMAGE="$(fetch DOCKER_IMAGE || true)"

[[ -n "${GCS_BUCKET}" ]]    || { echo "[startup] GCS_BUCKET missing"; exit 1; }
[[ -n "${DOCKER_IMAGE}" ]]  || { echo "[startup] DOCKER_IMAGE missing"; exit 1; }

# --- gcsfuse install (DLVM has it usually; install if missing) ---
if ! command -v gcsfuse >/dev/null; then
  echo "[startup] installing gcsfuse"
  export GCSFUSE_REPO=gcsfuse-$(lsb_release -c -s)
  echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
  curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
  sudo apt-get update && sudo apt-get install -y gcsfuse
fi

# --- mount bucket ---
MOUNT=/mnt/gcs-data
mkdir -p "${MOUNT}"
if ! mountpoint -q "${MOUNT}"; then
  echo "[startup] gcsfuse mount ${GCS_BUCKET} -> ${MOUNT}"
  gcsfuse --implicit-dirs --file-mode=644 --dir-mode=755 "${GCS_BUCKET}" "${MOUNT}"
fi
ls "${MOUNT}" | head -10

# --- docker auth + image pull ---
REGION="$(echo "${DOCKER_IMAGE}" | cut -d- -f1)"
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet || true
echo "[startup] docker pull ${DOCKER_IMAGE}"
docker pull "${DOCKER_IMAGE}" || true

echo "[startup] $(date) — ready"
