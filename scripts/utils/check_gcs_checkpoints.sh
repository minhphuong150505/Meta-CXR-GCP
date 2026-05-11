#!/usr/bin/env bash
# List checkpoints currently stored on GCS, grouped by run.
set -euo pipefail
source "$(dirname "$0")/../common.sh"

PREFIX="gs://${GCS_BUCKET}/checkpoints"
log "Listing ${PREFIX}/"
gsutil ls "${PREFIX}/" 2>/dev/null | while read -r RUN_URI; do
  echo ""
  echo "## ${RUN_URI}"
  gsutil ls -lh "${RUN_URI}" 2>/dev/null | sed 's/^/  /'
done
