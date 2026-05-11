#!/usr/bin/env bash
# Resume training from latest GCS checkpoint. Thin wrapper around 04_train.sh.
#
# Usage:
#   ./scripts/04_resume_training.sh                          # auto-pick best.pth
#   ./scripts/04_resume_training.sh checkpoint_last.pth      # specific file
#   RUN_ID=my-run ./scripts/04_resume_training.sh
set -euo pipefail
source "$(dirname "$0")/common.sh"

RESUME_FROM="${1:-}"   # empty = let notebook auto-pick best.pth via gcs_checkpoint

if [[ -z "${RUN_ID:-}" ]]; then
  die "Set RUN_ID (the original run id) before resuming. e.g. RUN_ID=mimic_cxr_20260512_120000 $0"
fi

log "Resume run ${RUN_ID} from: ${RESUME_FROM:-<auto-pick best.pth>}"
RESUME_FROM="${RESUME_FROM}" RUN_ID="${RUN_ID}" exec "$(dirname "$0")/04_train.sh"
