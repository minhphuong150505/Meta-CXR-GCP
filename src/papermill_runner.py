"""CLI wrapper: papermill notebooks/META_CXR_gcp.ipynb với params từ YAML + .env.

Cũng install SIGTERM handler để force-flush checkpoint khi spot eviction.

Usage:
    python -m src.papermill_runner --run-id myrun [--resume-from checkpoint_last.pth] [--smoke]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from src.config_loader import REPO_ROOT, load_config
from src.gcs_checkpoint import install_sigterm_flush


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=os.environ.get("RUN_ID", "default-run"))
    ap.add_argument("--resume-from", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    output_dir = cfg.training.run.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Register SIGTERM handler — fires if spot VM is being preempted.
    install_sigterm_flush(
        run_id=args.run_id,
        local_dir=output_dir,
        gcs_prefix=cfg.checkpoint.gcs_prefix,
        preserve_files=list(cfg.checkpoint.preserve_files),
        keep_last_n=cfg.checkpoint.keep_last_n_epoch_ckpts,
        flush_seconds=cfg.checkpoint.upload.sigterm_flush_seconds,
    )

    nb_in = REPO_ROOT / "notebooks" / "META_CXR_gcp.ipynb"
    nb_out = REPO_ROOT / "output" / f"{args.run_id}.ipynb"
    nb_out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "papermill",
        str(nb_in), str(nb_out),
        "-p", "RUN_ID", args.run_id,
        "-p", "RESUME_FROM", args.resume_from,
        "-p", "SMOKE_TEST", "True" if args.smoke else "False",
        "--log-output", "--log-level", "INFO",
    ]
    print("papermill cmd:", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    sys.exit(main())
