#!/usr/bin/env python3
"""Convert Kaggle notebook -> GCP notebook in-place. Touches only 7 cells + injects 1.

Usage: python3 scripts/utils/adapt_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[2] / "notebooks" / "META_CXR_gcp.ipynb"


def cell_code(src: str, *, parameters: bool = False) -> dict:
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }
    if parameters:
        cell["metadata"]["tags"] = ["parameters"]
    return cell


def cell_md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


PARAMETERS_CELL = """\
# === Papermill parameters cell ===
# Override these from CLI: papermill ... -p RUN_ID my-run -p RESUME_FROM checkpoint_last.pth
RUN_ID = "mimic_cxr_2gpu_stage1"
RESUME_FROM = ""        # "" = auto-pick best.pth from GCS; or "checkpoint_last.pth" / "checkpoint_N.pth"
BATCH_SIZE = None       # None = giữ value trong configs/training.yaml
MAX_EPOCH = None        # None = giữ value trong configs/training.yaml
SMOKE_TEST = False      # True = override max_epoch=1, batch_size=2 cho smoke run

import os, sys
os.environ.setdefault("RUN_ID", RUN_ID)

# Install SIGTERM flush handler (spot eviction safety).
sys.path.insert(0, "/workspace/Meta-CXR-GCP")
try:
    import yaml
    from src.gcs_checkpoint import install_sigterm_flush
    with open("/workspace/Meta-CXR-GCP/configs/checkpoint.yaml") as _f:
        _CKPT = yaml.safe_load(_f)
    install_sigterm_flush(
        run_id=RUN_ID,
        local_dir=_CKPT["local_dir"],
        gcs_prefix=_CKPT["gcs_prefix"],
        preserve_files=_CKPT["preserve_files"],
        keep_last_n=_CKPT["keep_last_n_epoch_ckpts"],
        flush_seconds=_CKPT["upload"]["sigterm_flush_seconds"],
    )
    print("SIGTERM flush handler installed.")
except Exception as _e:
    print(f"Warning: SIGTERM handler not installed: {_e}")
"""

CELL_4_WANDB = """\
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path("/workspace/Meta-CXR-GCP/.env"), override=False)
load_dotenv(override=False)  # fallback: CWD .env

if not os.environ.get("WANDB_API_KEY"):
    raise RuntimeError(
        "WANDB_API_KEY not set. Add to .env or pass via docker -e WANDB_API_KEY=..."
    )

import wandb
wandb.login()
print("wandb: Logged in successfully")
"""

CELL_6_REPO = """\
import os

# Repo đã có sẵn trong Docker image; chỉ chdir.
REPO_DIR = os.environ.get("REPO_DIR", "/workspace/Meta-CXR-GCP")
os.chdir(REPO_DIR)
print(f"Working directory: {os.getcwd()}")
!ls -la
"""

CELL_8_VERIFY = """\
import os
import yaml

with open("configs/data.yaml") as f:
    DATA_CFG = yaml.safe_load(f)

MOUNT = DATA_CFG["mount_point"]
PATHS = DATA_CFG["paths"]

assert os.path.ismount(MOUNT) or os.path.isdir(MOUNT), (
    f"GCS mount missing at {MOUNT}. Run gcsfuse before launching notebook "
    f"(see scripts/vm_startup.sh)."
)

# CSVs phải tồn tại.
for fname in DATA_CFG["required_csvs"]:
    p = os.path.join(MOUNT, fname)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing CSV on GCS mount: {p}")
print("All required CSVs present.")

# Images + reports roots.
for k in ("images_root", "reports_root"):
    p = PATHS[k]
    if not os.path.isdir(p):
        raise FileNotFoundError(f"{k} not found: {p}")
print(f"images_root:  {PATHS['images_root']}")
print(f"reports_root: {PATHS['reports_root']}")

# Export env vars expected downstream (data/lavis loaders đọc qua env_config.yaml — set ở cell sau).
os.environ["KAGGLE_INPUT"] = MOUNT       # giữ tên biến để code cũ không phải đổi
os.environ["IMAGE_ROOT"]   = MOUNT
os.environ["REPORTS_ROOT"] = PATHS["reports_root"]
os.environ["REPORTS_CSV"]  = PATHS["reports_csv"]

import pandas as pd
df = pd.read_csv(PATHS["reports_csv"])
print(f"\\nLoaded reports CSV: {len(df)} rows")
print(df[["Img_Folder", "Img_Filename"]].head(3).to_string())
"""

CELL_10_ENVCFG = """\
import os
import subprocess
import yaml

# Java auto-detect (Docker image cài openjdk-8 ở /usr/lib/jvm/...).
result = subprocess.run(
    "readlink -f $(which java) | sed 's|/bin/java||'",
    shell=True, capture_output=True, text=True
)
java_home = result.stdout.strip() or "/usr/lib/jvm/java-8-openjdk-amd64/jre"
java_path = java_home + "/bin:"

with open("configs/data.yaml") as f:
    DATA = yaml.safe_load(f)
with open("configs/training.yaml") as f:
    TRAIN = yaml.safe_load(f)

P = DATA["paths"]
OUTPUT_DIR = TRAIN["run"]["output_dir"]

env_config = {
    "paths": {
        "data_root":          P["data_root"],
        "mimic_cxr_jpg_root": P["mimic_cxr_jpg_root"],
        "split_csv":          P["split_csv"],
        "reports_csv":        P["reports_csv"],
        "chexpert_csv":       P["chexpert_csv"],
        "metadata_csv":       P["metadata_csv"],
        "output_dir":         OUTPUT_DIR,
        "checkpoint_dir":     OUTPUT_DIR,
    },
    "wandb": {
        "entity":  os.environ.get("WANDB_ENTITY", ""),
        "project": os.environ.get("WANDB_PROJECT", "meta-cxr"),
    },
    "java": {"home": java_home, "path": java_path},
}

os.makedirs("configs", exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open("configs/env_config.yaml", "w") as f:
    yaml.safe_dump(env_config, f, sort_keys=False)

print("Written configs/env_config.yaml:")
print(yaml.safe_dump(env_config, sort_keys=False))
"""

CELL_12_LAUNCH = """\
import os
import subprocess
import sys
import yaml

REPO_DIR = os.environ.get("REPO_DIR", "/workspace/Meta-CXR-GCP")

with open("configs/training.yaml") as f:
    TRAIN = yaml.safe_load(f)
with open("configs/checkpoint.yaml") as f:
    CKPT_CFG = yaml.safe_load(f)

OUTPUT_DIR = TRAIN["run"]["output_dir"]
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Resume detection: prefer best.pth từ GCS (qua gcsfuse mount) ──
from src.gcs_checkpoint import resolve_resume_checkpoint   # noqa: E402

resume_path = resolve_resume_checkpoint(
    run_id=RUN_ID,
    override=RESUME_FROM or None,
    local_output_dir=OUTPUT_DIR,
)
resume_args = ["--options", f"run.resume_ckpt_path={resume_path}"] if resume_path else []

# Smoke-test overrides (papermill param).
override_args = []
if SMOKE_TEST:
    override_args += ["--options", "run.max_epoch=1", "run.batch_size_train=2", "run.batch_size_eval=2"]
else:
    if MAX_EPOCH:
        override_args += ["--options", f"run.max_epoch={MAX_EPOCH}"]
    if BATCH_SIZE:
        override_args += ["--options", f"run.batch_size_train={BATCH_SIZE}"]

cmd = [
    sys.executable, "-m", "torch.distributed.run",
    "--standalone",
    "--nproc_per_node=2",
    "--master_port=12355",
    "-m", "pretraining.train",
    "--cfg-path", "pretraining/configs/mimic_cxr_2gpu.yaml",
    "--options", f"run.output_dir={OUTPUT_DIR}",
] + resume_args + override_args

print("Launch command:")
print(" ".join(cmd))
print("\\n" + "=" * 60 + "\\n")

env = os.environ.copy()
env["PYTHONPATH"] = REPO_DIR
env["RUN_ID"] = RUN_ID

proc = subprocess.Popen(
    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, bufsize=1, cwd=REPO_DIR, env=env,
)
for line in proc.stdout:
    print(line, end="", flush=True)
proc.wait()
print(f"\\n{'=' * 60}\\nTraining exited with code: {proc.returncode}")
"""

CELL_16_UPLOAD = """\
import os
import yaml

from src.gcs_checkpoint import upload_run_to_gcs

with open("configs/training.yaml") as f:
    TRAIN = yaml.safe_load(f)
with open("configs/checkpoint.yaml") as f:
    CKPT = yaml.safe_load(f)

upload_run_to_gcs(
    run_id=RUN_ID,
    local_dir=TRAIN["run"]["output_dir"],
    gcs_prefix=CKPT["gcs_prefix"],
    preserve_files=CKPT["preserve_files"],
    keep_last_n=CKPT["keep_last_n_epoch_ckpts"],
    verify=CKPT["upload"]["verify_after_upload"],
)
print("Checkpoint sync to GCS complete.")
"""


def main():
    nb = json.loads(NB_PATH.read_text())
    cells = nb["cells"]

    # Cell 0: title.
    cells[0] = cell_md("# META-CXR Training on GCP Compute Engine (2x T4 spot)\n")

    # Inject parameters cell as new cell at index 1 (becomes cell 1, pushing rest by 1).
    # To keep indices in plan stable, insert AFTER cell 0 — i.e. index 1.
    cells.insert(1, cell_code(PARAMETERS_CELL, parameters=True))

    # Now original index N → N+1. Replacements (using NEW indices):
    # Original  2 (deps)      -> new 3   : keep as-is
    # Original  4 (wandb)     -> new 5
    # Original  6 (repo)      -> new 7
    # Original  8 (verify)    -> new 9
    # Original 10 (envcfg)    -> new 11
    # Original 12 (launch)    -> new 13
    # Original 16 (push)      -> new 17

    cells[5]  = cell_code(CELL_4_WANDB)
    cells[7]  = cell_code(CELL_6_REPO)
    cells[9]  = cell_code(CELL_8_VERIFY)
    cells[11] = cell_code(CELL_10_ENVCFG)
    cells[13] = cell_code(CELL_12_LAUNCH)
    cells[17] = cell_code(CELL_16_UPLOAD)

    # Also retitle markdown headers for parity.
    if cells[16]["cell_type"] == "markdown":
        cells[16] = cell_md("## Cell 8 — Upload checkpoints to GCS\n")

    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"Adapted: {NB_PATH}")


if __name__ == "__main__":
    main()
