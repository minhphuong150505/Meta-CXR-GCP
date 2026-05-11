"""GCS checkpoint sync.

Behavior (mirror configs/checkpoint.yaml):
- `checkpoint_best.pth`  → upload, NEVER delete on GCS or local.
- `checkpoint_last.pth`  → upload (overwrite), keep.
- `checkpoint_<N>.pth`   → upload, rolling-delete to keep `keep_last_n` newest on GCS.

`upload_run_to_gcs` calls `gsutil cp` then `gsutil stat` to verify size BEFORE
any delete. We shell out to `gsutil` (already in the deep-learning image) thay vì
google-cloud-storage SDK để giảm thêm dep và để command tự retry network blips.

`resolve_resume_checkpoint` looks up the run prefix on GCS and downloads the
chosen file to local before training starts.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

EPOCH_RE = re.compile(r"^checkpoint_(\d+)\.pth$")


def _bucket() -> str:
    b = os.environ.get("GCS_BUCKET")
    if not b:
        raise RuntimeError("GCS_BUCKET env var not set (see .env)")
    return b


def _gsutil(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    cmd = ["gsutil", *args]
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def gcs_uri(prefix: str, run_id: str, filename: str = "") -> str:
    base = f"gs://{_bucket()}/{prefix.strip('/')}/{run_id}"
    return f"{base}/{filename}" if filename else base + "/"


def upload_file(local: Path, remote: str, verify: bool = True) -> None:
    _gsutil("cp", str(local), remote)
    if verify:
        local_size = local.stat().st_size
        stat = _gsutil("stat", remote)
        m = re.search(r"Content-Length:\s*(\d+)", stat.stdout)
        if not m or int(m.group(1)) != local_size:
            raise RuntimeError(
                f"Upload size mismatch for {remote}: local={local_size}, "
                f"stat output={stat.stdout!r}"
            )


def list_gcs(prefix: str) -> list[str]:
    res = _gsutil("ls", prefix, check=False)
    if res.returncode != 0:
        return []
    return [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]


def upload_run_to_gcs(
    *,
    run_id: str,
    local_dir: str | Path,
    gcs_prefix: str,
    preserve_files: Iterable[str],
    keep_last_n: int,
    verify: bool = True,
) -> None:
    """Sync local_dir → gs://<bucket>/<gcs_prefix>/<run_id>/.

    Uploads every .pth file. After upload:
      - Files in `preserve_files` are never deleted (best, last).
      - `checkpoint_<N>.pth` files are kept rolling — only `keep_last_n` newest
        (by epoch number) remain on GCS. Older ones get `gsutil rm`'d.
    """
    local_dir = Path(local_dir)
    preserve = set(preserve_files)
    if not local_dir.is_dir():
        print(f"[gcs_checkpoint] local_dir missing: {local_dir} — nothing to upload")
        return

    pth_files = sorted(local_dir.glob("*.pth"))
    if not pth_files:
        print(f"[gcs_checkpoint] no .pth in {local_dir}")
        return

    for f in pth_files:
        remote = gcs_uri(gcs_prefix, run_id, f.name)
        print(f"[gcs_checkpoint] upload {f.name} ({f.stat().st_size/1e6:.1f} MB) -> {remote}")
        upload_file(f, remote, verify=verify)

    # Rolling delete: only checkpoint_<N>.pth on GCS, keep N newest by epoch num.
    remote_list = list_gcs(gcs_uri(gcs_prefix, run_id))
    numbered = []
    for uri in remote_list:
        name = uri.rsplit("/", 1)[-1]
        m = EPOCH_RE.match(name)
        if m:
            numbered.append((int(m.group(1)), uri))
    numbered.sort()  # ascending epoch
    to_delete = numbered[:-keep_last_n] if len(numbered) > keep_last_n else []
    for epoch, uri in to_delete:
        name = uri.rsplit("/", 1)[-1]
        if name in preserve:
            continue  # paranoid double-check
        print(f"[gcs_checkpoint] rolling-delete epoch {epoch}: {uri}")
        _gsutil("rm", uri, check=False)


def resolve_resume_checkpoint(
    *,
    run_id: str,
    override: str | None,
    local_output_dir: str | Path,
    gcs_prefix: str = "checkpoints",
    prefer_order: Iterable[str] = ("checkpoint_best.pth", "checkpoint_last.pth"),
) -> str | None:
    """Pick a checkpoint to resume from and download it locally.

    Args:
        override: explicit filename (e.g. "checkpoint_last.pth" or "checkpoint_5.pth").
                  If empty/None: walk prefer_order and pick first found on GCS.

    Returns local path string (for runner_base.py resume_ckpt_path), or None.
    """
    local_output_dir = Path(local_output_dir)
    local_output_dir.mkdir(parents=True, exist_ok=True)
    base = gcs_uri(gcs_prefix, run_id)

    candidates: list[str]
    if override:
        candidates = [override]
    else:
        candidates = list(prefer_order)

    for fname in candidates:
        remote = f"{base}{fname}"
        check = _gsutil("stat", remote, check=False)
        if check.returncode == 0:
            local = local_output_dir / fname
            print(f"[gcs_checkpoint] resume <- {remote}")
            _gsutil("cp", remote, str(local))
            return str(local)

    print(f"[gcs_checkpoint] no resume candidate at {base} — train from scratch")
    return None


def install_sigterm_flush(
    *,
    run_id: str,
    local_dir: str | Path,
    gcs_prefix: str,
    preserve_files: Iterable[str],
    keep_last_n: int,
    flush_seconds: int = 25,
) -> None:
    """Register a SIGTERM handler that force-flushes current ckpts to GCS.

    GCE spot eviction sends SIGTERM with ~30s grace before SIGKILL. We try to
    upload whatever is in local_dir within `flush_seconds`. Not transactional —
    last-effort save.
    """
    def _handler(signum, frame):  # noqa: ARG001
        deadline = time.time() + flush_seconds
        print(f"[gcs_checkpoint] SIGTERM caught — flushing to GCS (deadline {flush_seconds}s)", flush=True)
        try:
            upload_run_to_gcs(
                run_id=run_id, local_dir=local_dir, gcs_prefix=gcs_prefix,
                preserve_files=preserve_files, keep_last_n=keep_last_n, verify=False,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[gcs_checkpoint] flush failed: {e}", flush=True)
        remaining = deadline - time.time()
        print(f"[gcs_checkpoint] flush done ({remaining:.1f}s spare). Exiting.", flush=True)
        sys.exit(143)  # 128 + SIGTERM

    signal.signal(signal.SIGTERM, _handler)
