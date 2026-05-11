"""Unit tests cho gcs_checkpoint, mock gsutil bằng monkeypatch trên subprocess.run.

Không gọi GCS thật.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import gcs_checkpoint as gc  # noqa: E402


class FakeGsutil:
    """In-memory gsutil mock. Tracks uploads/deletes/stat for assertions."""

    def __init__(self):
        self.store: dict[str, int] = {}  # uri -> size
        self.deleted: list[str] = []

    def __call__(self, cmd, *, check=True, capture_output=True, text=True):
        assert cmd[0] == "gsutil"
        sub = cmd[1]
        rc, out, err = 0, "", ""
        if sub == "cp":
            src, dst = cmd[2], cmd[3]
            if src.startswith("gs://"):  # download
                if src not in self.store:
                    rc = 1
                else:
                    Path(dst).write_bytes(b"x" * self.store[src])
            else:  # upload
                size = Path(src).stat().st_size
                self.store[dst] = size
        elif sub == "stat":
            uri = cmd[2]
            if uri in self.store:
                out = f"Content-Length:        {self.store[uri]}\n"
            else:
                rc = 1
        elif sub == "ls":
            prefix = cmd[2]
            matches = [u for u in self.store if u.startswith(prefix)]
            if not matches:
                rc = 1
            out = "\n".join(matches)
        elif sub == "rm":
            uri = cmd[2]
            self.store.pop(uri, None)
            self.deleted.append(uri)
        else:
            raise AssertionError(f"unexpected gsutil sub: {sub}")
        return subprocess.CompletedProcess(cmd, rc, stdout=out, stderr=err)


@pytest.fixture
def fake_gsutil(monkeypatch):
    os.environ["GCS_BUCKET"] = "test-bucket"
    fake = FakeGsutil()
    monkeypatch.setattr(subprocess, "run", fake)
    return fake


def _mkckpt(d: Path, name: str, size: int = 1024) -> Path:
    p = d / name
    p.write_bytes(b"x" * size)
    return p


def test_upload_preserves_best_and_last(fake_gsutil, tmp_path):
    for n in ("checkpoint_best.pth", "checkpoint_last.pth",
              "checkpoint_1.pth", "checkpoint_2.pth"):
        _mkckpt(tmp_path, n)
    gc.upload_run_to_gcs(
        run_id="r1", local_dir=tmp_path, gcs_prefix="checkpoints",
        preserve_files=["checkpoint_best.pth", "checkpoint_last.pth"],
        keep_last_n=5,
    )
    keys = set(fake_gsutil.store)
    assert any(k.endswith("checkpoint_best.pth") for k in keys)
    assert any(k.endswith("checkpoint_last.pth") for k in keys)
    assert fake_gsutil.deleted == [], "Nothing should be deleted yet (within keep_last_n)"


def test_rolling_delete_keeps_newest_n(fake_gsutil, tmp_path):
    # 7 epoch ckpts, keep_last_n=5 → epoch 1,2 should be deleted.
    for i in range(1, 8):
        _mkckpt(tmp_path, f"checkpoint_{i}.pth")
    _mkckpt(tmp_path, "checkpoint_best.pth")
    gc.upload_run_to_gcs(
        run_id="r1", local_dir=tmp_path, gcs_prefix="checkpoints",
        preserve_files=["checkpoint_best.pth", "checkpoint_last.pth"],
        keep_last_n=5,
    )
    deleted_names = [d.rsplit("/", 1)[-1] for d in fake_gsutil.deleted]
    assert deleted_names == ["checkpoint_1.pth", "checkpoint_2.pth"]
    # best.pth still present.
    assert any(k.endswith("checkpoint_best.pth") for k in fake_gsutil.store)


def test_best_never_deleted_even_if_named_in_pattern(fake_gsutil, tmp_path):
    _mkckpt(tmp_path, "checkpoint_best.pth")
    gc.upload_run_to_gcs(
        run_id="r1", local_dir=tmp_path, gcs_prefix="checkpoints",
        preserve_files=["checkpoint_best.pth", "checkpoint_last.pth"],
        keep_last_n=0,
    )
    assert fake_gsutil.deleted == []
    assert any(k.endswith("checkpoint_best.pth") for k in fake_gsutil.store)


def test_resume_prefers_best_over_last(fake_gsutil, tmp_path):
    base = "gs://test-bucket/checkpoints/r1"
    fake_gsutil.store[f"{base}/checkpoint_best.pth"] = 2048
    fake_gsutil.store[f"{base}/checkpoint_last.pth"] = 1024
    path = gc.resolve_resume_checkpoint(
        run_id="r1", override=None, local_output_dir=tmp_path,
    )
    assert path and path.endswith("checkpoint_best.pth")
    assert Path(path).stat().st_size == 2048


def test_resume_override_specific(fake_gsutil, tmp_path):
    base = "gs://test-bucket/checkpoints/r1"
    fake_gsutil.store[f"{base}/checkpoint_best.pth"] = 2048
    fake_gsutil.store[f"{base}/checkpoint_3.pth"] = 512
    path = gc.resolve_resume_checkpoint(
        run_id="r1", override="checkpoint_3.pth", local_output_dir=tmp_path,
    )
    assert path and path.endswith("checkpoint_3.pth")


def test_resume_none_when_empty(fake_gsutil, tmp_path):
    path = gc.resolve_resume_checkpoint(run_id="empty", override=None, local_output_dir=tmp_path)
    assert path is None


def test_upload_verify_size_mismatch_raises(fake_gsutil, tmp_path, monkeypatch):
    p = _mkckpt(tmp_path, "checkpoint_best.pth", size=4096)

    # Make stat report wrong size.
    original = fake_gsutil.__call__

    def buggy(cmd, **kw):
        res = original(cmd, **kw)
        if cmd[1] == "stat":
            res = subprocess.CompletedProcess(
                cmd, 0, stdout="Content-Length: 999\n", stderr=""
            )
        return res

    monkeypatch.setattr(subprocess, "run", buggy)
    with pytest.raises(RuntimeError, match="size mismatch"):
        gc.upload_file(p, "gs://test-bucket/checkpoints/r1/checkpoint_best.pth", verify=True)
