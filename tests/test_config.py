"""Smoke tests cho config_loader."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config_loader import load_config, require_env  # noqa: E402


def test_load_all_sections():
    cfg = load_config()
    assert "gcp" in cfg and "training" in cfg and "data" in cfg and "checkpoint" in cfg


def test_data_paths_mount_under_gcs():
    cfg = load_config(sections=["data"])
    assert cfg.data.mount_point == "/mnt/gcs-data"
    for key, val in cfg.data.paths.items():
        assert str(val).startswith("/mnt/gcs-data"), f"{key} not under mount: {val}"


def test_training_output_dir_not_kaggle():
    cfg = load_config(sections=["training"])
    out = cfg.training.run.output_dir
    assert "/kaggle/" not in out, f"Leftover Kaggle path: {out}"


def test_checkpoint_preserves_best():
    cfg = load_config(sections=["checkpoint"])
    assert "checkpoint_best.pth" in cfg.checkpoint.preserve_files


def test_require_env_raises_when_missing(monkeypatch):
    monkeypatch.delenv("DEFINITELY_NOT_SET_VAR", raising=False)
    with pytest.raises(RuntimeError):
        require_env("DEFINITELY_NOT_SET_VAR")


def test_gcp_zones_fallback_list():
    cfg = load_config(sections=["gcp"])
    assert len(cfg.gcp.zones) >= 2, "Need fallback zones for spot capacity"
