"""Config loader: merge configs/*.yaml với .env (dotenv) qua OmegaConf.

Usage:
    from src.config_loader import load_config
    cfg = load_config()                          # all sections
    cfg = load_config(sections=["data", "gcp"])  # subset
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from omegaconf import DictConfig, OmegaConf

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv optional khi env đã set qua docker -e
    def load_dotenv(*_a, **_kw):
        return False

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_SECTIONS = ("gcp", "training", "data", "checkpoint")


def load_config(
    sections: Iterable[str] = DEFAULT_SECTIONS,
    dotenv_path: str | Path | None = None,
) -> DictConfig:
    load_dotenv(dotenv_path or REPO_ROOT / ".env", override=False)
    merged = OmegaConf.create({})
    for name in sections:
        path = CONFIG_DIR / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Missing config: {path}")
        merged[name] = OmegaConf.load(path)
    # Resolve interpolations (${oc.env:...}) eagerly so callers get plain values.
    OmegaConf.resolve(merged)
    return merged


def require_env(*keys: str) -> dict[str, str]:
    """Return env vars, raising if any are missing/empty. Use ở entrypoints."""
    missing = [k for k in keys if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing required env vars: {missing}. Check .env.")
    return {k: os.environ[k] for k in keys}


if __name__ == "__main__":
    cfg = load_config()
    print(OmegaConf.to_yaml(cfg))
