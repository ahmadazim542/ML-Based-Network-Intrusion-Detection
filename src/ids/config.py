"""Load the YAML experiment config and resolve paths against the project root."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


class Config(dict):
    """dict with attribute access (cfg.split.test_size) that stays YAML/JSON serialisable."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Config({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load configs/default.yaml, then layer the YAML at `path` and keyword overrides on top."""
    with open(DEFAULT_CONFIG) as f:
        data = yaml.safe_load(f)
    if path is not None and resolve(path) != DEFAULT_CONFIG:
        with open(resolve(path)) as f:
            data = _merge(data, yaml.safe_load(f) or {})
    data = _merge(data, overrides)
    cfg = _wrap(data)
    for key in ("raw", "processed", "results"):
        cfg.paths[key] = str(resolve(cfg.paths[key]))
    cfg.dataset.dir = str(resolve(cfg.dataset.dir))
    cfg.cross_dataset.target_dir = str(resolve(cfg.cross_dataset.target_dir))
    return cfg


def results_dir(cfg: Config, *parts: str) -> Path:
    """results/<dataset>__<task>__<split>/<parts...>, created on demand."""
    run = f"{cfg.dataset.name}__{cfg.task}__{cfg.split.strategy}"
    p = Path(cfg.paths.results, run, *parts)
    p.mkdir(parents=True, exist_ok=True)
    return p
