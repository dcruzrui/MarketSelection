"""Lightweight config helpers for the public demo.

Scripts in this repo can be tuned in three ways (highest priority wins):
1) Environment variables (fast experiments)
2) config.yaml (recommended for repo-wide tuning)
3) Script defaults

This module is intentionally dependency-light (only PyYAML).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def _as_bool(x: Any, default: bool = False) -> bool:
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("1", "true", "yes", "y", "t"):
        return True
    if s in ("0", "false", "no", "n", "f"):
        return False
    return default


def cfg_get(cfg: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def resolve_repo_path(repo_root: Path, p: Any, default: Optional[Path] = None) -> Optional[Path]:
    """Resolve a repo-relative path from config/env into an absolute Path."""
    if p is None:
        return default
    s = str(p).strip()
    if not s:
        return default
    path = Path(s)
    if not path.is_absolute():
        path = repo_root / path
    return path


def load_repo_config(repo_root: Path) -> Dict[str, Any]:
    """Load config.yaml from repo root unless CONFIG_PATH is set."""
    if yaml is None:
        return {}
    cfg_path = os.environ.get("CONFIG_PATH", "").strip()
    if cfg_path:
        p = Path(cfg_path)
        if not p.is_absolute():
            p = repo_root / p
    else:
        p = repo_root / "config.yaml"
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def env_or_cfg(env_key: str, cfg: Dict[str, Any], cfg_key: str, default: Any = None) -> Any:
    """Return env var if set, else cfg value if present, else default."""
    if env_key in os.environ and str(os.environ.get(env_key, "")).strip() != "":
        return os.environ.get(env_key)
    v = cfg_get(cfg, cfg_key, None)
    return default if v is None else v


def env_or_cfg_bool(env_key: str, cfg: Dict[str, Any], cfg_key: str, default: bool = False) -> bool:
    if env_key in os.environ and str(os.environ.get(env_key, "")).strip() != "":
        return _as_bool(os.environ.get(env_key), default)
    return _as_bool(cfg_get(cfg, cfg_key, default), default)


def env_or_cfg_float(env_key: str, cfg: Dict[str, Any], cfg_key: str, default: float) -> float:
    v = env_or_cfg(env_key, cfg, cfg_key, default)
    try:
        return float(v)
    except Exception:
        return float(default)


def env_or_cfg_int(env_key: str, cfg: Dict[str, Any], cfg_key: str, default: int) -> int:
    v = env_or_cfg(env_key, cfg, cfg_key, default)
    try:
        return int(float(v))
    except Exception:
        return int(default)


def env_or_cfg_list(env_key: str, cfg: Dict[str, Any], cfg_key: str, default: Sequence[Any]) -> list:
    if env_key in os.environ and str(os.environ.get(env_key, "")).strip() != "":
        raw = str(os.environ.get(env_key))
        # Accept CSV: "6,8,10" -> [6,8,10]
        items = [x.strip() for x in raw.split(",") if x.strip() != ""]
        out = []
        for it in items:
            try:
                out.append(int(it))
            except Exception:
                try:
                    out.append(float(it))
                except Exception:
                    out.append(it)
        return out
    v = cfg_get(cfg, cfg_key, None)
    if v is None:
        return list(default)
    if isinstance(v, (list, tuple)):
        return list(v)
    return list(default)
