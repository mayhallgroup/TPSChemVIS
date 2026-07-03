"""Persistent user config at ~/.asbuilder/config.json."""
from __future__ import annotations

import json
from pathlib import Path

_CONFIG_DIR = Path.home() / ".asbuilder"
_CONFIG_FILE = _CONFIG_DIR / "config.json"

TPSCHEM_REPO = "https://github.com/arnab82/TPSChem.jl.git"
DEFAULT_INSTALL_DIR = _CONFIG_DIR / "TPSChem.jl"

VIBEMOL_REPO = "https://github.com/evangelistalab/vibemol.git"
VIBEMOL_DIR = _CONFIG_DIR / "vibemol"


def load() -> dict:
    if _CONFIG_FILE.exists():
        try:
            return json.loads(_CONFIG_FILE.read_text())
        except Exception:
            return {}
    return {}


def save(data: dict) -> None:
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _CONFIG_FILE.write_text(json.dumps(data, indent=2))


def get(key: str, default=None):
    return load().get(key, default)


def set_value(key: str, value) -> None:
    data = load()
    data[key] = value
    save(data)


def julia_project() -> Path | None:
    p = get("julia_project")
    if p:
        path = Path(p)
        if (path / "Project.toml").exists():
            return path
    return None


def julia_bin() -> str:
    return get("julia_bin", "julia")


_MAX_RECENT = 8


def recent_projects() -> list[Path]:
    """Return existing recent project paths, most-recent first."""
    return [Path(p) for p in get("recent_projects", []) if Path(p).exists()]


def add_recent_project(path: Path) -> None:
    """Prepend path to the recent-projects list, capped at _MAX_RECENT."""
    current = [p for p in get("recent_projects", []) if p != str(path)]
    set_value("recent_projects", [str(path)] + current[:_MAX_RECENT - 1])
