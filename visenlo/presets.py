"""Settings presets stored as JSON under the user's config directory.

``~/.config/visenlo/presets/<name>.json`` (or ``%APPDATA%/visenlo`` on
Windows). A special ``last`` preset is written automatically after each run.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path


def config_dir() -> Path:
    override = os.environ.get("VISENLO_CONFIG_DIR")
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / "visenlo"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "visenlo"


def presets_dir() -> Path:
    path = config_dir() / "presets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._ -]+", "", str(name)).strip().replace(" ", "_")
    return slug[:48] or "preset"


def list_presets() -> list[str]:
    return sorted(p.stem for p in presets_dir().glob("*.json"))


def save_preset(name: str, settings: dict) -> str:
    path = presets_dir() / f"{_slug(name)}.json"
    path.write_text(json.dumps(settings, indent=2, sort_keys=True))
    return path.stem


def load_preset(name: str) -> dict:
    path = presets_dir() / f"{_slug(name)}.json"
    if not path.is_file():
        raise FileNotFoundError(f"No preset named '{name}'")
    data = json.loads(path.read_text())
    return data if isinstance(data, dict) else {}


def delete_preset(name: str) -> bool:
    path = presets_dir() / f"{_slug(name)}.json"
    if path.is_file():
        path.unlink()
        return True
    return False
