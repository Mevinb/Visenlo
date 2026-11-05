#!/usr/bin/env python3
"""Build a source-only ZIP without models, local runtimes, or personal images.

Run ``python scripts/make_dist.py`` to create ``dist/Visenlo-v1.zip``.
Only the app's source entry points and source directories are packaged. This is
intentionally narrower than a Git archive: untracked inputs and local settings
must never become part of a distributable download.
"""
from __future__ import annotations

import fnmatch
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent.parent
ROOT_FILES = {
    "app.py", "app.css", "launcher.py", "__init__.py", "README.md", "requirements.txt",
    "requirements-base.txt", "requirements-gpu.txt", "requirements-dev.txt",
    "install.sh", "install.bat", "install.ps1", "run.sh", "run.bat",
    "Dockerfile", "docker-compose.yml", ".dockerignore", ".gitignore",
    "LICENSE", "LICENSE.md", "LICENSE.txt",
}
SOURCE_DIRS = {"visenlo", "scripts", "docs", "tests"}
SOURCE_SUFFIXES = {".py", ".md", ".txt", ".sh", ".bat", ".ps1", ".css", ".js", ".html", ".toml", ".yaml", ".yml"}
EXCLUDE_DIRS = {
    ".venv", ".venvs", "venv", "env", ".git", "__pycache__", "node_modules",
    ".ruff_cache", ".pytest_cache", ".mypy_cache", "dist", "build", "outputs",
    "output", "results", "flagged", ".gradio", "models", "testdata", "logs",
    "screenshots", "artifacts", ".aws", ".codex", ".agents",
}


def should_exclude(path: Path, root: Path = ROOT) -> bool:
    """Return true for anything outside the explicit distributable source set."""
    rel = path.relative_to(root)
    if any(fnmatch.fnmatch(part.lower(), pattern) for part in rel.parts
           for pattern in ("secrets*", "credentials*", "*.pem", "*.key", "id_rsa*", "id_ed25519*")):
        return True
    if path.is_symlink() or any(part in EXCLUDE_DIRS for part in rel.parts):
        return True
    if any(part.startswith(".") for part in rel.parts) and str(rel) not in ROOT_FILES:
        return True
    if len(rel.parts) == 1:
        return rel.name not in ROOT_FILES
    return rel.parts[0] not in SOURCE_DIRS or path.suffix.lower() not in SOURCE_SUFFIXES


def iter_distribution_files(root: Path = ROOT):
    """Walk only source folders; never descend into model or runtime trees."""
    root = Path(root)
    for name in sorted(ROOT_FILES):
        path = root / name
        if path.is_file() and not should_exclude(path, root):
            yield path
    for name in sorted(SOURCE_DIRS):
        source = root / name
        if not source.is_dir() or source.is_symlink():
            continue
        for directory, dirs, files in os.walk(source, followlinks=False):
            base = Path(directory)
            dirs[:] = sorted(d for d in dirs if d not in EXCLUDE_DIRS
                             and not d.startswith(".") and not (base / d).is_symlink())
            for filename in sorted(files):
                path = base / filename
                if not should_exclude(path, root):
                    yield path


def create_archive(root: Path, destination: Path) -> Path:
    root, destination = Path(root), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in iter_distribution_files(root):
                archive.write(path, Path("Visenlo") / path.relative_to(root))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main():
    out = create_archive(ROOT, ROOT / "dist" / "Visenlo-v1.zip")
    print(f"[ok] Created {out} ({out.stat().st_size / 1e6:.2f} MB)")
    print("Contains app source, launchers, scripts, documentation and dependency list.")
    print("Excludes all local runtimes, weights, settings, input images and outputs.")
    print("Install dependencies and explicitly set up model weights after extraction.")


if __name__ == "__main__":
    main()
