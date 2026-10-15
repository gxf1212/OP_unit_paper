"""Portable resolution of the MolBlender package on ``sys.path``.

MolBlender is not pip-installed in this project; it lives as a sibling source
repository. To keep run scripts free of machine-specific absolute paths, its
``src`` directory is located in the following order:

1. ``$MOLBLENDER_ROOT`` (explicit override, points at the MolBlender repo root).
2. A sibling repository: walk up from the calling script until a
   ``MolBlender/src`` directory is found (``OP_unit`` and ``MolBlender`` are
   expected to share a common parent directory).
3. Assume ``molblender`` is already importable (e.g. an installed package).

Call :func:`ensure_molblender_on_path` once at import time, passing the script's
own directory.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def resolve_molblender_src(script_dir: Path) -> Path | None:
    """Return the MolBlender ``src`` directory, or ``None`` if not found locally."""
    env_root = os.environ.get("MOLBLENDER_ROOT")
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root).expanduser() / "src")
    for parent in [script_dir, *script_dir.parents]:
        candidates.append(parent / "MolBlender" / "src")
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def ensure_molblender_on_path(script_dir: Path) -> None:
    """Insert MolBlender's ``src`` on ``sys.path`` if it is available locally."""
    src = resolve_molblender_src(script_dir)
    if src is not None and str(src) not in sys.path:
        sys.path.insert(0, str(src))
