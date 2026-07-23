from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import re
from typing import Any

from .io import write_json


SAMPLE_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")
SAMPLE_DIRECTORIES = (
    "reference",
    "sources",
    "analysis",
    "design",
    "diagnostics",
    "jobs",
    "effects",
    "candidates",
    "reports",
)


def initialize_sample_workspace(
    samples_root: Path,
    sample_id: str,
    source_video: Path,
) -> dict[str, Any]:
    """Create an isolated work area for one input transition video."""
    if not SAMPLE_ID_PATTERN.fullmatch(sample_id):
        raise ValueError("sample ID must use lowercase letters, digits, underscores, or hyphens")
    if not source_video.is_file():
        raise FileNotFoundError(f"sample video does not exist: {source_video}")

    sample_dir = samples_root / sample_id
    manifest_file = sample_dir / "sample_workspace.json"
    if manifest_file.exists():
        raise ValueError(f"sample workspace already exists: {sample_dir}")

    directories = {name: sample_dir / name for name in SAMPLE_DIRECTORIES}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    manifest = {
        "artifact_type": "transition_sample_workspace",
        "artifact_version": 1,
        "sample_id": sample_id,
        "source_video": str(source_video.resolve()),
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "directories": {name: str(path) for name, path in directories.items()},
    }
    write_json(manifest_file, manifest)
    return {
        "status": "succeeded",
        "sample_directory": str(sample_dir),
        "manifest_file": str(manifest_file),
        "directories": manifest["directories"],
    }
