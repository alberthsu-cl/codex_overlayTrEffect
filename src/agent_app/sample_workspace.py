from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import re
from typing import Any

from .harness_bridge import load_harness_modules
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
    workspace_root: Path | None = None,
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

    catalog_sync: dict[str, Any] | None = None
    if workspace_root is not None:
        modules = load_harness_modules(workspace_root)
        source_manifest_path = workspace_root / "harness" / "configs" / "effect_catalog_sources.json"
        catalog_result = modules["sync_effect_catalog_sources"](
            workspace_root,
            source_manifest_path=source_manifest_path,
        )
        write_json(source_manifest_path, catalog_result["manifest"])
        catalog_path = workspace_root / "harness" / "configs" / "effect_catalog.json"
        catalog = modules["build_effect_catalog"](
            workspace_root,
            source_manifest_path=source_manifest_path,
        )
        write_json(catalog_path, catalog)
        catalog_sync = {
            "source_manifest": str(source_manifest_path),
            "catalog": str(catalog_path),
            "discovered_fx_count": len(catalog_result["discovered_fx_ids"]),
            "added_fx_ids": catalog_result["added_fx_ids"],
            "removed_fx_ids": catalog_result["removed_fx_ids"],
        }

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
    if catalog_sync is not None:
        manifest["catalog_sync"] = catalog_sync
    write_json(manifest_file, manifest)
    return {
        "status": "succeeded",
        "sample_directory": str(sample_dir),
        "manifest_file": str(manifest_file),
        "directories": manifest["directories"],
        "catalog_sync": catalog_sync,
    }
