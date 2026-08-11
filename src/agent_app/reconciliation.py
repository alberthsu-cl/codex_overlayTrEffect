from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import validate_transition_analysis
from .io import load_json, write_json

_MISSING = object()

_SCALAR_FIELDS = (
    "transition.structure_type",
    "transition.region_count",
    "planner_hints.recommended_effect_family",
    "planner_hints.family_status",
    "planner_hints.new_effect_needed",
    "planner_hints.implementation_status",
    "evaluation_policy.motion_topology.mode",
)

_SET_FIELDS = (
    "transition.motion_axes",
    "planner_hints.visual_primitives",
)


def reconcile_transition_analyses(analysis_files: list[Path]) -> dict[str, Any]:
    if not analysis_files:
        raise ValueError("reconcile-analyses requires at least one --analysis file")

    samples: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    for analysis_file in analysis_files:
        payload = load_json(analysis_file)
        file_issues = validate_transition_analysis(payload)
        if file_issues:
            issues.append(f"{analysis_file}: " + "; ".join(file_issues))
            continue
        sample_id = _unique_sample_id(payload, payloads)
        payloads[sample_id] = payload
        transition = payload.get("transition", {})
        samples.append(
            {
                "sample_id": sample_id,
                "analysis_file": str(analysis_file),
                "input_video": payload.get("input_video"),
                "style_label": transition.get("style_label"),
                "confidence": transition.get("confidence"),
            }
        )
    if issues:
        raise ValueError("invalid transition analysis artifact(s): " + " | ".join(issues))

    total = len(payloads)
    convergent: dict[str, Any] = {}
    divergent: dict[str, Any] = {}

    signal_keys: set[str] = set()
    for payload in payloads.values():
        signal_keys |= set(payload.get("visual_signals", {}).keys())
    scalar_fields = list(_SCALAR_FIELDS) + [f"visual_signals.{key}" for key in sorted(signal_keys)]

    for field in scalar_fields:
        _classify_scalar_field(payloads, field, total, convergent, divergent)

    for field in _SET_FIELDS:
        per_sample_sets = {
            sample_id: set(value)
            for sample_id, payload in payloads.items()
            if isinstance(value := _get_path(payload, field), list)
        }
        if not per_sample_sets:
            continue
        union: set[str] = set()
        for value_set in per_sample_sets.values():
            union |= value_set
        convergent_values = []
        divergent_values = {}
        for item in sorted(union):
            supporting = sorted(
                sample_id for sample_id, value_set in per_sample_sets.items() if item in value_set
            )
            if len(supporting) == total:
                convergent_values.append(item)
            else:
                divergent_values[item] = {"support": len(supporting), "total": total, "samples": supporting}
        if convergent_values:
            convergent[field] = {"values": convergent_values, "total": total}
        if divergent_values:
            divergent[field] = divergent_values

    limitations_by_sample = {
        sample_id: list(payload.get("limitations", [])) for sample_id, payload in payloads.items()
    }

    return {
        "artifact_type": "cross_sample_consensus",
        "artifact_version": 1,
        "sample_count": total,
        "samples": samples,
        "convergent": convergent,
        "divergent": divergent,
        "limitations_by_sample": limitations_by_sample,
    }


def reconcile_and_write(analysis_files: list[Path], output_file: Path) -> dict[str, Any]:
    consensus = reconcile_transition_analyses(analysis_files)
    write_json(output_file, consensus)
    return {"status": "succeeded", "output": str(output_file), "consensus": consensus}


def _classify_scalar_field(
    payloads: dict[str, dict[str, Any]],
    field: str,
    total: int,
    convergent: dict[str, Any],
    divergent: dict[str, Any],
) -> None:
    present = {
        sample_id: value
        for sample_id, payload in payloads.items()
        if (value := _get_path(payload, field)) is not _MISSING
    }
    if not present:
        return
    distinct_values = _distinct(present.values())
    if len(distinct_values) == 1:
        convergent[field] = {
            "value": distinct_values[0],
            "support": len(present),
            "total": total,
        }
    else:
        divergent[field] = {
            "total": total,
            "by_value": _group_by_value(present),
        }


def _unique_sample_id(payload: dict[str, Any], existing: dict[str, Any]) -> str:
    base = Path(str(payload.get("input_video", "sample"))).stem or "sample"
    candidate = base
    suffix = 2
    while candidate in existing:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def _get_path(payload: dict[str, Any], dotted_path: str) -> Any:
    node: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


def _distinct(values: Any) -> list[Any]:
    seen: list[Any] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def _group_by_value(present: dict[str, Any]) -> list[dict[str, Any]]:
    groups: list[tuple[Any, list[str]]] = []
    for sample_id, value in present.items():
        for existing_value, sample_ids in groups:
            if existing_value == value:
                sample_ids.append(sample_id)
                break
        else:
            groups.append((value, [sample_id]))
    return [
        {"value": value, "samples": sorted(sample_ids)}
        for value, sample_ids in groups
    ]
