from __future__ import annotations

from typing import Any


def validate_transition_analysis(payload: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    _require_value(payload, "artifact_type", "transition_structure", issues)
    _require_value(payload, "artifact_version", 1, issues)
    for field in (
        "input_video",
        "video_metadata",
        "transition",
        "visual_signals",
        "frame_progress_mapping",
        "evidence",
        "limitations",
        "planner_hints",
    ):
        _require_field(payload, field, issues)

    video_metadata = payload.get("video_metadata")
    if isinstance(video_metadata, dict):
        _require_field(video_metadata, "frame_count", issues, prefix="video_metadata")
        if not isinstance(video_metadata.get("frame_count"), int) or video_metadata.get("frame_count", 0) < 1:
            issues.append("video_metadata.frame_count must be a positive integer")

    transition = payload.get("transition")
    if isinstance(transition, dict):
        for field in ("style_label", "summary", "start_frame", "end_frame", "confidence"):
            _require_field(transition, field, issues, prefix="transition")
        _validate_confidence(transition.get("confidence"), "transition.confidence", issues)

    planner_hints = payload.get("planner_hints")
    if isinstance(planner_hints, dict):
        for field in (
            "recommended_effect_family",
            "family_status",
            "visual_primitives",
            "new_effect_needed",
            "implementation_status",
        ):
            _require_field(planner_hints, field, issues, prefix="planner_hints")
        if planner_hints.get("family_status") not in {"known", "unknown"}:
            issues.append("planner_hints.family_status must be 'known' or 'unknown'")
        if planner_hints.get("implementation_status") not in {
            "supported", "unsupported", "review_required"
        }:
            issues.append("planner_hints.implementation_status is invalid")
        if not isinstance(planner_hints.get("visual_primitives"), list):
            issues.append("planner_hints.visual_primitives must be an array")

    return issues


def validate_effect_design(payload: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    _require_value(payload, "artifact_type", "effect_design", issues)
    _require_value(payload, "artifact_version", 1, issues)
    for field in ("analysis_artifact", "decision", "target_effect", "design_notes"):
        _require_field(payload, field, issues)

    decision = payload.get("decision")
    if isinstance(decision, dict):
        _require_field(decision, "action", issues, prefix="decision")
        _require_field(decision, "confidence", issues, prefix="decision")
        if decision.get("action") not in {
            "reuse_existing_effect",
            "tune_existing_effect",
            "implement_new_effect",
        }:
            issues.append("decision.action is not a supported effect-design action")
        _validate_confidence(decision.get("confidence"), "decision.confidence", issues)

    target_effect = payload.get("target_effect")
    if isinstance(target_effect, dict):
        _require_field(target_effect, "family", issues, prefix="target_effect")

    seed = payload.get("implementation_seed")
    if seed is not None:
        if not isinstance(seed, dict):
            issues.append("implementation_seed must be an object")
        else:
            for field in ("family", "required_shader_capabilities"):
                _require_field(seed, field, issues, prefix="implementation_seed")
            if not isinstance(seed.get("required_shader_capabilities"), list):
                issues.append("implementation_seed.required_shader_capabilities must be an array")
            if isinstance(target_effect, dict) and isinstance(seed.get("family"), str):
                if target_effect.get("family") != seed["family"]:
                    issues.append("target_effect.family must match implementation_seed.family")
            variant = payload.get("source_variant")
            if isinstance(variant, dict) and isinstance(seed.get("template_effect_id"), str):
                if isinstance(target_effect, dict) and target_effect.get("closest_existing_effect_id") != seed["template_effect_id"]:
                    issues.append("implementation_seed.template_effect_id must match target_effect.closest_existing_effect_id")

    design_notes = payload.get("design_notes")
    if isinstance(design_notes, dict):
        for field in ("must_preserve", "approximations", "risks"):
            _require_field(design_notes, field, issues, prefix="design_notes")

    return issues


def build_render_job(
    analysis: dict[str, Any],
    design: dict[str, Any],
    source_a: str,
    source_b: str,
    reference_transition: str | None,
    width: int,
    height: int,
    fps: int,
    frame_count: int,
    progress_schedule: list[float] | None = None,
) -> dict[str, Any]:
    analysis_issues = validate_transition_analysis(analysis)
    design_issues = validate_effect_design(design)
    if analysis_issues or design_issues:
        details = analysis_issues + design_issues
        raise ValueError("invalid Codex artifacts: " + "; ".join(details))

    decision = design["decision"]
    action = decision["action"]
    planner_hints = analysis["planner_hints"]
    implementation_status = planner_hints.get("implementation_status")
    if action == "implement_new_effect" and implementation_status != "supported":
        raise ValueError(
            "new-effect execution requires planner_hints.implementation_status=supported; "
            f"received {implementation_status!r}"
        )
    target_effect = design["target_effect"]
    effect_id = target_effect.get("effect_id") or target_effect.get("closest_existing_effect_id")
    if not effect_id:
        raise ValueError(
            "effect design must provide target_effect.effect_id or "
            "target_effect.closest_existing_effect_id for rendering"
        )

    family = target_effect.get("family") or "single_pass"
    category = target_effect.get("expected_runtime_shape") or family
    if category == "single_pass_fullscreen":
        category = "single_pass"
    job_name = _job_name(family, action)
    return {
        "job_name": job_name,
        "effect": {
            "fx_id": effect_id,
            "category": category,
            "effect_spec": None,
            "uniforms": {"progress": 0.0},
        },
        "inputs": {
            "source_a": source_a,
            "source_b": source_b,
            "reference_transition": reference_transition,
        },
        "render": {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "output_format": "png_sequence",
            **({"progress_schedule": progress_schedule} if progress_schedule is not None else {}),
        },
        "planning": {
            "source": "agent_effect_design",
            "decision": action,
            "decision_confidence": decision.get("confidence"),
            "analysis_artifact": design.get("analysis_artifact"),
            "analysis_style_label": analysis["transition"].get("style_label"),
            "analysis_summary": analysis["transition"].get("summary"),
            "analysis_family_status": planner_hints.get("family_status"),
            "analysis_visual_primitives": planner_hints.get("visual_primitives", []),
            "analysis_implementation_status": implementation_status,
            "design_reason": decision.get("reason"),
            "must_preserve": design["design_notes"].get("must_preserve", []),
            "approximations": design["design_notes"].get("approximations", []),
            "risks": design["design_notes"].get("risks", []),
        },
    }


def _require_field(payload: dict[str, Any], field: str, issues: list[str], prefix: str = "") -> None:
    if field not in payload:
        issues.append(f"{prefix + '.' if prefix else ''}{field} is required")


def _require_value(payload: dict[str, Any], field: str, expected: Any, issues: list[str]) -> None:
    if payload.get(field) != expected:
        issues.append(f"{field} must be {expected!r}")


def _validate_confidence(value: Any, field: str, issues: list[str]) -> None:
    if not isinstance(value, (int, float)) or not 0 <= value <= 1:
        issues.append(f"{field} must be a number between 0 and 1")


def _job_name(family: str, action: str) -> str:
    safe_family = "".join(character if character.isalnum() else "_" for character in family).strip("_")
    safe_action = "".join(character if character.isalnum() else "_" for character in action).strip("_")
    return f"agent_{safe_action}_{safe_family or 'effect'}"
