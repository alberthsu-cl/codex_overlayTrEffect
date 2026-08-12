from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import re
import shutil
from typing import Any

from .io import load_json, write_json


STATE_FILE_NAME = "candidate_state.json"
ITERATION_PATTERN = re.compile(r"^iteration_(\d+)_.*\.json$")
HYPOTHESIS_CATEGORIES = (
    "timing",
    "regions",
    "displacement",
    "blur",
    "blend",
    "shader_structure",
    "uv_mapping",
    "other",
)
SSIM_IMPROVEMENT = 0.001
SSIM_REGRESSION_TOLERANCE = 0.003
MSE_IMPROVEMENT_RATIO = 0.01
MSE_REGRESSION_TOLERANCE = 0.03
MOTION_SIMILARITY_IMPROVEMENT = 0.02
MOTION_SIMILARITY_REGRESSION_TOLERANCE = 0.03
ENDPOINT_MSE_TOLERANCE = 1.0
ENDPOINT_SSIM_TOLERANCE = 0.999
SELECTION_METRICS = (
    "mse",
    "mae",
    "ssim",
    "motion_similarity",
    "peak_mse",
    "peak_ssim",
    "foreground_body_rotation_error",
    "foreground_body_scale_error",
    "foreground_body_translation_error",
    "foreground_body_pivot_error",
    "foreground_body_rotation_direction_agreement",
    "geometry_similarity",
    "salient_centroid_distance_error",
    "salient_centroid_max_distance_error",
    "salient_coverage_error",
    "salient_rotation_error",
    "salient_rotation_max_error",
    "motion_blur_direction_agreement",
    "slice_structure_similarity",
    "rgb_separation_similarity",
    "rgb_slice_coherence",
    "dense_rgb_slice_similarity",
)
LOWER_IS_BETTER_METRICS = {
    "mse",
    "mae",
    "peak_mse",
    "foreground_body_rotation_error",
    "foreground_body_scale_error",
    "foreground_body_translation_error",
    "foreground_body_pivot_error",
    "salient_centroid_distance_error",
    "salient_centroid_max_distance_error",
    "salient_coverage_error",
    "salient_rotation_error",
    "salient_rotation_max_error",
}
TRANSLATION_DELTA_LIMIT_PIXELS = 2.0
PIVOT_DELTA_LIMIT_PIXELS = 8.0
ROTATION_DELTA_LIMIT_DEGREES = 10.0
SCALE_DELTA_LIMIT_RATIO = 0.15
# A similarity transform's pivot is the fixed point of (I - scale*R), so the
# solve degenerates as the transform approaches identity: a body that barely
# rotates reports a pivot hundreds of pixels away from anything meaningful.
# Require a measurable departure from identity before pivot error is allowed to
# drive prioritisation.
PIVOT_CONDITIONING_MIN_ROTATION_DEGREES = 1.0
PIVOT_CONDITIONING_MIN_SCALE_DELTA = 0.01
# A signed-direction disagreement has no magnitude to scale, but it is a
# qualitative error, so it outranks a merely-over-threshold magnitude.
DIRECTION_DISAGREEMENT_SEVERITY = 3.0
# Severity is preferably measured against the reference's own magnitude: an error
# is serious when it is a large fraction of the motion actually present, which is
# comparable across pixels, degrees and ratios in a way raw thresholds are not.
RELATIVE_ERROR_LIMIT = 0.25
RELATIVE_ROTATION_FLOOR_DEGREES = 1.0
RELATIVE_SCALE_FLOOR_RATIO = 0.01
RELATIVE_TRANSLATION_FLOOR_PIXELS = 1.0
# Consecutive tradeoffs mean every hypothesis is moving some metric while
# regressing another - the signature of tuning inside measurement noise.  After
# this many in a row, the next request demands a structural re-derivation
# instead of another constant nudge.
CONSECUTIVE_TRADEOFF_ESCALATION_LIMIT = 3
METRIC_IMPROVEMENT_THRESHOLDS = {
    "motion_similarity": MOTION_SIMILARITY_IMPROVEMENT,
    "foreground_body_rotation_direction_agreement": 0.05,
    "geometry_similarity": 0.02,
    "slice_structure_similarity": 0.02,
    "rgb_separation_similarity": 0.02,
    "rgb_slice_coherence": 0.02,
    "dense_rgb_slice_similarity": 0.02,
}
METRIC_REGRESSION_TOLERANCES = {
    "motion_similarity": MOTION_SIMILARITY_REGRESSION_TOLERANCE,
    "foreground_body_rotation_direction_agreement": 0.05,
    "geometry_similarity": 0.03,
    "slice_structure_similarity": 0.04,
    "rgb_separation_similarity": 0.04,
    "rgb_slice_coherence": 0.04,
    "dense_rgb_slice_similarity": 0.03,
}
# foreground_body_translation_error/pivot_error/scale_error can all have a
# baseline value that is already tiny (ORB similarity-transform fits on a
# near-static or barely-scaled body routinely land under 0.2px / 0.005 ratio).
# A purely relative tolerance (MSE_REGRESSION_TOLERANCE) breaks down there: a
# 0.0003 scale-ratio wobble - well within the detector's own measurement noise
# - reads as a 10%+ "regression" and trips the guardrail on noise alone,
# blocking otherwise-clean iterations regardless of hypothesis (confirmed
# twice on classic_spin_01/SpinBlur_01: first on translation_error, then again
# on scale_error once that was fixed). Require the absolute delta to also
# clear a floor before a relative regression counts. RELATIVE_TRANSLATION_FLOOR_PIXELS
# and RELATIVE_SCALE_FLOOR_RATIO are reused here since they already define
# this codebase's noise floors for a closely related purpose (severity
# relative-scaling below).
METRIC_REGRESSION_ABSOLUTE_FLOORS = {
    "foreground_body_translation_error": RELATIVE_TRANSLATION_FLOOR_PIXELS,
    "foreground_body_pivot_error": RELATIVE_TRANSLATION_FLOOR_PIXELS,
    "foreground_body_scale_error": RELATIVE_SCALE_FLOOR_RATIO,
}


def set_candidate_baseline(
    candidate_manifest_file: Path,
    iteration: int,
    report_file: Path,
    source_dir: Path | None = None,
) -> dict[str, Any]:
    state = _load_or_create_state(candidate_manifest_file)
    metrics = _metrics_from_report(load_json(report_file))
    if not _endpoints_are_exact(metrics):
        raise ValueError("cannot set a baseline when endpoint checks exceed stable-frame tolerance")
    snapshot_dir = _snapshot_baseline_sources(
        candidate_manifest_file,
        iteration,
        source_dir or candidate_manifest_file.parent,
    )
    state["baseline"] = {
        "iteration": iteration,
        "report_file": str(report_file),
        "metrics": metrics,
        "source_snapshot": str(snapshot_dir),
        "selected_at": _timestamp(),
        "selection_policy": _resolve_selection_policy(candidate_manifest_file),
    }
    _upsert_history(
        state,
        iteration,
        "baseline",
        "accepted",
        metrics,
        str(report_file),
        str(snapshot_dir),
    )
    _upsert_shortlist(state, iteration, "accepted", metrics, str(report_file))
    _refresh_rejected_budget(state)
    _write_state(candidate_manifest_file, state)
    return {"status": "succeeded", "state": state}


def start_refinement_phase(
    candidate_manifest_file: Path,
    name: str,
    baseline_iteration: int,
    report_file: Path,
    max_iterations: int,
    max_rejected: int,
    source_dir: Path | None = None,
) -> dict[str, Any]:
    if not re.fullmatch(r"[a-z0-9_]+", name):
        raise ValueError("phase name must use lowercase letters, digits, and underscores")
    if max_iterations < 1 or max_rejected < 1:
        raise ValueError("phase budgets must be positive")
    state = _load_or_create_state(candidate_manifest_file)
    if any(item.get("name") == name for item in state.get("phases", [])):
        raise ValueError(
            f"phase name already exists: {name!r}; use the existing request packet "
            "for the active phase or choose a new phase name"
        )
    baseline_result = set_candidate_baseline(
        candidate_manifest_file=candidate_manifest_file,
        iteration=baseline_iteration,
        report_file=report_file,
        source_dir=source_dir,
    )
    state = baseline_result["state"]
    known_iterations = [number for number, _ in _iteration_records(candidate_manifest_file.parent)]
    known_iterations.extend(
        int(item["iteration"])
        for item in state["history"]
        if isinstance(item.get("iteration"), int)
    )
    first_iteration = max(known_iterations, default=0) + 1
    previous_phase = _active_phase(state)
    closed_at = _timestamp()
    if previous_phase is not None:
        previous_phase["status"] = "closed"
        previous_phase["closed_at"] = closed_at
        previous_phase["closed_reason"] = "superseded_by_new_phase"
    phase = {
        "name": name,
        "baseline_iteration": baseline_iteration,
        "first_iteration": first_iteration,
        "max_iterations": max_iterations,
        "max_rejected": max_rejected,
        "started_at": closed_at,
        "status": "active",
    }
    state.setdefault("phases", [])
    state["phases"].append(phase)
    state["active_phase"] = name
    state["budgets"] = {
        "phase": name,
        "max_iterations": max_iterations,
        "max_rejected": max_rejected,
        "attempted_so_far": 0,
        "rejected_so_far": 0,
    }
    _write_state(candidate_manifest_file, state)
    return {"status": "succeeded", "phase": phase, "state_file": str(_state_file(candidate_manifest_file))}


def resume_candidate_refinement(
    candidate_manifest_file: Path,
    analysis_file: Path,
    design_file: Path,
    phase_name: str,
    max_iterations: int,
    max_rejected: int,
) -> dict[str, Any]:
    """Restart refinement from the selected baseline and create its first packet."""
    state = _load_or_create_state(candidate_manifest_file)
    profile = state.get("evaluation_profile")
    if not isinstance(profile, dict):
        raise ValueError("candidate has no evaluation profile; run candidate-set-evaluation-profile first")
    baseline = state.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("candidate has no selected baseline")
    report_file = Path(str(baseline.get("report_file", "")))
    if not report_file.is_file():
        raise FileNotFoundError(f"selected baseline report was not found: {report_file}")

    phase = start_refinement_phase(
        candidate_manifest_file=candidate_manifest_file,
        name=phase_name,
        baseline_iteration=int(baseline["iteration"]),
        report_file=report_file,
        max_iterations=max_iterations,
        max_rejected=max_rejected,
    )
    restoration = restore_candidate_baseline(candidate_manifest_file)
    packet = build_next_iteration_packet(
        candidate_manifest_file=candidate_manifest_file,
        analysis_file=analysis_file,
        design_file=design_file,
        max_iterations=max_iterations,
        max_rejected=max_rejected,
        evaluate_after_edit=True,
    )
    return {
        "status": "succeeded",
        "phase": phase["phase"],
        "restoration": restoration,
        "packet": packet,
    }


def restore_candidate_baseline(candidate_manifest_file: Path) -> dict[str, Any]:
    state = _load_or_create_state(candidate_manifest_file)
    baseline = state.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("candidate has no selected baseline")
    snapshot_dir = Path(str(baseline.get("source_snapshot", "")))
    if not snapshot_dir.is_dir():
        raise FileNotFoundError("selected baseline has no source snapshot")
    restored = _restore_sources_from_snapshot(candidate_manifest_file, snapshot_dir)
    return {
        "status": "succeeded",
        "effect_id": state["effect_id"],
        "baseline_iteration": baseline["iteration"],
        "source_snapshot": str(snapshot_dir),
        "restored_files": restored,
    }


def human_accept_candidate(
    candidate_manifest_file: Path,
    iteration: int,
    reviewer: str,
    reason: str,
) -> dict[str, Any]:
    """Record a human visual acceptance for the currently selected baseline."""
    reviewer = reviewer.strip()
    reason = reason.strip()
    if not reviewer:
        raise ValueError("reviewer must not be empty")
    if not reason:
        raise ValueError("reason must not be empty")

    state = _load_or_create_state(candidate_manifest_file)
    baseline = state.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("candidate has no selected baseline to accept")
    if int(baseline.get("iteration", -1)) != iteration:
        raise ValueError("human acceptance must reference the selected baseline iteration")

    accepted_at = _timestamp()
    acceptance = {
        "iteration": iteration,
        "status": "human_accepted",
        "reviewer": reviewer,
        "reason": reason,
        "accepted_at": accepted_at,
        "report_file": baseline.get("report_file", ""),
        "source_snapshot": baseline.get("source_snapshot", ""),
        "automated_metrics": "diagnostic_only",
    }
    state["human_acceptance"] = acceptance
    for entry in state["history"]:
        if int(entry.get("iteration", -1)) == iteration:
            entry["status"] = "human_accepted"
            entry["human_review"] = acceptance
    _upsert_shortlist(
        state,
        iteration,
        "human_accepted",
        baseline.get("metrics", {}),
        str(baseline.get("report_file", "")),
    )
    phase = _active_phase(state) or _phase_by_name(
        state,
        str(state.get("budgets", {}).get("phase", "")),
    )
    if phase is not None:
        phase["status"] = "closed"
        phase["closed_at"] = accepted_at
        phase["closed_reason"] = "human_accepted"
        first_iteration = int(phase["first_iteration"])
        phase_history = [
            item for item in state["history"] if int(item.get("iteration", 0)) >= first_iteration
        ]
        state["budgets"] = {
            "phase": phase["name"],
            "max_iterations": int(phase["max_iterations"]),
            "max_rejected": int(phase["max_rejected"]),
            "attempted_so_far": len(phase_history),
            "rejected_so_far": sum(1 for item in phase_history if item.get("status") == "rejected"),
            "status": "closed",
        }
    state["active_phase"] = None
    _write_state(candidate_manifest_file, state)
    return {
        "status": "succeeded",
        "acceptance": acceptance,
        "state_file": str(_state_file(candidate_manifest_file)),
    }


def build_next_iteration_packet(
    candidate_manifest_file: Path,
    analysis_file: Path,
    design_file: Path,
    max_iterations: int,
    max_rejected: int,
    evaluate_after_edit: bool = False,
) -> dict[str, Any]:
    state = _load_or_create_state(candidate_manifest_file)
    selection_policy = _resolve_selection_policy(candidate_manifest_file)
    if "dense_rgb_slice_similarity" in selection_policy.get("primary_metrics", []):
        _hydrate_dense_rgb_slice_state(state)
    candidate_dir = candidate_manifest_file.parent
    iteration_records = _iteration_records(candidate_dir)
    known_iterations = [item[0] for item in iteration_records]
    known_iterations.extend(
        int(item["iteration"])
        for item in state["history"]
        if isinstance(item.get("iteration"), int)
    )
    if state["baseline"] is not None:
        known_iterations.append(int(state["baseline"]["iteration"]))
    last_iteration = max(known_iterations, default=0)
    next_iteration = last_iteration + 1
    phase = _active_phase(state)
    if phase is not None:
        first_iteration = int(phase["first_iteration"])
        attempted_count = sum(
            1 for item in state["history"] if int(item.get("iteration", 0)) >= first_iteration
        )
        rejected_count = sum(
            1
            for item in state["history"]
            if int(item.get("iteration", 0)) >= first_iteration and item.get("status") == "rejected"
        )
        phase_max_iterations = int(phase["max_iterations"])
        phase_max_rejected = int(phase["max_rejected"])
        if attempted_count >= phase_max_iterations:
            _close_phase_for_budget(
                candidate_manifest_file,
                state,
                phase,
                attempted_count,
                rejected_count,
                "iteration_budget_exhausted",
            )
            raise ValueError(f"phase iteration budget exhausted: {attempted_count} reaches {phase_max_iterations}")
        if rejected_count >= phase_max_rejected:
            _close_phase_for_budget(
                candidate_manifest_file,
                state,
                phase,
                attempted_count,
                rejected_count,
                "rejected_iteration_budget_exhausted",
            )
            raise ValueError(f"phase rejected-iteration budget exhausted: {rejected_count} reaches {phase_max_rejected}")
        blocked_categories = _blocked_categories(state, first_iteration)
        tradeoff_streak = _consecutive_tradeoffs(state, first_iteration)
        budget = {
            "phase": phase["name"],
            "max_iterations": phase_max_iterations,
            "max_rejected": phase_max_rejected,
            "attempted_so_far": attempted_count,
            "rejected_so_far": rejected_count,
            "consecutive_tradeoffs": tradeoff_streak,
        }
    else:
        if next_iteration > max_iterations:
            raise ValueError(
                f"iteration budget exhausted: next iteration {next_iteration} exceeds {max_iterations}"
            )
        rejected_count = sum(1 for item in state["history"] if item.get("status") == "rejected")
        if rejected_count >= max_rejected:
            raise ValueError(
                f"rejected-iteration budget exhausted: {rejected_count} reaches {max_rejected}"
            )
        blocked_categories = _blocked_categories(state)
        tradeoff_streak = _consecutive_tradeoffs(state)
        budget = {
            "max_iterations": max_iterations,
            "max_rejected": max_rejected,
            "rejected_so_far": rejected_count,
            "consecutive_tradeoffs": tradeoff_streak,
        }
    latest_report = _latest_report(candidate_dir / "evaluations")
    latest_video = _latest_video(candidate_dir / "evaluations")
    latest_comparison = _latest_comparison_video(candidate_dir / "evaluations")
    latest_motion = _latest_motion_video(candidate_dir / "evaluations")
    reference_diagnostics, reference_diagnostic_video = _reference_diagnostics(analysis_file)
    reference_edge_diagnostics = _reference_edge_diagnostics(analysis_file)
    reference_grid_density_diagnostics = _reference_grid_density_diagnostics(analysis_file)
    refinement_priority = _motion_refinement_priority(state)
    refinement_findings = _ranked_findings(state)
    packet_file = candidate_dir / "packets" / f"iteration_{next_iteration:03d}_packet.json"
    prompt_file = candidate_dir / "packets" / f"iteration_{next_iteration:03d}_agent_request.md"
    candidate = load_json(candidate_manifest_file)
    evaluation_command = None
    continuation_command = None
    if evaluate_after_edit:
        profile = state.get("evaluation_profile")
        if not isinstance(profile, dict):
            raise ValueError("candidate has no evaluation profile; run candidate-set-evaluation-profile first")
        evaluation_command = _evaluation_command(
            profile,
            next_iteration,
            analysis_file,
            design_file,
            max_iterations,
            max_rejected,
        )
        continuation_command = _continuation_command(
            candidate_manifest_file,
            analysis_file,
            design_file,
            max_iterations,
            max_rejected,
        )
    state["budgets"] = budget
    packet = {
        "artifact_type": "candidate_iteration_packet",
        "artifact_version": 1,
        "effect_id": candidate["effect_id"],
        "iteration": next_iteration,
        "candidate_manifest": str(candidate_manifest_file),
        "analysis_file": str(analysis_file),
        "design_file": str(design_file),
        "candidate_files": candidate["candidate_files"],
        "latest_report": str(latest_report) if latest_report else None,
        "latest_candidate_video": str(latest_video) if latest_video else None,
        "latest_comparison_video": str(latest_comparison) if latest_comparison else None,
        "latest_motion_video": str(latest_motion) if latest_motion else None,
        "reference_diagnostics_file": str(reference_diagnostics) if reference_diagnostics else None,
        "reference_diagnostics_video": str(reference_diagnostic_video) if reference_diagnostic_video else None,
        "reference_edge_diagnostics_file": str(reference_edge_diagnostics) if reference_edge_diagnostics else None,
        "reference_grid_density_diagnostics_file": (
            str(reference_grid_density_diagnostics) if reference_grid_density_diagnostics else None
        ),
        "refinement_priority": refinement_priority,
        # Every over-threshold signal, not only the winner: a lower-ranked entry
        # can still be the real defect when thresholds across different units are
        # not equally calibrated.
        "refinement_findings": refinement_findings,
        # Measured per-frame reference targets, so an iteration can fit a curve to
        # the reference instead of nudging a coefficient and re-measuring.
        "reference_curves": _reference_curves(state),
        "prompt_files": _select_prompt_files(
            analysis_file,
            include_edge_diagnostics=bool(reference_edge_diagnostics),
        ),
        "baseline": state["baseline"],
        "history": state["history"],
        "shortlist": state["shortlist"],
        "active_phase": phase,
        "allowed_hypothesis_categories": [
            category for category in HYPOTHESIS_CATEGORIES if category not in blocked_categories
        ],
        "blocked_hypothesis_categories": blocked_categories,
        "budgets": state["budgets"],
        "escalation": {
            "consecutive_tradeoffs": tradeoff_streak,
            "limit": CONSECUTIVE_TRADEOFF_ESCALATION_LIMIT,
            "structural_review_required": tradeoff_streak >= CONSECUTIVE_TRADEOFF_ESCALATION_LIMIT,
        },
        "evaluation_after_edit": evaluate_after_edit,
        "evaluation_command": evaluation_command,
        "continuation_command": continuation_command,
    }
    packet["packet_file"] = str(packet_file)
    write_json(packet_file, packet)
    prompt_file.write_text(_refinement_request(packet, candidate_dir), encoding="utf-8")
    state["last_packet"] = str(packet_file)
    _write_state(candidate_manifest_file, state)
    return {
        "status": "succeeded",
        "controller_status": "ready",
        "iteration": next_iteration,
        "packet_file": str(packet_file),
        "prompt_file": str(prompt_file),
        "blocked_hypothesis_categories": blocked_categories,
        "evaluation_after_edit": evaluate_after_edit,
    }


def set_evaluation_profile(
    candidate_manifest_file: Path,
    profile: dict[str, Any],
) -> dict[str, Any]:
    required_strings = (
        "manifest",
        "job",
        "reference",
        "output_root",
        "backup_root",
        "msbuild",
        "renderer",
    )
    for key in required_strings:
        if not isinstance(profile.get(key), str) or not profile[key]:
            raise ValueError(f"evaluation profile requires {key}")
        if re.search(r"<[^>]+>", profile[key]):
            raise ValueError(f"evaluation profile has unresolved placeholder in {key}")
    for key in ("width", "height"):
        if not isinstance(profile.get(key), int) or profile[key] < 1:
            raise ValueError(f"evaluation profile requires a positive {key}")
    start, end = profile.get("frame_start"), profile.get("frame_end")
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
        raise ValueError("evaluation profile requires ordered non-negative frame_start and frame_end")
    state = _load_or_create_state(candidate_manifest_file)
    state["evaluation_profile"] = profile
    _write_state(candidate_manifest_file, state)
    return {"status": "succeeded", "state_file": str(_state_file(candidate_manifest_file)), "profile": profile}


def continue_candidate_refinement(
    candidate_manifest_file: Path,
    analysis_file: Path,
    design_file: Path,
    max_iterations: int,
    max_rejected: int,
) -> dict[str, Any]:
    """Prepare one bounded next iteration from the latest recorded evaluation."""
    state = _load_or_create_state(candidate_manifest_file)
    if state.get("human_acceptance"):
        raise ValueError("candidate has been human accepted; start a new refinement phase before continuing")

    evaluated = [
        item
        for item in state["history"]
        if item.get("status") in {"accepted", "rejected", "tradeoff"}
        and item.get("hypothesis_category") != "baseline"
        and isinstance(item.get("iteration"), int)
    ]
    if not evaluated:
        raise ValueError("candidate has no completed refinement evaluation to continue from")
    latest = max(evaluated, key=lambda item: int(item["iteration"]))
    outcome = str(latest["status"])
    restoration = None
    if outcome in {"rejected", "tradeoff"}:
        restoration = restore_candidate_baseline(candidate_manifest_file)

    packet = build_next_iteration_packet(
        candidate_manifest_file=candidate_manifest_file,
        analysis_file=analysis_file,
        design_file=design_file,
        max_iterations=max_iterations,
        max_rejected=max_rejected,
        evaluate_after_edit=True,
    )
    return {
        "status": "succeeded",
        "previous_iteration": latest["iteration"],
        "previous_outcome": outcome,
        "restored_baseline": restoration is not None,
        "restoration": restoration,
        "next_iteration": packet["iteration"],
        "packet_file": packet["packet_file"],
        "prompt_file": packet["prompt_file"],
    }


def record_candidate_evaluation(
    candidate_manifest_file: Path,
    iteration: int,
    report_file: Path,
) -> dict[str, Any]:
    state = _load_or_create_state(candidate_manifest_file)
    iteration_file = _find_iteration_file(candidate_manifest_file.parent, iteration)
    record = load_json(iteration_file)
    category = record.get("hypothesis_category") or _legacy_category(record, iteration_file.name)
    if category not in HYPOTHESIS_CATEGORIES:
        raise ValueError(
            f"iteration {iteration} must declare hypothesis_category from: {', '.join(HYPOTHESIS_CATEGORIES)}"
        )
    metrics = _metrics_from_report(load_json(report_file))
    selection_policy = _resolve_selection_policy(candidate_manifest_file)
    if "dense_rgb_slice_similarity" in selection_policy.get("primary_metrics", []):
        _hydrate_dense_rgb_slice_state(state)
    outcome, reason, decision = _select_outcome_with_decision(
        state.get("baseline"), metrics, selection_policy
    )
    # Preserve every evaluated shader before a later continuation restores a baseline.
    # Tradeoff candidates may become eligible after a selection-policy revision.
    snapshot_dir = _snapshot_evaluated_sources(candidate_manifest_file, iteration)
    record["evaluation"] = metrics
    record["status"] = outcome
    record["reason"] = reason
    record["selection_decision"] = decision
    record["source_snapshot"] = str(snapshot_dir)
    write_json(iteration_file, record)
    _upsert_history(
        state,
        iteration,
        category,
        outcome,
        metrics,
        str(report_file),
        str(snapshot_dir),
    )
    if outcome in {"accepted", "tradeoff"}:
        _upsert_shortlist(state, iteration, outcome, metrics, str(report_file))
    if outcome == "accepted":
        snapshot_dir = _snapshot_baseline_sources(
            candidate_manifest_file,
            iteration,
            candidate_manifest_file.parent,
        )
        state["baseline"] = {
            "iteration": iteration,
            "report_file": str(report_file),
            "metrics": metrics,
            "source_snapshot": str(snapshot_dir),
            "selected_at": _timestamp(),
            "selection_policy": selection_policy,
        }
    _refresh_rejected_budget(state)
    _write_state(candidate_manifest_file, state)
    return {
        "status": outcome,
        "reason": reason,
        "selection_policy": selection_policy,
        "selection_decision": decision,
        "state_file": str(_state_file(candidate_manifest_file)),
    }


def candidate_status(candidate_manifest_file: Path) -> dict[str, Any]:
    state = _load_or_create_state(candidate_manifest_file)
    return {
        "status": "succeeded",
        "effect_id": state["effect_id"],
        "state_file": str(_state_file(candidate_manifest_file)),
        "baseline": state["baseline"],
        "human_acceptance": state.get("human_acceptance"),
        "history": state["history"],
        "shortlist": state["shortlist"],
        "budgets": state.get("budgets"),
        "phases": state.get("phases", []),
        "active_phase": _active_phase(state),
        "evaluation_profile_configured": isinstance(state.get("evaluation_profile"), dict),
        "blocked_hypothesis_categories": _blocked_categories(
            state,
            int(_active_phase(state)["first_iteration"]) if _active_phase(state) else None,
        ),
        "last_packet": state.get("last_packet"),
    }


def reassess_candidate_history(candidate_manifest_file: Path) -> dict[str, Any]:
    """Replay saved evaluations under the current policy without changing candidate files or state."""
    return _reassess_candidate_history(candidate_manifest_file, apply_best=False)


def apply_reassessed_baseline(candidate_manifest_file: Path) -> dict[str, Any]:
    """Restore the best policy-eligible historical snapshot as the selected baseline."""
    return _reassess_candidate_history(candidate_manifest_file, apply_best=True)


def _reassess_candidate_history(
    candidate_manifest_file: Path,
    apply_best: bool,
) -> dict[str, Any]:
    state = _load_or_create_state(candidate_manifest_file)
    initial_baseline = _initial_history_baseline(candidate_manifest_file, state)
    if initial_baseline is None:
        raise ValueError("candidate has no selected baseline")
    policy = _resolve_selection_policy(candidate_manifest_file)
    prospective = dict(initial_baseline)
    entries: list[dict[str, Any]] = []
    for item in sorted(state.get("history", []), key=lambda value: int(value.get("iteration", -1))):
        if item.get("hypothesis_category") == "baseline" or int(item.get("iteration", -1)) <= int(initial_baseline["iteration"]):
            continue
        report_name = item.get("report_file")
        if not isinstance(report_name, str) or not Path(report_name).is_file():
            continue
        metrics = _metrics_from_report(load_json(Path(report_name)))
        outcome, reason, decision = _select_outcome_with_decision(prospective, metrics, policy)
        entry = {
            "iteration": item.get("iteration"),
            "recorded_status": item.get("status"),
            "reassessed_status": outcome,
            "reason": reason,
            "selection_decision": decision,
        }
        snapshot_dir = _history_snapshot(candidate_manifest_file, item)
        entry["source_snapshot"] = str(snapshot_dir) if snapshot_dir is not None else None
        entry["eligible_for_baseline"] = snapshot_dir is not None
        if outcome == "accepted" and snapshot_dir is None:
            entry["reassessed_status"] = "ineligible"
            entry["reason"] = "would improve under policy, but no preserved source snapshot is available"
        entries.append(entry)
        if outcome == "accepted" and snapshot_dir is not None:
            prospective = {
                "iteration": item["iteration"],
                "metrics": metrics,
                "report_file": str(report_name),
                "source_snapshot": str(snapshot_dir),
            }
    result = {
        "status": "succeeded",
        "mode": "applied" if apply_best else "preview_only",
        "selection_policy": policy,
        "starting_baseline_iteration": initial_baseline["iteration"],
        "recommended_baseline_iteration": prospective["iteration"],
        "iterations": entries,
    }
    if not apply_best:
        return result
    snapshot_dir = Path(str(prospective.get("source_snapshot", "")))
    if not snapshot_dir.is_dir():
        raise ValueError("no policy-eligible historical candidate has a preserved source snapshot")
    _restore_sources_from_snapshot(candidate_manifest_file, snapshot_dir)
    selected_at = _timestamp()
    state["baseline"] = {
        "iteration": prospective["iteration"],
        "report_file": prospective.get("report_file", initial_baseline.get("report_file", "")),
        "metrics": prospective["metrics"],
        "source_snapshot": str(snapshot_dir),
        "selected_at": selected_at,
        "selection_policy": policy,
        "selection_source": "historical_reassessment",
    }
    active_phase = _active_phase(state)
    if active_phase is not None:
        active_phase["status"] = "closed"
        active_phase["closed_at"] = selected_at
        active_phase["closed_reason"] = "historical_rebaseline"
    state["historical_rebaseline"] = {
        "iteration": prospective["iteration"],
        "starting_baseline_iteration": initial_baseline["iteration"],
        "selection_policy": policy,
        "applied_at": selected_at,
    }
    _upsert_shortlist(
        state,
        int(prospective["iteration"]),
        "accepted",
        prospective["metrics"],
        str(prospective.get("report_file", "")),
    )
    _refresh_rejected_budget(state)
    _write_state(candidate_manifest_file, state)
    result["restored_baseline_iteration"] = prospective["iteration"]
    result["source_snapshot"] = str(snapshot_dir)
    return result


def _load_or_create_state(candidate_manifest_file: Path) -> dict[str, Any]:
    state_file = _state_file(candidate_manifest_file)
    if state_file.exists():
        state = load_json(state_file)
    else:
        candidate = load_json(candidate_manifest_file)
        state = {
            "artifact_type": "candidate_refinement_state",
            "artifact_version": 1,
            "effect_id": candidate["effect_id"],
            "baseline": None,
            "history": [],
            "shortlist": [],
            "last_packet": None,
        }
    state.setdefault("shortlist", [])
    state.setdefault("phases", [])
    if _import_legacy_history(candidate_manifest_file.parent, state):
        _write_state(candidate_manifest_file, state)
    return state


def _write_state(candidate_manifest_file: Path, state: dict[str, Any]) -> None:
    write_json(_state_file(candidate_manifest_file), state)


def _snapshot_baseline_sources(
    candidate_manifest_file: Path,
    iteration: int,
    source_dir: Path,
) -> Path:
    candidate = load_json(candidate_manifest_file)
    candidate_files = [Path(path) for path in candidate.get("candidate_files", [])]
    if not candidate_files:
        raise ValueError("candidate manifest has no candidate files")
    snapshot_dir = candidate_manifest_file.parent / "baselines" / f"iteration_{iteration:03d}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for candidate_file in candidate_files:
        source = source_dir / candidate_file.name
        if not source.exists():
            raise FileNotFoundError(f"baseline source is missing {source.name}")
        shutil.copyfile(source, snapshot_dir / candidate_file.name)
    return snapshot_dir


def _snapshot_evaluated_sources(candidate_manifest_file: Path, iteration: int) -> Path:
    """Keep the exact candidate that produced an evaluation, regardless of its outcome."""
    candidate = load_json(candidate_manifest_file)
    candidate_files = [Path(path) for path in candidate.get("candidate_files", [])]
    if not candidate_files:
        raise ValueError("candidate manifest has no candidate files")
    snapshot_dir = candidate_manifest_file.parent / "snapshots" / f"iteration_{iteration:03d}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for candidate_file in candidate_files:
        if not candidate_file.is_file():
            raise FileNotFoundError(f"candidate source is missing {candidate_file.name}")
        shutil.copyfile(candidate_file, snapshot_dir / candidate_file.name)
    return snapshot_dir


def _restore_sources_from_snapshot(candidate_manifest_file: Path, snapshot_dir: Path) -> list[str]:
    candidate = load_json(candidate_manifest_file)
    candidate_files = [Path(path) for path in candidate.get("candidate_files", [])]
    target_files = [Path(path) for path in candidate.get("target_files", [])]
    if not candidate_files or len(candidate_files) != len(target_files):
        raise ValueError("candidate manifest has invalid candidate and target files")
    restored: list[str] = []
    for candidate_file, target_file in zip(candidate_files, target_files):
        source = snapshot_dir / candidate_file.name
        if not source.exists():
            raise FileNotFoundError(f"baseline snapshot is missing {source.name}")
        shutil.copyfile(source, candidate_file)
        shutil.copyfile(source, target_file)
        restored.append(str(candidate_file))
    return restored


def _history_snapshot(candidate_manifest_file: Path, item: dict[str, Any]) -> Path | None:
    stored = item.get("source_snapshot")
    if isinstance(stored, str) and Path(stored).is_dir():
        return Path(stored)
    iteration = item.get("iteration")
    if isinstance(iteration, int):
        baseline_snapshot = candidate_manifest_file.parent / "baselines" / f"iteration_{iteration:03d}"
        if baseline_snapshot.is_dir():
            return baseline_snapshot
    return None


def _initial_history_baseline(
    candidate_manifest_file: Path,
    state: dict[str, Any],
) -> dict[str, Any] | None:
    candidates = [
        item for item in state.get("history", [])
        if item.get("hypothesis_category") == "baseline" and isinstance(item.get("metrics"), dict)
    ]
    if candidates:
        item = min(candidates, key=lambda value: int(value.get("iteration", 0)))
        snapshot = _history_snapshot(candidate_manifest_file, item)
        if snapshot is not None:
            return {
                "iteration": int(item["iteration"]),
                "metrics": item["metrics"],
                "report_file": str(item.get("report_file", "")),
                "source_snapshot": str(snapshot),
            }
    baseline = state.get("baseline")
    return dict(baseline) if isinstance(baseline, dict) else None


def _state_file(candidate_manifest_file: Path) -> Path:
    return candidate_manifest_file.parent / STATE_FILE_NAME


def _iteration_records(candidate_dir: Path) -> list[tuple[int, Path]]:
    records: list[tuple[int, Path]] = []
    for path in candidate_dir.glob("iteration_*.json"):
        match = ITERATION_PATTERN.match(path.name)
        if match:
            records.append((int(match.group(1)), path))
    return sorted(records)


def _find_iteration_file(candidate_dir: Path, iteration: int) -> Path:
    matches = [path for number, path in _iteration_records(candidate_dir) if number == iteration]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one iteration record for iteration {iteration}")
    return matches[0]


def _metrics_from_report(report: dict[str, Any]) -> dict[str, Any]:
    score = report.get("score", report)
    window = score.get("transition_window")
    if not isinstance(window, dict):
        raise ValueError("evaluation report is missing transition_window metrics")
    endpoints = score.get("endpoint_checks")
    if not isinstance(endpoints, dict):
        raise ValueError("evaluation report is missing endpoint checks")
    metrics = {
        "mse": _number(window, "mse"),
        "mae": _number(window, "mae"),
        "psnr_db": _number(window, "psnr_db"),
        "ssim": _number(window, "ssim"),
        "transition_window": {
            "frame_start": window.get("frame_start"),
            "frame_end": window.get("frame_end"),
            "frame_count": window.get("frame_count"),
        },
        "endpoint_checks": endpoints,
    }
    motion = score.get("motion_metrics")
    if isinstance(motion, dict):
        motion_metrics = {
            "motion_similarity": _number(motion, "motion_similarity"),
            "direction_agreement": _number(motion, "direction_agreement"),
        }
        if isinstance(motion.get("flow_vector_mae"), (int, float)):
            motion_metrics["flow_vector_mae"] = _number(motion, "flow_vector_mae")
        if isinstance(motion.get("motion_region_iou"), (int, float)):
            motion_metrics["motion_region_iou"] = _number(motion, "motion_region_iou")
        if isinstance(motion.get("reliable_motion_coverage"), (int, float)):
            motion_metrics["reliable_motion_coverage"] = _number(motion, "reliable_motion_coverage")
        if isinstance(motion.get("horizontal_shift_mae"), (int, float)):
            motion_metrics["horizontal_shift_mae"] = _number(motion, "horizontal_shift_mae")
        metrics["motion"] = motion_metrics
    topology = score.get("motion_topology")
    if isinstance(topology, dict):
        metrics["motion_topology"] = topology
    geometry = score.get("motion_geometry")
    if isinstance(geometry, dict):
        metrics["motion_geometry"] = geometry
    body_geometry = score.get("foreground_body_transform")
    if isinstance(body_geometry, dict):
        metrics["foreground_body_transform"] = body_geometry
        metrics.update(_foreground_body_selection_metrics(body_geometry))
    centroid_tracking = score.get("salient_centroid_tracking")
    if isinstance(centroid_tracking, dict):
        metrics["salient_centroid_tracking"] = centroid_tracking
        metrics.update(_salient_centroid_selection_metrics(centroid_tracking))
    rotation_tracking = score.get("salient_rotation_tracking")
    if isinstance(rotation_tracking, dict):
        metrics["salient_rotation_tracking"] = rotation_tracking
        metrics.update(_salient_rotation_selection_metrics(rotation_tracking))
    geometry_summary = score.get("motion_geometry")
    if isinstance(geometry_summary, dict):
        similarity = geometry_summary.get("geometry_similarity")
        if isinstance(similarity, (int, float)):
            metrics["geometry_similarity"] = float(similarity)
    angular_motion = score.get("angular_motion")
    if isinstance(angular_motion, dict):
        metrics["angular_motion"] = angular_motion
    regional_motion = score.get("regional_motion")
    if isinstance(regional_motion, dict):
        metrics["regional_motion"] = regional_motion
    edge_content_policy = score.get("edge_content_policy")
    if isinstance(edge_content_policy, dict):
        metrics["edge_content_policy"] = edge_content_policy
    rgb_slices = score.get("dense_rgb_slices")
    if isinstance(rgb_slices, dict):
        metrics["dense_rgb_slices"] = rgb_slices
        confidence = rgb_slices.get("confidence")
        if rgb_slices.get("status") == "estimated" and isinstance(confidence, (int, float)):
            for metric in (
                "slice_structure_similarity",
                "rgb_separation_similarity",
                "rgb_slice_coherence",
                "dense_rgb_slice_similarity",
            ):
                value = rgb_slices.get(metric)
                if isinstance(value, (int, float)):
                    metrics[metric] = float(value)
    blur_direction = score.get("motion_blur_direction")
    if isinstance(blur_direction, dict):
        metrics["motion_blur_direction"] = blur_direction
        if blur_direction.get("status") == "estimated" and isinstance(
            blur_direction.get("direction_agreement"), (int, float)
        ):
            metrics["motion_blur_direction_agreement"] = float(blur_direction["direction_agreement"])
    phase_scores = score.get("phase_scores")
    if isinstance(phase_scores, dict):
        metrics["phase_scores"] = phase_scores
        peak = phase_scores.get("peak")
        if isinstance(peak, dict):
            for metric in ("mse", "ssim"):
                value = peak.get(metric)
                if isinstance(value, (int, float)):
                    metrics[f"peak_{metric}"] = float(value)
    return metrics


def _hydrate_dense_rgb_slice_state(state: dict[str, Any]) -> None:
    """Backfill the new metric for pre-feature reports without rerendering.

    Historical reports retain their candidate/reference frame paths, so the
    diagnostic can be recomputed from immutable render artifacts. This lets the
    first post-upgrade refinement request and evaluation compare against the
    established baseline instead of spending an iteration merely bootstrapping
    the new metric.
    """
    records: list[dict[str, Any]] = []
    baseline = state.get("baseline")
    if isinstance(baseline, dict):
        records.append(baseline)
    history = [item for item in state.get("history", []) if isinstance(item, dict)]
    if history:
        records.append(max(history, key=lambda item: int(item.get("iteration", -1))))
    for record in records:
        metrics = record.get("metrics")
        report_file = record.get("report_file")
        if not isinstance(metrics, dict) or "dense_rgb_slice_similarity" in metrics:
            continue
        if not isinstance(report_file, str) or not Path(report_file).is_file():
            continue
        diagnostic = _score_dense_rgb_slices_from_report(Path(report_file))
        if diagnostic is None:
            continue
        metrics["dense_rgb_slices"] = diagnostic
        if diagnostic.get("status") != "estimated":
            continue
        for metric in (
            "slice_structure_similarity",
            "rgb_separation_similarity",
            "rgb_slice_coherence",
            "dense_rgb_slice_similarity",
        ):
            value = diagnostic.get(metric)
            if isinstance(value, (int, float)):
                metrics[metric] = float(value)


def _score_dense_rgb_slices_from_report(report_file: Path) -> dict[str, Any] | None:
    report = load_json(report_file)
    score = report.get("score", report)
    if not isinstance(score, dict):
        return None
    existing = score.get("dense_rgb_slices")
    if isinstance(existing, dict):
        return existing
    candidate = score.get("candidate")
    reference = score.get("reference")
    window = score.get("transition_window")
    width = score.get("width")
    height = score.get("height")
    if not (
        isinstance(candidate, str)
        and isinstance(reference, str)
        and isinstance(window, dict)
        and isinstance(width, int)
        and isinstance(height, int)
    ):
        return None
    candidate_path = Path(candidate)
    reference_path = Path(reference)
    if not candidate_path.exists() or not reference_path.exists():
        return None
    from .harness_bridge import load_harness_modules

    workspace_root = Path(__file__).resolve().parents[3]
    scorer = load_harness_modules(workspace_root).get("score_dense_rgb_slices")
    if scorer is None:
        return None
    return scorer(
        candidate=candidate_path,
        reference=reference_path,
        width=width,
        height=height,
        frame_start=int(window.get("frame_start", 0)),
        frame_end=int(window.get("frame_end", 0)),
    )


def _number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"evaluation report has invalid {key}")
    return float(value)


def _foreground_body_selection_metrics(body: dict[str, Any]) -> dict[str, float]:
    """Flatten phase-specific transform agreement into policy-selectable metrics."""
    phases = body.get("phases")
    phase_values = phases.values() if isinstance(phases, dict) else []
    rotation_errors: list[float] = []
    scale_errors: list[float] = []
    translation_errors: list[float] = []
    pivot_errors: list[float] = []
    direction_matches = 0
    direction_observations = 0
    for phase in phase_values:
        if not isinstance(phase, dict):
            continue
        candidate = phase.get("candidate")
        reference = phase.get("reference")
        if not isinstance(candidate, dict) or not isinstance(reference, dict):
            continue
        candidate_rotation = _nested_number(candidate, "rotation_field", "mean_degrees")
        reference_rotation = _nested_number(reference, "rotation_field", "mean_degrees")
        if candidate_rotation is not None and reference_rotation is not None:
            rotation_errors.append(abs(candidate_rotation - reference_rotation))
            if abs(reference_rotation) >= 1.0:
                direction_observations += 1
                if candidate_rotation * reference_rotation > 0:
                    direction_matches += 1
        candidate_scale = _nested_number(candidate, "radial_scale_field", "mean_ratio")
        reference_scale = _nested_number(reference, "radial_scale_field", "mean_ratio")
        if candidate_scale is not None and reference_scale is not None:
            scale_errors.append(abs(candidate_scale - reference_scale))
        candidate_translation = _translation_vector(candidate)
        reference_translation = _translation_vector(reference)
        if candidate_translation is not None and reference_translation is not None:
            translation_errors.append(
                ((candidate_translation[0] - reference_translation[0]) ** 2
                 + (candidate_translation[1] - reference_translation[1]) ** 2) ** 0.5
            )
        candidate_pivot = _pivot_vector(candidate)
        reference_pivot = _pivot_vector(reference)
        if candidate_pivot is not None and reference_pivot is not None:
            pivot_errors.append(
                ((candidate_pivot[0] - reference_pivot[0]) ** 2
                 + (candidate_pivot[1] - reference_pivot[1]) ** 2) ** 0.5
            )
    result: dict[str, float] = {}
    if rotation_errors:
        result["foreground_body_rotation_error"] = sum(rotation_errors) / len(rotation_errors)
    if scale_errors:
        result["foreground_body_scale_error"] = sum(scale_errors) / len(scale_errors)
    if translation_errors:
        result["foreground_body_translation_error"] = sum(translation_errors) / len(translation_errors)
    if pivot_errors:
        result["foreground_body_pivot_error"] = sum(pivot_errors) / len(pivot_errors)
    if direction_observations:
        result["foreground_body_rotation_direction_agreement"] = direction_matches / direction_observations
    return result


def _salient_centroid_selection_metrics(tracking: dict[str, Any]) -> dict[str, float]:
    """Expose centroid agreement only when enough unclipped frames were compared.

    A low-confidence trajectory must not reach the selection policy: it would let
    a candidate be accepted on the strength of two or three usable frames.
    """
    if tracking.get("status") != "succeeded":
        return {}
    result: dict[str, float] = {}
    mean_distance = tracking.get("mean_centroid_distance_pixels")
    if isinstance(mean_distance, (int, float)):
        result["salient_centroid_distance_error"] = float(mean_distance)
    max_distance = tracking.get("max_centroid_distance_pixels")
    if isinstance(max_distance, (int, float)):
        result["salient_centroid_max_distance_error"] = float(max_distance)
    coverage_error = tracking.get("mean_coverage_error")
    if isinstance(coverage_error, (int, float)):
        result["salient_coverage_error"] = float(coverage_error)
    return result


def _salient_rotation_selection_metrics(tracking: dict[str, Any]) -> dict[str, float]:
    """Expose absolute-turn agreement only when the unwrap chain was reliable."""
    if tracking.get("status") != "succeeded":
        return {}
    result: dict[str, float] = {}
    mean_error = tracking.get("mean_absolute_rotation_error_degrees")
    if isinstance(mean_error, (int, float)):
        result["salient_rotation_error"] = float(mean_error)
    max_error = tracking.get("max_absolute_rotation_error_degrees")
    if isinstance(max_error, (int, float)):
        result["salient_rotation_max_error"] = float(max_error)
    return result


def _nested_number(payload: dict[str, Any], group: str, key: str) -> float | None:
    value = payload.get(group)
    if not isinstance(value, dict) or not isinstance(value.get(key), (int, float)):
        return None
    return float(value[key])


def _translation_vector(payload: dict[str, Any]) -> tuple[float, float] | None:
    x = _nested_number(payload, "translation_field", "mean_dx_pixels")
    y = _nested_number(payload, "translation_field", "mean_dy_pixels")
    return (x, y) if x is not None and y is not None else None


def _pivot_vector(payload: dict[str, Any]) -> tuple[float, float] | None:
    x = _nested_number(payload, "pivot_field", "x_pixels")
    y = _nested_number(payload, "pivot_field", "y_pixels")
    return (x, y) if x is not None and y is not None else None


def _endpoints_are_exact(metrics: dict[str, Any]) -> bool:
    """Permit insignificant video-reference compression variance at stable endpoints."""
    endpoints = metrics["endpoint_checks"]
    for key in ("before_transition", "after_transition"):
        endpoint = endpoints.get(key)
        if not isinstance(endpoint, dict):
            return False
        mse = endpoint.get("mse")
        ssim = endpoint.get("ssim")
        if not isinstance(mse, (int, float)) or not isinstance(ssim, (int, float)):
            return False
        if mse > ENDPOINT_MSE_TOLERANCE or ssim < ENDPOINT_SSIM_TOLERANCE:
            return False
    return True


def _select_outcome(baseline: dict[str, Any] | None, metrics: dict[str, Any]) -> tuple[str, str]:
    outcome, reason, _ = _select_outcome_with_decision(baseline, metrics, _legacy_selection_policy())
    return outcome, reason


def _select_outcome_with_decision(
    baseline: dict[str, Any] | None,
    metrics: dict[str, Any],
    selection_policy: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    if not _endpoints_are_exact(metrics):
        return "rejected", "endpoint checks exceed stable-frame tolerance", {
            "policy": selection_policy,
            "endpoint_checks": "failed",
        }
    topology = metrics.get("motion_topology")
    if (
        isinstance(topology, dict)
        and topology.get("status") == "structural_mismatch"
        and topology.get("enforcement") == "hard"
    ):
        return "rejected", "candidate violates the required reference motion topology", {
            "policy": selection_policy,
            "endpoint_checks": "passed",
            "topology": "hard_mismatch",
        }
    if baseline is None:
        return "accepted", "first valid evaluation becomes the baseline", {
            "policy": selection_policy,
            "endpoint_checks": "passed",
            "comparison": "initial_baseline",
        }
    previous = baseline["metrics"]
    deltas = _selection_deltas(metrics, previous)
    primary = [metric for metric in selection_policy["primary_metrics"] if metric in deltas]
    guardrails = [metric for metric in selection_policy["guardrail_metrics"] if metric in deltas]
    improved = [metric for metric in primary if deltas[metric]["materially_improved"]]
    guardrail_failures = [metric for metric in guardrails if deltas[metric]["regressed_beyond_guardrail"]]
    decision = {
        "policy": selection_policy,
        "endpoint_checks": "passed",
        "baseline_iteration": baseline.get("iteration"),
        "metric_deltas": deltas,
        "primary_metrics_available": primary,
        "materially_improved_primary_metrics": improved,
        "guardrail_failures": guardrail_failures,
    }
    if improved and not guardrail_failures:
        return "accepted", "improved a policy-primary metric within configured guardrails", decision
    any_improved = [metric for metric, value in deltas.items() if value["improved"]]
    if any_improved:
        decision["improved_metrics"] = any_improved
        if guardrail_failures:
            return "tradeoff", "primary or advisory metrics improved but a policy guardrail regressed", decision
        return "tradeoff", "only advisory metrics improved or no primary improvement met the policy threshold", decision
    return "rejected", "no policy-primary or advisory metric improved against the accepted baseline", decision


def _selection_deltas(metrics: dict[str, Any], previous: dict[str, Any]) -> dict[str, dict[str, Any]]:
    deltas: dict[str, dict[str, Any]] = {}
    for metric in SELECTION_METRICS:
        current = _selection_metric_value(metrics, metric)
        baseline = _selection_metric_value(previous, metric)
        if not isinstance(current, (int, float)) or not isinstance(baseline, (int, float)):
            continue
        if metric in LOWER_IS_BETTER_METRICS:
            relative = (float(current) - float(baseline)) / max(abs(float(baseline)), 1e-9)
            absolute_delta = float(current) - float(baseline)
            absolute_floor = METRIC_REGRESSION_ABSOLUTE_FLOORS.get(metric)
            regressed_beyond_guardrail = relative > MSE_REGRESSION_TOLERANCE
            if absolute_floor is not None:
                regressed_beyond_guardrail = regressed_beyond_guardrail and absolute_delta > absolute_floor
            deltas[metric] = {
                "baseline": float(baseline), "current": float(current), "relative_change": relative,
                "improved": relative < 0.0,
                "materially_improved": relative <= -MSE_IMPROVEMENT_RATIO,
                "regressed_beyond_guardrail": regressed_beyond_guardrail,
            }
        else:
            delta = float(current) - float(baseline)
            threshold = METRIC_IMPROVEMENT_THRESHOLDS.get(metric, SSIM_IMPROVEMENT)
            tolerance = METRIC_REGRESSION_TOLERANCES.get(metric, SSIM_REGRESSION_TOLERANCE)
            deltas[metric] = {
                "baseline": float(baseline), "current": float(current), "delta": delta,
                "improved": delta > 0.0,
                "materially_improved": delta >= threshold,
                "regressed_beyond_guardrail": delta < -tolerance,
            }
    return deltas


def _selection_metric_value(metrics: dict[str, Any], metric: str) -> Any:
    if metric == "motion_similarity":
        motion = metrics.get("motion")
        return motion.get(metric) if isinstance(motion, dict) else None
    return metrics.get(metric)


def _legacy_selection_policy() -> dict[str, Any]:
    return {
        "profile": "legacy_pareto",
        "source": "legacy_default",
        "primary_metrics": ["ssim", "mse", "motion_similarity"],
        "guardrail_metrics": ["ssim", "mse", "motion_similarity"],
        "advisory_metrics": ["mae", "peak_mse", "peak_ssim"],
    }


def _resolve_selection_policy(candidate_manifest_file: Path) -> dict[str, Any]:
    candidate = load_json(candidate_manifest_file)
    artifacts: list[tuple[str, dict[str, Any]]] = []
    sources: list[dict[str, Any]] = [candidate]
    source_manifest = candidate.get("source_manifest")
    if isinstance(source_manifest, str) and Path(source_manifest).is_file():
        generated = load_json(Path(source_manifest))
        if isinstance(generated, dict):
            sources.append(generated)
    sample_root = _candidate_sample_root(candidate_manifest_file)
    if sample_root is not None:
        sources.append(
            {
                "design_artifact": str(sample_root / "design" / "effect_design.json"),
                "analysis_artifact": str(sample_root / "analysis" / "transition_structure.json"),
            }
        )
    for label, key in (("effect_design", "design_artifact"), ("transition_analysis", "analysis_artifact")):
        value = next((source.get(key) for source in sources if isinstance(source.get(key), str)), None)
        if isinstance(value, str) and Path(value).is_file():
            artifact = load_json(Path(value))
            if isinstance(artifact, dict):
                artifacts.append((label, artifact))
    for label, artifact in artifacts:
        policy = artifact.get("evaluation_policy")
        selection = policy.get("selection") if isinstance(policy, dict) else None
        normalized = _normalize_selection_policy(selection, f"{label}.evaluation_policy.selection")
        if normalized is not None:
            if _artifact_requires_dense_rgb_slices(artifact):
                return _with_dense_rgb_slice_selection(normalized)
            return normalized
    vocabulary = " ".join(_artifact_selection_vocabulary(artifact) for _, artifact in artifacts).casefold()
    transform_markers = ("rotation", "rotate", "scale", "perspective", "card", "flip", "reflection")
    if any(marker in vocabulary for marker in transform_markers):
        return {
            "profile": "transform",
            "source": "artifact_inference",
            "primary_metrics": [
                "foreground_body_rotation_error",
                "foreground_body_translation_error",
                "foreground_body_scale_error",
                "foreground_body_pivot_error",
            ],
            "guardrail_metrics": [],
            "advisory_metrics": ["mse", "mae", "peak_mse", "ssim", "peak_ssim", "motion_similarity"],
        }
    if _vocabulary_requires_dense_rgb_slices(vocabulary):
        return _with_dense_rgb_slice_selection(_legacy_selection_policy())
    return _legacy_selection_policy()


def _artifact_requires_dense_rgb_slices(artifact: dict[str, Any]) -> bool:
    signals = artifact.get("visual_signals")
    explicit_rgb = isinstance(signals, dict) and signals.get("rgb_split") is True
    vocabulary = _artifact_selection_vocabulary(artifact).casefold()
    return _vocabulary_requires_dense_rgb_slices(vocabulary, explicit_rgb=explicit_rgb)


def _vocabulary_requires_dense_rgb_slices(vocabulary: str, explicit_rgb: bool = False) -> bool:
    has_rgb = explicit_rgb or any(marker in vocabulary for marker in ("rgb", "channel separation", "chromatic"))
    has_slices = any(marker in vocabulary for marker in ("slice", "strip", "segmented"))
    has_density = any(marker in vocabulary for marker in ("dense", "many", "hierarchy", "fragment"))
    return has_rgb and has_slices and has_density


def _with_dense_rgb_slice_selection(policy: dict[str, Any]) -> dict[str, Any]:
    result = dict(policy)
    result["profile"] = "dense_rgb_slices"
    result["source"] = f"{policy.get('source', 'artifact_inference')} (dense RGB-slice objective)"
    result["primary_metrics"] = list(dict.fromkeys([
        "dense_rgb_slice_similarity", *policy.get("primary_metrics", [])
    ]))
    result["guardrail_metrics"] = list(dict.fromkeys([
        "dense_rgb_slice_similarity", "ssim", "mse", *policy.get("guardrail_metrics", [])
    ]))
    result["advisory_metrics"] = list(dict.fromkeys([
        *policy.get("advisory_metrics", []),
        "slice_structure_similarity",
        "rgb_separation_similarity",
        "rgb_slice_coherence",
    ]))
    return result


def _candidate_sample_root(candidate_manifest_file: Path) -> Path | None:
    candidate_dir = candidate_manifest_file.parent
    candidates_dir = candidate_dir.parent
    if candidates_dir.name != "candidates":
        return None
    sample_root = candidates_dir.parent
    if (sample_root / "analysis" / "transition_structure.json").is_file() or (
        sample_root / "design" / "effect_design.json"
    ).is_file():
        return sample_root
    return None


def _normalize_selection_policy(policy: Any, source: str) -> dict[str, Any] | None:
    if not isinstance(policy, dict):
        return None
    def metrics_for(key: str) -> list[str]:
        values = policy.get(key, [])
        return [value for value in values if isinstance(value, str) and value in SELECTION_METRICS]
    primary = metrics_for("primary_metrics")
    if not primary:
        return None
    profile = str(policy.get("profile", "custom"))
    guardrails = metrics_for("guardrail_metrics")
    advisory = metrics_for("advisory_metrics")
    # Perspective, rotation, and flip peaks commonly contain intentionally dark or
    # heavily blurred faces. SSIM remains useful context there, but must not veto
    # substantial pixel-error improvement.
    if profile == "transform":
        moved = [metric for metric in guardrails if metric in {"ssim", "peak_ssim"}]
        guardrails = [metric for metric in guardrails if metric not in moved]
        advisory = list(dict.fromkeys([*advisory, *moved]))
        if "foreground_body_rotation_direction_agreement" not in primary:
            guardrails = list(dict.fromkeys([
                *guardrails,
                "foreground_body_rotation_direction_agreement",
            ]))
        guardrails = list(dict.fromkeys([
            *guardrails,
            "foreground_body_rotation_error",
            "foreground_body_translation_error",
            "foreground_body_scale_error",
        ]))
        if moved:
            source = f"{source} (transform SSIM advisory)"
    return {
        "profile": profile,
        "source": source,
        "primary_metrics": primary,
        "guardrail_metrics": guardrails,
        "advisory_metrics": advisory,
    }


def _artifact_selection_vocabulary(artifact: dict[str, Any]) -> str:
    transition = artifact.get("transition") if isinstance(artifact.get("transition"), dict) else {}
    hints = artifact.get("planner_hints") if isinstance(artifact.get("planner_hints"), dict) else {}
    design = artifact.get("design_notes") if isinstance(artifact.get("design_notes"), dict) else {}
    seed = artifact.get("implementation_seed") if isinstance(artifact.get("implementation_seed"), dict) else {}
    values: list[Any] = [transition, hints, design, seed, artifact.get("target_effect")]
    return " ".join(str(value) for value in values)


def _motion_delta(metrics: dict[str, Any], previous: dict[str, Any]) -> float | None:
    motion = metrics.get("motion")
    previous_motion = previous.get("motion")
    if not isinstance(motion, dict) or not isinstance(previous_motion, dict):
        return None
    return float(motion["motion_similarity"]) - float(previous_motion["motion_similarity"])


def _upsert_history(
    state: dict[str, Any],
    iteration: int,
    category: str,
    status: str,
    metrics: dict[str, Any],
    report_file: str,
    source_snapshot: str | None = None,
) -> None:
    item = {
        "iteration": iteration,
        "hypothesis_category": category,
        "status": status,
        "metrics": metrics,
        "report_file": report_file,
        "recorded_at": _timestamp(),
    }
    if source_snapshot:
        item["source_snapshot"] = source_snapshot
    state["history"] = [entry for entry in state["history"] if entry.get("iteration") != iteration]
    state["history"].append(item)
    state["history"].sort(key=lambda entry: int(entry["iteration"]))


def _upsert_shortlist(
    state: dict[str, Any],
    iteration: int,
    status: str,
    metrics: dict[str, Any],
    report_file: str,
) -> None:
    item = {
        "iteration": iteration,
        "status": status,
        "metrics": metrics,
        "report_file": report_file,
    }
    state["shortlist"] = [entry for entry in state["shortlist"] if entry.get("iteration") != iteration]
    state["shortlist"].append(item)
    state["shortlist"].sort(key=lambda entry: int(entry["iteration"]))


def _refresh_rejected_budget(state: dict[str, Any]) -> None:
    budgets = state.get("budgets")
    if not isinstance(budgets, dict):
        return
    phase = _active_phase(state)
    if phase is None:
        budgets["rejected_so_far"] = sum(
            1 for item in state["history"] if item.get("status") == "rejected"
        )
        return
    first_iteration = int(phase["first_iteration"])
    phase_history = [
        item for item in state["history"] if int(item.get("iteration", 0)) >= first_iteration
    ]
    budgets["attempted_so_far"] = len(phase_history)
    budgets["rejected_so_far"] = sum(1 for item in phase_history if item.get("status") == "rejected")


def _close_phase_for_budget(
    candidate_manifest_file: Path,
    state: dict[str, Any],
    phase: dict[str, Any],
    attempted_count: int,
    rejected_count: int,
    reason: str,
) -> None:
    phase["status"] = "closed"
    phase["closed_at"] = _timestamp()
    phase["closed_reason"] = reason
    state["active_phase"] = None
    state["budgets"] = {
        "phase": phase["name"],
        "max_iterations": int(phase["max_iterations"]),
        "max_rejected": int(phase["max_rejected"]),
        "attempted_so_far": attempted_count,
        "rejected_so_far": rejected_count,
        "status": "closed",
        "closed_reason": reason,
    }
    _write_state(candidate_manifest_file, state)


def _import_legacy_history(candidate_dir: Path, state: dict[str, Any]) -> bool:
    known = {item.get("iteration") for item in state["history"]}
    changed = False
    for iteration, path in _iteration_records(candidate_dir):
        if iteration in known:
            continue
        record = load_json(path)
        outcome = _legacy_outcome(record)
        if outcome is None:
            continue
        category = _legacy_category(record, path.name)
        _upsert_history(
            state,
            iteration,
            category,
            outcome,
            record.get("evaluation", {}),
            "",
        )
        changed = True
    return changed


def _legacy_outcome(record: dict[str, Any]) -> str | None:
    status = record.get("status")
    if status in {"accepted", "rejected", "tradeoff"}:
        return status
    evaluation = record.get("evaluation")
    if isinstance(evaluation, dict) and evaluation.get("status") in {"accepted", "rejected", "tradeoff"}:
        return evaluation["status"]
    return None


def _legacy_category(record: dict[str, Any], filename: str) -> str:
    category = record.get("hypothesis_category")
    if category in HYPOTHESIS_CATEGORIES:
        return category
    normalized = filename.lower()
    if "blur" in normalized:
        return "blur"
    if "boundary" in normalized or "region" in normalized:
        return "regions"
    if "displacement" in normalized:
        return "displacement"
    if "mix" in normalized or "timing" in normalized:
        return "timing"
    return "other"


def _active_phase(state: dict[str, Any]) -> dict[str, Any] | None:
    active_name = state.get("active_phase")
    if not isinstance(active_name, str):
        return None
    for phase in state.get("phases", []):
        if isinstance(phase, dict) and phase.get("name") == active_name and phase.get("status") == "active":
            return phase
    return None


def _phase_by_name(state: dict[str, Any], name: str) -> dict[str, Any] | None:
    if not name:
        return None
    for phase in state.get("phases", []):
        if isinstance(phase, dict) and phase.get("name") == name:
            return phase
    return None


def _magnitude_findings(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect every over-threshold geometry disagreement with a comparable severity.

    Severity is the value divided by that signal's own threshold, so errors
    measured in pixels, degrees and ratios can be ranked against each other.
    """
    findings: list[dict[str, Any]] = []

    geometry = metrics.get("motion_geometry")
    if isinstance(geometry, dict) and geometry.get("status") == "geometry_mismatch":
        candidate = geometry.get("candidate") if isinstance(geometry.get("candidate"), dict) else {}
        reference = geometry.get("reference") if isinstance(geometry.get("reference"), dict) else {}
        confidence = min(
            float(candidate.get("confidence", 0.0)),
            float(reference.get("confidence", 0.0)),
        )
        if confidence >= 0.5:
            position = {
                "level": "high",
                "focus": "transform_position",
                "reason": "reliable geometry shows incorrect transform position or translation direction",
                "geometry": geometry,
                "recommended_categories": ["displacement", "shader_structure"],
            }
            translation_delta = geometry.get("translation_delta_pixels")
            if isinstance(translation_delta, (int, float)) and translation_delta > TRANSLATION_DELTA_LIMIT_PIXELS:
                findings.append(
                    {**position, "severity": translation_delta / TRANSLATION_DELTA_LIMIT_PIXELS}
                )
            if geometry.get("translation_direction_agreement") is False:
                findings.append({**position, "severity": DIRECTION_DISAGREEMENT_SEVERITY})
            pivot_delta = geometry.get("pivot_delta_pixels")
            if (
                isinstance(pivot_delta, (int, float))
                and pivot_delta > PIVOT_DELTA_LIMIT_PIXELS
                and _pivot_is_conditioned(candidate)
                and _pivot_is_conditioned(reference)
            ):
                findings.append({**position, "severity": pivot_delta / PIVOT_DELTA_LIMIT_PIXELS})
            if not findings:
                findings.append(
                    {
                        "level": "high",
                        "focus": "motion_geometry",
                        "reason": "reliable reference and candidate geometry estimates disagree",
                        "geometry": geometry,
                        "recommended_categories": ["displacement", "regions", "shader_structure"],
                        "severity": 1.0,
                    }
                )

    body_geometry = metrics.get("foreground_body_transform")
    if isinstance(body_geometry, dict) and body_geometry.get("status") == "estimated":
        confidence = body_geometry.get("confidence", 0.0)
        if isinstance(confidence, (int, float)) and confidence >= 0.5:
            phase_values = body_geometry.get("phases") if isinstance(body_geometry.get("phases"), dict) else {}
            phase_items = [item for item in phase_values.values() if isinstance(item, dict)]
            for key, limit, focus, reason, magnitude in (
                (
                    "rotation_delta_degrees",
                    ROTATION_DELTA_LIMIT_DEGREES,
                    "transform_rotation",
                    "feature-tracked body rotation disagrees with the reference",
                    _reference_rotation_magnitude,
                ),
                (
                    "scale_delta_ratio",
                    SCALE_DELTA_LIMIT_RATIO,
                    "transform_scale",
                    "feature-tracked body scale disagrees with the reference",
                    _reference_scale_magnitude,
                ),
                (
                    "translation_delta_pixels",
                    TRANSLATION_DELTA_LIMIT_PIXELS,
                    "foreground_body_transform",
                    "feature-tracked body position disagrees with the reference",
                    _reference_translation_magnitude,
                ),
            ):
                worst_item, worst = None, 0.0
                for item in phase_items:
                    value = abs(float(item.get(key, 0.0)))
                    if value >= worst:
                        worst_item, worst = item, value
                if worst <= limit or worst_item is None:
                    continue
                observed: dict[str, Any] = {key: worst, "limit": limit}
                # Prefer severity relative to how much the reference itself moves.
                # Absolute-over-threshold ranking is only as fair as the thresholds,
                # and a tight pixel limit next to a loose degree limit silently
                # ranked a same-sized rotation error six times lower than position.
                reference_magnitude = magnitude(worst_item)
                if reference_magnitude is not None:
                    relative = worst / reference_magnitude
                    severity = relative / RELATIVE_ERROR_LIMIT
                    observed["reference_magnitude"] = reference_magnitude
                    observed["relative_error"] = relative
                    observed["severity_basis"] = "relative"
                else:
                    severity = worst / limit
                    observed["severity_basis"] = "absolute"
                findings.append(
                    {
                        "level": "high",
                        "focus": focus,
                        "reason": reason,
                        "geometry": body_geometry,
                        "recommended_categories": ["displacement", "shader_structure"],
                        "severity": severity,
                        "observed": observed,
                    }
                )
    return findings


def _reference_rotation_magnitude(phase: dict[str, Any]) -> float | None:
    """How much the reference body rotates, when that is large enough to divide by."""
    reference = phase.get("reference") if isinstance(phase.get("reference"), dict) else {}
    rotation = _nested_number(reference, "rotation_field", "mean_degrees")
    if rotation is None or abs(rotation) < RELATIVE_ROTATION_FLOOR_DEGREES:
        return None
    return abs(rotation)


def _reference_scale_magnitude(phase: dict[str, Any]) -> float | None:
    """The reference's departure from unit scale, which is what a ratio error is against."""
    reference = phase.get("reference") if isinstance(phase.get("reference"), dict) else {}
    scale = _nested_number(reference, "radial_scale_field", "mean_ratio")
    if scale is None or abs(scale - 1.0) < RELATIVE_SCALE_FLOOR_RATIO:
        return None
    return abs(scale - 1.0)


def _reference_translation_magnitude(phase: dict[str, Any]) -> float | None:
    reference = phase.get("reference") if isinstance(phase.get("reference"), dict) else {}
    magnitude = _nested_number(reference, "translation_field", "magnitude_pixels")
    if magnitude is None or magnitude < RELATIVE_TRANSLATION_FLOOR_PIXELS:
        return None
    return magnitude


def _ranked_findings(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Summarise every over-threshold geometry signal, most severe first."""
    scored = [
        item
        for item in state.get("history", [])
        if isinstance(item, dict)
        and item.get("hypothesis_category") != "baseline"
        and isinstance(item.get("metrics"), dict)
    ]
    if not scored:
        return []
    latest = max(scored, key=lambda item: int(item.get("iteration", -1)))
    strongest: dict[str, dict[str, Any]] = {}
    for finding in _magnitude_findings(latest["metrics"]):
        focus = str(finding["focus"])
        previous = strongest.get(focus)
        if previous is None or float(finding["severity"]) > float(previous["severity"]):
            strongest[focus] = {
                "focus": focus,
                "severity": finding["severity"],
                "reason": finding["reason"],
                "observed": finding.get("observed"),
            }
    return sorted(strongest.values(), key=lambda finding: -float(finding["severity"]))


def _reference_curves(state: dict[str, Any]) -> dict[str, Any]:
    """Per-frame reference targets alongside the candidate's own measurements.

    These are absolute, physically meaningful quantities - visible body scale from
    coverage, body centre, and total turn - so a hypothesis can fit the reference's
    actual curve rather than guess a direction and wait for the next score.
    """
    scored = [
        item
        for item in state.get("history", [])
        if isinstance(item, dict)
        and item.get("hypothesis_category") != "baseline"
        and isinstance(item.get("metrics"), dict)
    ]
    if not scored:
        return {"status": "unavailable", "reason": "no evaluated iteration yet"}
    metrics = max(scored, key=lambda item: int(item.get("iteration", -1)))["metrics"]
    centroid = metrics.get("salient_centroid_tracking")
    rotation = metrics.get("salient_rotation_tracking")
    rows: dict[int, dict[str, Any]] = {}
    if isinstance(centroid, dict):
        for sample in centroid.get("samples") or []:
            if not isinstance(sample, dict) or sample.get("status") != "evaluated":
                continue
            frame = sample.get("frame_index")
            if not isinstance(frame, int):
                continue
            row = rows.setdefault(frame, {"frame_index": frame})
            for side in ("reference", "candidate"):
                body = sample.get(side) or {}
                coverage = body.get("coverage")
                if isinstance(coverage, (int, float)) and coverage > 0:
                    # A uniformly scaled full-frame body covers scale^2 of the view.
                    row[f"{side}_scale"] = float(coverage) ** 0.5
                for key, target in (("centroid_x", "centre_x"), ("centroid_y", "centre_y")):
                    value = body.get(key)
                    if isinstance(value, (int, float)):
                        row[f"{side}_{target}"] = float(value)
    if isinstance(rotation, dict):
        for sample in rotation.get("samples") or []:
            if not isinstance(sample, dict):
                continue
            frame = sample.get("frame_index")
            if not isinstance(frame, int):
                continue
            row = rows.setdefault(frame, {"frame_index": frame})
            for side in ("reference", "candidate"):
                fit = sample.get(side) or {}
                angle = fit.get("unwrapped_degrees")
                if isinstance(angle, (int, float)):
                    row[f"{side}_rotation_degrees"] = float(angle)
                # Measured body scale, which unlike sqrt(coverage) is not
                # confounded by frame-edge clipping of a rotated body.
                measured_scale = fit.get("scale")
                if isinstance(measured_scale, (int, float)):
                    row[f"{side}_measured_scale"] = float(measured_scale)
    if not rows:
        return {"status": "unavailable", "reason": "no per-frame reference measurements available"}
    return {
        "status": "available",
        "note": (
            "reference_* columns are the measured target; candidate_* are this candidate's "
            "current values. scale is sqrt(visible coverage) and is only a single-body proxy "
            "while one body is visible. rotation is total turn, not a per-frame rate."
        ),
        "frames": [rows[frame] for frame in sorted(rows)],
    }


def _pivot_is_conditioned(transform: dict[str, Any]) -> bool:
    """Report whether a transform is far enough from identity to place a pivot.

    Near identity the pivot solve is ill-conditioned and its output is dominated
    by noise, so pivot error must not be allowed to outrank real rotation or
    scale disagreement.
    """
    rotation = _nested_number(transform, "rotation_field", "mean_degrees")
    scale = _nested_number(transform, "radial_scale_field", "mean_ratio")
    if rotation is not None and abs(rotation) >= PIVOT_CONDITIONING_MIN_ROTATION_DEGREES:
        return True
    return scale is not None and abs(scale - 1.0) >= PIVOT_CONDITIONING_MIN_SCALE_DELTA


def _consecutive_tradeoffs(state: dict[str, Any], first_iteration: int | None = None) -> int:
    """Count tradeoffs since the most recent accepted iteration in this phase."""
    evaluated = [
        item
        for item in state.get("history", [])
        if isinstance(item, dict)
        and item.get("status") in {"accepted", "rejected", "tradeoff"}
        and item.get("hypothesis_category") != "baseline"
        and isinstance(item.get("iteration"), int)
        and (first_iteration is None or int(item["iteration"]) >= first_iteration)
    ]
    streak = 0
    for item in sorted(evaluated, key=lambda item: int(item["iteration"]), reverse=True):
        if item["status"] != "tradeoff":
            break
        streak += 1
    return streak


def _blocked_categories(state: dict[str, Any], first_iteration: int | None = None) -> list[str]:
    blocked: list[str] = []
    for category in HYPOTHESIS_CATEGORIES:
        rejected = sum(
            1
            for item in state["history"]
            if item.get("hypothesis_category") == category
            and item.get("status") == "rejected"
            and (first_iteration is None or int(item.get("iteration", 0)) >= first_iteration)
        )
        if rejected >= 3:
            blocked.append(category)
    return blocked


def _latest_report(evaluations_dir: Path) -> Path | None:
    reports = list(evaluations_dir.glob("*/reports/candidate_iteration_report.json"))
    return max(reports, key=lambda path: path.stat().st_mtime) if reports else None


def _latest_video(evaluations_dir: Path) -> Path | None:
    videos = list(evaluations_dir.glob("*/artifacts/rendered_transition.mp4"))
    return max(videos, key=lambda path: path.stat().st_mtime) if videos else None


def _latest_comparison_video(evaluations_dir: Path) -> Path | None:
    videos = list(evaluations_dir.glob("*/artifacts/comparison_transition_window.mp4"))
    return max(videos, key=lambda path: path.stat().st_mtime) if videos else None


def _latest_motion_video(evaluations_dir: Path) -> Path | None:
    videos = list(evaluations_dir.glob("*/artifacts/motion_diagnostics.mp4"))
    return max(videos, key=lambda path: path.stat().st_mtime) if videos else None


def _reference_diagnostics(analysis_file: Path) -> tuple[Path | None, Path | None]:
    sample_dir = analysis_file.parent.parent
    diagnostics_file = sample_dir / "diagnostics" / "reference_motion_diagnostics.json"
    if not diagnostics_file.exists():
        return None, None
    try:
        diagnostics = load_json(diagnostics_file)
    except (OSError, ValueError):
        return diagnostics_file, None
    video = diagnostics.get("video")
    video_file = Path(video["file"]) if isinstance(video, dict) and isinstance(video.get("file"), str) else None
    return diagnostics_file, video_file if video_file and video_file.exists() else None


def _reference_edge_diagnostics(analysis_file: Path) -> Path | None:
    diagnostics_file = analysis_file.parent.parent / "diagnostics" / "edge_content_diagnostics.json"
    return diagnostics_file if diagnostics_file.is_file() else None


def _reference_grid_density_diagnostics(analysis_file: Path) -> Path | None:
    diagnostics_file = analysis_file.parent.parent / "diagnostics" / "grid_density_diagnostics.json"
    return diagnostics_file if diagnostics_file.is_file() else None


def _motion_refinement_priority(state: dict[str, Any]) -> dict[str, Any]:
    """Prioritize motion geometry only when its diagnostics are reliable."""
    scored = [
        item
        for item in state.get("history", [])
        if isinstance(item, dict)
        and item.get("hypothesis_category") != "baseline"
        and isinstance(item.get("metrics"), dict)
    ]
    if not scored:
        return {"level": "normal", "reason": "no prior refinement motion metrics"}
    latest = max(scored, key=lambda item: int(item.get("iteration", -1)))
    rgb_slices = latest["metrics"].get("dense_rgb_slices")
    if isinstance(rgb_slices, dict):
        confidence = rgb_slices.get("confidence")
        composite = rgb_slices.get("dense_rgb_slice_similarity")
        if (
            rgb_slices.get("status") == "estimated"
            and isinstance(confidence, (int, float))
            and confidence >= 0.55
            and isinstance(composite, (int, float))
            and composite < 0.82
        ):
            components = {
                "slice_structure": rgb_slices.get("slice_structure_similarity"),
                "rgb_separation": rgb_slices.get("rgb_separation_similarity"),
                "rgb_slice_coherence": rgb_slices.get("rgb_slice_coherence"),
            }
            available = {
                key: float(value) for key, value in components.items() if isinstance(value, (int, float))
            }
            weakest = min(available, key=available.get) if available else "dense_rgb_slices"
            categories = {
                "slice_structure": ["regions", "shader_structure"],
                "rgb_separation": ["displacement", "shader_structure"],
                "rgb_slice_coherence": ["regions", "displacement", "shader_structure"],
            }.get(weakest, ["regions", "displacement", "shader_structure"])
            return {
                "level": "high",
                "focus": "dense_rgb_slices",
                "reason": "reference-relative scoring identifies dense RGB-slice structure as the dominant deficit",
                "dense_rgb_slices": rgb_slices,
                "weakest_component": weakest,
                "recommended_categories": categories,
            }
    motion = latest["metrics"].get("motion")
    topology = latest["metrics"].get("motion_topology")
    if isinstance(topology, dict) and topology.get("status") == "structural_mismatch":
        evidence_count = topology.get("evidence_pair_count")
        region_match_rate = topology.get("candidate_region_match_rate")
        direction_match_rate = topology.get("direction_match_rate")
        if (
            isinstance(evidence_count, int)
            and evidence_count >= 2
            and isinstance(region_match_rate, (int, float))
            and region_match_rate >= 0.5
            and isinstance(direction_match_rate, (int, float))
            and direction_match_rate <= 0.25
        ):
            return {
                "level": "high",
                "focus": "signed_direction",
                "reason": "candidate broadly matches reference regions but fails their signed motion directions",
                "topology": topology,
                "recommended_categories": ["displacement", "regions", "shader_structure"],
            }
        return {
            "level": "high",
            "focus": "motion_topology",
            "reason": "candidate collapses a required reference motion topology",
            "topology": topology,
            "recommended_categories": ["shader_structure", "regions"],
        }
    angular_motion = latest["metrics"].get("angular_motion")
    if isinstance(angular_motion, dict) and angular_motion.get("status") == "direction_mismatch":
        confidence = angular_motion.get("confidence")
        if isinstance(confidence, (int, float)) and confidence >= 0.35:
            return {
                "level": "high",
                "focus": "angular_direction",
                "reason": "reliable signed angular motion disagrees between the reference and candidate",
                "angular_motion": angular_motion,
                "recommended_categories": ["displacement", "shader_structure"],
            }
    # Magnitude-based signals are ranked by how far past their own threshold they
    # sit, rather than by the order they are tested.  First-match-wins ordering
    # previously let motion_geometry's pivot term pre-empt the rotation and scale
    # tests below it, so a degenerate pivot could hide a large rotation error
    # indefinitely.
    findings = _magnitude_findings(latest["metrics"])
    if findings:
        return max(findings, key=lambda finding: float(finding["severity"]))
    regional_motion = latest["metrics"].get("regional_motion")
    if isinstance(regional_motion, dict) and regional_motion.get("status") == "direction_mismatch":
        return {
            "level": "high",
            "focus": "regional_direction",
            "reason": "candidate regional motion axis or continuous direction disagrees with the reference",
            "regional_motion": regional_motion,
            "recommended_categories": ["displacement", "regions"],
        }
    edge_content = latest["metrics"].get("edge_content_policy")
    if isinstance(edge_content, dict):
        reference_policy = edge_content.get("reference") if isinstance(edge_content.get("reference"), dict) else {}
        candidate_policy = edge_content.get("candidate") if isinstance(edge_content.get("candidate"), dict) else {}
        recommended = reference_policy.get("recommended_policy")
        confidence = reference_policy.get("confidence")
        observed = candidate_policy.get("policy")
        if (
            reference_policy.get("status") == "estimated"
            and isinstance(recommended, str)
            and recommended in {"clamp", "mirror", "repeat"}
            and isinstance(confidence, (int, float))
            and confidence >= 0.7
            and recommended != observed
        ):
            return {
                "level": "high",
                "focus": "edge_content_policy",
                "reason": "reliable source-edge evidence disagrees with the candidate UV edge policy",
                "edge_content_policy": edge_content,
                "recommended_categories": ["uv_mapping", "shader_structure"],
            }
    if not isinstance(motion, dict):
        return {"level": "normal", "reason": "latest evaluation has no motion metrics"}
    coverage = motion.get("reliable_motion_coverage")
    agreement = motion.get("direction_agreement")
    if (
        isinstance(coverage, (int, float))
        and isinstance(agreement, (int, float))
        and coverage >= 0.6
        and agreement < 0.6
    ):
        return {
            "level": "high",
            "focus": "motion_geometry",
            "reason": "reliable motion diagnostics show weak direction agreement",
            "reliable_motion_coverage": float(coverage),
            "direction_agreement": float(agreement),
            "recommended_categories": ["regions", "displacement"],
        }
    return {
        "level": "normal",
        "reason": "motion diagnostics do not currently require geometry-first refinement",
    }


def _evaluation_command(
    profile: dict[str, Any],
    iteration: int,
    analysis_file: Path,
    design_file: Path,
    max_iterations: int,
    max_rejected: int,
) -> str:
    backup_dir = Path(profile["backup_root"]) / f"iteration_{iteration:03d}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    lines = [
        "conda run -n harness python agent/src/main.py candidate-evaluate `",
        f'  --manifest "{profile["manifest"]}" `',
        f'  --job "{profile["job"]}" `',
        f'  --reference "{profile["reference"]}" `',
        f'  --output-root "{profile["output_root"]}" `',
        f'  --backup-dir "{backup_dir}" `',
        f'  --msbuild "{profile["msbuild"]}" `',
        f'  --renderer "{profile["renderer"]}" `',
        f'  --configuration "{profile.get("configuration", "Debug")}" `',
        f'  --platform "{profile.get("platform", "x64")}" `',
        f'  --width {profile["width"]} --height {profile["height"]} `',
        f'  --frame-start {profile["frame_start"]} --frame-end {profile["frame_end"]} `',
        f"  --iteration {iteration} `",
        f'  --continue-analysis "{analysis_file}" `',
        f'  --continue-design "{design_file}" `',
        f"  --continue-max-iterations {max_iterations} `",
        f"  --continue-max-rejected {max_rejected}",
    ]
    if profile.get("calibrate_progress", True):
        lines[-1] += " `"
        lines.append("  --calibrate-progress")
    return "\n".join(lines)


def _continuation_command(
    candidate_manifest_file: Path,
    analysis_file: Path,
    design_file: Path,
    max_iterations: int,
    max_rejected: int,
) -> str:
    return "\n".join(
        [
            "conda run -n harness python agent/src/main.py candidate-continue `",
            f'  --manifest "{candidate_manifest_file}" `',
            f'  --analysis "{analysis_file}" `',
            f'  --design "{design_file}" `',
            f"  --max-iterations {max_iterations} --max-rejected {max_rejected}",
        ]
    )


def _refinement_request(packet: dict[str, Any], candidate_dir: Path) -> str:
    allowed = ", ".join(packet["allowed_hypothesis_categories"])
    prompt_files = packet.get("prompt_files") or ["agent/prompts/agent_effect_refinement_prompt.md"]
    prompt_lines = "\n".join(f"- {path}" for path in prompt_files)
    return f"""Read:
{prompt_lines}
- {packet['analysis_file']}
- {packet['design_file']}
- {packet['packet_file'] if 'packet_file' in packet else 'the iteration packet JSON'}
- {packet['latest_report'] or 'no previous evaluation report'}
- {packet['latest_candidate_video'] or 'no previous candidate video'}
- {packet['latest_comparison_video'] or 'no previous comparison video'}
- {packet['latest_motion_video'] or 'no previous motion diagnostic video'}
- {packet['reference_diagnostics_file'] or 'no reference motion diagnostics'}
- {packet['reference_diagnostics_video'] or 'no reference motion diagnostic video'}
- {packet['reference_edge_diagnostics_file'] or 'no reference edge-content diagnostics'}
- {packet.get('reference_grid_density_diagnostics_file') or 'no reference grid-density diagnostics'}

Edit only:
{candidate_dir}

Refine {packet['effect_id']} for iteration {packet['iteration']}.

{_refinement_priority_instruction(packet.get('refinement_priority'), Path(packet['analysis_file']) if packet.get('analysis_file') else None)}
{_findings_instruction(packet.get('refinement_findings'))}
{_reference_curve_instruction(packet.get('reference_curves'))}
{_escalation_instruction(packet.get('escalation'))}
Choose exactly one hypothesis category from: {allowed}.
Do not repeat a rejected category unless you provide new visual evidence.
Preserve the FX ID, class names, endpoint behavior, and candidate workspace boundary.
Create or update exactly one iteration_{packet['iteration']:03d}_*.json record with
`hypothesis_category`, `visual_hypothesis`, `changed_files`, and expected outcome.
{_evaluation_instruction(packet)}
"""


def _findings_instruction(findings: Any) -> str:
    """List the competing signals so a lower-ranked one can still be chosen."""
    if not isinstance(findings, list) or len(findings) < 2:
        return ""
    lines = "\n".join(
        f"- {finding.get('focus')}: severity {float(finding.get('severity', 0.0)):.1f}x its threshold"
        + (f" ({finding.get('observed')})" if finding.get("observed") else "")
        for finding in findings
        if isinstance(finding, dict)
    )
    return (
        "\nAll over-threshold geometry signals this iteration, most severe first:\n"
        f"{lines}\n"
        "Severity is each value divided by its own threshold, and those thresholds are not equally calibrated "
        "across pixels, degrees and ratios, so the top entry is a suggestion rather than a verdict. If a "
        "lower-ranked signal is a larger error relative to the reference's own magnitude, address that instead "
        "and say why.\n"
    )


def _reference_curve_instruction(curves: Any) -> str:
    """Point the iteration at the measured target curve, so it can fit rather than guess."""
    if not isinstance(curves, dict) or curves.get("status") != "available":
        return ""
    frames = curves.get("frames")
    if not isinstance(frames, list) or not frames:
        return ""
    columns = [
        ("reference_scale", "candidate_scale", "scale"),
        ("reference_centre_x", "candidate_centre_x", "centre_x"),
        ("reference_rotation_degrees", "candidate_rotation_degrees", "turn_deg"),
    ]
    header = "frame | " + " | ".join(
        f"{label:>18}" for _, _, label in columns
    )
    lines = [header, "-" * len(header)]
    for row in frames:
        cells = []
        for reference_key, candidate_key, _ in columns:
            reference_value = row.get(reference_key)
            candidate_value = row.get(candidate_key)
            if isinstance(reference_value, (int, float)) and isinstance(candidate_value, (int, float)):
                cells.append(f"{reference_value:8.3f}/{candidate_value:<9.3f}")
            elif isinstance(reference_value, (int, float)):
                cells.append(f"{reference_value:8.3f}/{'-':<9}")
            else:
                cells.append(f"{'-':>8}/{'-':<9}")
        lines.append(f"{row.get('frame_index'):>5} | " + " | ".join(cells))
    table = "\n".join(lines)
    return (
        "\nMeasured reference targets, shown as reference/candidate per frame:\n"
        f"{table}\n"
        f"{curves.get('note', '')}\n"
        "Fit the shader's curve to these measurements - solve for the magnitude and easing shape that "
        "reproduces the reference column - rather than scaling an existing constant by a percentage and "
        "waiting for the next score. If a column shows the candidate easing the wrong way (the reference "
        "accelerating while the candidate decelerates, or vice versa), the exponent is wrong, not just the "
        "magnitude.\n"
    )


def _escalation_instruction(escalation: Any) -> str:
    """Demand a structural re-derivation once nudging has plainly stalled."""
    if not isinstance(escalation, dict):
        return ""
    streak = escalation.get("consecutive_tradeoffs")
    if not isinstance(streak, int) or streak <= 0:
        return ""
    if not escalation.get("structural_review_required"):
        return (
            f"\nThis phase has {streak} consecutive tradeoff outcome(s). Each one improved a metric while "
            "regressing another, so the current lever may be near its useful limit. Prefer a hypothesis that "
            "changes the model rather than one that rescales the same constant.\n"
        )
    return (
        f"\nSTRUCTURAL REVIEW REQUIRED: {streak} consecutive tradeoffs, at or past the limit of "
        f"{escalation.get('limit')}. Do not tune another constant this iteration. Every hypothesis has been "
        "moving one metric while regressing another, which is the signature of optimising inside measurement "
        "noise. Instead do one of the following and say which you chose:\n"
        "- Re-derive a curve from the reference itself (measure the quantity per frame and fit it) rather than "
        "adjusting a coefficient by a percentage.\n"
        "- Question the analysis and design artifacts: check whether a `must_preserve` claim or the transition "
        "summary is contradicted by measurement, and report the contradiction instead of coding against it.\n"
        "- Replace the model for the failing property (its easing shape, its pivot/translation parameterisation, "
        "or the number of bodies composited) rather than its magnitude.\n"
        "- If a selection metric itself looks degenerate or unit-ambiguous, say so and propose the measurement "
        "that would settle it.\n"
    )


def _select_prompt_files(analysis_file: Path, include_edge_diagnostics: bool) -> list[str]:
    """Select only the compact prompt modules relevant to this transition."""
    files = [
        "agent/prompts/agent_effect_refinement_prompt.md",
        "agent/prompts/base/refinement_contract.md",
        "agent/prompts/diagnostics/motion_geometry.md",
        "agent/prompts/diagnostics/optical_flow.md",
    ]
    try:
        analysis = load_json(analysis_file)
    except (OSError, ValueError):
        analysis = {}
    transition = analysis.get("transition") if isinstance(analysis, dict) else {}
    hints = analysis.get("planner_hints") if isinstance(analysis, dict) else {}
    text = " ".join(
        str(value).lower()
        for value in (
            transition.get("structure_type", "") if isinstance(transition, dict) else "",
            transition.get("summary", "") if isinstance(transition, dict) else "",
            hints.get("recommended_effect_family", "") if isinstance(hints, dict) else "",
        )
    )
    if any(token in text for token in ("rotation", "scale", "affine", "rotate", "flip")):
        files.append("agent/prompts/families/affine_transform.md")
    elif any(token in text for token in ("slide", "wipe", "split", "region", "band")):
        files.append("agent/prompts/families/segmented_motion.md")
    else:
        files.append("agent/prompts/families/general_motion.md")
    if include_edge_diagnostics:
        files.append("agent/prompts/diagnostics/edge_content_policy.md")
    return files


_CONSENSUS_FOCUS_KEYWORDS = {
    "motion_topology": ("motion_axes", "region", "direction"),
    "signed_direction": ("motion_axes", "direction"),
    "regional_direction": ("motion_axes", "direction", "region"),
    "angular_direction": ("motion_axes", "direction", "rotation"),
    "dense_rgb_slices": ("rgb", "slice"),
}


def _load_cross_sample_consensus(analysis_file: Path | None) -> dict[str, Any] | None:
    """Load the sibling cross_sample_consensus.json for this candidate's analysis, if any.

    Most candidates are built from a single sample and have no consensus file;
    this returns None in that case rather than requiring one.
    """
    if analysis_file is None:
        return None
    consensus_file = Path(analysis_file).parent / "cross_sample_consensus.json"
    if not consensus_file.exists():
        return None
    try:
        payload = load_json(consensus_file)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("artifact_type") != "cross_sample_consensus":
        return None
    return payload


def _consensus_annotation(priority: Any, consensus: dict[str, Any] | None) -> str:
    """Cross-check the current refinement focus against the multi-sample consensus.

    Purely advisory text appended to the generated request - it never changes
    acceptance mechanics, only tells whoever is choosing a hypothesis whether
    the property they are about to spend an iteration on is a real, agreed
    effect trait or a sample-specific quirk of the primary sample alone.
    """
    if not consensus or not isinstance(priority, dict):
        return ""
    keywords = _CONSENSUS_FOCUS_KEYWORDS.get(priority.get("focus"))
    if not keywords:
        return ""
    sample_count = consensus.get("sample_count", "?")
    convergent = consensus.get("convergent") if isinstance(consensus.get("convergent"), dict) else {}
    divergent = consensus.get("divergent") if isinstance(consensus.get("divergent"), dict) else {}

    def _matches(key: str) -> bool:
        lowered = key.lower()
        return any(keyword in lowered for keyword in keywords)

    convergent_hits = [key for key in convergent if _matches(key)]
    divergent_hits = [key for key in divergent if _matches(key)]
    if not convergent_hits and not divergent_hits:
        return ""

    parts = []
    if divergent_hits:
        named = ", ".join(f"`{key}`" for key in divergent_hits[:3])
        parts.append(
            f"the cross-sample consensus ({sample_count} samples) marks {named} as sample-specific (not agreed "
            "across every sample) - further precision here has limited payoff for generalizing this effect; weigh "
            "that against spending the remaining iteration budget on a convergent property instead"
        )
    if convergent_hits:
        named = ", ".join(f"`{key}`" for key in convergent_hits[:3])
        parts.append(
            f"the consensus marks {named} as convergent (agreed across all {sample_count} samples) - this is a "
            "real effect property, not sample noise, and safe to invest iteration budget in"
        )
    return "\nCross-sample consensus note: " + "; ".join(parts) + "."


def _refinement_priority_instruction(priority: Any, analysis_file: Path | None = None) -> str:
    text = _refinement_priority_base_text(priority)
    consensus = _load_cross_sample_consensus(analysis_file)
    return text + _consensus_annotation(priority, consensus)


def _refinement_priority_base_text(priority: Any) -> str:
    if not isinstance(priority, dict) or priority.get("level") != "high":
        return (
            "Current refinement priority: normal. Read motion_geometry before choosing a hypothesis. If translation, "
            "pivot, rotation sign, scale, or reflection disagrees with adequate confidence, change the corresponding "
            "transform/displacement source code before using timing, blur, or blend."
        )
    categories = ", ".join(str(category) for category in priority.get("recommended_categories", []))
    if priority.get("focus") == "motion_topology":
        topology = priority.get("topology") if isinstance(priority.get("topology"), dict) else {}
        evidence_count = topology.get("evidence_pair_count", 0)
        region_match_rate = topology.get("candidate_region_match_rate", 0.0)
        direction_match_rate = topology.get("direction_match_rate", 0.0)
        return (
            "Current refinement priority: high motion topology. The reference requires multiple spatial direction groups, "
            f"but the candidate does not satisfy that contract across {evidence_count} evidence pairs "
            f"(region-topology match {region_match_rate:.3f}, direction match {direction_match_rate:.3f}). "
            "First model the groups with signed displacement and straight-line partitions; use an arbitrary region mask "
            "only when repeated reference evidence rules out a piecewise-linear partition."
        )
    if priority.get("focus") == "dense_rgb_slices":
        diagnostic = priority.get("dense_rgb_slices") if isinstance(priority.get("dense_rgb_slices"), dict) else {}
        return (
            "Current refinement priority: dense RGB-separated slice fidelity. The reference-relative composite is "
            f"{diagnostic.get('dense_rgb_slice_similarity', 'unknown')} with confidence "
            f"{diagnostic.get('confidence', 'unknown')}; its weakest component is "
            f"`{priority.get('weakest_component', 'unknown')}`. Prefer one of: {categories}. "
            "Match the reference slice density, RGB offset magnitude, and their spatial overlap; do not maximize "
            "noise, slice count, or chromatic offset beyond the measured target."
        )
    if priority.get("focus") == "signed_direction":
        topology = priority.get("topology") if isinstance(priority.get("topology"), dict) else {}
        evidence_count = topology.get("evidence_pair_count", 0)
        region_match_rate = topology.get("candidate_region_match_rate", 0.0)
        direction_match_rate = topology.get("direction_match_rate", 0.0)
        return (
            "Current refinement priority: high signed direction. The candidate already broadly matches the reference "
            f"region layout across {evidence_count} reliable evidence pairs (region-topology match {region_match_rate:.3f}), "
            f"but its direction match is {direction_match_rate:.3f}. Correct each reliable region's signed displacement "
            "and motion axis before changing region boundaries, blur, blend, or sampler behavior. Use the per-pair "
            "reference and candidate vectors in the report; do not replace continuous vectors with fixed direction buckets. "
            "For segmented motion, retain the simplest straight-line partition until repeated evidence requires a more complex boundary."
        )
    if priority.get("focus") == "motion_geometry":
        geometry = priority.get("geometry") if isinstance(priority.get("geometry"), dict) else {}
        rotation_delta = geometry.get("rotation_delta_degrees", "unknown")
        scale_delta = geometry.get("scale_delta_ratio", "unknown")
        return (
            "Current refinement priority: high motion geometry. The candidate and reference transformation estimates "
            f"differ by {rotation_delta} degrees of rotation and {scale_delta} scale ratio. "
            "Inspect rotation, scale, reflection, and spatial-displacement evidence before tuning blur or blend."
        )
    if priority.get("focus") == "transform_rotation":
        observed = priority.get("observed") if isinstance(priority.get("observed"), dict) else {}
        return (
            "Current refinement priority: transform rotation. The worst phase disagrees with the reference by "
            f"{observed.get('rotation_delta_degrees', 'unknown')} degrees against a {observed.get('limit', 'unknown')} "
            "degree limit. Note that this figure is a PER-FRAME-PAIR rate, not a total turn, so it accumulates over "
            "the phase: a few degrees per pair is tens or hundreds of degrees of accumulated error. Check the total "
            "rotation the reference body reaches, not only its rate, and correct the rotation magnitude and its easing "
            "shape before spending the iteration on position, blur, or blend."
        )
    if priority.get("focus") == "transform_scale":
        observed = priority.get("observed") if isinstance(priority.get("observed"), dict) else {}
        return (
            "Current refinement priority: transform scale. The worst phase disagrees with the reference by a "
            f"{observed.get('scale_delta_ratio', 'unknown')} scale ratio against a {observed.get('limit', 'unknown')} "
            "limit. Compare the body's visible extent per frame against the reference, including whether it should "
            "reach zero at all, and correct the scale curve's floor and easing shape before tuning position or timing."
        )
    if priority.get("focus") == "transform_position":
        geometry = priority.get("geometry") if isinstance(priority.get("geometry"), dict) else {}
        return (
            "Current refinement priority: transform position. The candidate's translation error is "
            f"{geometry.get('translation_delta_pixels', 'unknown')} pixels. Inspect the transform pivot, signed "
            "translation vector, and per-region origins. Use `displacement` when the transform structure exists; "
            "use `shader_structure` only when the pivot/translation model is missing. Do not spend this iteration "
            "on timing, blur, or blend."
        )
    if priority.get("focus") == "angular_direction":
        angular = priority.get("angular_motion") if isinstance(priority.get("angular_motion"), dict) else {}
        reference = angular.get("reference") if isinstance(angular.get("reference"), dict) else {}
        candidate = angular.get("candidate") if isinstance(angular.get("candidate"), dict) else {}
        return (
            "Current refinement priority: high signed angular direction. The reference indicates "
            f"`{reference.get('direction', 'indeterminate')}` rotation while the candidate indicates "
            f"`{candidate.get('direction', 'indeterminate')}` (confidence {angular.get('confidence', 'unknown')}). "
            "Reverse the sign of an existing centered rotation transform under `displacement`; choose "
            "`shader_structure` only if the pivot or rotation transform is missing. Do not change blur, blend, "
            "or region masks until the signed rotation direction is correct."
        )
    if priority.get("focus") == "foreground_body_transform":
        geometry = priority.get("geometry") if isinstance(priority.get("geometry"), dict) else {}
        phases = geometry.get("phases") if isinstance(geometry.get("phases"), dict) else {}
        return (
            "Current refinement priority: foreground-body affine transform. The feature-tracked visible body has a "
            "phase-specific transform comparison. Read the outgoing, midpoint, and incoming values in `phases`; "
            "do not average opposed phases into one transform. Use the per-phase body evidence to correct "
            "the 2D pivot, signed angle, scale, or translation in shader code. Do not treat this as a 3D problem, "
            "and do not change blur or blend first."
        )
    if priority.get("focus") == "regional_direction":
        regional = priority.get("regional_motion") if isinstance(priority.get("regional_motion"), dict) else {}
        return (
            "Current refinement priority: high regional direction. The candidate regional motion differs from the "
            f"reference by {regional.get('direction_delta_degrees', 'unknown')} degrees and has axis agreement "
            f"{regional.get('axis_agreement', 'unknown')}. Preserve continuous signed regional vectors; do not "
            "replace them with fixed four- or eight-direction buckets."
        )
    if priority.get("focus") == "edge_content_policy":
        edge_content = priority.get("edge_content_policy") if isinstance(priority.get("edge_content_policy"), dict) else {}
        reference = edge_content.get("reference") if isinstance(edge_content.get("reference"), dict) else {}
        candidate = edge_content.get("candidate") if isinstance(edge_content.get("candidate"), dict) else {}
        return (
            "Current refinement priority: high edge-content policy. Reference source-edge evidence supports "
            f"`{reference.get('recommended_policy', 'unknown')}` with confidence {reference.get('confidence', 'unknown')}; "
            f"the candidate currently appears to use `{candidate.get('policy', 'unknown')}`. Verify the rendered edge "
            "evidence, then implement an explicit UV mapping policy. Do not change direction, line partitions, blur, or blend."
        )
    coverage = priority.get("reliable_motion_coverage")
    agreement = priority.get("direction_agreement")
    return (
        "Current refinement priority: high motion geometry. "
        f"Reliable motion coverage is {coverage:.3f} and direction agreement is {agreement:.3f}. "
        f"Investigate {categories} before blur or blend unless direct visual evidence rules out a direction or region mismatch."
    )


def _evaluation_instruction(packet: dict[str, Any]) -> str:
    command = packet.get("evaluation_command")
    if not isinstance(command, str):
        return "Do not run evaluation; the controller will run it after the edit."
    continuation = packet.get("continuation_command")
    continuation_instruction = "Read the resulting controller outcome and stop."
    if isinstance(continuation, str):
        continuation_instruction = f"""A completed evaluation is not a failure merely because its score is poor:

- `accepted`, `rejected`, and `tradeoff` are all completed controller outcomes.
- Progress calibration and its linear probe are normal evaluation stages, not failures.
- Do not stop, restore sources yourself, or start a new phase after any of those outcomes.

This evaluation command already invokes controller continuation after `accepted`, `rejected`, or `tradeoff`. Do not run the continuation command a second time. Read its result, then read and execute the newly generated request when the parent goal asks for a bounded run.

For reference, the continuation performed by the command is:

```powershell
{continuation}
```

The continuation command restores the selected baseline when required and creates the next agent request. If the parent request asks for a bounded multi-iteration refinement run, read and execute that next request in the same agent session. Otherwise, stop after the continuation command returns."""
    build_failure_instruction = f"""

If `candidate-evaluate` fails during the build for iteration {packet['iteration']}, do not start a new phase or skip the iteration. Read the generated `iteration_{packet['iteration']:03d}_build_repair_*.md` request, repair the compilation issue in the candidate workspace, and rerun `candidate-evaluate` for iteration {packet['iteration']} using a new backup directory. The controller restores registered target files automatically; keep the existing FX ID."""
    return f"""After editing, run exactly this evaluation command:

```powershell
{command}
```

{continuation_instruction}
{build_failure_instruction}"""


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
