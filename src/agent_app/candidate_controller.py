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
    }
    _upsert_history(state, iteration, "baseline", "accepted", metrics, str(report_file))
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
    phase = {
        "name": name,
        "baseline_iteration": baseline_iteration,
        "first_iteration": first_iteration,
        "max_iterations": max_iterations,
        "max_rejected": max_rejected,
        "started_at": _timestamp(),
        "status": "active",
    }
    state.setdefault("phases", [])
    state["phases"] = [item for item in state["phases"] if item.get("name") != name]
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


def restore_candidate_baseline(candidate_manifest_file: Path) -> dict[str, Any]:
    state = _load_or_create_state(candidate_manifest_file)
    baseline = state.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("candidate has no selected baseline")
    snapshot_dir = Path(str(baseline.get("source_snapshot", "")))
    if not snapshot_dir.is_dir():
        raise FileNotFoundError("selected baseline has no source snapshot")
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
            raise ValueError(f"phase iteration budget exhausted: {attempted_count} reaches {phase_max_iterations}")
        if rejected_count >= phase_max_rejected:
            raise ValueError(f"phase rejected-iteration budget exhausted: {rejected_count} reaches {phase_max_rejected}")
        blocked_categories = _blocked_categories(state, first_iteration)
        budget = {
            "phase": phase["name"],
            "max_iterations": phase_max_iterations,
            "max_rejected": phase_max_rejected,
            "attempted_so_far": attempted_count,
            "rejected_so_far": rejected_count,
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
        budget = {
            "max_iterations": max_iterations,
            "max_rejected": max_rejected,
            "rejected_so_far": rejected_count,
        }
    latest_report = _latest_report(candidate_dir / "evaluations")
    latest_video = _latest_video(candidate_dir / "evaluations")
    latest_comparison = _latest_comparison_video(candidate_dir / "evaluations")
    latest_motion = _latest_motion_video(candidate_dir / "evaluations")
    reference_diagnostics, reference_diagnostic_video = _reference_diagnostics(analysis_file)
    packet_file = candidate_dir / "packets" / f"iteration_{next_iteration:03d}_packet.json"
    prompt_file = candidate_dir / "packets" / f"iteration_{next_iteration:03d}_codex_request.md"
    candidate = load_json(candidate_manifest_file)
    evaluation_command = None
    continuation_command = None
    if evaluate_after_edit:
        profile = state.get("evaluation_profile")
        if not isinstance(profile, dict):
            raise ValueError("candidate has no evaluation profile; run candidate-set-evaluation-profile first")
        evaluation_command = _evaluation_command(profile, next_iteration)
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
        "baseline": state["baseline"],
        "history": state["history"],
        "shortlist": state["shortlist"],
        "active_phase": phase,
        "allowed_hypothesis_categories": [
            category for category in HYPOTHESIS_CATEGORIES if category not in blocked_categories
        ],
        "blocked_hypothesis_categories": blocked_categories,
        "budgets": state["budgets"],
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
    outcome, reason = _select_outcome(state.get("baseline"), metrics)
    record["evaluation"] = metrics
    record["status"] = outcome
    record["reason"] = reason
    write_json(iteration_file, record)
    _upsert_history(state, iteration, category, outcome, metrics, str(report_file))
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
        }
    _refresh_rejected_budget(state)
    _write_state(candidate_manifest_file, state)
    return {"status": outcome, "reason": reason, "state_file": str(_state_file(candidate_manifest_file))}


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
    return metrics


def _number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"evaluation report has invalid {key}")
    return float(value)


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
    if not _endpoints_are_exact(metrics):
        return "rejected", "endpoint checks exceed stable-frame tolerance"
    if baseline is None:
        return "accepted", "first valid evaluation becomes the baseline"
    previous = baseline["metrics"]
    ssim_delta = metrics["ssim"] - previous["ssim"]
    mse_change = (metrics["mse"] - previous["mse"]) / previous["mse"]
    motion_delta = _motion_delta(metrics, previous)

    materially_improved = (
        ssim_delta >= SSIM_IMPROVEMENT
        or mse_change <= -MSE_IMPROVEMENT_RATIO
        or (motion_delta is not None and motion_delta >= MOTION_SIMILARITY_IMPROVEMENT)
    )
    within_guardrails = (
        ssim_delta >= -SSIM_REGRESSION_TOLERANCE
        and mse_change <= MSE_REGRESSION_TOLERANCE
        and (
            motion_delta is None
            or motion_delta >= -MOTION_SIMILARITY_REGRESSION_TOLERANCE
        )
    )
    if materially_improved and within_guardrails:
        return "accepted", "improved a primary image or motion metric within Pareto guardrails"

    any_improved = (
        ssim_delta > 0
        or mse_change < 0
        or (motion_delta is not None and motion_delta > 0)
    )
    if any_improved:
        return "tradeoff", "one image or motion metric improved while another regressed"
    return "rejected", "image and motion metrics regressed against the accepted baseline"


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
) -> None:
    item = {
        "iteration": iteration,
        "hypothesis_category": category,
        "status": status,
        "metrics": metrics,
        "report_file": report_file,
        "recorded_at": _timestamp(),
    }
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


def _evaluation_command(profile: dict[str, Any], iteration: int) -> str:
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
        f"  --iteration {iteration}",
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
    return f"""Read:
- agent/prompts/codex_effect_refinement_prompt.md
- {packet['analysis_file']}
- {packet['design_file']}
- {packet['packet_file'] if 'packet_file' in packet else 'the iteration packet JSON'}
- {packet['latest_report'] or 'no previous evaluation report'}
- {packet['latest_candidate_video'] or 'no previous candidate video'}
- {packet['latest_comparison_video'] or 'no previous comparison video'}
- {packet['latest_motion_video'] or 'no previous motion diagnostic video'}
- {packet['reference_diagnostics_file'] or 'no reference motion diagnostics'}
- {packet['reference_diagnostics_video'] or 'no reference motion diagnostic video'}

Edit only:
{candidate_dir}

Refine {packet['effect_id']} for iteration {packet['iteration']}.

Choose exactly one hypothesis category from: {allowed}.
Do not repeat a rejected category unless you provide new visual evidence.
Preserve the FX ID, class names, endpoint behavior, and candidate workspace boundary.
Create or update exactly one iteration_{packet['iteration']:03d}_*.json record with
`hypothesis_category`, `visual_hypothesis`, `changed_files`, and expected outcome.
{_evaluation_instruction(packet)}
"""


def _evaluation_instruction(packet: dict[str, Any]) -> str:
    command = packet.get("evaluation_command")
    if not isinstance(command, str):
        return "Do not run evaluation; the controller will run it after the edit."
    continuation = packet.get("continuation_command")
    continuation_instruction = "Read the resulting controller outcome and stop."
    if isinstance(continuation, str):
        continuation_instruction = f"""Read the resulting controller outcome. If it is `accepted`, `rejected`, or `tradeoff`, run exactly this continuation command:

```powershell
{continuation}
```

The continuation command restores the selected baseline when required and creates the next Codex request. Stop after it returns; do not edit the next iteration."""
    return f"""After editing, run exactly this evaluation command:

```powershell
{command}
```

{continuation_instruction}"""


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
