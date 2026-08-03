from __future__ import annotations

from datetime import UTC, datetime
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from .harness_bridge import load_harness_modules
from .io import load_json, write_json
from .artifacts import build_render_job
from .candidate_controller import record_candidate_evaluation


def prepare_reference(
    workspace_root: Path,
    source_video: Path,
    output_dir: Path,
    fps: int,
    width: int,
    height: int,
    target_frame_count: int,
    ffmpeg_path: str | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> dict[str, Any]:
    modules = load_harness_modules(workspace_root)
    result = modules["prepare_reference_transition"](
        source_video=source_video,
        output_dir=output_dir,
        fps=fps,
        width=width,
        height=height,
        target_frame_count=target_frame_count,
        ffmpeg_path=ffmpeg_path,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    return {
        "status": "succeeded",
        "message": result.message,
        "output_dir": str(result.output_dir),
        "manifest_file": str(result.manifest_file),
        "frame_count": result.frame_count,
        "detected_start_frame": result.detected_start_frame,
        "detected_end_frame": result.detected_end_frame,
        "detected_frame_count": result.detected_frame_count,
    }


def prepare_sources(
    source_video: Path,
    output_root: Path,
    start_frame: int | None,
    end_frame: int | None,
    frame_count: int,
    width: int,
    height: int,
    ffmpeg_path: str | None = None,
    analysis_file: Path | None = None,
    reference_manifest_file: Path | None = None,
) -> dict[str, Any]:
    if (
        analysis_file is None
        and reference_manifest_file is None
        and start_frame is not None
        and end_frame is not None
        and (start_frame < 0 or end_frame < 0 or end_frame < start_frame)
    ):
        raise ValueError("source frame boundaries must be non-negative and ordered")
    if not source_video.exists():
        raise FileNotFoundError(f"source video does not exist: {source_video}")
    ffmpeg_executable = ffmpeg_path or shutil.which("ffmpeg")
    if not ffmpeg_executable:
        raise RuntimeError("ffmpeg is required for source preparation but was not found on PATH")

    selection: dict[str, Any]
    if analysis_file is not None or reference_manifest_file is not None:
        if analysis_file is None or reference_manifest_file is None:
            raise ValueError("analysis and reference manifest are both required for mapped source boundaries")
        if start_frame is not None or end_frame is not None:
            raise ValueError("mapped source boundaries cannot be combined with --start-frame or --end-frame")
        selection = resolve_source_boundaries(analysis_file, reference_manifest_file)
        start_frame = int(selection["source_a_frame"])
        end_frame = int(selection["source_b_frame"])
    elif start_frame is None and end_frame is None:
        source_video_frame_count = _probe_video_frame_count(ffmpeg_executable, source_video)
        start_frame = 0
        end_frame = source_video_frame_count - 1
        selection = {
            "mode": "video_endpoints",
            "source_video_frame_count": source_video_frame_count,
        }
    else:
        if start_frame is None or end_frame is None:
            raise ValueError("original-video source selection requires both --start-frame and --end-frame")
        selection = {"mode": "original_video_frames"}

    if start_frame < 0 or end_frame < 0 or end_frame < start_frame:
        raise ValueError("source frame boundaries must be non-negative and ordered")
    if frame_count < 2:
        raise ValueError("frame_count must be at least 2")
    output_root.mkdir(parents=True, exist_ok=True)
    source_a_dir = output_root / "source_a"
    source_b_dir = output_root / "source_b"
    source_a_dir.mkdir(parents=True, exist_ok=True)
    source_b_dir.mkdir(parents=True, exist_ok=True)
    _clear_png_frames(source_a_dir)
    _clear_png_frames(source_b_dir)

    source_a_frame = output_root / "source_a_frame.png"
    source_b_frame = output_root / "source_b_frame.png"
    _extract_single_frame(
        ffmpeg_executable, source_video, start_frame, source_a_frame, width, height
    )
    _extract_single_frame(
        ffmpeg_executable, source_video, end_frame, source_b_frame, width, height
    )
    for index in range(frame_count):
        shutil.copyfile(source_a_frame, source_a_dir / f"frame_{index:04d}.png")
        shutil.copyfile(source_b_frame, source_b_dir / f"frame_{index:04d}.png")

    manifest = {
        "artifact_type": "source_pair",
        "artifact_version": 1,
        "source_video": str(source_video),
        "fps_assumption": "source frame indexes refer to normalized video frames",
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "source_a_frame": start_frame,
        "source_b_frame": end_frame,
        "source_a": str(source_a_dir),
        "source_b": str(source_b_dir),
        "ffmpeg": ffmpeg_executable,
        "selection": selection,
    }
    manifest_file = output_root / "source_pair_manifest.json"
    write_json(manifest_file, manifest)
    source_a_frame.unlink(missing_ok=True)
    source_b_frame.unlink(missing_ok=True)
    return {
        "status": "succeeded",
        "output_root": str(output_root),
        "manifest_file": str(manifest_file),
        "source_a": str(source_a_dir),
        "source_b": str(source_b_dir),
        "frame_count": frame_count,
        "source_a_frame": start_frame,
        "source_b_frame": end_frame,
        "selection": selection,
    }


def resolve_source_boundaries(analysis_file: Path, reference_manifest_file: Path) -> dict[str, Any]:
    """Map prepared-reference stable boundaries back to original source-video frames."""
    analysis = load_json(analysis_file)
    transition = analysis.get("transition")
    if not isinstance(transition, dict):
        raise ValueError("analysis artifact is missing transition")
    source_a_reference_frame = transition.get("stable_source_a_end_frame")
    source_b_reference_frame = transition.get("stable_source_b_start_frame")
    if not isinstance(source_a_reference_frame, int) or not isinstance(source_b_reference_frame, int):
        raise ValueError("analysis must provide stable_source_a_end_frame and stable_source_b_start_frame")

    reference_manifest = load_json(reference_manifest_file)
    mapping = reference_manifest.get("frame_progress_mapping")
    if not isinstance(mapping, list):
        raise ValueError("reference manifest is missing frame_progress_mapping")
    source_by_output_frame = {
        item.get("output_frame"): item.get("normalized_clip_source_frame")
        for item in mapping
        if isinstance(item, dict)
        and isinstance(item.get("output_frame"), int)
        and isinstance(item.get("normalized_clip_source_frame"), int)
    }
    source_a_frame = source_by_output_frame.get(source_a_reference_frame)
    source_b_frame = source_by_output_frame.get(source_b_reference_frame)
    if not isinstance(source_a_frame, int) or not isinstance(source_b_frame, int):
        raise ValueError("analysis stable source boundaries are outside the prepared-reference mapping")
    if source_b_frame < source_a_frame:
        raise ValueError("mapped source boundaries are reversed")
    return {
        "mode": "prepared_reference_mapping",
        "analysis_file": str(analysis_file),
        "reference_manifest_file": str(reference_manifest_file),
        "source_a_reference_frame": source_a_reference_frame,
        "source_b_reference_frame": source_b_reference_frame,
        "source_a_frame": source_a_frame,
        "source_b_frame": source_b_frame,
    }


def analyze_reference_diagnostics(
    workspace_root: Path,
    reference: Path,
    output_dir: Path,
    width: int,
    height: int,
    frame_start: int = 0,
    frame_end: int | None = None,
    ffmpeg_path: str | None = None,
) -> dict[str, Any]:
    """Create deterministic reference-only motion evidence for agent review."""
    if output_dir.exists() and output_dir.is_file():
        raise ValueError(f"reference diagnostics output must be a directory: {output_dir}")
    if output_dir.suffix.lower() in {".json", ".mp4", ".png"}:
        raise ValueError(
            "reference diagnostics --output-dir must be a directory path, not a file path: "
            f"{output_dir}"
        )
    modules = load_harness_modules(workspace_root)
    frames_dir = output_dir / "reference_motion_frames"
    result = modules["analyze_reference_motion"](
        reference=reference,
        output_dir=frames_dir,
        width=width,
        height=height,
        frame_start=frame_start,
        frame_end=frame_end,
        ffmpeg_path=ffmpeg_path,
    )
    reference_manifest = reference / "reference_transition_manifest.json"
    fps = 30
    if reference_manifest.exists():
        manifest = load_json(reference_manifest)
        if isinstance(manifest.get("fps"), int) and manifest["fps"] > 0:
            fps = manifest["fps"]
    ffmpeg_executable = ffmpeg_path or shutil.which("ffmpeg")
    video = (
        _encode_png_sequence(
            frames_dir,
            fps=fps,
            ffmpeg_executable=ffmpeg_executable,
            output_file=output_dir / "reference_motion_diagnostics.mp4",
        )
        if ffmpeg_executable
        else {"status": "skipped", "message": "ffmpeg was not found; PNG diagnostics remain available"}
    )
    result["video"] = video
    source_directories = [reference.parent / "sources" / "source_a", reference.parent / "sources" / "source_b"]
    edge_policy_analyzer = modules.get("analyze_edge_content_policy")
    edge_policy = (
        edge_policy_analyzer(
            reference=reference,
            source_directories=source_directories,
            width=width,
            height=height,
            frame_start=frame_start,
            frame_end=frame_end,
            ffmpeg_path=ffmpeg_path,
        )
        if edge_policy_analyzer is not None and any(directory.is_dir() for directory in source_directories)
        else {
            "artifact_type": "edge_content_policy_diagnostics",
            "status": "not_applicable",
            "reason": "prepared source A/B directories are not available beside the reference",
        }
    )
    edge_output_file = output_dir / "edge_content_diagnostics.json"
    write_json(edge_output_file, edge_policy)
    result["edge_content_policy"] = edge_policy
    output_file = output_dir / "reference_motion_diagnostics.json"
    write_json(output_file, result)
    return {
        "status": "succeeded",
        "diagnostics": result,
        "output_file": str(output_file),
        "edge_output_file": str(edge_output_file),
    }


def ensure_reference_diagnostics(
    workspace_root: Path,
    reference: Path,
    width: int = 1920,
    height: int = 1080,
    ffmpeg_path: str | None = None,
) -> dict[str, Any]:
    """Ensure canonical reference diagnostics exist before a refinement phase."""
    output_dir = reference.resolve().parent / "diagnostics"
    diagnostics_file = output_dir / "reference_motion_diagnostics.json"
    edge_diagnostics_file = output_dir / "edge_content_diagnostics.json"
    source_directories = [reference.parent / "sources" / "source_a", reference.parent / "sources" / "source_b"]
    needs_edge_diagnostics = any(directory.is_dir() for directory in source_directories)
    if diagnostics_file.is_file():
        try:
            payload = load_json(diagnostics_file)
            if (
                payload.get("artifact_type") == "reference_motion_diagnostics"
                and isinstance(payload.get("pairs"), list)
                and isinstance(payload.get("summary"), dict)
                and isinstance(payload["summary"].get("topology_contract"), dict)
                and isinstance(payload["summary"].get("motion_geometry"), dict)
                and isinstance((payload["summary"]["motion_geometry"].get("translation_field")), dict)
                and isinstance(payload["summary"].get("angular_motion"), dict)
                and isinstance(payload["summary"].get("angular_motion_phases"), list)
                and (not needs_edge_diagnostics or edge_diagnostics_file.is_file())
            ):
                return {
                    "status": "ready",
                    "regenerated": False,
                    "output_file": str(diagnostics_file),
                    "edge_output_file": str(edge_diagnostics_file) if edge_diagnostics_file.is_file() else None,
                }
        except (OSError, ValueError):
            pass

    result = analyze_reference_diagnostics(
        workspace_root=workspace_root,
        reference=reference,
        output_dir=output_dir,
        width=width,
        height=height,
        ffmpeg_path=ffmpeg_path,
    )
    return {
        "status": "ready",
        "regenerated": True,
        "output_file": result["output_file"],
        "edge_output_file": result.get("edge_output_file"),
    }


def retrieve_effect(
    workspace_root: Path,
    analysis_file: Path,
    output_file: Path,
) -> dict[str, Any]:
    modules = load_harness_modules(workspace_root)
    analysis = load_json(analysis_file)
    planner_hints = analysis.get("planner_hints")
    if not isinstance(planner_hints, dict):
        raise ValueError("analysis artifact is missing planner_hints")
    family = planner_hints.get("recommended_effect_family")
    if not isinstance(family, str) or not family:
        raise ValueError("analysis planner_hints.recommended_effect_family is required")

    catalog = modules["build_effect_catalog"](workspace_root)
    selected = modules["select_effect_candidate"](catalog, style=family, input_kind="real")
    if selected is None:
        result = {
            "status": "not_found",
            "analysis": str(analysis_file),
            "requested_family": family,
            "requested_effect_id": planner_hints.get("recommended_effect_id"),
            "catalog_registration_count": catalog.get("registration_count"),
        }
    else:
        requested_effect_id = planner_hints.get("recommended_effect_id")
        result = {
            "status": "retrieved",
            "analysis": str(analysis_file),
            "requested_family": family,
            "requested_effect_id": requested_effect_id,
            "selected": selected,
            "exact_id_match": (
                requested_effect_id is None
                or selected.get("fx_id") == requested_effect_id
                or selected.get("effect_id") == requested_effect_id
            ),
        }
    write_json(output_file, result)
    return result


def benchmark_effects(
    workspace_root: Path,
    analysis_file: Path,
    source_a: str,
    source_b: str,
    reference_transition: Path,
    output_root: Path,
    output_file: Path,
    family: str | None,
    width: int,
    height: int,
    fps: int,
    frame_count: int,
    renderer: str | None,
    ffmpeg_path: str | None,
) -> dict[str, Any]:
    modules = load_harness_modules(workspace_root)
    analysis = load_json(analysis_file)
    planner_hints = analysis.get("planner_hints", {})
    selected_family = family or planner_hints.get("recommended_effect_family")
    if not isinstance(selected_family, str) or not selected_family:
        raise ValueError("a candidate family is required through --family or planner_hints")

    catalog = modules["build_effect_catalog"](workspace_root)
    candidates_by_fx_id: dict[str, dict[str, Any]] = {}
    aliases_by_fx_id: dict[str, list[str]] = {}
    for effect in catalog.get("effects", []):
        if effect.get("effect_source") != "builtin" or effect.get("family") != selected_family:
            continue
        effect_id = effect.get("fx_id")
        if not isinstance(effect_id, str) or not effect_id:
            continue
        if effect_id not in candidates_by_fx_id:
            candidates_by_fx_id[effect_id] = effect
            aliases_by_fx_id[effect_id] = []
        else:
            aliases_by_fx_id[effect_id].append(str(effect.get("effect_id")))
    candidates = list(candidates_by_fx_id.values())
    if not candidates:
        raise ValueError(f"no built-in effects found for family: {selected_family}")

    results: list[dict[str, Any]] = []
    for candidate in candidates:
        effect_id = candidate.get("fx_id")
        if not isinstance(effect_id, str) or not effect_id:
            continue
        job = {
            "job_name": f"agent_benchmark_{candidate['effect_id']}",
            "effect": {
                "fx_id": effect_id,
                "category": candidate.get("family", "unknown"),
                "effect_spec": None,
                "uniforms": {"progress": 0.0},
            },
            "inputs": {
                "source_a": source_a,
                "source_b": source_b,
                "reference_transition": str(reference_transition),
            },
            "render": {
                "width": width,
                "height": height,
                "fps": fps,
                "frame_count": frame_count,
                "output_format": "png_sequence",
            },
            "planning": {
                "source": "agent_catalog_benchmark",
                "family": selected_family,
                "catalog_effect_id": candidate.get("effect_id"),
            },
        }
        job_file = output_root / "jobs" / f"{candidate['effect_id']}.json"
        write_json(job_file, job)
        render_result = render_job(
            workspace_root=workspace_root,
            job_file=job_file,
            output_root=output_root / "runs",
            renderer=renderer,
            ffmpeg_path=ffmpeg_path,
        )
        candidate_result: dict[str, Any] = {
            "effect_id": candidate.get("effect_id"),
            "fx_id": effect_id,
            "catalog_aliases": aliases_by_fx_id.get(effect_id, []),
            "render": render_result,
        }
        if render_result.get("status") == "succeeded":
            run_root = Path(render_result["workspace"])
            score_file = run_root / "reports" / "score.json"
            candidate_result["score"] = score_candidate(
                workspace_root=workspace_root,
                candidate=Path(render_result["artifacts_dir"]),
                reference=reference_transition,
                output_file=score_file,
                width=width,
                height=height,
                frame_count=frame_count,
                require_exact_frame_count=True,
                ffmpeg_path=ffmpeg_path,
            )
        results.append(candidate_result)

    ranked = sorted(
        results,
        key=lambda item: (
            item.get("score", {}).get("mse", float("inf")),
            -item.get("score", {}).get("ssim", -1.0),
        ),
    )
    report = {
        "report_type": "agent_effect_benchmark",
        "report_version": 1,
        "status": "succeeded",
        "analysis": str(analysis_file),
        "family": selected_family,
        "candidate_count": len(results),
        "ranked_candidates": ranked,
    }
    write_json(output_file, report)
    return report


def render_job(
    workspace_root: Path,
    job_file: Path,
    output_root: Path,
    renderer: str | None,
    ffmpeg_path: str | None = None,
    run_name_prefix: str | None = None,
) -> dict[str, Any]:
    modules = load_harness_modules(workspace_root)
    job = modules["load_render_job"](job_file)
    run_name = run_name_prefix or job.job_name
    run_root = output_root / f"{run_name}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    inputs_dir = run_root / "inputs"
    render_dir = run_root / "render"
    reports_dir = run_root / "reports"
    artifacts_dir = run_root / "artifacts"
    for path in (inputs_dir, render_dir, reports_dir, artifacts_dir):
        path.mkdir(parents=True, exist_ok=False)

    workspace = modules["JobWorkspace"](
        root=run_root,
        inputs_dir=inputs_dir,
        render_dir=render_dir,
        reports_dir=reports_dir,
        artifacts_dir=artifacts_dir,
    )
    invocation = modules["prepare_render_invocation"](
        repo_root=workspace_root,
        workspace=workspace,
        job=job,
        renderer_executable=renderer,
    )
    video = (
        _encode_artifact_video(
            artifacts_dir=invocation.expected_output_dir,
            fps=job.render.fps,
            ffmpeg_path=ffmpeg_path,
        )
        if invocation.status == "succeeded"
        else {"status": "skipped", "message": "render did not succeed"}
    )
    result = {
        "status": invocation.status,
        "message": invocation.message,
        "job_file": str(job_file),
        "workspace": str(run_root),
        "request_file": str(invocation.request_file),
        "result_file": str(invocation.result_file),
        "artifacts_dir": str(invocation.expected_output_dir),
        "renderer": invocation.renderer_executable,
        "exit_code": invocation.exit_code,
        "produced_frame_count": invocation.produced_frame_count,
        "expected_frame_count": invocation.expected_frame_count,
        "stdout": invocation.stdout,
        "stderr": invocation.stderr,
        "renderer_result": invocation.renderer_result,
        "video": video,
    }
    write_json(run_root / "render_report.json", result)
    return result


def _encode_artifact_video(
    artifacts_dir: Path,
    fps: int,
    ffmpeg_path: str | None,
) -> dict[str, Any]:
    """Encode a successful PNG sequence for quick visual review."""
    first_frame = artifacts_dir / "frame_0000.png"
    if not first_frame.exists():
        return {
            "status": "skipped",
            "message": f"expected first rendered frame was not found: {first_frame}",
        }

    ffmpeg_executable = ffmpeg_path or shutil.which("ffmpeg")
    if not ffmpeg_executable:
        return {
            "status": "skipped",
            "message": "ffmpeg was not found; PNG artifacts remain available",
        }

    return _encode_png_sequence(artifacts_dir, fps, ffmpeg_executable, artifacts_dir / "rendered_transition.mp4")


def _encode_png_sequence(
    frames_dir: Path,
    fps: int,
    ffmpeg_executable: str,
    output_file: Path,
) -> dict[str, Any]:
    frame_pattern = "frame_%04d.png"
    completed = subprocess.run(
        [
            ffmpeg_executable,
            "-y",
            "-framerate",
            str(fps),
            "-start_number",
            "0",
            "-i",
            frame_pattern,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            output_file.name,
        ],
        cwd=frames_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "status": "failed",
            "message": "ffmpeg could not encode the rendered PNG sequence",
            "exit_code": completed.returncode,
            "stderr": completed.stderr,
        }
    return {
        "status": "succeeded",
        "file": str(output_file),
        "fps": fps,
        "exit_code": completed.returncode,
    }


def _create_comparison_assets(
    artifacts_dir: Path,
    reference_dir: Path,
    fps: int,
    frame_start: int | None,
    frame_end: int | None,
    ffmpeg_path: str | None,
) -> dict[str, Any]:
    ffmpeg_executable = ffmpeg_path or shutil.which("ffmpeg")
    if not ffmpeg_executable:
        return {"status": "skipped", "message": "ffmpeg was not found"}
    if not (artifacts_dir / "frame_0000.png").exists() or not (reference_dir / "frame_0000.png").exists():
        return {"status": "skipped", "message": "candidate or reference PNG sequence is incomplete"}

    reference_video = artifacts_dir / "reference_transition.mp4"
    full_comparison = artifacts_dir / "comparison_side_by_side.mp4"
    outputs = {
        "reference_video": _run_ffmpeg(
            [
                "-y", "-framerate", str(fps), "-start_number", "0", "-i", str(reference_dir / "frame_%04d.png"),
                "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", str(reference_video),
            ],
            ffmpeg_executable,
        ),
        "side_by_side_video": _run_ffmpeg(
            _comparison_command(
                artifacts_dir, reference_dir, fps, 0, None, full_comparison
            ),
            ffmpeg_executable,
        ),
    }
    if frame_start is not None and frame_end is not None:
        if frame_start < 0 or frame_end < frame_start:
            raise ValueError("comparison frame window is invalid")
        window_video = artifacts_dir / "comparison_transition_window.mp4"
        outputs["transition_window_video"] = _run_ffmpeg(
            _comparison_command(
                artifacts_dir,
                reference_dir,
                fps,
                frame_start,
                frame_end - frame_start + 1,
                window_video,
            ),
            ffmpeg_executable,
        )
    return {"status": "succeeded", "fps": fps, **outputs}


def _comparison_command(
    artifacts_dir: Path,
    reference_dir: Path,
    fps: int,
    start_number: int,
    frame_count: int | None,
    output_file: Path,
) -> list[str]:
    command = [
        "-y",
        "-framerate", str(fps), "-start_number", str(start_number), "-i", str(artifacts_dir / "frame_%04d.png"),
        "-framerate", str(fps), "-start_number", str(start_number), "-i", str(reference_dir / "frame_%04d.png"),
        "-filter_complex", "[0:v][1:v]hstack=inputs=2",
    ]
    if frame_count is not None:
        command.extend(["-frames:v", str(frame_count)])
    command.extend(["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", str(output_file)])
    return command


def _run_ffmpeg(arguments: list[str], ffmpeg_executable: str) -> dict[str, Any]:
    completed = subprocess.run(
        [ffmpeg_executable, *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "status": "succeeded" if completed.returncode == 0 else "failed",
        "file": arguments[-1],
        "exit_code": completed.returncode,
        "stderr": completed.stderr if completed.returncode else "",
    }


def score_candidate(
    workspace_root: Path,
    candidate: Path,
    reference: Path,
    output_file: Path,
    width: int,
    height: int,
    frame_count: int | None,
    require_exact_frame_count: bool,
    ffmpeg_path: str | None = None,
    frame_start: int | None = None,
    frame_end: int | None = None,
    endpoint_frame_count: int = 3,
    analysis_file: Path | None = None,
    design_file: Path | None = None,
    source_files: list[Path] | None = None,
    sampler_source: Path | None = None,
    source_directories: list[Path] | None = None,
) -> dict[str, Any]:
    modules = load_harness_modules(workspace_root)
    score = modules["score_frame_sequences"](
        candidate=candidate,
        reference=reference,
        width=width,
        height=height,
        frame_count=frame_count,
        ffmpeg_path=ffmpeg_path,
        require_exact_frame_count=require_exact_frame_count,
    ).to_dict()
    score["candidate"] = str(candidate)
    score["reference"] = str(reference)
    sampler_diagnostics = modules.get("analyze_sampler_repetition")
    if sampler_diagnostics is not None and (source_files or sampler_source):
        score["sampler_repetition"] = sampler_diagnostics(
            source_files=source_files,
            sampler_source=sampler_source,
        )
    if frame_start is not None or frame_end is not None:
        score.update(
            _build_windowed_score(
                score,
                frame_start=frame_start,
                frame_end=frame_end,
                endpoint_frame_count=endpoint_frame_count,
            )
        )
        phase_scores = _build_phase_scores(score, analysis_file)
        if phase_scores is not None:
            score["phase_scores"] = phase_scores
        # The default source mode intentionally uses video frame 0 and the
        # final decoded frame.  A prepared reference may instead be a compact
        # interior transition window, whose adjacent "stable" frames are not
        # pixel-identical to those declared render endpoints.  Endpoint
        # validation must therefore prove the shader reaches its configured
        # A/B inputs, rather than accidentally reject a correct endpoint for
        # a source-video timing difference.
        if source_directories and len(source_directories) >= 2:
            score["endpoint_checks"] = _build_source_endpoint_checks(
                candidate=candidate,
                source_a=source_directories[0],
                source_b=source_directories[1],
                width=width,
                height=height,
                frame_start=score["transition_window"]["frame_start"],
                frame_end=score["transition_window"]["frame_end"],
                endpoint_frame_count=endpoint_frame_count,
                ffmpeg_path=ffmpeg_path,
            )
        score["transition_diagnostics"] = _build_transition_diagnostics(score)
        edge_policy_analyzer = modules.get("analyze_edge_content_policy")
        if edge_policy_analyzer is not None and source_directories:
            reference_policy = edge_policy_analyzer(
                reference=reference,
                source_directories=source_directories,
                width=width,
                height=height,
                frame_start=score["transition_window"]["frame_start"],
                frame_end=score["transition_window"]["frame_end"],
                ffmpeg_path=ffmpeg_path,
                output_dir=candidate / "edge_content_diagnostics",
            )
            score["edge_content_policy"] = {
                "reference": reference_policy,
                "candidate": _candidate_edge_policy(score.get("sampler_repetition")),
            }
        centroid_scorer = modules.get("score_salient_centroid_tracking")
        if centroid_scorer is not None:
            score["salient_centroid_tracking"] = centroid_scorer(
                candidate=candidate,
                reference=reference,
                width=width,
                height=height,
                frame_start=score["transition_window"]["frame_start"],
                frame_end=score["transition_window"]["frame_end"],
                ffmpeg_path=ffmpeg_path,
            )
        motion_scorer = modules.get("score_motion")
        if motion_scorer is not None:
            motion_metrics = motion_scorer(
                candidate=candidate,
                reference=reference,
                width=width,
                height=height,
                frame_start=score["transition_window"]["frame_start"],
                frame_end=score["transition_window"]["frame_end"],
                ffmpeg_path=ffmpeg_path,
            )
            score["motion_metrics"] = motion_metrics
            geometry = _score_motion_geometry(reference, motion_metrics)
            if geometry is not None:
                score["motion_geometry"] = geometry
            body_geometry = _score_foreground_body_transform(reference, motion_metrics)
            if body_geometry is not None:
                score["foreground_body_transform"] = body_geometry
            angular_motion = _score_angular_motion(
                reference, motion_metrics, analysis_file=analysis_file, design_file=design_file
            )
            if angular_motion is not None:
                score["angular_motion"] = angular_motion
            regional_motion = _score_regional_motion(reference, motion_metrics)
            if regional_motion is not None:
                score["regional_motion"] = regional_motion
            topology = _score_motion_topology(
                reference,
                motion_metrics,
                analysis_file=analysis_file,
                design_file=design_file,
            )
            if topology is not None:
                score["motion_topology"] = topology
            score["transition_diagnostics"]["worst_motion_pairs"] = sorted(
                motion_metrics["pairs"],
                key=lambda pair: float(pair.get("vector_mae", pair.get("mean_shift_error", 0.0))),
                reverse=True,
            )[:5]
            motion_visualizer = modules.get("create_motion_visualizations")
            if motion_visualizer is not None:
                visualizations = motion_visualizer(
                    candidate=candidate,
                    reference=reference,
                    output_dir=candidate / "motion_diagnostics",
                    width=width,
                    height=height,
                    frame_start=score["transition_window"]["frame_start"],
                    frame_end=score["transition_window"]["frame_end"],
                    ffmpeg_path=ffmpeg_path,
                )
                if visualizations.get("status") == "succeeded":
                    ffmpeg_executable = ffmpeg_path or shutil.which("ffmpeg")
                    if ffmpeg_executable:
                        visualizations["video"] = _encode_png_sequence(
                            candidate / "motion_diagnostics",
                            fps=30,
                            ffmpeg_executable=ffmpeg_executable,
                            output_file=candidate / "motion_diagnostics.mp4",
                        )
                score["motion_visualizations"] = visualizations
    score["status"] = "succeeded"
    write_json(output_file, score)
    return score


def _candidate_edge_policy(sampler_diagnostics: Any) -> dict[str, Any]:
    """Describe only source-level candidate evidence; rendered behavior remains authoritative."""
    if not isinstance(sampler_diagnostics, dict):
        return {"policy": "unknown", "reason": "candidate sampler source was not available"}
    if sampler_diagnostics.get("uv_wrapping_construct_count", 0):
        return {
            "policy": "repeat",
            "reason": "candidate contains modulo-like UV mapping",
            "confidence": "advisory",
        }
    modes = sampler_diagnostics.get("address_modes")
    values = set(modes.values()) if isinstance(modes, dict) else set()
    if len(values) == 1 and values.intersection({"CLAMP", "MIRROR", "WRAP", "BORDER"}):
        return {
            "policy": next(iter(values)).lower(),
            "reason": "candidate relies on a shared sampler address mode",
            "confidence": "advisory",
        }
    return {"policy": "unknown", "reason": "candidate edge policy is not explicit"}


def _score_motion_topology(
    reference: Path,
    motion_metrics: dict[str, Any],
    analysis_file: Path | None = None,
    design_file: Path | None = None,
) -> dict[str, Any] | None:
    diagnostics_file = reference.parent / "diagnostics" / "reference_motion_diagnostics.json"
    if not diagnostics_file.exists():
        return None
    diagnostics = load_json(diagnostics_file)
    contract = (diagnostics.get("summary") or {}).get("topology_contract")
    if not isinstance(contract, dict) or contract.get("status") != "required":
        return None
    policy = _resolve_motion_topology_policy(analysis_file, design_file, contract)
    if policy["mode"] == "disabled":
        return {
            "status": "not_applicable",
            "reason": "motion topology is disabled for this transition structure",
            "policy": policy,
            "contract": contract,
        }
    evidence = contract.get("evidence_pairs")
    pairs = motion_metrics.get("pairs")
    if not isinstance(evidence, list) or not isinstance(pairs, list):
        return None
    pair_by_range = {
        (pair.get("from_frame"), pair.get("to_frame")): pair
        for pair in pairs
        if isinstance(pair, dict)
    }
    minimum_regions = int(contract.get("minimum_concurrent_regions", 2))
    observed = [
        pair_by_range.get((item.get("from_frame"), item.get("to_frame")))
        for item in evidence
        if isinstance(item, dict)
    ]
    observed = [pair for pair in observed if isinstance(pair, dict)]
    if not observed:
        return None
    region_match_rate = sum(
        int(pair.get("matched_direction_region_count", 0)) >= minimum_regions
        for pair in observed
    ) / len(observed)
    direction_match_rate = sum(
        float(pair.get("direction_agreement", 0.0)) >= 0.5 for pair in observed
    ) / len(observed)
    return {
        "contract": contract,
        "evidence_pair_count": len(observed),
        "candidate_region_match_rate": region_match_rate,
        "direction_match_rate": direction_match_rate,
        "enforcement": policy["mode"],
        "policy": policy,
        "status": (
            "satisfied"
            if region_match_rate >= policy["minimum_region_match_rate"]
            and direction_match_rate >= policy["minimum_direction_match_rate"]
            else "structural_mismatch"
        ),
    }


def _resolve_motion_topology_policy(
    analysis_file: Path | None,
    design_file: Path | None,
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Resolve whether motion topology is relevant to this candidate."""
    for source_name, artifact_file in (("effect_design", design_file), ("transition_analysis", analysis_file)):
        if artifact_file is None or not artifact_file.exists():
            continue
        try:
            artifact = load_json(artifact_file)
        except (OSError, ValueError):
            continue
        policy = artifact.get("evaluation_policy")
        if not isinstance(policy, dict):
            continue
        topology = policy.get("motion_topology")
        if isinstance(topology, dict):
            resolved = _normalize_motion_topology_policy(topology, source_name)
            if resolved is not None:
                return resolved

    if analysis_file is not None and analysis_file.exists():
        try:
            analysis = load_json(analysis_file)
        except (OSError, ValueError):
            analysis = {}
        transition = analysis.get("transition") if isinstance(analysis, dict) else None
        if isinstance(transition, dict):
            region_count = transition.get("region_count")
            structure = str(transition.get("structure_type", "")).casefold()
            geometry = str(transition.get("split_geometry", "")).casefold()
            axes = {
                str(axis).casefold()
                for axis in transition.get("motion_axes", [])
                if isinstance(axis, str)
            }
            segmented = (
                isinstance(region_count, int)
                and region_count >= 2
            ) or any(
                marker in structure or marker in geometry
                for marker in ("split", "band", "quadrant", "region", "mask")
            )
            if segmented:
                return _normalize_motion_topology_policy(
                    {"mode": "advisory"},
                    "transition_analysis.inferred",
                    inferred_axes=sorted(axes),
                )
            return _normalize_motion_topology_policy(
                {"mode": "disabled"},
                "transition_analysis.inferred",
            )

    return _normalize_motion_topology_policy(
        {"mode": "disabled"},
        "no_artifact_policy",
    )


def _normalize_motion_topology_policy(
    policy: dict[str, Any],
    source: str,
    inferred_axes: list[str] | None = None,
) -> dict[str, Any] | None:
    mode = policy.get("mode")
    if mode not in {"disabled", "advisory", "hard"}:
        return None
    region_threshold = policy.get("minimum_region_match_rate", 0.5)
    direction_threshold = policy.get("minimum_direction_match_rate", 0.5)
    if not isinstance(region_threshold, (int, float)) or not 0 <= region_threshold <= 1:
        return None
    if not isinstance(direction_threshold, (int, float)) or not 0 <= direction_threshold <= 1:
        return None
    result = {
        "mode": mode,
        "minimum_region_match_rate": float(region_threshold),
        "minimum_direction_match_rate": float(direction_threshold),
        "source": source,
    }
    if inferred_axes:
        result["inferred_motion_axes"] = inferred_axes
    return result


def _score_motion_geometry(reference: Path, motion_metrics: dict[str, Any]) -> dict[str, Any] | None:
    """Compare candidate transformation cues with the cached reference geometry."""
    diagnostics_file = reference.parent / "diagnostics" / "reference_motion_diagnostics.json"
    if not diagnostics_file.exists():
        return None
    diagnostics = load_json(diagnostics_file)
    expected = (diagnostics.get("summary") or {}).get("motion_geometry")
    candidate = motion_metrics.get("motion_geometry")
    if not isinstance(expected, dict) or not isinstance(candidate, dict):
        return None
    if expected.get("status") != "estimated" or candidate.get("status") != "estimated":
        return {
            "status": "needs_review",
            "reference": expected,
            "candidate": candidate,
        }

    reference_rotation = float((expected.get("rotation_field") or {}).get("mean_degrees", 0.0))
    candidate_rotation = float((candidate.get("rotation_field") or {}).get("mean_degrees", 0.0))
    rotation_delta = abs((candidate_rotation - reference_rotation + 180.0) % 360.0 - 180.0)
    reference_scale = float((expected.get("radial_scale_field") or {}).get("mean_ratio", 1.0))
    candidate_scale = float((candidate.get("radial_scale_field") or {}).get("mean_ratio", 1.0))
    scale_delta = abs(candidate_scale - reference_scale)
    reference_translation = expected.get("translation_field") or {}
    candidate_translation = candidate.get("translation_field") or {}
    reference_pivot = expected.get("pivot_field") or {}
    candidate_pivot = candidate.get("pivot_field") or {}
    translation_dx = float(candidate_translation.get("mean_dx_pixels", 0.0)) - float(
        reference_translation.get("mean_dx_pixels", 0.0)
    )
    translation_dy = float(candidate_translation.get("mean_dy_pixels", 0.0)) - float(
        reference_translation.get("mean_dy_pixels", 0.0)
    )
    translation_delta = (translation_dx * translation_dx + translation_dy * translation_dy) ** 0.5
    pivot_delta = None
    if reference_pivot.get("status") == "estimated" and candidate_pivot.get("status") == "estimated":
        pivot_dx = float(candidate_pivot.get("x_pixels", 0.0)) - float(reference_pivot.get("x_pixels", 0.0))
        pivot_dy = float(candidate_pivot.get("y_pixels", 0.0)) - float(reference_pivot.get("y_pixels", 0.0))
        pivot_delta = (pivot_dx * pivot_dx + pivot_dy * pivot_dy) ** 0.5
    reference_translation_magnitude = float(reference_translation.get("magnitude_pixels", 0.0))
    candidate_translation_magnitude = float(candidate_translation.get("magnitude_pixels", 0.0))
    translation_direction_agreement = True
    translation_direction_delta = 0.0
    if reference_translation_magnitude > 2.0 and candidate_translation_magnitude > 2.0:
        reference_vector = (
            float(reference_translation.get("mean_dx_pixels", 0.0)),
            float(reference_translation.get("mean_dy_pixels", 0.0)),
        )
        candidate_vector = (
            float(candidate_translation.get("mean_dx_pixels", 0.0)),
            float(candidate_translation.get("mean_dy_pixels", 0.0)),
        )
        reference_angle = math.degrees(math.atan2(reference_vector[1], reference_vector[0]))
        candidate_angle = math.degrees(math.atan2(candidate_vector[1], candidate_vector[0]))
        translation_direction_delta = abs((candidate_angle - reference_angle + 180.0) % 360.0 - 180.0)
        translation_direction_agreement = (
            reference_vector[0] * candidate_vector[0]
            + reference_vector[1] * candidate_vector[1]
            > 0.0
            and translation_direction_delta <= 30.0
        )
    reference_flip = bool((expected.get("reflection_or_flip") or {}).get("detected", False))
    candidate_flip = bool((candidate.get("reflection_or_flip") or {}).get("detected", False))
    rotation_direction_agreement = True
    if abs(reference_rotation) > 1.0 and abs(candidate_rotation) > 1.0:
        rotation_direction_agreement = reference_rotation * candidate_rotation > 0.0
    geometry_components = [
        max(0.0, 1.0 - rotation_delta / 30.0),
        max(0.0, 1.0 - translation_delta / 20.0),
        max(0.0, 1.0 - scale_delta / 0.30),
        1.0 if reference_flip == candidate_flip else 0.0,
    ]
    geometry_similarity = sum(geometry_components) / len(geometry_components)
    geometry_mismatch = (
        rotation_delta > 10.0
        or not rotation_direction_agreement
        or translation_delta > 2.0
        or (pivot_delta is not None and pivot_delta > 8.0)
        or not translation_direction_agreement
        or scale_delta > 0.15
        or reference_flip != candidate_flip
    )
    return {
        "status": "geometry_mismatch" if geometry_mismatch else "satisfied",
        "reference": expected,
        "candidate": candidate,
        "rotation_delta_degrees": rotation_delta,
        "rotation_direction_agreement": rotation_direction_agreement,
        "scale_delta_ratio": scale_delta,
        "translation_delta_pixels": translation_delta,
        "translation_delta_dx_pixels": translation_dx,
        "translation_delta_dy_pixels": translation_dy,
        "translation_direction_agreement": translation_direction_agreement,
        "translation_direction_delta_degrees": translation_direction_delta,
        "pivot_delta_pixels": pivot_delta,
        "geometry_similarity": geometry_similarity,
        "reflection_agreement": reference_flip == candidate_flip,
    }


def _score_foreground_body_transform(reference: Path, motion_metrics: dict[str, Any]) -> dict[str, Any] | None:
    """Compare feature-tracked body transforms without making them hard gates."""
    diagnostics_file = reference.parent / "diagnostics" / "reference_motion_diagnostics.json"
    if not diagnostics_file.exists():
        return None
    diagnostics = load_json(diagnostics_file)
    expected = (diagnostics.get("summary") or {}).get("foreground_body_transform")
    candidate = motion_metrics.get("foreground_body_transform")
    expected_phases = (diagnostics.get("summary") or {}).get("foreground_body_transform_phases")
    candidate_phases = motion_metrics.get("foreground_body_transform_phases")
    if isinstance(expected_phases, dict) and isinstance(candidate_phases, dict):
        phases = {
            phase: _compare_body_transform_phase(expected_phases.get(phase), candidate_phases.get(phase))
            for phase in ("outgoing", "midpoint", "incoming")
        }
        reliable = [item for item in phases.values() if item.get("status") == "estimated"]
        return {
            "status": "estimated" if reliable else "advisory_indeterminate",
            "detector": "orb_feature_similarity",
            "phases": phases,
            "reference": expected,
            "candidate": candidate,
            "confidence": min([float(item.get("confidence", 0.0)) for item in reliable] or [0.0]),
        }
    if not isinstance(expected, dict) or not isinstance(candidate, dict):
        return None
    if expected.get("status") != "estimated" or candidate.get("status") != "estimated":
        return {"status": "advisory_indeterminate", "reference": expected, "candidate": candidate}
    expected_rotation = float((expected.get("rotation_field") or {}).get("mean_degrees", 0.0))
    candidate_rotation = float((candidate.get("rotation_field") or {}).get("mean_degrees", 0.0))
    rotation_delta = abs((candidate_rotation - expected_rotation + 180.0) % 360.0 - 180.0)
    expected_scale = float((expected.get("radial_scale_field") or {}).get("mean_ratio", 1.0))
    candidate_scale = float((candidate.get("radial_scale_field") or {}).get("mean_ratio", 1.0))
    expected_translation = expected.get("translation_field") or {}
    candidate_translation = candidate.get("translation_field") or {}
    dx = float(candidate_translation.get("mean_dx_pixels", 0.0)) - float(expected_translation.get("mean_dx_pixels", 0.0))
    dy = float(candidate_translation.get("mean_dy_pixels", 0.0)) - float(expected_translation.get("mean_dy_pixels", 0.0))
    translation_delta = math.hypot(dx, dy)
    expected_flip = bool((expected.get("reflection_or_flip") or {}).get("detected", False))
    candidate_flip = bool((candidate.get("reflection_or_flip") or {}).get("detected", False))
    return {
        "status": "estimated",
        "detector": "orb_feature_similarity",
        "reference": expected,
        "candidate": candidate,
        "rotation_delta_degrees": rotation_delta,
        "scale_delta_ratio": abs(candidate_scale - expected_scale),
        "translation_delta_pixels": translation_delta,
        "translation_delta_dx_pixels": dx,
        "translation_delta_dy_pixels": dy,
        "reflection_agreement": expected_flip == candidate_flip,
        "confidence": min(float(expected.get("confidence", 0.0)), float(candidate.get("confidence", 0.0))),
    }


def _compare_body_transform_phase(expected: Any, candidate: Any) -> dict[str, Any]:
    if not isinstance(expected, dict) or not isinstance(candidate, dict):
        return {"status": "needs_review", "reason": "phase has no reliable body transform"}
    if expected.get("status") != "estimated" or candidate.get("status") != "estimated":
        return {"status": "needs_review", "reference": expected, "candidate": candidate}
    expected_rotation = float((expected.get("rotation_field") or {}).get("mean_degrees", 0.0))
    candidate_rotation = float((candidate.get("rotation_field") or {}).get("mean_degrees", 0.0))
    expected_translation = expected.get("translation_field") or {}
    candidate_translation = candidate.get("translation_field") or {}
    dx = float(candidate_translation.get("mean_dx_pixels", 0.0)) - float(expected_translation.get("mean_dx_pixels", 0.0))
    dy = float(candidate_translation.get("mean_dy_pixels", 0.0)) - float(expected_translation.get("mean_dy_pixels", 0.0))
    expected_scale = float((expected.get("radial_scale_field") or {}).get("mean_ratio", 1.0))
    candidate_scale = float((candidate.get("radial_scale_field") or {}).get("mean_ratio", 1.0))
    return {
        "status": "estimated",
        "reference": expected,
        "candidate": candidate,
        "confidence": min(float(expected.get("confidence", 0.0)), float(candidate.get("confidence", 0.0))),
        "rotation_delta_degrees": abs((candidate_rotation - expected_rotation + 180.0) % 360.0 - 180.0),
        "scale_delta_ratio": abs(candidate_scale - expected_scale),
        "translation_delta_pixels": math.hypot(dx, dy),
        "translation_delta_dx_pixels": dx,
        "translation_delta_dy_pixels": dy,
    }


def _score_angular_motion(
    reference: Path,
    motion_metrics: dict[str, Any],
    analysis_file: Path | None,
    design_file: Path | None,
) -> dict[str, Any] | None:
    """Compare signed angular evidence only for rotation-like transition designs."""
    if not _is_rotation_like_transition(analysis_file, design_file):
        return None
    diagnostics_file = reference.parent / "diagnostics" / "reference_motion_diagnostics.json"
    if not diagnostics_file.exists():
        return None
    diagnostics = load_json(diagnostics_file)
    expected = (diagnostics.get("summary") or {}).get("angular_motion")
    candidate = motion_metrics.get("angular_motion")
    if not isinstance(expected, dict) or not isinstance(candidate, dict):
        return None
    expected_phases = (diagnostics.get("summary") or {}).get("angular_motion_phases", [])
    candidate_phases = motion_metrics.get("angular_motion_phases", [])
    phase_comparison = _compare_angular_motion_phases(expected_phases, candidate_phases)
    if phase_comparison:
        confidence = min(float(item["confidence"]) for item in phase_comparison)
        direction_match = all(bool(item["direction_match"]) for item in phase_comparison)
        return {
            "status": "satisfied" if direction_match else "direction_mismatch",
            "reference": expected,
            "candidate": candidate,
            "phases": phase_comparison,
            "direction_match": direction_match,
            "confidence": confidence,
            "enforcement": "advisory",
        }
    expected_confidence = float(expected.get("confidence", 0.0))
    candidate_confidence = float(candidate.get("confidence", 0.0))
    if expected.get("status") != "estimated" or candidate.get("status") != "estimated":
        return {
            "status": "advisory_indeterminate",
            "reference": expected,
            "candidate": candidate,
            "reason": "signed angular evidence is not reliable enough to direct refinement",
        }
    direction_match = expected.get("direction") == candidate.get("direction")
    confidence = min(expected_confidence, candidate_confidence)
    return {
        "status": "satisfied" if direction_match else "direction_mismatch",
        "reference": expected,
        "candidate": candidate,
        "direction_match": direction_match,
        "confidence": confidence,
        "enforcement": "advisory",
    }


def _compare_angular_motion_phases(expected: Any, candidate: Any) -> list[dict[str, Any]]:
    if not isinstance(expected, list) or not isinstance(candidate, list):
        return []
    candidate_by_name = {
        item.get("name"): item for item in candidate if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    comparisons: list[dict[str, Any]] = []
    for reference_phase in expected:
        if not isinstance(reference_phase, dict) or reference_phase.get("status") != "estimated":
            continue
        name = reference_phase.get("name")
        candidate_phase = candidate_by_name.get(name)
        if not isinstance(candidate_phase, dict) or candidate_phase.get("status") != "estimated":
            continue
        confidence = min(
            float(reference_phase.get("confidence", 0.0)),
            float(candidate_phase.get("confidence", 0.0)),
        )
        if confidence < 0.35:
            continue
        comparisons.append(
            {
                "name": name,
                "reference": reference_phase,
                "candidate": candidate_phase,
                "direction_match": reference_phase.get("direction") == candidate_phase.get("direction"),
                "confidence": confidence,
            }
        )
    return comparisons


def _is_rotation_like_transition(analysis_file: Path | None, design_file: Path | None) -> bool:
    vocabulary: list[str] = []
    for artifact_file in (analysis_file, design_file):
        if artifact_file is None or not artifact_file.is_file():
            continue
        try:
            artifact = load_json(artifact_file)
        except (OSError, ValueError):
            continue
        vocabulary.append(str(artifact.get("transition", {})))
        vocabulary.append(str(artifact.get("planner_hints", {})))
        vocabulary.append(str(artifact.get("target_effect", {})))
        vocabulary.append(str(artifact.get("implementation_seed", {})))
    text = " ".join(vocabulary).casefold()
    return any(marker in text for marker in ("rotate", "rotation", "scale", "perspective", "card", "flip", "reflection"))


def _score_regional_motion(reference: Path, motion_metrics: dict[str, Any]) -> dict[str, Any] | None:
    diagnostics_file = reference.parent / "diagnostics" / "reference_motion_diagnostics.json"
    if not diagnostics_file.exists():
        return None
    diagnostics = load_json(diagnostics_file)
    expected = (diagnostics.get("summary") or {}).get("regional_motion")
    candidate = motion_metrics.get("regional_motion")
    if not isinstance(expected, dict) or not isinstance(candidate, dict):
        return None
    if expected.get("status") != "estimated" or candidate.get("status") != "estimated":
        return {"status": "needs_review", "reference": expected, "candidate": candidate}
    expected_angle = float(expected.get("direction_degrees", 0.0))
    candidate_angle = float(candidate.get("direction_degrees", 0.0))
    angle_delta = abs((candidate_angle - expected_angle + 180.0) % 360.0 - 180.0)
    expected_axis = expected.get("dominant_axis")
    candidate_axis = candidate.get("dominant_axis")
    return {
        "status": "satisfied" if angle_delta <= 20.0 and expected_axis == candidate_axis else "direction_mismatch",
        "reference": expected,
        "candidate": candidate,
        "direction_delta_degrees": angle_delta,
        "axis_agreement": expected_axis == candidate_axis,
    }


def _build_windowed_score(
    score: dict[str, Any],
    frame_start: int | None,
    frame_end: int | None,
    endpoint_frame_count: int,
) -> dict[str, Any]:
    frames = score.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("windowed scoring requires per-frame metrics")
    if endpoint_frame_count < 1:
        raise ValueError("endpoint_frame_count must be at least 1")

    last_index = len(frames) - 1
    start = 0 if frame_start is None else frame_start
    end = last_index if frame_end is None else frame_end
    if start < 0 or end < 0 or start > end or end > last_index:
        raise ValueError(f"score window must be within frame indexes 0 through {last_index}")

    window_frames = frames[start : end + 1]
    before_frames = frames[max(0, start - endpoint_frame_count) : start]
    after_frames = frames[end + 1 : end + 1 + endpoint_frame_count]
    return {
        "transition_window": {
            "frame_start": start,
            "frame_end": end,
            **_aggregate_frame_scores(window_frames),
        },
        "endpoint_checks": {
            "requested_frame_count": endpoint_frame_count,
            "before_transition": _aggregate_frame_scores(before_frames) if before_frames else None,
            "after_transition": _aggregate_frame_scores(after_frames) if after_frames else None,
        },
    }


def _build_source_endpoint_checks(
    candidate: Path,
    source_a: Path,
    source_b: Path,
    width: int,
    height: int,
    frame_start: int,
    frame_end: int,
    endpoint_frame_count: int,
    ffmpeg_path: str | None,
) -> dict[str, Any]:
    """Score stable candidate edges against the job's declared A/B endpoints."""
    from overlay_harness.evaluator import decode_frame_rgb, discover_frames, score_rgb_buffers

    candidate_frames = discover_frames(candidate)
    source_a_frames = discover_frames(source_a)
    source_b_frames = discover_frames(source_b)
    if not source_a_frames or not source_b_frames:
        raise ValueError("source endpoint checks require non-empty source A and source B directories")
    before_indexes = list(range(max(0, frame_start - endpoint_frame_count), frame_start))
    after_indexes = list(range(frame_end + 1, min(len(candidate_frames), frame_end + 1 + endpoint_frame_count)))
    ffmpeg_executable = ffmpeg_path or shutil.which("ffmpeg")

    def aggregate(indexes: list[int], endpoint_frame: Path) -> dict[str, Any] | None:
        if not indexes:
            return None
        endpoint_rgb = decode_frame_rgb(ffmpeg_executable, endpoint_frame, width, height)
        frames = []
        for index in indexes:
            candidate_rgb = decode_frame_rgb(ffmpeg_executable, candidate_frames[index], width, height)
            frame_score = score_rgb_buffers(candidate_rgb, endpoint_rgb, width, height)
            frames.append(
                {
                    "candidate_frame": str(candidate_frames[index]),
                    "reference_frame": str(endpoint_frame),
                    "mse": frame_score["mse"],
                    "mae": frame_score["mae"],
                    "psnr_db": frame_score["psnr_db"],
                    "ssim": frame_score["ssim"],
                }
            )
        return _aggregate_frame_scores(frames)

    return {
        "requested_frame_count": endpoint_frame_count,
        "reference_mode": "declared_render_sources",
        "source_a": str(source_a),
        "source_b": str(source_b),
        "before_transition": aggregate(before_indexes, source_a_frames[0]),
        "after_transition": aggregate(after_indexes, source_b_frames[0]),
    }


def _build_transition_diagnostics(score: dict[str, Any]) -> dict[str, Any]:
    window = score["transition_window"]
    frames = score["frames"]
    start = int(window["frame_start"])
    end = int(window["frame_end"])
    details = [
        {
            "frame_index": index,
            "mse": frame["mse"],
            "mae": frame["mae"],
            "ssim": frame["ssim"],
            "candidate_frame": frame.get("candidate_frame"),
            "reference_frame": frame.get("reference_frame"),
        }
        for index, frame in enumerate(frames[start : end + 1], start=start)
    ]
    return {
        "frame_count": len(details),
        "worst_mse_frames": sorted(details, key=lambda item: float(item["mse"]), reverse=True)[:5],
        "lowest_ssim_frames": sorted(
            details,
            key=lambda item: float(item["ssim"]) if item["ssim"] is not None else float("inf"),
        )[:5],
    }


def _build_phase_scores(score: dict[str, Any], analysis_file: Path | None) -> dict[str, Any] | None:
    """Aggregate onset, peak, and settle scores from analysis progress coordinates."""
    if analysis_file is None or not analysis_file.is_file():
        return None
    try:
        analysis = load_json(analysis_file)
    except (OSError, ValueError):
        return None
    mapping = analysis.get("frame_progress_mapping") if isinstance(analysis, dict) else None
    if not isinstance(mapping, list):
        return None
    progress_by_frame = {
        int(item["frame_index"]): float(item["normalized_progress"])
        for item in mapping
        if isinstance(item, dict)
        and isinstance(item.get("frame_index"), int)
        and isinstance(item.get("normalized_progress"), (int, float))
    }
    window = score.get("transition_window")
    frames = score.get("frames")
    if not isinstance(window, dict) or not isinstance(frames, list) or not progress_by_frame:
        return None
    start = int(window["frame_start"])
    end = int(window["frame_end"])
    grouped: dict[str, list[dict[str, Any]]] = {"onset": [], "peak": [], "settle": []}
    for index in range(start, end + 1):
        progress = progress_by_frame.get(index)
        if progress is None or index >= len(frames):
            continue
        phase = "onset" if progress < 0.25 else "peak" if progress <= 0.75 else "settle"
        grouped[phase].append(frames[index])
    result = {phase: _aggregate_frame_scores(items) for phase, items in grouped.items() if items}
    return result or None


def _aggregate_frame_scores(frames: list[dict[str, Any]]) -> dict[str, Any]:
    if not frames:
        raise ValueError("cannot aggregate an empty frame score list")
    mse = sum(float(frame["mse"]) for frame in frames) / len(frames)
    mae = sum(float(frame["mae"]) for frame in frames) / len(frames)
    ssim_values = [float(frame["ssim"]) for frame in frames if frame.get("ssim") is not None]
    return {
        "frame_count": len(frames),
        "mse": mse,
        "mae": mae,
        "psnr_db": 10 * math.log10((255.0 * 255.0) / mse) if mse > 0 else None,
        "ssim": sum(ssim_values) / len(ssim_values) if ssim_values else None,
    }


def build_report(
    analysis_file: Path,
    design_file: Path,
    render_file: Path,
    score_file: Path,
    output_file: Path,
) -> dict[str, Any]:
    report = {
        "report_type": "agent_transition_regression",
        "report_version": 1,
        "status": "succeeded",
        "analysis": load_json(analysis_file),
        "effect_design": load_json(design_file),
        "render": load_json(render_file),
        "score": load_json(score_file),
    }
    write_json(output_file, report)
    return report


def evaluate_candidate(
    workspace_root: Path,
    candidate_manifest_file: Path,
    job_file: Path,
    reference: Path,
    output_root: Path,
    backup_dir: Path,
    msbuild: str,
    configuration: str,
    platform: str,
    renderer: str | None,
    width: int,
    height: int,
    frame_count: int | None,
    ffmpeg_path: str | None,
    restore: bool = False,
    frame_start: int | None = None,
    frame_end: int | None = None,
    endpoint_frame_count: int = 3,
    iteration: int | None = None,
    calibrate_progress: bool = False,
) -> dict[str, Any]:
    """Stage, build, render, and score a candidate, optionally restoring it afterward."""
    candidate = load_json(candidate_manifest_file)
    candidate_files = [Path(path) for path in candidate.get("candidate_files", [])]
    target_files = [Path(path) for path in candidate.get("target_files", [])]
    if not candidate_files or len(candidate_files) != len(target_files):
        raise ValueError("candidate manifest has invalid candidate and target files")
    if any(not path.exists() for path in candidate_files + target_files):
        raise FileNotFoundError("candidate or registered target source file is missing")

    backup_dir.mkdir(parents=True, exist_ok=True)
    backups: list[tuple[Path, Path]] = []
    for target_path in target_files:
        backup_path = backup_dir / target_path.name
        if backup_path.exists():
            raise FileExistsError(f"refusing to overwrite backup file: {backup_path}")
        shutil.copyfile(target_path, backup_path)
        backups.append((target_path, backup_path))

    target_dir = target_files[0].parent
    target_root = target_dir.parent
    dll_path = target_root / "x64" / configuration / "OverlayTrPlugInFx.dll"
    build_dll_path = target_dir / "x64" / configuration / "OverlayTrPlugInFx.dll"
    dll_backup = backup_dir / dll_path.name
    had_dll = dll_path.exists()
    if had_dll:
        shutil.copyfile(dll_path, dll_backup)

    try:
        for candidate_path, target_path in zip(candidate_files, target_files):
            shutil.copyfile(candidate_path, target_path)
        build = subprocess.run(
            [
                msbuild,
                str(target_dir / "OverlayTrPlugInFx.vcxproj"),
                f"/p:Configuration={configuration}",
                f"/p:Platform={platform}",
                "/t:Rebuild",
                "/m",
            ],
            cwd=workspace_root,
            env=_normalized_windows_environment(),
            capture_output=True,
            text=True,
            check=False,
        )
        if build.returncode != 0:
            repair_request, failure_file = _write_build_failure_repair_artifacts(
                candidate_manifest_file=candidate_manifest_file,
                candidate=candidate,
                candidate_files=candidate_files,
                backup_dir=backup_dir,
                iteration=iteration,
                stdout=build.stdout,
                stderr=build.stderr,
            )
            # Keep the failed candidate sources for the agent to repair, but do not
            # leave the registered target or production DLL in a broken state.
            for target_path, backup_path in backups:
                shutil.copyfile(backup_path, target_path)
            if had_dll:
                shutil.copyfile(dll_backup, dll_path)
            raise RuntimeError(
                "msbuild failed; target sources were restored. "
                f"Agent repair request: {repair_request}\n"
                f"Build failure report: {failure_file}\n"
                f"{build.stdout}\n{build.stderr}"
            )
        if not build_dll_path.exists():
            raise FileNotFoundError(f"candidate build did not produce plugin DLL: {build_dll_path}")
        shutil.copyfile(build_dll_path, dll_path)

        evaluation_job_file = job_file
        calibration: dict[str, Any] | None = None
        if calibrate_progress:
            calibration = calibrate_candidate_progress(
                workspace_root=workspace_root,
                candidate_manifest_file=candidate_manifest_file,
                job_file=job_file,
                output_dir=backup_dir / "progress_calibration",
                renderer=renderer,
                width=width,
                height=height,
                frame_count=frame_count,
                ffmpeg_path=ffmpeg_path,
                frame_start=frame_start,
                frame_end=frame_end,
            )
            evaluation_job_file = Path(calibration["aligned_job_file"])

        render_result = render_job(
            workspace_root=workspace_root,
            job_file=evaluation_job_file,
            output_root=output_root,
            renderer=renderer,
            ffmpeg_path=ffmpeg_path,
            run_name_prefix=_candidate_evaluation_run_name(
                candidate_manifest_file=candidate_manifest_file,
                iteration=iteration,
            ),
        )
        if render_result.get("status") != "succeeded":
            raise RuntimeError(f"candidate render failed: {render_result.get('message')}")
        run_root = Path(render_result["workspace"])
        render_job_definition = load_json(evaluation_job_file)
        planning = render_job_definition.get("planning")
        planning = planning if isinstance(planning, dict) else {}
        inputs = render_job_definition.get("inputs")
        inputs = inputs if isinstance(inputs, dict) else {}
        source_directories = []
        for key in ("source_a", "source_b"):
            source = inputs.get(key)
            if not isinstance(source, str):
                continue
            source_path = Path(source)
            source_directories.append(source_path if source_path.is_absolute() else workspace_root / source_path)
        score_file = run_root / "reports" / "score.json"
        score_result = score_candidate(
            workspace_root=workspace_root,
            candidate=Path(render_result["artifacts_dir"]),
            reference=reference,
            output_file=score_file,
            width=width,
            height=height,
            frame_count=frame_count,
            require_exact_frame_count=False,
            ffmpeg_path=ffmpeg_path,
            frame_start=frame_start,
            frame_end=frame_end,
            endpoint_frame_count=endpoint_frame_count,
            analysis_file=Path(candidate.get("analysis_artifact") or planning.get("analysis_artifact"))
            if isinstance(candidate.get("analysis_artifact") or planning.get("analysis_artifact"), str)
            else None,
            design_file=Path(candidate.get("design_artifact") or planning.get("design_artifact"))
            if isinstance(candidate.get("design_artifact") or planning.get("design_artifact"), str)
            else None,
            source_files=candidate_files,
            sampler_source=target_files[0].parent / "FxBase.cpp",
            source_directories=source_directories,
        )
        render_settings = render_job_definition.get("render", {})
        fps = render_settings.get("fps", 30)
        if not isinstance(fps, int) or fps < 1:
            raise ValueError("candidate evaluation job has invalid render.fps")
        comparison = _create_comparison_assets(
            artifacts_dir=Path(render_result["artifacts_dir"]),
            reference_dir=reference,
            fps=fps,
            frame_start=frame_start,
            frame_end=frame_end,
            ffmpeg_path=ffmpeg_path,
        )
        write_json(run_root / "reports" / "comparison_assets.json", comparison)
        report_file = run_root / "reports" / "candidate_iteration_report.json"
        report = build_report(
            analysis_file=Path(candidate.get("analysis_artifact", "")),
            design_file=Path(candidate.get("design_artifact", "")),
            render_file=run_root / "render_report.json",
            score_file=score_file,
            output_file=report_file,
        ) if candidate.get("analysis_artifact") and candidate.get("design_artifact") else {
            "report_type": "agent_candidate_iteration",
            "report_version": 1,
            "status": "succeeded",
            "effect_id": candidate.get("effect_id"),
            "render": render_result,
            "score": score_result,
        }
        report["comparison"] = comparison
        if calibration is not None:
            calibration_file = run_root / "reports" / "progress_calibration.json"
            write_json(calibration_file, calibration)
            report["progress_calibration"] = {
                **calibration,
                "artifact_file": str(calibration_file),
            }
        write_json(report_file, report)
        controller = (
            record_candidate_evaluation(
                candidate_manifest_file=candidate_manifest_file,
                iteration=iteration,
                report_file=report_file,
            )
            if iteration is not None
            else None
        )
        return {
            "status": "succeeded",
            "report": report,
            "report_file": str(report_file),
            "controller": controller,
        }
    finally:
        if restore:
            for target_path, backup_path in backups:
                shutil.copyfile(backup_path, target_path)
            if had_dll:
                shutil.copyfile(dll_backup, dll_path)


def _write_build_failure_repair_artifacts(
    candidate_manifest_file: Path,
    candidate: dict[str, Any],
    candidate_files: list[Path],
    backup_dir: Path,
    iteration: int | None,
    stdout: str,
    stderr: str,
) -> tuple[Path, Path]:
    """Persist compiler evidence and an agent-only repair request after a failed build."""
    failure_dir = backup_dir / "build_failure"
    failure_dir.mkdir(parents=True, exist_ok=True)
    failure_file = failure_dir / "build_failure.json"
    write_json(
        failure_file,
        {
            "artifact_type": "candidate_build_failure",
            "artifact_version": 1,
            "iteration": iteration,
            "effect_id": candidate.get("effect_id"),
            "candidate_manifest": str(candidate_manifest_file),
            "candidate_files": [str(path) for path in candidate_files],
            "stdout_file": str(failure_dir / "msbuild.stdout.log"),
            "stderr_file": str(failure_dir / "msbuild.stderr.log"),
            "status": "build_failed",
        },
    )
    (failure_dir / "msbuild.stdout.log").write_text(stdout or "", encoding="utf-8")
    (failure_dir / "msbuild.stderr.log").write_text(stderr or "", encoding="utf-8")

    candidate_dir = candidate_manifest_file.parent
    packet_dir = candidate_dir / "packets"
    packet_dir.mkdir(parents=True, exist_ok=True)
    iteration_label = f"{int(iteration):03d}" if iteration is not None else "unknown"
    request_file = packet_dir / f"iteration_{iteration_label}_build_repair_{backup_dir.name}.md"
    request_file.write_text(
        "\n".join(
            [
                "# Build Failure Repair",
                "",
                "Read:",
                f"- {candidate_manifest_file}",
                f"- {failure_file}",
                f"- {failure_dir / 'msbuild.stdout.log'}",
                f"- {failure_dir / 'msbuild.stderr.log'}",
                *[f"- {path}" for path in candidate_files],
                "",
                "Edit only the candidate workspace represented by the manifest.",
                "Fix compilation errors reported by MSBuild without changing the FX ID, class names, endpoint behavior, or candidate boundary.",
                "Do not perform visual refinement or create a new iteration record.",
                "The controller will rerun candidate-evaluate for the same iteration after this repair.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return request_file, failure_file


def calibrate_candidate_progress(
    workspace_root: Path,
    candidate_manifest_file: Path,
    job_file: Path,
    output_dir: Path,
    renderer: str | None,
    width: int,
    height: int,
    frame_count: int | None,
    ffmpeg_path: str | None,
    frame_start: int | None = None,
    frame_end: int | None = None,
) -> dict[str, Any]:
    """Probe a candidate at linear progress and derive an evaluation-local schedule."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite calibration directory: {output_dir}")
    job = load_json(job_file)
    render = job.get("render")
    inputs = job.get("inputs")
    if not isinstance(render, dict) or not isinstance(inputs, dict):
        raise ValueError("candidate evaluation job is missing render or inputs")
    count = frame_count or render.get("frame_count")
    if not isinstance(count, int) or count < 2:
        raise ValueError("candidate evaluation job has invalid render.frame_count")
    source_a = _resolve_workspace_path(workspace_root, inputs.get("source_a"), "inputs.source_a")
    source_b = _resolve_workspace_path(workspace_root, inputs.get("source_b"), "inputs.source_b")

    output_dir.mkdir(parents=True)
    probe_job = {**job, "render": {**render}}
    probe_job["render"].pop("progress_schedule", None)
    probe_job["job_name"] = f"{job.get('job_name', 'candidate')}_linear_probe"
    probe_job_file = output_dir / "linear_probe_job.json"
    write_json(probe_job_file, probe_job)
    probe = render_job(
        workspace_root=workspace_root,
        job_file=probe_job_file,
        output_root=output_dir / "runs",
        renderer=renderer,
        ffmpeg_path=ffmpeg_path,
    )
    if probe.get("status") != "succeeded":
        raise RuntimeError(f"progress calibration probe failed: {probe.get('message')}")

    probe_frames = Path(probe["artifacts_dir"])
    mae_to_a, mae_to_b, source_mae = _probe_endpoint_distances(
        probe_frames=probe_frames,
        source_a=source_a,
        source_b=source_b,
        frame_count=count,
    )
    calibration = _detect_progress_calibration(
        mae_to_a=mae_to_a,
        mae_to_b=mae_to_b,
        source_mae=source_mae,
    )
    reference_window = _reference_output_window(
        candidate_manifest_file=candidate_manifest_file,
        reference_path=_resolve_workspace_path(workspace_root, inputs.get("reference_transition"), "inputs.reference_transition"),
        frame_count=count,
        analysis_file=(job.get("planning") or {}).get("analysis_artifact") if isinstance(job.get("planning"), dict) else None,
        requested_frame_start=frame_start,
        requested_frame_end=frame_end,
    )
    schedule = _build_progress_schedule(
        frame_count=count,
        frame_start=reference_window["frame_start"],
        frame_end=reference_window["frame_end"],
        progress_start=calibration["active_progress_start"],
        progress_end=calibration["active_progress_end"],
    )
    aligned_job = {**job, "render": {**render, "progress_schedule": schedule}}
    aligned_job_file = output_dir / "aligned_evaluation_job.json"
    write_json(aligned_job_file, aligned_job)
    result = {
        "artifact_type": "candidate_progress_calibration",
        "artifact_version": 1,
        "status": calibration["status"],
        "method": "linear_probe_endpoint_distance",
        "probe_job_file": str(probe_job_file),
        "probe_render": probe,
        "reference_window": reference_window,
        "candidate_interval": calibration,
        "aligned_job_file": str(aligned_job_file),
    }
    write_json(output_dir / "progress_calibration.json", result)
    return result


def _detect_progress_calibration(
    mae_to_a: list[float], mae_to_b: list[float], source_mae: float
) -> dict[str, Any]:
    if len(mae_to_a) != len(mae_to_b) or len(mae_to_a) < 2:
        raise ValueError("progress calibration requires matching probe endpoint scores")
    threshold = max(2.0, source_mae * 0.03)
    active = [min(distance_a, distance_b) > threshold for distance_a, distance_b in zip(mae_to_a, mae_to_b)]
    indexes = [index for index, is_active in enumerate(active) if is_active]
    count = len(active)
    if len(indexes) < 2 or source_mae <= 2.0:
        return {
            "status": "needs_review",
            "confidence": 0.0,
            "reason": "probe did not show a reliable interval distinct from both endpoints; using linear progress",
            "active_frame_start": 0,
            "active_frame_end": count - 1,
            "active_progress_start": 0.0,
            "active_progress_end": 1.0,
            "threshold_mae": threshold,
            "source_endpoint_mae": source_mae,
        }
    start, end = indexes[0], indexes[-1]
    confidence = 0.9 if start > 0 and end < count - 1 else 0.65
    return {
        "status": "succeeded",
        "confidence": confidence,
        "reason": "detected frames distinct from both stable endpoint sources",
        "active_frame_start": start,
        "active_frame_end": end,
        "active_progress_start": start / (count - 1),
        "active_progress_end": end / (count - 1),
        "threshold_mae": threshold,
        "source_endpoint_mae": source_mae,
    }


def _probe_endpoint_distances(
    probe_frames: Path, source_a: Path, source_b: Path, frame_count: int
) -> tuple[list[float], list[float], float]:
    """Measure only the low-resolution endpoint distance required for calibration."""
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("OpenCV and NumPy are required for progress calibration") from error

    def read(path: Path) -> Any:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"could not decode calibration frame: {path}")
        return cv2.resize(image, (64, 36), interpolation=cv2.INTER_AREA).astype(numpy.float32)

    a = read(source_a / "frame_0000.png")
    b = read(source_b / "frame_0000.png")
    mae_to_a: list[float] = []
    mae_to_b: list[float] = []
    for index in range(frame_count):
        frame = read(probe_frames / f"frame_{index:04d}.png")
        mae_to_a.append(float(numpy.mean(numpy.abs(frame - a))))
        mae_to_b.append(float(numpy.mean(numpy.abs(frame - b))))
    return mae_to_a, mae_to_b, float(numpy.mean(numpy.abs(a - b)))


def _reference_output_window(
    candidate_manifest_file: Path,
    reference_path: Path,
    frame_count: int,
    analysis_file: str | None = None,
    requested_frame_start: int | None = None,
    requested_frame_end: int | None = None,
) -> dict[str, Any]:
    if requested_frame_start is not None or requested_frame_end is not None:
        if requested_frame_start is None or requested_frame_end is None:
            raise ValueError("progress calibration requires both frame_start and frame_end")
        if (
            requested_frame_start < 0
            or requested_frame_end < requested_frame_start
            or requested_frame_end >= frame_count
        ):
            raise ValueError("progress calibration frame range must be within the render frame count")
        return {
            "frame_start": requested_frame_start,
            "frame_end": requested_frame_end,
            "source": "evaluation_arguments",
        }

    manifest = load_json(candidate_manifest_file)
    analysis_file = analysis_file or manifest.get("analysis_artifact")
    reference_manifest = load_json(reference_path / "reference_transition_manifest.json")
    mapping = reference_manifest.get("frame_progress_mapping")
    if not isinstance(analysis_file, str) or not isinstance(mapping, list):
        return {"frame_start": 0, "frame_end": frame_count - 1, "source": "full_render_fallback"}
    transition = load_json(Path(analysis_file)).get("transition", {})
    start_source, end_source = transition.get("start_frame"), transition.get("end_frame")
    if not isinstance(start_source, int) or not isinstance(end_source, int):
        return {"frame_start": 0, "frame_end": frame_count - 1, "source": "full_render_fallback"}
    if 0 <= start_source <= end_source < frame_count:
        return {
            "frame_start": start_source,
            "frame_end": end_source,
            "source": "analysis_output_frames",
        }
    matched = [item.get("output_frame") for item in mapping if isinstance(item, dict) and start_source <= item.get("normalized_clip_source_frame", -1) <= end_source]
    if not matched or not all(isinstance(item, int) for item in matched):
        return {"frame_start": 0, "frame_end": frame_count - 1, "source": "full_render_fallback"}
    return {
        "frame_start": min(matched),
        "frame_end": max(matched),
        "source": "legacy_analysis_source_frames",
    }


def _resolve_workspace_path(workspace_root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"candidate evaluation job has invalid {field}")
    path = Path(value)
    return path if path.is_absolute() else workspace_root / path


def _candidate_evaluation_run_name(
    candidate_manifest_file: Path,
    iteration: int | None,
) -> str | None:
    """Build a readable evaluation prefix from its iteration record and effect job."""
    if iteration is None:
        return None

    iteration_records = sorted(
        candidate_manifest_file.parent.glob(f"iteration_{iteration:03d}_*.json")
    )
    if len(iteration_records) != 1:
        return None

    return iteration_records[0].stem


def build_job_from_artifacts(
    analysis_file: Path,
    design_file: Path,
    source_a: str,
    source_b: str,
    reference_transition: str | None,
    output_file: Path,
    width: int,
    height: int,
    fps: int,
    frame_count: int | None,
    progress_frame_start: int | None = None,
    progress_frame_end: int | None = None,
    progress_value_start: float | None = None,
    progress_value_end: float | None = None,
) -> dict[str, Any]:
    resolved_frame_count = frame_count or _reference_frame_count(reference_transition) or 30
    progress_schedule = _build_progress_schedule(
        frame_count=resolved_frame_count,
        frame_start=progress_frame_start,
        frame_end=progress_frame_end,
        progress_start=progress_value_start,
        progress_end=progress_value_end,
    )
    job = build_render_job(
        analysis=load_json(analysis_file),
        design=load_json(design_file),
        source_a=source_a,
        source_b=source_b,
        reference_transition=reference_transition,
        width=width,
        height=height,
        fps=fps,
        frame_count=resolved_frame_count,
        progress_schedule=progress_schedule,
    )
    job["planning"]["analysis_artifact"] = str(analysis_file.resolve())
    job["planning"]["design_artifact"] = str(design_file.resolve())
    write_json(output_file, job)
    return job


def _build_progress_schedule(
    frame_count: int,
    frame_start: int | None,
    frame_end: int | None,
    progress_start: float | None,
    progress_end: float | None,
) -> list[float] | None:
    values = (frame_start, frame_end, progress_start, progress_end)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(
            "progress scheduling requires frame start/end and progress start/end together"
        )
    assert frame_start is not None and frame_end is not None
    assert progress_start is not None and progress_end is not None
    if frame_start < 0 or frame_end < frame_start or frame_end >= frame_count:
        raise ValueError("progress frame range must be within the render frame count")
    if not 0 <= progress_start <= progress_end <= 1:
        raise ValueError("progress values must be non-decreasing and within 0 through 1")
    span = frame_end - frame_start
    return [
        0.0
        if frame_index < frame_start
        else 1.0
        if frame_index > frame_end
        else progress_start
        if span == 0
        else progress_start + (progress_end - progress_start) * (frame_index - frame_start) / span
        for frame_index in range(frame_count)
    ]


def _reference_frame_count(reference_transition: str | None) -> int | None:
    if not reference_transition:
        return None
    reference_path = Path(reference_transition)
    manifest_path = (
        reference_path / "reference_transition_manifest.json"
        if reference_path.is_dir()
        else reference_path
    )
    if not manifest_path.exists():
        return None
    manifest = load_json(manifest_path)
    frame_count = manifest.get("frame_count")
    return frame_count if isinstance(frame_count, int) and frame_count >= 2 else None


def _normalized_windows_environment() -> dict[str, str]:
    """Avoid ProcessStartInfo failures from duplicate case-insensitive env keys."""
    normalized: dict[str, str] = {}
    seen: set[str] = set()
    for key, value in os.environ.items():
        folded = key.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        normalized[key] = value
    return normalized


def _extract_single_frame(
    ffmpeg_executable: str,
    source_video: Path,
    frame_index: int,
    output_file: Path,
    width: int,
    height: int,
) -> None:
    completed = subprocess.run(
        [
            ffmpeg_executable,
            "-v",
            "error",
            "-y",
            "-i",
            str(source_video),
            "-vf",
            f"select='eq(n\\,{frame_index})',scale={width}:{height}",
            "-frames:v",
            "1",
            str(output_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to extract source frame {frame_index}: {completed.stderr.strip()}"
        )
    if not output_file.exists():
        raise RuntimeError(f"ffmpeg produced no frame for source index {frame_index}")


def _probe_video_frame_count(ffmpeg_executable: str, source_video: Path) -> int:
    ffprobe_executable = Path(ffmpeg_executable).with_name("ffprobe.exe")
    if not ffprobe_executable.exists():
        ffprobe_executable = Path(shutil.which("ffprobe") or "")
    if not ffprobe_executable.exists():
        raise RuntimeError("ffprobe is required to select the final source-video frame")
    completed = subprocess.run(
        [
            str(ffprobe_executable),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames,nb_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(source_video),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe failed to count source frames: {completed.stderr.strip()}")
    for value in completed.stdout.splitlines():
        try:
            frame_count = int(value.strip())
        except ValueError:
            continue
        if frame_count > 0:
            return frame_count
    raise RuntimeError("ffprobe did not return a positive source-video frame count")


def _clear_png_frames(directory: Path) -> None:
    for path in directory.glob("frame_*.png"):
        path.unlink()
