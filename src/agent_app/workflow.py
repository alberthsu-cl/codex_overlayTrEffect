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
    start_frame: int,
    end_frame: int,
    frame_count: int,
    width: int,
    height: int,
    ffmpeg_path: str | None = None,
) -> dict[str, Any]:
    if start_frame < 0 or end_frame < 0 or end_frame < start_frame:
        raise ValueError("source frame boundaries must be non-negative and ordered")
    if frame_count < 2:
        raise ValueError("frame_count must be at least 2")
    if not source_video.exists():
        raise FileNotFoundError(f"source video does not exist: {source_video}")

    ffmpeg_executable = ffmpeg_path or shutil.which("ffmpeg")
    if not ffmpeg_executable:
        raise RuntimeError("ffmpeg is required for source preparation but was not found on PATH")

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
) -> dict[str, Any]:
    modules = load_harness_modules(workspace_root)
    job = modules["load_render_job"](job_file)
    run_root = output_root / f"{job.job_name}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
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
    if frame_start is not None or frame_end is not None:
        score.update(
            _build_windowed_score(
                score,
                frame_start=frame_start,
                frame_end=frame_end,
                endpoint_frame_count=endpoint_frame_count,
            )
        )
        score["transition_diagnostics"] = _build_transition_diagnostics(score)
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
            raise RuntimeError(f"msbuild failed:\n{build.stdout}\n{build.stderr}")
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
            )
            evaluation_job_file = Path(calibration["aligned_job_file"])

        render_result = render_job(
            workspace_root=workspace_root,
            job_file=evaluation_job_file,
            output_root=output_root,
            renderer=renderer,
            ffmpeg_path=ffmpeg_path,
        )
        if render_result.get("status") != "succeeded":
            raise RuntimeError(f"candidate render failed: {render_result.get('message')}")
        run_root = Path(render_result["workspace"])
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
        )
        render_job_definition = load_json(evaluation_job_file)
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
) -> dict[str, int]:
    manifest = load_json(candidate_manifest_file)
    analysis_file = analysis_file or manifest.get("analysis_artifact")
    reference_manifest = load_json(reference_path / "reference_transition_manifest.json")
    mapping = reference_manifest.get("frame_progress_mapping")
    if not isinstance(analysis_file, str) or not isinstance(mapping, list):
        return {"frame_start": 0, "frame_end": frame_count - 1}
    transition = load_json(Path(analysis_file)).get("transition", {})
    start_source, end_source = transition.get("start_frame"), transition.get("end_frame")
    if not isinstance(start_source, int) or not isinstance(end_source, int):
        return {"frame_start": 0, "frame_end": frame_count - 1}
    matched = [item.get("output_frame") for item in mapping if isinstance(item, dict) and start_source <= item.get("normalized_clip_source_frame", -1) <= end_source]
    if not matched or not all(isinstance(item, int) for item in matched):
        return {"frame_start": 0, "frame_end": frame_count - 1}
    return {"frame_start": min(matched), "frame_end": max(matched)}


def _resolve_workspace_path(workspace_root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"candidate evaluation job has invalid {field}")
    path = Path(value)
    return path if path.is_absolute() else workspace_root / path


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


def _clear_png_frames(directory: Path) -> None:
    for path in directory.glob("frame_*.png"):
        path.unlink()
