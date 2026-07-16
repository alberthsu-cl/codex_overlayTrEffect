from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import shutil
import subprocess
from typing import Any

from .harness_bridge import load_harness_modules
from .io import load_json, write_json
from .artifacts import build_render_job


def prepare_reference(
    workspace_root: Path,
    source_video: Path,
    output_dir: Path,
    fps: int,
    width: int,
    height: int,
    target_frame_count: int,
    ffmpeg_path: str | None = None,
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
    }
    write_json(run_root / "render_report.json", result)
    return result


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
    score["status"] = "succeeded"
    write_json(output_file, score)
    return score


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
) -> dict[str, Any]:
    """Temporarily stage, build, render, score, and restore a candidate."""
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
                "/m",
            ],
            cwd=workspace_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if build.returncode != 0:
            raise RuntimeError(f"msbuild failed:\n{build.stdout}\n{build.stderr}")

        render_result = render_job(
            workspace_root=workspace_root,
            job_file=job_file,
            output_root=output_root,
            renderer=renderer,
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
        )
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
        if not report_file.exists():
            write_json(report_file, report)
        return {"status": "succeeded", "report": report, "report_file": str(report_file)}
    finally:
        for target_path, backup_path in backups:
            shutil.copyfile(backup_path, target_path)
        if had_dll:
            shutil.copyfile(dll_backup, dll_path)


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
) -> dict[str, Any]:
    resolved_frame_count = frame_count or _reference_frame_count(reference_transition) or 30
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
    )
    write_json(output_file, job)
    return job


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
