from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .harness_bridge import load_harness_modules
from .io import load_json, write_json


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

