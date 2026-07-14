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
