from __future__ import annotations

import argparse
import json
from pathlib import Path

from .workflow import (
    build_job_from_artifacts,
    build_report,
    benchmark_effects,
    prepare_reference,
    prepare_sources,
    retrieve_effect,
    render_job,
    score_candidate,
)
from .codegen import generate_effect, register_effect


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    workspace_root = Path(__file__).resolve().parents[3]

    try:
        if args.command == "prepare":
            result = prepare_reference(
                workspace_root=workspace_root,
                source_video=Path(args.source_video).resolve(),
                output_dir=Path(args.output_dir).resolve(),
                fps=args.fps,
                width=args.width,
                height=args.height,
                target_frame_count=args.target_frame_count,
                ffmpeg_path=args.ffmpeg,
            )
        elif args.command == "prepare-sources":
            result = prepare_sources(
                source_video=Path(args.source_video).resolve(),
                output_root=Path(args.output_root).resolve(),
                start_frame=args.start_frame,
                end_frame=args.end_frame,
                frame_count=args.frame_count,
                width=args.width,
                height=args.height,
                ffmpeg_path=args.ffmpeg,
            )
        elif args.command == "retrieve":
            result = retrieve_effect(
                workspace_root=workspace_root,
                analysis_file=Path(args.analysis).resolve(),
                output_file=Path(args.output).resolve(),
            )
        elif args.command == "benchmark":
            result = benchmark_effects(
                workspace_root=workspace_root,
                analysis_file=Path(args.analysis).resolve(),
                source_a=args.source_a,
                source_b=args.source_b,
                reference_transition=Path(args.reference_transition).resolve(),
                output_root=Path(args.output_root).resolve(),
                output_file=Path(args.output).resolve(),
                family=args.family,
                width=args.width,
                height=args.height,
                fps=args.fps,
                frame_count=args.frame_count,
                renderer=args.renderer or _default_renderer(workspace_root),
                ffmpeg_path=args.ffmpeg,
            )
        elif args.command == "render":
            renderer = args.renderer or _default_renderer(workspace_root)
            result = render_job(
                workspace_root=workspace_root,
                job_file=Path(args.job).resolve(),
                output_root=Path(args.output_root).resolve(),
                renderer=renderer,
            )
        elif args.command == "score":
            result = score_candidate(
                workspace_root=workspace_root,
                candidate=Path(args.candidate).resolve(),
                reference=Path(args.reference).resolve(),
                output_file=Path(args.output).resolve(),
                width=args.width,
                height=args.height,
                frame_count=args.frame_count,
                require_exact_frame_count=args.require_exact_frame_count,
                ffmpeg_path=args.ffmpeg,
            )
        elif args.command == "build-job":
            result = build_job_from_artifacts(
                analysis_file=Path(args.analysis).resolve(),
                design_file=Path(args.design).resolve(),
                source_a=args.source_a,
                source_b=args.source_b,
                reference_transition=args.reference_transition,
                output_file=Path(args.output).resolve(),
                width=args.width,
                height=args.height,
                fps=args.fps,
                frame_count=args.frame_count,
            )
            result = {"status": "succeeded", "job": result, "output": str(Path(args.output).resolve())}
        elif args.command == "generate":
            result = {
                "status": "succeeded",
                "manifest": generate_effect(
                    design_file=Path(args.design).resolve(),
                    output_dir=Path(args.output_dir).resolve(),
                    template_root=Path(args.template_root).resolve()
                    if args.template_root
                    else workspace_root / "overlaytrengine" / "OverlayTrPlugInFx",
                    manifest_file=Path(args.manifest).resolve(),
                    force=args.force,
                ),
            }
        elif args.command == "register":
            result = {
                "status": "succeeded",
                "registration": register_effect(
                    manifest_file=Path(args.manifest).resolve(),
                    target_root=Path(args.target_root).resolve(),
                ),
            }
        else:
            result = build_report(
                analysis_file=Path(args.analysis).resolve(),
                design_file=Path(args.design).resolve(),
                render_file=Path(args.render_report).resolve(),
                score_file=Path(args.score_report).resolve(),
                output_file=Path(args.output).resolve(),
            )
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as error:
        parser.exit(1, f"agent: error: {error}\n")

    print(json.dumps(result, indent=2))
    return 0 if result.get("status") in {"succeeded", "blocked"} else 1


def _default_renderer(workspace_root: Path) -> str | None:
    path = workspace_root / "harness" / "native_renderer" / "build" / "x64" / "Debug" / "OverlayTrHarnessRenderer.exe"
    return str(path) if path.exists() else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Codex-driven transition effect workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="prepare normalized reference transition frames")
    prepare.add_argument("--source-video", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--width", type=int, default=1920)
    prepare.add_argument("--height", type=int, default=1080)
    prepare.add_argument("--fps", type=int, default=30)
    prepare.add_argument("--target-frame-count", type=int, default=30)
    prepare.add_argument("--ffmpeg")

    prepare_sources_cmd = subparsers.add_parser(
        "prepare-sources",
        help="extract source A/B frames from the analysis boundaries and repeat them",
    )
    prepare_sources_cmd.add_argument("--source-video", required=True)
    prepare_sources_cmd.add_argument("--output-root", required=True)
    prepare_sources_cmd.add_argument("--start-frame", type=int, required=True)
    prepare_sources_cmd.add_argument("--end-frame", type=int, required=True)
    prepare_sources_cmd.add_argument("--frame-count", type=int, default=30)
    prepare_sources_cmd.add_argument("--width", type=int, default=1920)
    prepare_sources_cmd.add_argument("--height", type=int, default=1080)
    prepare_sources_cmd.add_argument("--ffmpeg")

    retrieve_cmd = subparsers.add_parser(
        "retrieve",
        help="retrieve the closest built-in effect from the analysis family",
    )
    retrieve_cmd.add_argument("--analysis", required=True)
    retrieve_cmd.add_argument("--output", required=True)

    benchmark_cmd = subparsers.add_parser(
        "benchmark",
        help="render and score every built-in effect in an effect family",
    )
    benchmark_cmd.add_argument("--analysis", required=True)
    benchmark_cmd.add_argument("--source-a", required=True)
    benchmark_cmd.add_argument("--source-b", required=True)
    benchmark_cmd.add_argument("--reference-transition", required=True)
    benchmark_cmd.add_argument("--output-root", required=True)
    benchmark_cmd.add_argument("--output", required=True)
    benchmark_cmd.add_argument("--family")
    benchmark_cmd.add_argument("--width", type=int, default=1920)
    benchmark_cmd.add_argument("--height", type=int, default=1080)
    benchmark_cmd.add_argument("--fps", type=int, default=30)
    benchmark_cmd.add_argument("--frame-count", type=int, default=30)
    benchmark_cmd.add_argument("--renderer")
    benchmark_cmd.add_argument("--ffmpeg")

    render = subparsers.add_parser("render", help="render a JSON job with the existing headless renderer")
    render.add_argument("--job", required=True)
    render.add_argument("--output-root", required=True)
    render.add_argument("--renderer")

    score = subparsers.add_parser("score", help="score candidate frames against reference frames")
    score.add_argument("--candidate", required=True)
    score.add_argument("--reference", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--width", type=int, required=True)
    score.add_argument("--height", type=int, required=True)
    score.add_argument("--frame-count", type=int)
    score.add_argument("--require-exact-frame-count", action="store_true")
    score.add_argument("--ffmpeg")

    build_job = subparsers.add_parser(
        "build-job",
        help="build an existing-effect render job from Codex analysis and design artifacts",
    )
    build_job.add_argument("--analysis", required=True)
    build_job.add_argument("--design", required=True)
    build_job.add_argument("--source-a", required=True)
    build_job.add_argument("--source-b", required=True)
    build_job.add_argument("--reference-transition")
    build_job.add_argument("--output", required=True)
    build_job.add_argument("--width", type=int, default=1920)
    build_job.add_argument("--height", type=int, default=1080)
    build_job.add_argument("--fps", type=int, default=30)
    build_job.add_argument(
        "--frame-count",
        type=int,
        help="override the render frame count; otherwise use the reference manifest, then 30",
    )

    generate = subparsers.add_parser(
        "generate",
        help="generate isolated C++/HLSL sources from a supported effect template",
    )
    generate.add_argument("--design", required=True)
    generate.add_argument("--output-dir", required=True)
    generate.add_argument("--manifest", required=True)
    generate.add_argument("--template-root")
    generate.add_argument("--force", action="store_true")

    register = subparsers.add_parser(
        "register",
        help="copy a generated package into OverlayTrPlugInFx and register it",
    )
    register.add_argument("--manifest", required=True)
    register.add_argument("--target-root", required=True)

    report = subparsers.add_parser("report", help="combine analysis, design, render, and score artifacts")
    report.add_argument("--analysis", required=True)
    report.add_argument("--design", required=True)
    report.add_argument("--render-report", required=True)
    report.add_argument("--score-report", required=True)
    report.add_argument("--output", required=True)

    return parser
