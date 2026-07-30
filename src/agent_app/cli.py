from __future__ import annotations

import argparse
import json
from pathlib import Path

from .workflow import (
    build_job_from_artifacts,
    build_report,
    benchmark_effects,
    evaluate_candidate,
    analyze_reference_diagnostics,
    ensure_reference_diagnostics,
    prepare_reference,
    prepare_sources,
    retrieve_effect,
    render_job,
    score_candidate,
)
from .codegen import (
    generate_effect,
    initialize_candidate,
    promote_candidate,
    register_effect,
)
from .candidate_controller import (
    apply_reassessed_baseline,
    build_next_iteration_packet,
    candidate_status,
    continue_candidate_refinement,
    human_accept_candidate,
    reassess_candidate_history,
    record_candidate_evaluation,
    restore_candidate_baseline,
    set_candidate_baseline,
    set_evaluation_profile,
    resume_candidate_refinement,
    start_refinement_phase,
)
from .sample_workspace import initialize_sample_workspace


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    workspace_root = Path(__file__).resolve().parents[3]

    try:
        if args.command == "sample-init":
            result = initialize_sample_workspace(
                samples_root=Path(args.output_root).resolve()
                if args.output_root
                else workspace_root / "agent" / "work" / "samples",
                sample_id=args.sample_id,
                source_video=Path(args.source_video).resolve(),
                workspace_root=workspace_root,
            )
        elif args.command == "prepare":
            result = prepare_reference(
                workspace_root=workspace_root,
                source_video=Path(args.source_video).resolve(),
                output_dir=Path(args.output_dir).resolve(),
                fps=args.fps,
                width=args.width,
                height=args.height,
                target_frame_count=args.target_frame_count,
                ffmpeg_path=args.ffmpeg,
                start_frame=args.start_frame,
                end_frame=args.end_frame,
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
                analysis_file=Path(args.analysis).resolve() if args.analysis else None,
                reference_manifest_file=Path(args.reference_manifest).resolve()
                if args.reference_manifest
                else None,
            )
        elif args.command == "reference-diagnostics":
            reference_dir = Path(args.reference).resolve()
            canonical_output_dir = reference_dir.parent / "diagnostics"
            if args.output_dir:
                requested_output_dir = Path(args.output_dir).resolve()
                if requested_output_dir != canonical_output_dir:
                    raise ValueError(
                        "reference diagnostics must be written to the canonical folder beside "
                        f"the reference directory: {canonical_output_dir}"
                    )
            result = analyze_reference_diagnostics(
                workspace_root=workspace_root,
                reference=reference_dir,
                output_dir=canonical_output_dir,
                width=args.width,
                height=args.height,
                frame_start=args.frame_start,
                frame_end=args.frame_end,
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
                ffmpeg_path=args.ffmpeg,
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
                frame_start=args.frame_start,
                frame_end=args.frame_end,
                endpoint_frame_count=args.endpoint_frame_count,
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
                progress_frame_start=args.progress_frame_start,
                progress_frame_end=args.progress_frame_end,
                progress_value_start=args.progress_value_start,
                progress_value_end=args.progress_value_end,
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
        elif args.command == "candidate-init":
            result = {
                "status": "succeeded",
                "candidate": initialize_candidate(
                    manifest_file=Path(args.manifest).resolve(),
                    output_dir=Path(args.output_dir).resolve(),
                    force=args.force,
                ),
            }
        elif args.command == "candidate-promote":
            result = promote_candidate(
                candidate_manifest_file=Path(args.manifest).resolve(),
                backup_dir=Path(args.backup_dir).resolve(),
            )
        elif args.command == "candidate-evaluate":
            result = evaluate_candidate(
                workspace_root=workspace_root,
                candidate_manifest_file=Path(args.manifest).resolve(),
                job_file=Path(args.job).resolve(),
                reference=Path(args.reference).resolve(),
                output_root=Path(args.output_root).resolve(),
                backup_dir=Path(args.backup_dir).resolve(),
                msbuild=args.msbuild,
                configuration=args.configuration,
                platform=args.platform,
                renderer=args.renderer or _default_renderer(workspace_root),
                width=args.width,
                height=args.height,
                frame_count=args.frame_count,
                ffmpeg_path=args.ffmpeg,
                restore=args.restore,
                frame_start=args.frame_start,
                frame_end=args.frame_end,
                endpoint_frame_count=args.endpoint_frame_count,
                iteration=args.iteration,
                calibrate_progress=args.calibrate_progress,
            )
        elif args.command == "candidate-set-baseline":
            result = set_candidate_baseline(
                candidate_manifest_file=Path(args.manifest).resolve(),
                iteration=args.iteration,
                report_file=Path(args.report).resolve(),
                source_dir=Path(args.source_dir).resolve() if args.source_dir else None,
            )
        elif args.command == "candidate-set-evaluation-profile":
            manifest_file = Path(args.manifest).resolve()
            result = set_evaluation_profile(
                candidate_manifest_file=manifest_file,
                profile={
                    "manifest": str(manifest_file),
                    "job": str(Path(args.job).resolve()),
                    "reference": str(Path(args.reference).resolve()),
                    "output_root": str(Path(args.output_root).resolve()),
                    "backup_root": str(Path(args.backup_root).resolve()),
                    "msbuild": args.msbuild,
                    "renderer": args.renderer or _default_renderer(workspace_root),
                    "configuration": args.configuration,
                    "platform": args.platform,
                    "width": args.width,
                    "height": args.height,
                    "frame_start": args.frame_start,
                    "frame_end": args.frame_end,
                    "calibrate_progress": not args.no_calibrate_progress,
                },
            )
        elif args.command == "candidate-restore-baseline":
            result = restore_candidate_baseline(Path(args.manifest).resolve())
        elif args.command == "candidate-human-accept":
            result = human_accept_candidate(
                candidate_manifest_file=Path(args.manifest).resolve(),
                iteration=args.iteration,
                reviewer=args.reviewer,
                reason=args.reason,
            )
        elif args.command == "candidate-start-phase":
            result = start_refinement_phase(
                candidate_manifest_file=Path(args.manifest).resolve(),
                name=args.name,
                baseline_iteration=args.baseline_iteration,
                report_file=Path(args.report).resolve(),
                max_iterations=args.max_iterations,
                max_rejected=args.max_rejected,
                source_dir=Path(args.source_dir).resolve() if args.source_dir else None,
            )
        elif args.command == "candidate-resume":
            manifest_file = Path(args.manifest).resolve()
            analysis_file = Path(args.analysis).resolve()
            profile_state = json.loads((manifest_file.parent / "candidate_state.json").read_text(encoding="utf-8"))
            profile = profile_state.get("evaluation_profile")
            if not isinstance(profile, dict):
                raise ValueError("candidate has no evaluation profile; run candidate-set-evaluation-profile first")
            diagnostics = ensure_reference_diagnostics(
                workspace_root=workspace_root,
                reference=Path(str(profile["reference"])).resolve(),
                width=int(profile["width"]),
                height=int(profile["height"]),
            )
            result = resume_candidate_refinement(
                candidate_manifest_file=manifest_file,
                analysis_file=analysis_file,
                design_file=Path(args.design).resolve(),
                phase_name=args.phase,
                max_iterations=args.max_iterations,
                max_rejected=args.max_rejected,
            )
            result["diagnostics"] = diagnostics
        elif args.command == "candidate-record-score":
            result = record_candidate_evaluation(
                candidate_manifest_file=Path(args.manifest).resolve(),
                iteration=args.iteration,
                report_file=Path(args.report).resolve(),
            )
        elif args.command == "candidate-reassess":
            manifest_file = Path(args.manifest).resolve()
            result = (
                apply_reassessed_baseline(manifest_file)
                if args.apply_best
                else reassess_candidate_history(manifest_file)
            )
        elif args.command == "candidate-next":
            result = build_next_iteration_packet(
                candidate_manifest_file=Path(args.manifest).resolve(),
                analysis_file=Path(args.analysis).resolve(),
                design_file=Path(args.design).resolve(),
                max_iterations=args.max_iterations,
                max_rejected=args.max_rejected,
                evaluate_after_edit=args.evaluate_after_edit,
            )
        elif args.command == "candidate-continue":
            manifest_file = Path(args.manifest).resolve()
            state_file = manifest_file.parent / "candidate_state.json"
            diagnostics = None
            if state_file.is_file():
                profile_state = json.loads(state_file.read_text(encoding="utf-8"))
                profile = profile_state.get("evaluation_profile")
                if isinstance(profile, dict):
                    diagnostics = ensure_reference_diagnostics(
                        workspace_root=workspace_root,
                        reference=Path(str(profile["reference"])).resolve(),
                        width=int(profile["width"]),
                        height=int(profile["height"]),
                    )
            result = continue_candidate_refinement(
                candidate_manifest_file=manifest_file,
                analysis_file=Path(args.analysis).resolve(),
                design_file=Path(args.design).resolve(),
                max_iterations=args.max_iterations,
                max_rejected=args.max_rejected,
            )
            if diagnostics is not None:
                result["diagnostics"] = diagnostics
        elif args.command == "candidate-status":
            result = candidate_status(Path(args.manifest).resolve())
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

    sample_init = subparsers.add_parser(
        "sample-init",
        help="create an isolated work area for one sample transition video",
    )
    sample_init.add_argument("--sample-id", required=True)
    sample_init.add_argument("--source-video", required=True)
    sample_init.add_argument(
        "--output-root",
        help="parent folder for sample workspaces; defaults to agent/work/samples",
    )

    prepare = subparsers.add_parser("prepare", help="prepare normalized reference transition frames")
    prepare.add_argument("--source-video", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--width", type=int, default=1920)
    prepare.add_argument("--height", type=int, default=1080)
    prepare.add_argument("--fps", type=int, default=30)
    prepare.add_argument("--target-frame-count", type=int, default=30)
    prepare.add_argument(
        "--start-frame",
        type=int,
        help="manual normalized-video transition start frame; requires --end-frame",
    )
    prepare.add_argument(
        "--end-frame",
        type=int,
        help="manual normalized-video transition end frame; requires --start-frame",
    )
    prepare.add_argument("--ffmpeg")

    prepare_sources_cmd = subparsers.add_parser(
        "prepare-sources",
        help="extract stable source A/B frames and repeat them",
    )
    prepare_sources_cmd.add_argument("--source-video", required=True)
    prepare_sources_cmd.add_argument("--output-root", required=True)
    prepare_sources_cmd.add_argument(
        "--start-frame",
        type=int,
        help="stable source-A frame in the original source video; requires --end-frame",
    )
    prepare_sources_cmd.add_argument(
        "--end-frame",
        type=int,
        help="stable source-B frame in the original source video; requires --start-frame",
    )
    prepare_sources_cmd.add_argument(
        "--analysis",
        help="transition_structure.json with prepared-reference stable A/B boundary frames",
    )
    prepare_sources_cmd.add_argument(
        "--reference-manifest",
        help="prepared reference_transition_manifest.json used to map analysis frames to original video frames",
    )
    prepare_sources_cmd.add_argument("--frame-count", type=int, default=30)
    prepare_sources_cmd.add_argument("--width", type=int, default=1920)
    prepare_sources_cmd.add_argument("--height", type=int, default=1080)
    prepare_sources_cmd.add_argument("--ffmpeg")

    reference_diagnostics = subparsers.add_parser(
        "reference-diagnostics",
        help="analyze prepared reference frames into flow, dynamic-region, and confidence evidence",
    )
    reference_diagnostics.add_argument("--reference", required=True)
    reference_diagnostics.add_argument(
        "--output-dir",
        help="must be the diagnostics folder beside the prepared reference directory",
    )
    reference_diagnostics.add_argument("--width", type=int, default=1920)
    reference_diagnostics.add_argument("--height", type=int, default=1080)
    reference_diagnostics.add_argument("--frame-start", type=int, default=0)
    reference_diagnostics.add_argument("--frame-end", type=int)
    reference_diagnostics.add_argument("--ffmpeg")

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
    render.add_argument("--ffmpeg")

    score = subparsers.add_parser("score", help="score candidate frames against reference frames")
    score.add_argument("--candidate", required=True)
    score.add_argument("--reference", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--width", type=int, required=True)
    score.add_argument("--height", type=int, required=True)
    score.add_argument("--frame-count", type=int)
    score.add_argument("--require-exact-frame-count", action="store_true")
    score.add_argument("--ffmpeg")
    score.add_argument("--frame-start", type=int)
    score.add_argument("--frame-end", type=int)
    score.add_argument("--endpoint-frame-count", type=int, default=3)

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
    build_job.add_argument("--progress-frame-start", type=int)
    build_job.add_argument("--progress-frame-end", type=int)
    build_job.add_argument("--progress-value-start", type=float)
    build_job.add_argument("--progress-value-end", type=float)

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

    candidate_init = subparsers.add_parser(
        "candidate-init",
        help="copy a registered effect into an isolated refinement workspace",
    )
    candidate_init.add_argument("--manifest", required=True)
    candidate_init.add_argument("--output-dir", required=True)
    candidate_init.add_argument("--force", action="store_true")

    candidate_promote = subparsers.add_parser(
        "candidate-promote",
        help="back up and promote a reviewed candidate into the registered FX",
    )
    candidate_promote.add_argument("--manifest", required=True)
    candidate_promote.add_argument("--backup-dir", required=True)

    candidate_evaluate = subparsers.add_parser(
        "candidate-evaluate",
        help="stage, build, render, and score a candidate",
    )
    candidate_evaluate.add_argument("--manifest", required=True)
    candidate_evaluate.add_argument("--job", required=True)
    candidate_evaluate.add_argument("--reference", required=True)
    candidate_evaluate.add_argument("--output-root", required=True)
    candidate_evaluate.add_argument("--backup-dir", required=True)
    candidate_evaluate.add_argument("--msbuild", default="msbuild")
    candidate_evaluate.add_argument("--configuration", default="Debug")
    candidate_evaluate.add_argument("--platform", default="x64")
    candidate_evaluate.add_argument("--renderer")
    candidate_evaluate.add_argument("--width", type=int, required=True)
    candidate_evaluate.add_argument("--height", type=int, required=True)
    candidate_evaluate.add_argument("--frame-count", type=int)
    candidate_evaluate.add_argument("--ffmpeg")
    candidate_evaluate.add_argument("--frame-start", type=int)
    candidate_evaluate.add_argument("--frame-end", type=int)
    candidate_evaluate.add_argument("--endpoint-frame-count", type=int, default=3)
    candidate_evaluate.add_argument(
        "--calibrate-progress",
        action="store_true",
        help="linear-probe the staged candidate and use an evaluation-local derived progress schedule",
    )
    candidate_evaluate.add_argument(
        "--iteration",
        type=int,
        help="record this evaluation in the controller state and matching iteration record",
    )
    candidate_evaluate.add_argument(
        "--restore",
        action="store_true",
        help="restore registered target sources and the Debug plugin DLL after evaluation",
    )

    candidate_baseline = subparsers.add_parser(
        "candidate-set-baseline",
        help="select a valid scored iteration as the controller baseline",
    )
    candidate_baseline.add_argument("--manifest", required=True)
    candidate_baseline.add_argument("--iteration", required=True, type=int)
    candidate_baseline.add_argument("--report", required=True)
    candidate_baseline.add_argument(
        "--source-dir",
        help="source directory to snapshot as the selected baseline; defaults to the candidate workspace",
    )

    candidate_evaluation_profile = subparsers.add_parser(
        "candidate-set-evaluation-profile",
        help="store the reusable one-shot candidate evaluation command inputs",
    )
    candidate_evaluation_profile.add_argument("--manifest", required=True)
    candidate_evaluation_profile.add_argument("--job", required=True)
    candidate_evaluation_profile.add_argument("--reference", required=True)
    candidate_evaluation_profile.add_argument("--output-root", required=True)
    candidate_evaluation_profile.add_argument("--backup-root", required=True)
    candidate_evaluation_profile.add_argument("--msbuild", default="msbuild")
    candidate_evaluation_profile.add_argument("--configuration", default="Debug")
    candidate_evaluation_profile.add_argument("--platform", default="x64")
    candidate_evaluation_profile.add_argument("--renderer")
    candidate_evaluation_profile.add_argument("--width", type=int, required=True)
    candidate_evaluation_profile.add_argument("--height", type=int, required=True)
    candidate_evaluation_profile.add_argument("--frame-start", type=int, required=True)
    candidate_evaluation_profile.add_argument("--frame-end", type=int, required=True)
    candidate_evaluation_profile.add_argument(
        "--no-calibrate-progress",
        action="store_true",
        help="omit progress calibration from the generated one-shot evaluation command",
    )

    candidate_restore = subparsers.add_parser(
        "candidate-restore-baseline",
        help="restore candidate and registered sources from the selected baseline snapshot",
    )
    candidate_restore.add_argument("--manifest", required=True)

    candidate_human_accept = subparsers.add_parser(
        "candidate-human-accept",
        help="record human visual acceptance for the selected candidate baseline and close its phase",
    )
    candidate_human_accept.add_argument("--manifest", required=True)
    candidate_human_accept.add_argument("--iteration", required=True, type=int)
    candidate_human_accept.add_argument("--reviewer", required=True)
    candidate_human_accept.add_argument("--reason", required=True)

    candidate_phase = subparsers.add_parser(
        "candidate-start-phase",
        help="start a bounded refinement phase with a newly scored baseline",
    )
    candidate_phase.add_argument("--manifest", required=True)
    candidate_phase.add_argument("--name", required=True)
    candidate_phase.add_argument("--baseline-iteration", required=True, type=int)
    candidate_phase.add_argument("--report", required=True)
    candidate_phase.add_argument("--max-iterations", required=True, type=int)
    candidate_phase.add_argument("--max-rejected", required=True, type=int)
    candidate_phase.add_argument("--source-dir")

    candidate_record_score = subparsers.add_parser(
        "candidate-record-score",
        help="record an artifact-only score for an existing candidate iteration",
    )
    candidate_record_score.add_argument("--manifest", required=True)
    candidate_record_score.add_argument("--iteration", required=True, type=int)
    candidate_record_score.add_argument("--report", required=True)

    candidate_reassess = subparsers.add_parser(
        "candidate-reassess",
        help="replay historical evaluated candidates under the current selection policy",
    )
    candidate_reassess.add_argument("--manifest", required=True)
    candidate_reassess.add_argument(
        "--apply-best",
        action="store_true",
        help="restore the best eligible saved candidate and set it as the selected baseline",
    )

    candidate_next = subparsers.add_parser(
        "candidate-next",
        help="prepare the next refinement packet and Codex request",
    )
    candidate_next.add_argument("--manifest", required=True)
    candidate_next.add_argument("--analysis", required=True)
    candidate_next.add_argument("--design", required=True)
    candidate_next.add_argument("--max-iterations", type=int, default=20)
    candidate_next.add_argument("--max-rejected", type=int, default=8)
    candidate_next.add_argument(
        "--evaluate-after-edit",
        action="store_true",
        help="include one configured candidate evaluation command in the generated Codex request",
    )

    candidate_continue = subparsers.add_parser(
        "candidate-continue",
        help="restore after a rejected/tradeoff evaluation and prepare the next edit-and-evaluate request",
    )
    candidate_continue.add_argument("--manifest", required=True)
    candidate_continue.add_argument("--analysis", required=True)
    candidate_continue.add_argument("--design", required=True)
    candidate_continue.add_argument("--max-iterations", type=int, default=20)
    candidate_continue.add_argument("--max-rejected", type=int, default=8)

    candidate_resume = subparsers.add_parser(
        "candidate-resume",
        help="verify diagnostics, restart a bounded phase from the selected baseline, and prepare its first request",
    )
    candidate_resume.add_argument("--manifest", required=True)
    candidate_resume.add_argument("--analysis", required=True)
    candidate_resume.add_argument("--design", required=True)
    candidate_resume.add_argument("--phase", required=True)
    candidate_resume.add_argument("--max-iterations", type=int, default=6)
    candidate_resume.add_argument("--max-rejected", type=int, default=3)

    candidate_status_cmd = subparsers.add_parser(
        "candidate-status",
        help="show controller baseline, history, budgets, and blocked hypothesis categories",
    )
    candidate_status_cmd.add_argument("--manifest", required=True)

    report = subparsers.add_parser("report", help="combine analysis, design, render, and score artifacts")
    report.add_argument("--analysis", required=True)
    report.add_argument("--design", required=True)
    report.add_argument("--render-report", required=True)
    report.add_argument("--score-report", required=True)
    report.add_argument("--output", required=True)

    return parser
