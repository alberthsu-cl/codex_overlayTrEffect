from __future__ import annotations

from pathlib import Path
import json
import sys
import unittest
import uuid
from unittest.mock import patch


AGENT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(AGENT_SRC))

from agent_app.workflow import (
    _encode_artifact_video,
    _create_comparison_assets,
    _build_progress_schedule,
    _detect_progress_calibration,
    build_report,
    prepare_reference,
    prepare_sources,
    retrieve_effect,
    score_candidate,
)
from agent_app.artifacts import build_render_job
from agent_app.candidate_controller import (
    _endpoints_are_exact,
    build_next_iteration_packet,
    human_accept_candidate,
    record_candidate_evaluation,
    restore_candidate_baseline,
    set_candidate_baseline,
    start_refinement_phase,
)
from agent_app.sample_workspace import initialize_sample_workspace
from agent_app.codegen import (
    _is_automatic_variant_replacement,
    _resolve_variant_template_path,
    _update_project_filters,
)


class WorkflowTests(unittest.TestCase):
    def test_controller_allows_negligible_endpoint_compression_variance(self) -> None:
        metrics = {
            "endpoint_checks": {
                "before_transition": {"mse": 0.0, "ssim": 1.0},
                "after_transition": {"mse": 0.056, "ssim": 0.999998},
            }
        }
        self.assertTrue(_endpoints_are_exact(metrics))
        metrics["endpoint_checks"]["after_transition"] = {"mse": 1.1, "ssim": 0.999998}
        self.assertFalse(_endpoints_are_exact(metrics))

    def test_progress_calibration_detects_visible_internal_interval(self) -> None:
        calibration = _detect_progress_calibration(
            mae_to_a=[0.0, 0.0, 0.0, 5.0, 14.0, 30.0, 40.0, 28.0, 12.0, 4.0, 0.0, 0.0],
            mae_to_b=[80.0, 80.0, 80.0, 70.0, 55.0, 35.0, 25.0, 14.0, 5.0, 0.0, 0.0, 0.0],
            source_mae=80.0,
        )
        self.assertEqual(calibration["status"], "succeeded")
        self.assertEqual(calibration["active_frame_start"], 3)
        self.assertEqual(calibration["active_frame_end"], 8)
        self.assertAlmostEqual(calibration["active_progress_start"], 3 / 11)
        self.assertAlmostEqual(calibration["active_progress_end"], 8 / 11)

    def test_progress_calibration_falls_back_when_endpoints_are_indistinct(self) -> None:
        calibration = _detect_progress_calibration(
            mae_to_a=[0.0, 0.2, 0.4],
            mae_to_b=[0.3, 0.2, 0.0],
            source_mae=0.4,
        )
        self.assertEqual(calibration["status"], "needs_review")
        self.assertEqual(calibration["active_progress_start"], 0.0)
        self.assertEqual(calibration["active_progress_end"], 1.0)

    def test_progress_schedule_holds_and_stretches_shader_interval(self) -> None:
        schedule = _build_progress_schedule(
            frame_count=60,
            frame_start=14,
            frame_end=43,
            progress_start=24 / 59,
            progress_end=40 / 59,
        )
        self.assertEqual(len(schedule), 60)
        self.assertEqual(schedule[0], 0.0)
        self.assertEqual(schedule[13], 0.0)
        self.assertAlmostEqual(schedule[14], 24 / 59)
        self.assertAlmostEqual(schedule[43], 40 / 59)
        self.assertEqual(schedule[44], 1.0)
        self.assertEqual(schedule[59], 1.0)

    def test_prepare_reference_forwards_manual_transition_window(self) -> None:
        result = type(
            "ReferenceResult",
            (),
            {
                "message": "prepared",
                "output_dir": Path("reference"),
                "manifest_file": Path("reference/reference_transition_manifest.json"),
                "frame_count": 30,
                "detected_start_frame": 64,
                "detected_end_frame": 93,
                "detected_frame_count": 30,
            },
        )()
        prepare = unittest.mock.Mock(return_value=result)
        with patch("agent_app.workflow.load_harness_modules", return_value={"prepare_reference_transition": prepare}):
            prepare_reference(
                workspace_root=Path("workspace"),
                source_video=Path("sample.mp4"),
                output_dir=Path("reference"),
                fps=30,
                width=1920,
                height=1080,
                target_frame_count=60,
                start_frame=64,
                end_frame=93,
            )
        self.assertEqual(prepare.call_args.kwargs["start_frame"], 64)
        self.assertEqual(prepare.call_args.kwargs["end_frame"], 93)

    def test_project_filters_accept_shared_model_generated_base_filter(self) -> None:
        base_stem = "TrModelGeneratedSeamlessSliding02"
        text = "\n".join(
            (
                '<Project xmlns="http://schemas.microsoft.com/developer/msbuild/2003">',
                "  <ItemGroup>",
                '    <Filter Include="Transition\\ModelGenerated">',
                "      <UniqueIdentifier>{b7cf5f0d-3d70-4b3c-8d4b-3d66d64a8421}</UniqueIdentifier>",
                "    </Filter>",
                "  </ItemGroup>",
                "  <ItemGroup>",
                f'    <ClCompile Include="{base_stem}.cpp">',
                "      <Filter>Transition\\ModelGenerated</Filter>",
                "    </ClCompile>",
                "  </ItemGroup>",
                "  <ItemGroup>",
                f'    <ClInclude Include="{base_stem}.h">',
                "      <Filter>Transition\\ModelGenerated</Filter>",
                "    </ClInclude>",
                "  </ItemGroup>",
                "  <ItemGroup>",
                f'    <FxCompile Include="{base_stem}_ps.hlsl">',
                "      <Filter>Transition\\ModelGenerated</Filter>",
                "    </FxCompile>",
                "  </ItemGroup>",
                "</Project>",
            )
        )
        updated = _update_project_filters(
            text,
            base_stem=base_stem,
            cpp_filename="TrModelGeneratedSeamlessSplitSlide01.cpp",
            header_filename="TrModelGeneratedSeamlessSplitSlide01.h",
            shader_filename="TrModelGeneratedSeamlessSplitSlide01_ps.hlsl",
        )
        self.assertIn('Include="TrModelGeneratedSeamlessSplitSlide01.cpp"', updated)
        self.assertIn('Include="TrModelGeneratedSeamlessSplitSlide01.h"', updated)
        self.assertIn('Include="TrModelGeneratedSeamlessSplitSlide01_ps.hlsl"', updated)
    def test_source_variant_accepts_repo_relative_template_path(self) -> None:
        root = Path(__file__).resolve().parents[1] / "work" / f"variant_path_test_{uuid.uuid4().hex}"
        template_root = root / "overlaytrengine" / "OverlayTrPlugInFx"
        try:
            template_root.mkdir(parents=True)
            expected = template_root / "TrExample.h"
            expected.write_text("example", encoding="utf-8")
            resolved = _resolve_variant_template_path(
                template_root,
                "overlaytrengine/OverlayTrPlugInFx/TrExample.h",
            )
            self.assertEqual(resolved, expected)
        finally:
            for path in sorted(root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            root.rmdir()

    def test_source_variant_ignores_automatic_renames(self) -> None:
        self.assertTrue(
            _is_automatic_variant_replacement(
                old="TrModelGeneratedSeamlessSliding02",
                new="TrModelGeneratedSeamlessSplitSlide01",
                base_stem="TrModelGeneratedSeamlessSliding02",
                base_effect_id="ModelGenerated\\SeamlessSliding_02",
                effect_id="ModelGenerated\\SeamlessSplitSlide_01",
                symbol="ModelGeneratedSeamlessSplitSlide01",
                class_name="CTrModelGeneratedSeamlessSplitSlide01",
                shader_symbol="g_Tr_ModelGeneratedSeamlessSplitSlide01_PS",
            )
        )

    def test_source_variant_uses_schema_closest_effect_id(self) -> None:
        target = {
            "closest_existing_effect_id": "ModelGenerated\\SeamlessSliding_02",
        }
        self.assertTrue(
            _is_automatic_variant_replacement(
                old=target["closest_existing_effect_id"],
                new="ModelGenerated\\SeamlessSplitSlide_01",
                base_stem="TrModelGeneratedSeamlessSliding02",
                base_effect_id=target.get("base_effect_id") or target.get("closest_existing_effect_id"),
                effect_id="ModelGenerated\\SeamlessSplitSlide_01",
                symbol="ModelGeneratedSeamlessSplitSlide01",
                class_name="CTrModelGeneratedSeamlessSplitSlide01",
                shader_symbol="g_Tr_ModelGeneratedSeamlessSplitSlide01_PS",
            )
        )
        self.assertTrue(
            _is_automatic_variant_replacement(
                old="ModelGenerated\\SeamlessSliding_02",
                new="ModelGenerated\\SeamlessSplitSlide_01",
                base_stem="TrModelGeneratedSeamlessSliding02",
                base_effect_id="ModelGenerated\\SeamlessSliding_02",
                effect_id="ModelGenerated\\SeamlessSplitSlide_01",
                symbol="ModelGeneratedSeamlessSplitSlide01",
                class_name="CTrModelGeneratedSeamlessSplitSlide01",
                shader_symbol="g_Tr_ModelGeneratedSeamlessSplitSlide01_PS",
            )
        )

    def test_sample_workspace_isolated_by_sample_id(self) -> None:
        root = Path(__file__).resolve().parents[1] / "work" / f"sample_workspace_test_{uuid.uuid4().hex}"
        source = root / "input.mp4"
        try:
            root.mkdir(parents=True)
            source.write_bytes(b"video")
            result = initialize_sample_workspace(root / "samples", "example_001", source)
            sample_dir = Path(result["sample_directory"])
            self.assertTrue((sample_dir / "sample_workspace.json").is_file())
            self.assertTrue((sample_dir / "reference").is_dir())
            self.assertTrue((sample_dir / "candidates").is_dir())
            with self.assertRaises(ValueError):
                initialize_sample_workspace(root / "samples", "example_001", source)
        finally:
            for path in sorted(root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            root.rmdir()

    def test_controller_tracks_baseline_and_evaluation_outcome(self) -> None:
        root = Path(__file__).resolve().parents[1] / "work" / f"controller_test_{uuid.uuid4().hex}"
        candidate_dir = root / "candidate"
        candidate_dir.mkdir(parents=True)
        manifest = candidate_dir / "candidate_manifest.json"
        candidate_source = candidate_dir / "Candidate.h"
        target_source = root / "target" / "Candidate.h"
        report = candidate_dir / "baseline_report.json"
        analysis = candidate_dir / "analysis.json"
        design = candidate_dir / "design.json"
        try:
            candidate_source.write_text("baseline", encoding="utf-8")
            target_source.parent.mkdir()
            target_source.write_text("baseline", encoding="utf-8")
            manifest.write_text(
                json.dumps(
                    {
                        "effect_id": "ModelGenerated\\Test",
                        "candidate_files": [str(candidate_source)],
                        "target_files": [str(target_source)],
                    }
                ),
                encoding="utf-8",
            )
            analysis.write_text("{}", encoding="utf-8")
            design.write_text("{}", encoding="utf-8")
            self._write_controller_report(report, mse=10.0, ssim=0.9, motion_similarity=0.7)
            state = set_candidate_baseline(manifest, iteration=1, report_file=report)
            self.assertEqual(state["status"], "succeeded")
            self.assertEqual(state["state"]["baseline"]["iteration"], 1)
            self.assertTrue((candidate_dir / "baselines" / "iteration_001" / "Candidate.h").exists())
            candidate_source.write_text("rejected", encoding="utf-8")
            target_source.write_text("rejected", encoding="utf-8")
            restored = restore_candidate_baseline(manifest)
            self.assertEqual(restored["baseline_iteration"], 1)
            self.assertEqual(candidate_source.read_text(encoding="utf-8"), "baseline")
            self.assertEqual(target_source.read_text(encoding="utf-8"), "baseline")

            packet = build_next_iteration_packet(
                candidate_manifest_file=manifest,
                analysis_file=analysis,
                design_file=design,
                max_iterations=3,
                max_rejected=2,
            )
            self.assertEqual(packet["iteration"], 2)
            self.assertTrue(Path(packet["prompt_file"]).exists())

            iteration_file = candidate_dir / "iteration_002_regions.json"
            iteration_file.write_text(
                json.dumps({"iteration": 2, "hypothesis_category": "regions", "status": "candidate_only"}),
                encoding="utf-8",
            )
            improved_report = candidate_dir / "improved_report.json"
            self._write_controller_report(improved_report, mse=10.2, ssim=0.902, motion_similarity=0.7)
            outcome = record_candidate_evaluation(manifest, 2, improved_report)
            self.assertEqual(outcome["status"], "accepted")
            record = json.loads(iteration_file.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "accepted")

            tradeoff_file = candidate_dir / "iteration_003_displacement.json"
            tradeoff_file.write_text(
                json.dumps({"iteration": 3, "hypothesis_category": "displacement", "status": "candidate_only"}),
                encoding="utf-8",
            )
            tradeoff_report = candidate_dir / "tradeoff_report.json"
            self._write_controller_report(tradeoff_report, mse=10.3, ssim=0.89, motion_similarity=0.75)
            tradeoff = record_candidate_evaluation(manifest, 3, tradeoff_report)
            self.assertEqual(tradeoff["status"], "tradeoff")
            state_data = json.loads((candidate_dir / "candidate_state.json").read_text(encoding="utf-8"))
            self.assertEqual([item["iteration"] for item in state_data["shortlist"]], [1, 2, 3])
        finally:
            for path in sorted(root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            root.rmdir()

    def test_controller_starts_independent_refinement_phase(self) -> None:
        root = Path(__file__).resolve().parents[1] / "work" / f"phase_test_{uuid.uuid4().hex}"
        candidate_dir = root / "candidate"
        candidate_dir.mkdir(parents=True)
        manifest = candidate_dir / "candidate_manifest.json"
        candidate_source = candidate_dir / "Candidate.h"
        target_source = root / "target" / "Candidate.h"
        report = candidate_dir / "baseline_report.json"
        analysis = candidate_dir / "analysis.json"
        design = candidate_dir / "design.json"
        try:
            candidate_source.write_text("baseline", encoding="utf-8")
            target_source.parent.mkdir()
            target_source.write_text("baseline", encoding="utf-8")
            manifest.write_text(
                json.dumps(
                    {
                        "effect_id": "ModelGenerated\\Test",
                        "candidate_files": [str(candidate_source)],
                        "target_files": [str(target_source)],
                    }
                ),
                encoding="utf-8",
            )
            analysis.write_text("{}", encoding="utf-8")
            design.write_text("{}", encoding="utf-8")
            self._write_controller_report(report, mse=10.0, ssim=0.9, motion_similarity=0.7)
            start = start_refinement_phase(
                candidate_manifest_file=manifest,
                name="optical_flow",
                baseline_iteration=1,
                report_file=report,
                max_iterations=2,
                max_rejected=1,
            )
            self.assertEqual(start["phase"]["first_iteration"], 2)
            packet = build_next_iteration_packet(
                candidate_manifest_file=manifest,
                analysis_file=analysis,
                design_file=design,
                max_iterations=1,
                max_rejected=1,
            )
            self.assertEqual(packet["iteration"], 2)
            packet_data = json.loads(Path(packet["packet_file"]).read_text(encoding="utf-8"))
            self.assertEqual(packet_data["active_phase"]["name"], "optical_flow")
            self.assertEqual(packet_data["budgets"]["rejected_so_far"], 0)
        finally:
            for path in sorted(root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            root.rmdir()

    def test_human_acceptance_closes_active_phase(self) -> None:
        root = Path(__file__).resolve().parents[1] / "work" / f"human_accept_test_{uuid.uuid4().hex}"
        candidate_dir = root / "candidate"
        candidate_dir.mkdir(parents=True)
        manifest = candidate_dir / "candidate_manifest.json"
        candidate_source = candidate_dir / "Candidate.h"
        target_source = root / "target" / "Candidate.h"
        report = candidate_dir / "baseline_report.json"
        try:
            candidate_source.write_text("baseline", encoding="utf-8")
            target_source.parent.mkdir()
            target_source.write_text("baseline", encoding="utf-8")
            manifest.write_text(
                json.dumps(
                    {
                        "effect_id": "ModelGenerated\\Test",
                        "candidate_files": [str(candidate_source)],
                        "target_files": [str(target_source)],
                    }
                ),
                encoding="utf-8",
            )
            self._write_controller_report(report, mse=10.0, ssim=0.9, motion_similarity=0.7)
            start_refinement_phase(
                candidate_manifest_file=manifest,
                name="visual_review",
                baseline_iteration=1,
                report_file=report,
                max_iterations=2,
                max_rejected=1,
            )
            result = human_accept_candidate(
                manifest,
                iteration=1,
                reviewer="Albert",
                reason="Acceptable at normal playback.",
            )
            self.assertEqual(result["acceptance"]["status"], "human_accepted")
            state = json.loads((candidate_dir / "candidate_state.json").read_text(encoding="utf-8"))
            self.assertIsNone(state["active_phase"])
            self.assertEqual(state["phases"][0]["status"], "closed")
            self.assertEqual(state["human_acceptance"]["iteration"], 1)
            self.assertEqual(state["history"][0]["status"], "human_accepted")
            self.assertEqual(state["budgets"]["attempted_so_far"], 0)
            self.assertEqual(state["budgets"]["rejected_so_far"], 0)
        finally:
            for path in sorted(root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            root.rmdir()

    def _write_controller_report(
        self,
        path: Path,
        mse: float,
        ssim: float,
        motion_similarity: float | None = None,
    ) -> None:
        score = {
            "transition_window": {
                "frame_start": 2,
                "frame_end": 4,
                "frame_count": 3,
                "mse": mse,
                "mae": 1.0,
                "psnr_db": 20.0,
                "ssim": ssim,
            },
            "endpoint_checks": {
                "before_transition": {"mse": 0.0, "ssim": 1.0},
                "after_transition": {"mse": 0.0, "ssim": 1.0},
            },
        }
        if motion_similarity is not None:
            score["motion_metrics"] = {
                "motion_similarity": motion_similarity,
                "flow_vector_mae": 1.0,
                "motion_region_iou": 1.0,
                "direction_agreement": 1.0,
            }
        path.write_text(
            json.dumps(
                {
                    "score": score
                }
            ),
            encoding="utf-8",
        )

    def test_encode_artifact_video_uses_png_sequence_and_job_fps(self) -> None:
        with self.subTest("successful encode"):
            artifacts_dir = Path(__file__).parent / "fixtures" / "render_artifacts"
            completed = type("Completed", (), {"returncode": 0, "stderr": ""})()
            with patch("agent_app.workflow.subprocess.run", return_value=completed) as run:
                result = _encode_artifact_video(
                    artifacts_dir=artifacts_dir,
                    fps=30,
                    ffmpeg_path="C:/tools/ffmpeg.exe",
                )

                self.assertEqual(result["status"], "succeeded")
                self.assertEqual(result["fps"], 30)
                self.assertEqual(run.call_args.args[0][0], "C:/tools/ffmpeg.exe")
                self.assertIn("frame_%04d.png", run.call_args.args[0])
                self.assertIn("rendered_transition.mp4", run.call_args.args[0])

    def test_create_comparison_assets_encodes_reference_and_window(self) -> None:
        artifacts_dir = Path(__file__).parent / "fixtures" / "render_artifacts"
        completed = type("Completed", (), {"returncode": 0, "stderr": ""})()
        with patch("agent_app.workflow.subprocess.run", return_value=completed) as run:
            result = _create_comparison_assets(
                artifacts_dir=artifacts_dir,
                reference_dir=artifacts_dir,
                fps=30,
                frame_start=2,
                frame_end=4,
                ffmpeg_path="C:/tools/ffmpeg.exe",
            )

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["reference_video"]["status"], "succeeded")
        self.assertEqual(result["transition_window_video"]["status"], "succeeded")
        self.assertEqual(run.call_count, 3)

    def test_retrieve_effect_returns_catalog_match(self) -> None:
        analysis = {
            "planner_hints": {
                "recommended_effect_family": "glitch",
                "recommended_effect_id": "fx-glitch",
            }
        }
        selected = {
            "effect_id": "builtin-glitch",
            "fx_id": "fx-glitch",
            "match_kind": "exact",
        }
        fake_modules = {
            "build_effect_catalog": lambda _: {"registration_count": 1},
            "select_effect_candidate": lambda *_args, **_kwargs: selected,
        }

        with patch("agent_app.workflow.load_json", return_value=analysis):
            with patch("agent_app.workflow.load_harness_modules", return_value=fake_modules):
                with patch("agent_app.workflow.write_json") as write_json:
                    result = retrieve_effect(
                        workspace_root=Path("D:/AI_Harness"),
                        analysis_file=Path("analysis.json"),
                        output_file=Path("retrieval.json"),
                    )

        self.assertEqual(result["status"], "retrieved")
        self.assertTrue(result["exact_id_match"])
        write_json.assert_called_once()

    def test_prepare_sources_rejects_reversed_boundaries(self) -> None:
        with self.assertRaisesRegex(ValueError, "boundaries"):
            prepare_sources(
                source_video=Path("missing.mp4"),
                output_root=Path("output"),
                start_frame=10,
                end_frame=9,
                frame_count=30,
                width=16,
                height=16,
            )

    def test_build_render_job_uses_existing_effect_design(self) -> None:
        analysis = {
            "artifact_type": "transition_structure",
            "artifact_version": 1,
            "input_video": "sample.mp4",
            "video_metadata": {"frame_count": 58},
            "transition": {
                "style_label": "seamless slide",
                "summary": "Source A slides away while source B enters.",
                "start_frame": 10,
                "end_frame": 45,
                "confidence": 0.9,
            },
            "visual_signals": {
                "cross_dissolve": False,
                "identity_morph": False,
                "scene_blend": False,
                "motion_smoothing": True,
                "lighting_continuity": True,
            },
            "frame_progress_mapping": [],
            "evidence": ["The image moves horizontally across the frame."],
            "limitations": [],
            "planner_hints": {
                "recommended_effect_family": "seamless_slide",
                "family_status": "known",
                "visual_primitives": ["translation"],
                "new_effect_needed": False,
                "implementation_status": "supported",
            },
        }
        design = {
            "artifact_type": "effect_design",
            "artifact_version": 1,
            "analysis_artifact": "analysis.json",
            "decision": {
                "action": "reuse_existing_effect",
                "confidence": 0.8,
                "reason": "The visible movement matches the built-in sliding effect.",
            },
            "target_effect": {
                "family": "seamless_slide",
                "effect_id": "CES_PlugIn_Seamless.dll\\DSP_TR_SeamlessSliding_LC",
                "expected_runtime_shape": "single_pass_fullscreen",
            },
            "design_notes": {
                "must_preserve": ["horizontal movement"],
                "approximations": [],
                "risks": [],
            },
        }

        job = build_render_job(
            analysis=analysis,
            design=design,
            source_a="source_a",
            source_b="source_b",
            reference_transition="reference_transition",
            width=1920,
            height=1080,
            fps=30,
            frame_count=30,
        )

        self.assertEqual(job["effect"]["fx_id"], design["target_effect"]["effect_id"])
        self.assertEqual(job["inputs"]["reference_transition"], "reference_transition")
        self.assertEqual(job["planning"]["decision"], "reuse_existing_effect")

    def test_build_render_job_accepts_compiled_new_effect(self) -> None:
        analysis = {
            "artifact_type": "transition_structure",
            "artifact_version": 1,
            "input_video": "sample.mp4",
            "video_metadata": {"frame_count": 2},
            "transition": {
                "style_label": "unknown",
                "summary": "Unknown effect.",
                "start_frame": 0,
                "end_frame": 1,
                "confidence": 0.2,
            },
            "visual_signals": {
                "cross_dissolve": False,
                "identity_morph": False,
                "scene_blend": False,
                "motion_smoothing": False,
                "lighting_continuity": False,
            },
            "frame_progress_mapping": [],
            "evidence": [],
            "limitations": [],
            "planner_hints": {
                "recommended_effect_family": "unknown",
                "family_status": "unknown",
                "visual_primitives": ["unknown"],
                "new_effect_needed": True,
                "implementation_status": "supported",
            },
        }
        design = {
            "artifact_type": "effect_design",
            "artifact_version": 1,
            "analysis_artifact": "analysis.json",
            "decision": {"action": "implement_new_effect", "confidence": 0.7},
            "target_effect": {
                "family": "seamless",
                "effect_id": "ModelGenerated\\Dissolve_01",
            },
            "design_notes": {"must_preserve": [], "approximations": [], "risks": []},
        }

        job = build_render_job(analysis, design, "source_a", "source_b", None, 16, 16, 30, 2)
        self.assertEqual(job["effect"]["fx_id"], design["target_effect"]["effect_id"])
        self.assertEqual(job["planning"]["decision"], "implement_new_effect")

    def test_build_render_job_blocks_unsupported_new_effect(self) -> None:
        analysis = {
            "artifact_type": "transition_structure",
            "artifact_version": 1,
            "input_video": "sample.mp4",
            "video_metadata": {"frame_count": 2},
            "transition": {
                "style_label": "unknown",
                "summary": "The effect cannot be represented by the current grammar.",
                "start_frame": 0,
                "end_frame": 1,
                "confidence": 0.4,
            },
            "visual_signals": {},
            "frame_progress_mapping": [],
            "evidence": [],
            "limitations": [],
            "planner_hints": {
                "recommended_effect_family": "unknown",
                "family_status": "unknown",
                "visual_primitives": ["unsupported_behavior"],
                "new_effect_needed": True,
                "implementation_status": "unsupported",
            },
        }
        design = {
            "artifact_type": "effect_design",
            "artifact_version": 1,
            "analysis_artifact": "analysis.json",
            "decision": {"action": "implement_new_effect", "confidence": 0.8},
            "target_effect": {"family": "unknown", "effect_id": "ModelGenerated\\Unknown_01"},
            "design_notes": {"must_preserve": [], "approximations": [], "risks": []},
        }

        with self.assertRaisesRegex(ValueError, "implementation_status=supported"):
            build_render_job(analysis, design, "source_a", "source_b", None, 16, 16, 30, 2)

    def test_score_candidate_writes_frame_and_aggregate_metrics(self) -> None:
        fake_score = type(
            "FakeScore",
            (),
            {
                "to_dict": lambda self: {
                    "frame_count": 4,
                    "mse": 3.0,
                    "frames": [
                        {"mse": 0.0, "mae": 0.0, "ssim": 1.0},
                        {"mse": 2.0, "mae": 1.0, "ssim": 0.8},
                        {"mse": 4.0, "mae": 2.0, "ssim": 0.6},
                        {"mse": 6.0, "mae": 3.0, "ssim": 0.4},
                    ],
                }
            },
        )()
        fake_modules = {
            "score_frame_sequences": lambda **_: fake_score,
            "score_motion": lambda **_: {
                "scorer": "opencv_farneback_dense_flow",
                "motion_similarity": 0.75,
                "flow_vector_mae": 2.0,
                "motion_region_iou": 0.5,
                "direction_agreement": 0.8,
                "pairs": [{"vector_mae": 2.0}],
            },
            "create_motion_visualizations": lambda **_: {
                "status": "succeeded",
                "frame_count": 1,
                "output_dir": "candidate/motion_diagnostics",
            },
        }

        with patch("agent_app.workflow.load_harness_modules", return_value=fake_modules):
            with patch("agent_app.workflow.write_json") as write_json:
                with patch(
                    "agent_app.workflow._encode_png_sequence",
                    return_value={"status": "succeeded", "file": "candidate/motion_diagnostics.mp4"},
                ):
                    result = score_candidate(
                        workspace_root=Path("D:/AI_Harness"),
                        candidate=Path("candidate"),
                        reference=Path("reference"),
                        output_file=Path("score.json"),
                        width=2,
                        height=2,
                        frame_count=4,
                        require_exact_frame_count=True,
                        frame_start=1,
                        frame_end=2,
                        endpoint_frame_count=1,
                    )

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["frame_count"], 4)
        self.assertEqual(result["mse"], 3.0)
        self.assertEqual(result["transition_window"]["mse"], 3.0)
        self.assertEqual(result["transition_window"]["frame_count"], 2)
        self.assertEqual(result["endpoint_checks"]["before_transition"]["mse"], 0.0)
        self.assertEqual(result["endpoint_checks"]["after_transition"]["mse"], 6.0)
        self.assertEqual(result["transition_diagnostics"]["worst_mse_frames"][0]["frame_index"], 2)
        self.assertEqual(result["motion_metrics"]["motion_similarity"], 0.75)
        self.assertEqual(result["transition_diagnostics"]["worst_motion_pairs"][0]["vector_mae"], 2.0)
        self.assertEqual(result["motion_visualizations"]["video"]["status"], "succeeded")
        write_json.assert_called_once()

    def test_build_report_preserves_all_artifacts(self) -> None:
        artifacts = {
            "analysis": {"artifact_type": "transition_structure"},
            "design": {"artifact_type": "effect_design"},
            "render": {"status": "blocked"},
            "score": {"status": "succeeded", "mse": 0.0},
        }

        def fake_load_json(path: Path):
            return artifacts[path.stem]

        with patch("agent_app.workflow.load_json", side_effect=fake_load_json):
            with patch("agent_app.workflow.write_json") as write_json:
                report = build_report(
                    analysis_file=Path("analysis.json"),
                    design_file=Path("design.json"),
                    render_file=Path("render.json"),
                    score_file=Path("score.json"),
                    output_file=Path("report.json"),
                )

        self.assertEqual(report["report_type"], "agent_transition_regression")
        self.assertEqual(report["analysis"]["artifact_type"], "transition_structure")
        self.assertEqual(report["effect_design"]["artifact_type"], "effect_design")
        write_json.assert_called_once()


if __name__ == "__main__":
    unittest.main()
