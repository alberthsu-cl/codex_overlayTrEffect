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
    build_report,
    prepare_sources,
    retrieve_effect,
    score_candidate,
)
from agent_app.artifacts import build_render_job
from agent_app.candidate_controller import (
    build_next_iteration_packet,
    record_candidate_evaluation,
    set_candidate_baseline,
)


class WorkflowTests(unittest.TestCase):
    def test_controller_tracks_baseline_and_evaluation_outcome(self) -> None:
        root = Path(__file__).resolve().parents[1] / "work" / f"controller_test_{uuid.uuid4().hex}"
        candidate_dir = root / "candidate"
        candidate_dir.mkdir(parents=True)
        manifest = candidate_dir / "candidate_manifest.json"
        report = candidate_dir / "baseline_report.json"
        analysis = candidate_dir / "analysis.json"
        design = candidate_dir / "design.json"
        try:
            manifest.write_text(json.dumps({"effect_id": "ModelGenerated\\Test", "candidate_files": []}), encoding="utf-8")
            analysis.write_text("{}", encoding="utf-8")
            design.write_text("{}", encoding="utf-8")
            self._write_controller_report(report, mse=10.0, ssim=0.9)
            state = set_candidate_baseline(manifest, iteration=1, report_file=report)
            self.assertEqual(state["status"], "succeeded")
            self.assertEqual(state["state"]["baseline"]["iteration"], 1)

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
            self._write_controller_report(improved_report, mse=9.0, ssim=0.91)
            outcome = record_candidate_evaluation(manifest, 2, improved_report)
            self.assertEqual(outcome["status"], "accepted")
            record = json.loads(iteration_file.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "accepted")
        finally:
            for path in sorted(root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            root.rmdir()

    def _write_controller_report(self, path: Path, mse: float, ssim: float) -> None:
        path.write_text(
            json.dumps(
                {
                    "score": {
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
        }

        with patch("agent_app.workflow.load_harness_modules", return_value=fake_modules):
            with patch("agent_app.workflow.write_json") as write_json:
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
