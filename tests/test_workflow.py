from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch


AGENT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(AGENT_SRC))

from agent_app.workflow import build_report, prepare_sources, retrieve_effect, score_candidate
from agent_app.artifacts import build_render_job


class WorkflowTests(unittest.TestCase):
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
            {"to_dict": lambda self: {"frame_count": 1, "mse": 0.0}},
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
                    frame_count=1,
                    require_exact_frame_count=True,
                )

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["frame_count"], 1)
        self.assertEqual(result["mse"], 0.0)
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
