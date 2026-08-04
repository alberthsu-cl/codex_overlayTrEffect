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
    _candidate_evaluation_run_name,
    _detect_progress_calibration,
    _reference_output_window,
    _score_motion_topology,
    ensure_reference_diagnostics,
    build_report,
    prepare_reference,
    prepare_sources,
    resolve_source_boundaries,
    retrieve_effect,
    score_candidate,
)
from agent_app.artifacts import build_render_job, validate_effect_design
from agent_app.cli import main as cli_main
from agent_app.candidate_controller import (
    CONSECUTIVE_TRADEOFF_ESCALATION_LIMIT,
    _consecutive_tradeoffs,
    _endpoints_are_exact,
    _escalation_instruction,
    _magnitude_findings,
    _metrics_from_report,
    _motion_refinement_priority,
    _pivot_is_conditioned,
    _normalize_selection_policy,
    _select_outcome,
    _select_outcome_with_decision,
    apply_reassessed_baseline,
    build_next_iteration_packet,
    human_accept_candidate,
    continue_candidate_refinement,
    reassess_candidate_history,
    record_candidate_evaluation,
    restore_candidate_baseline,
    set_candidate_baseline,
    set_evaluation_profile,
    start_refinement_phase,
)
from agent_app.sample_workspace import initialize_sample_workspace
from agent_app.codegen import (
    _is_automatic_variant_replacement,
    _resolve_variant_template_path,
    _update_project_filters,
)


class WorkflowTests(unittest.TestCase):
    @staticmethod
    def _geometry_metrics(*, pivot_delta: float, rotation: float, scale: float) -> dict:
        """Metrics shaped like a real report, with a tunable pivot and transform."""
        transform = {
            "confidence": 0.8,
            "rotation_field": {"mean_degrees": rotation},
            "radial_scale_field": {"mean_ratio": scale},
        }
        return {
            "motion_geometry": {
                "status": "geometry_mismatch",
                "candidate": transform,
                "reference": transform,
                "translation_delta_pixels": 0.5,
                "translation_direction_agreement": True,
                "pivot_delta_pixels": pivot_delta,
            },
            "foreground_body_transform": {
                "status": "estimated",
                "confidence": 0.8,
                "phases": {
                    "incoming": {
                        "rotation_delta_degrees": 40.0,
                        "scale_delta_ratio": 0.01,
                        "translation_delta_pixels": 0.5,
                    }
                },
            },
        }

    def test_degenerate_pivot_does_not_pre_empt_rotation(self) -> None:
        # A near-identity transform makes the pivot solve ill-conditioned, so its
        # huge delta must not outrank a real rotation disagreement.  This is the
        # ordering bug that hid a 4x rotation deficit for dozens of iterations.
        metrics = self._geometry_metrics(pivot_delta=200.0, rotation=0.05, scale=1.0004)
        priority = _motion_refinement_priority(
            {"history": [{"iteration": 1, "hypothesis_category": "displacement", "metrics": metrics}]}
        )
        self.assertEqual(priority["focus"], "transform_rotation")
        foci = {finding["focus"] for finding in _magnitude_findings(metrics)}
        self.assertNotIn("transform_position", foci)

    def test_conditioned_pivot_still_reports_position(self) -> None:
        # Once the body genuinely rotates, the pivot is meaningful again and a
        # large pivot delta should outrank a smaller rotation error.
        metrics = self._geometry_metrics(pivot_delta=200.0, rotation=12.0, scale=1.05)
        metrics["foreground_body_transform"]["phases"]["incoming"]["rotation_delta_degrees"] = 11.0
        priority = _motion_refinement_priority(
            {"history": [{"iteration": 1, "hypothesis_category": "displacement", "metrics": metrics}]}
        )
        self.assertEqual(priority["focus"], "transform_position")

    def test_pivot_conditioning_requires_departure_from_identity(self) -> None:
        self.assertFalse(
            _pivot_is_conditioned(
                {"rotation_field": {"mean_degrees": 0.1}, "radial_scale_field": {"mean_ratio": 1.001}}
            )
        )
        self.assertTrue(
            _pivot_is_conditioned(
                {"rotation_field": {"mean_degrees": 5.0}, "radial_scale_field": {"mean_ratio": 1.0}}
            )
        )
        self.assertTrue(
            _pivot_is_conditioned(
                {"rotation_field": {"mean_degrees": 0.0}, "radial_scale_field": {"mean_ratio": 1.2}}
            )
        )

    def test_consecutive_tradeoffs_counts_only_the_current_streak(self) -> None:
        history = [
            {"iteration": 1, "status": "tradeoff", "hypothesis_category": "displacement"},
            {"iteration": 2, "status": "accepted", "hypothesis_category": "displacement"},
            {"iteration": 3, "status": "tradeoff", "hypothesis_category": "displacement"},
            {"iteration": 4, "status": "tradeoff", "hypothesis_category": "displacement"},
        ]
        self.assertEqual(_consecutive_tradeoffs({"history": history}), 2)
        # An accepted iteration resets the streak.
        history.append({"iteration": 5, "status": "accepted", "hypothesis_category": "displacement"})
        self.assertEqual(_consecutive_tradeoffs({"history": history}), 0)
        # Phase scoping ignores iterations before the phase started.
        self.assertEqual(_consecutive_tradeoffs({"history": history}, first_iteration=3), 0)

    def test_escalation_demands_structural_review_past_the_limit(self) -> None:
        below = _escalation_instruction(
            {"consecutive_tradeoffs": 1, "limit": CONSECUTIVE_TRADEOFF_ESCALATION_LIMIT,
             "structural_review_required": False}
        )
        self.assertIn("consecutive tradeoff", below)
        self.assertNotIn("STRUCTURAL REVIEW REQUIRED", below)

        at_limit = _escalation_instruction(
            {"consecutive_tradeoffs": CONSECUTIVE_TRADEOFF_ESCALATION_LIMIT,
             "limit": CONSECUTIVE_TRADEOFF_ESCALATION_LIMIT,
             "structural_review_required": True}
        )
        self.assertIn("STRUCTURAL REVIEW REQUIRED", at_limit)
        self.assertIn("Do not tune another constant", at_limit)
        self.assertEqual(_escalation_instruction({"consecutive_tradeoffs": 0}), "")

    def test_transform_report_exposes_phase_selection_metrics(self) -> None:
        endpoints = {
            "before_transition": {"mse": 0.0, "ssim": 1.0},
            "after_transition": {"mse": 0.0, "ssim": 1.0},
        }
        def transform(rotation: float, dx: float, dy: float) -> dict:
            return {
                "rotation_field": {"mean_degrees": rotation},
                "radial_scale_field": {"mean_ratio": 1.0},
                "translation_field": {"mean_dx_pixels": dx, "mean_dy_pixels": dy},
                "pivot_field": {"x_pixels": 960.0, "y_pixels": 540.0},
            }
        report = {
            "score": {
                "transition_window": {"mse": 10.0, "mae": 2.0, "psnr_db": 30.0, "ssim": 0.8, "frame_start": 0, "frame_end": 1, "frame_count": 2},
                "endpoint_checks": endpoints,
                "foreground_body_transform": {
                    "phases": {
                        "outgoing": {"candidate": transform(5.0, 2.0, 3.0), "reference": transform(20.0, 10.0, 20.0)},
                        "incoming": {"candidate": transform(-4.0, 1.0, 4.0), "reference": transform(-18.0, 4.0, 18.0)},
                    }
                },
            }
        }
        metrics = _metrics_from_report(report)
        self.assertAlmostEqual(metrics["foreground_body_rotation_error"], 14.5)
        self.assertGreater(metrics["foreground_body_translation_error"], 0.0)
        self.assertEqual(metrics["foreground_body_rotation_direction_agreement"], 1.0)

    def test_transform_policy_treats_ssim_as_advisory(self) -> None:
        policy = _normalize_selection_policy(
            {
                "profile": "transform",
                "primary_metrics": ["mse", "mae"],
                "guardrail_metrics": ["peak_ssim"],
                "advisory_metrics": ["motion_similarity"],
            },
            "test",
        )

        self.assertIsNotNone(policy)
        self.assertEqual(
            policy["guardrail_metrics"],
            [
                "foreground_body_rotation_direction_agreement",
                "foreground_body_rotation_error",
                "foreground_body_translation_error",
                "foreground_body_scale_error",
            ],
        )
        self.assertEqual(policy["advisory_metrics"], ["motion_similarity", "peak_ssim"])

    def test_transform_selection_accepts_image_error_improvement_without_flow_guardrails(self) -> None:
        baseline = {
            "iteration": 0,
            "metrics": {
                "mse": 100.0,
                "mae": 10.0,
                "ssim": 0.50,
                "endpoint_checks": {
                    "before_transition": {"mse": 0.0, "ssim": 1.0},
                    "after_transition": {"mse": 0.0, "ssim": 1.0},
                },
                "motion": {"motion_similarity": 0.70},
            },
        }
        candidate = {
            "mse": 80.0,
            "mae": 8.0,
            "ssim": 0.40,
            "endpoint_checks": baseline["metrics"]["endpoint_checks"],
            "motion": {"motion_similarity": 0.50},
        }
        policy = {
            "profile": "transform",
            "source": "test",
            "primary_metrics": ["mse", "mae", "peak_mse"],
            "guardrail_metrics": [],
            "advisory_metrics": ["ssim", "motion_similarity"],
        }
        outcome, _, decision = _select_outcome_with_decision(baseline, candidate, policy)
        self.assertEqual(outcome, "accepted")
        self.assertEqual(decision["materially_improved_primary_metrics"], ["mse", "mae"])
        self.assertEqual(decision["guardrail_failures"], [])

    def test_candidate_evaluation_run_name_uses_iteration_record_stem(self) -> None:
        root = Path(__file__).resolve().parents[1] / "work" / f"evaluation_name_{uuid.uuid4().hex}"
        candidate_dir = root / "candidate"
        candidate_dir.mkdir(parents=True)
        manifest = candidate_dir / "candidate_manifest.json"
        job = root / "render_job.json"
        try:
            manifest.write_text("{}", encoding="utf-8")
            (candidate_dir / "iteration_030_displacement_band_magnitude.json").write_text("{}", encoding="utf-8")
            job.write_text(json.dumps({"job_name": "agent_tune_existing_effect_HorizontalSplitBlur"}), encoding="utf-8")

            result = _candidate_evaluation_run_name(manifest, 30)

            self.assertEqual(
                result,
                "iteration_030_displacement_band_magnitude",
            )
        finally:
            for path in sorted(root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            root.rmdir()

    def test_candidate_evaluation_run_name_falls_back_without_iteration_record(self) -> None:
        root = Path(__file__).resolve().parents[1] / "work" / f"evaluation_name_{uuid.uuid4().hex}"
        candidate_dir = root / "candidate"
        candidate_dir.mkdir(parents=True)
        manifest = candidate_dir / "candidate_manifest.json"
        job = root / "render_job.json"
        try:
            manifest.write_text("{}", encoding="utf-8")
            job.write_text(
                json.dumps({"job_name": "agent_tune_existing_effect_HorizontalSplitBlur"}),
                encoding="utf-8",
            )

            self.assertIsNone(_candidate_evaluation_run_name(manifest, 30))
        finally:
            for path in sorted(root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            root.rmdir()

    def test_ensure_reference_diagnostics_reuses_valid_canonical_artifact(self) -> None:
        root = Path(__file__).resolve().parents[1] / "work" / f"diagnostic_preflight_{uuid.uuid4().hex}"
        reference = root / "reference"
        diagnostics = root / "diagnostics"
        try:
            diagnostics.mkdir(parents=True)
            diagnostic_file = diagnostics / "reference_motion_diagnostics.json"
            diagnostic_file.write_text(
                json.dumps(
                    {
                        "artifact_type": "reference_motion_diagnostics",
                        "pairs": [],
                        "summary": {
                            "topology_contract": {"status": "not_required"},
                            "motion_geometry": {
                                "status": "needs_review",
                                "translation_field": {},
                            },
                            "angular_motion": {"status": "indeterminate"},
                            "angular_motion_phases": [],
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch("agent_app.workflow.analyze_reference_diagnostics") as analyze:
                result = ensure_reference_diagnostics(Path("D:/AI_Harness"), reference)

            self.assertFalse(result["regenerated"])
            self.assertEqual(result["output_file"], str(diagnostic_file))
            analyze.assert_not_called()
        finally:
            for path in sorted(root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            root.rmdir()

    def test_ensure_reference_diagnostics_regenerates_missing_canonical_artifact(self) -> None:
        root = Path(__file__).resolve().parents[1] / "work" / f"diagnostic_regenerate_{uuid.uuid4().hex}"
        reference = root / "reference"
        try:
            root.mkdir(parents=True)
            generated = root / "diagnostics" / "reference_motion_diagnostics.json"
            with patch(
                "agent_app.workflow.analyze_reference_diagnostics",
                return_value={"output_file": str(generated)},
            ) as analyze:
                result = ensure_reference_diagnostics(Path("D:/AI_Harness"), reference)

            self.assertTrue(result["regenerated"])
            analyze.assert_called_once()
            self.assertEqual(analyze.call_args.kwargs["output_dir"], root / "diagnostics")
        finally:
            for path in sorted(root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            root.rmdir()

    def test_effect_design_rejects_inconsistent_implementation_seed(self) -> None:
        design = {
            "artifact_type": "effect_design",
            "artifact_version": 1,
            "analysis_artifact": "analysis.json",
            "decision": {"action": "tune_existing_effect", "confidence": 0.8},
            "target_effect": {
                "family": "split_slide",
                "effect_id": "ModelGenerated\\SplitSlide_01",
                "closest_existing_effect_id": "ModelGenerated\\HorizontalSplitBlur_01",
            },
            "implementation_seed": {
                "family": "dissolve",
                "template_effect_id": "ModelGenerated\\Dissolve_01",
                "required_shader_capabilities": ["spatial_displacement"],
            },
            "design_notes": {"must_preserve": [], "approximations": [], "risks": []},
            "source_variant": {"base_stem": "TrModelGeneratedHorizontalSplitBlur01"},
        }

        issues = validate_effect_design(design)
        self.assertIn("target_effect.family must match implementation_seed.family", issues)
        self.assertIn(
            "implementation_seed.template_effect_id must match target_effect.closest_existing_effect_id", issues
        )

    def test_motion_topology_requires_candidate_direction_groups(self) -> None:
        root = Path(__file__).resolve().parents[1] / "work" / f"topology_score_{uuid.uuid4().hex}"
        reference = root / "reference"
        diagnostics = root / "diagnostics"
        try:
            diagnostics.mkdir(parents=True)
            (diagnostics / "reference_motion_diagnostics.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "topology_contract": {
                                "status": "required",
                                "minimum_concurrent_regions": 2,
                                "evidence_pairs": [{"from_frame": 4, "to_frame": 5}],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            analysis_file = root / "transition_structure.json"
            analysis_file.write_text(
                json.dumps(
                    {
                        "transition": {
                            "structure_type": "horizontal band split",
                            "region_count": 2,
                            "motion_axes": ["horizontal", "opposed"],
                            "split_geometry": "upper and lower bands",
                        }
                    }
                ),
                encoding="utf-8",
            )
            result = _score_motion_topology(
                reference,
                {
                    "pairs": [
                        {
                            "from_frame": 4,
                            "to_frame": 5,
                            "candidate_motion_region_count": 2,
                            "candidate_has_distinct_direction_groups": False,
                            "direction_agreement": 0.9,
                        }
                    ]
                },
                analysis_file=analysis_file,
            )
            self.assertEqual(result["status"], "structural_mismatch")
            self.assertEqual(result["candidate_region_match_rate"], 0.0)
            self.assertEqual(result["enforcement"], "advisory")
        finally:
            for path in sorted(root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            root.rmdir()

    def test_motion_topology_is_not_applicable_for_non_segmented_effect(self) -> None:
        root = Path(__file__).resolve().parents[1] / "work" / f"topology_policy_{uuid.uuid4().hex}"
        reference = root / "reference"
        diagnostics = root / "diagnostics"
        try:
            diagnostics.mkdir(parents=True)
            (diagnostics / "reference_motion_diagnostics.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "topology_contract": {
                                "status": "required",
                                "minimum_concurrent_regions": 2,
                                "evidence_pairs": [{"from_frame": 4, "to_frame": 5}],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            analysis_file = root / "transition_structure.json"
            analysis_file.write_text(
                json.dumps(
                    {
                        "transition": {
                            "structure_type": "dissolve",
                            "region_count": 1,
                            "motion_axes": [],
                            "split_geometry": None,
                        }
                    }
                ),
                encoding="utf-8",
            )

            result = _score_motion_topology(
                reference,
                {"pairs": [{"from_frame": 4, "to_frame": 5}]},
                analysis_file=analysis_file,
            )

            self.assertEqual(result["status"], "not_applicable")
            self.assertEqual(result["policy"]["mode"], "disabled")
        finally:
            for path in sorted(root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            root.rmdir()

    def test_effect_design_policy_overrides_inferred_motion_topology(self) -> None:
        root = Path(__file__).resolve().parents[1] / "work" / f"topology_policy_{uuid.uuid4().hex}"
        reference = root / "reference"
        diagnostics = root / "diagnostics"
        try:
            diagnostics.mkdir(parents=True)
            (diagnostics / "reference_motion_diagnostics.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "topology_contract": {
                                "status": "required",
                                "minimum_concurrent_regions": 2,
                                "evidence_pairs": [{"from_frame": 4, "to_frame": 5}],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            analysis_file = root / "transition_structure.json"
            analysis_file.write_text(
                json.dumps({"transition": {"structure_type": "horizontal band split", "region_count": 2}}),
                encoding="utf-8",
            )
            design_file = root / "effect_design.json"
            design_file.write_text(
                json.dumps({"evaluation_policy": {"motion_topology": {"mode": "disabled"}}}),
                encoding="utf-8",
            )

            result = _score_motion_topology(
                reference,
                {"pairs": [{"from_frame": 4, "to_frame": 5}]},
                analysis_file=analysis_file,
                design_file=design_file,
            )

            self.assertEqual(result["status"], "not_applicable")
            self.assertEqual(result["policy"]["source"], "effect_design")
        finally:
            for path in sorted(root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            root.rmdir()

    def test_controller_treats_advisory_motion_topology_mismatch_as_nonblocking(self) -> None:
        metrics = {
            "endpoint_checks": {
                "before_transition": {"mse": 0.0, "ssim": 1.0},
                "after_transition": {"mse": 0.0, "ssim": 1.0},
            },
            "motion_topology": {"status": "structural_mismatch"},
            "mse": 1.0,
            "ssim": 0.9,
        }

        outcome, reason = _select_outcome(None, metrics)
        self.assertEqual(outcome, "accepted")
        self.assertIn("first valid evaluation", reason)

    def test_controller_rejects_explicit_hard_motion_topology_mismatch(self) -> None:
        metrics = {
            "endpoint_checks": {
                "before_transition": {"mse": 0.0, "ssim": 1.0},
                "after_transition": {"mse": 0.0, "ssim": 1.0},
            },
            "motion_topology": {"status": "structural_mismatch", "enforcement": "hard"},
            "mse": 1.0,
            "ssim": 0.9,
        }

        outcome, reason = _select_outcome(None, metrics)
        self.assertEqual(outcome, "rejected")
        self.assertIn("motion topology", reason)

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

    def test_evaluation_profile_rejects_unresolved_placeholders(self) -> None:
        root = Path(__file__).resolve().parents[1] / "work" / f"profile_placeholder_{uuid.uuid4().hex}"
        candidate_dir = root / "candidate"
        candidate_dir.mkdir(parents=True)
        manifest = candidate_dir / "candidate_manifest.json"
        try:
            manifest.write_text(json.dumps({"effect_id": "ModelGenerated\\Test"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unresolved placeholder in job"):
                set_evaluation_profile(
                    manifest,
                    {
                        "manifest": str(manifest),
                        "job": str(root / "<sample-id>" / "render_job.json"),
                        "reference": str(root / "reference"),
                        "output_root": str(candidate_dir / "evaluations"),
                        "backup_root": str(candidate_dir / "backups"),
                        "msbuild": "msbuild.exe",
                        "renderer": "renderer.exe",
                        "width": 1920,
                        "height": 1080,
                        "frame_start": 14,
                        "frame_end": 44,
                    },
                )
        finally:
            for path in sorted(root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            root.rmdir()

    def test_reference_diagnostics_cli_defaults_to_sample_diagnostics_folder(self) -> None:
        reference = Path("D:/work/samples/example/reference")
        with patch("agent_app.cli.analyze_reference_diagnostics", return_value={"status": "succeeded"}) as analyze:
            with patch("builtins.print"):
                exit_code = cli_main(["reference-diagnostics", "--reference", str(reference)])

        self.assertEqual(exit_code, 0)
        self.assertEqual(analyze.call_args.kwargs["reference"], reference.resolve())
        self.assertEqual(analyze.call_args.kwargs["output_dir"], reference.resolve().parent / "diagnostics")

    def test_reference_diagnostics_cli_rejects_noncanonical_output_directory(self) -> None:
        reference = Path("D:/work/samples/example/reference")
        with patch("builtins.print"), self.assertRaises(SystemExit) as error:
            cli_main(
                [
                    "reference-diagnostics",
                    "--reference",
                    str(reference),
                    "--output-dir",
                    str(reference.parent / "analysis" / "diagnostics"),
                ]
            )

        self.assertEqual(error.exception.code, 1)

    def test_reference_diagnostics_rejects_file_looking_output_directory(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be a directory path"):
            from agent_app.workflow import analyze_reference_diagnostics

            analyze_reference_diagnostics(
                workspace_root=Path("D:/AI_Harness"),
                reference=Path("D:/work/samples/example/reference"),
                output_dir=Path("D:/work/samples/example/analysis/reference_motion_diagnostics.json"),
                width=1920,
                height=1080,
            )

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

    def test_motion_refinement_priority_requires_reliable_direction_mismatch(self) -> None:
        priority = _motion_refinement_priority(
            {
                "history": [
                    {
                        "iteration": 2,
                        "hypothesis_category": "blur",
                        "metrics": {
                            "motion": {
                                "reliable_motion_coverage": 0.95,
                                "direction_agreement": 0.38,
                            }
                        },
                    }
                ]
            }
        )
        self.assertEqual(priority["level"], "high")
        self.assertEqual(priority["recommended_categories"], ["regions", "displacement"])

    def test_motion_refinement_priority_prefers_signed_direction_over_topology(self) -> None:
        priority = _motion_refinement_priority(
            {
                "history": [
                    {
                        "iteration": 4,
                        "hypothesis_category": "regions",
                        "metrics": {
                            "motion_topology": {
                                "status": "structural_mismatch",
                                "evidence_pair_count": 3,
                                "candidate_region_match_rate": 2 / 3,
                                "direction_match_rate": 0.0,
                            }
                        },
                    }
                ]
            }
        )
        self.assertEqual(priority["focus"], "signed_direction")
        self.assertEqual(priority["recommended_categories"], ["displacement", "regions", "shader_structure"])

    def test_motion_refinement_priority_uses_uv_mapping_after_motion_is_aligned(self) -> None:
        priority = _motion_refinement_priority(
            {
                "history": [
                    {
                        "iteration": 5,
                        "hypothesis_category": "displacement",
                        "metrics": {
                            "edge_content_policy": {
                                "reference": {
                                    "status": "estimated",
                                    "recommended_policy": "repeat",
                                    "confidence": 0.82,
                                },
                                "candidate": {"policy": "mirror"},
                            }
                        },
                    }
                ]
            }
        )
        self.assertEqual(priority["focus"], "edge_content_policy")
        self.assertEqual(priority["recommended_categories"], ["uv_mapping", "shader_structure"])

    def test_motion_refinement_priority_can_focus_on_geometry_mismatch(self) -> None:
        priority = _motion_refinement_priority(
            {
                "history": [
                    {
                        "iteration": 3,
                        "hypothesis_category": "blend",
                        "metrics": {
                            "motion_geometry": {
                                "status": "geometry_mismatch",
                                "rotation_delta_degrees": 18.0,
                                "scale_delta_ratio": 0.22,
                                "reference": {"confidence": 0.9},
                                "candidate": {"confidence": 0.8},
                            }
                        },
                    }
                ]
            }
        )
        self.assertEqual(priority["focus"], "motion_geometry")
        self.assertEqual(priority["recommended_categories"], ["displacement", "regions", "shader_structure"])

    def test_motion_refinement_priority_focuses_on_transform_position(self) -> None:
        priority = _motion_refinement_priority(
            {
                "history": [{
                    "iteration": 3,
                    "hypothesis_category": "timing",
                    "metrics": {
                        "motion_geometry": {
                            "status": "geometry_mismatch",
                            "translation_delta_pixels": 14.0,
                            "translation_direction_agreement": False,
                            "reference": {"confidence": 0.9},
                            "candidate": {"confidence": 0.8},
                        }
                    },
                }]
            }
        )
        self.assertEqual(priority["focus"], "transform_position")
        self.assertEqual(priority["recommended_categories"], ["displacement", "shader_structure"])

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

    def test_progress_calibration_prefers_explicit_evaluation_window(self) -> None:
        window = _reference_output_window(
            candidate_manifest_file=Path("candidate_manifest.json"),
            reference_path=Path("reference"),
            frame_count=60,
            requested_frame_start=14,
            requested_frame_end=44,
        )
        self.assertEqual(window, {"frame_start": 14, "frame_end": 44, "source": "evaluation_arguments"})

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

    def test_sample_workspace_syncs_effect_catalog_once_when_workspace_root_is_provided(self) -> None:
        root = Path(__file__).resolve().parents[1] / "work" / f"sample_catalog_sync_{uuid.uuid4().hex}"
        source = root / "input.mp4"
        try:
            root.mkdir(parents=True)
            source.write_bytes(b"video")
            fake_modules = {
                "sync_effect_catalog_sources": lambda workspace_root, source_manifest_path: {
                    "manifest": {"catalog_type": "effect_catalog_sources", "registrations": []},
                    "discovered_fx_ids": ["ModelGenerated\\Example_01"],
                    "added_fx_ids": ["ModelGenerated\\Example_01"],
                    "removed_fx_ids": [],
                },
                "build_effect_catalog": lambda workspace_root, source_manifest_path: {
                    "catalog_type": "effect_catalog",
                    "registration_count": 1,
                },
            }
            with patch("agent_app.sample_workspace.load_harness_modules", return_value=fake_modules):
                result = initialize_sample_workspace(
                    root / "samples",
                    "example_002",
                    source,
                    workspace_root=root,
                )

            self.assertEqual(result["catalog_sync"]["discovered_fx_count"], 1)
            manifest = json.loads(Path(result["manifest_file"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["catalog_sync"]["added_fx_ids"], ["ModelGenerated\\Example_01"])
            self.assertTrue((root / "harness" / "configs" / "effect_catalog_sources.json").is_file())
            self.assertTrue((root / "harness" / "configs" / "effect_catalog.json").is_file())
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
            diagnostics_dir = root / "diagnostics"
            diagnostics_dir.mkdir()
            diagnostic_video = diagnostics_dir / "reference_motion_diagnostics.mp4"
            diagnostic_video.write_bytes(b"video")
            (diagnostics_dir / "reference_motion_diagnostics.json").write_text(
                json.dumps({"video": {"file": str(diagnostic_video)}}), encoding="utf-8"
            )
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
            packet_data = json.loads(Path(packet["packet_file"]).read_text(encoding="utf-8"))
            self.assertEqual(packet_data["reference_diagnostics_file"], str(diagnostics_dir / "reference_motion_diagnostics.json"))
            self.assertEqual(packet_data["reference_diagnostics_video"], str(diagnostic_video))

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

    def test_reassess_apply_restores_saved_historical_candidate(self) -> None:
        root = Path(__file__).resolve().parents[1] / "work" / f"reassess_test_{uuid.uuid4().hex}"
        candidate_dir = root / "candidate"
        candidate_dir.mkdir(parents=True)
        manifest = candidate_dir / "candidate_manifest.json"
        candidate_source = candidate_dir / "Candidate.h"
        target_source = root / "target" / "Candidate.h"
        baseline_report = candidate_dir / "baseline_report.json"
        improved_report = candidate_dir / "improved_report.json"
        design = candidate_dir / "design.json"
        generated_manifest = candidate_dir / "generated_manifest.json"
        try:
            candidate_source.write_text("baseline", encoding="utf-8")
            target_source.parent.mkdir()
            target_source.write_text("baseline", encoding="utf-8")
            design.write_text(
                json.dumps({"evaluation_policy": {"selection": {
                    "profile": "transform",
                    "primary_metrics": ["mse", "mae"],
                    "guardrail_metrics": ["peak_ssim"],
                    "advisory_metrics": [],
                }}}),
                encoding="utf-8",
            )
            generated_manifest.write_text(json.dumps({"design_artifact": str(design)}), encoding="utf-8")
            manifest.write_text(json.dumps({
                "effect_id": "ModelGenerated\\Test",
                "candidate_files": [str(candidate_source)],
                "target_files": [str(target_source)],
                "source_manifest": str(generated_manifest),
            }), encoding="utf-8")
            self._write_controller_report(baseline_report, mse=100.0, ssim=0.50)
            set_candidate_baseline(manifest, iteration=0, report_file=baseline_report)
            candidate_source.write_text("historical improvement", encoding="utf-8")
            (candidate_dir / "iteration_001_displacement.json").write_text(
                json.dumps({"iteration": 1, "hypothesis_category": "displacement"}), encoding="utf-8"
            )
            self._write_controller_report(improved_report, mse=80.0, ssim=0.40)
            record_candidate_evaluation(manifest, 1, improved_report)
            candidate_source.write_text("later edit", encoding="utf-8")

            preview = reassess_candidate_history(manifest)
            self.assertEqual(preview["recommended_baseline_iteration"], 1)
            applied = apply_reassessed_baseline(manifest)

            self.assertEqual(applied["restored_baseline_iteration"], 1)
            self.assertEqual(candidate_source.read_text(encoding="utf-8"), "historical improvement")
            self.assertEqual(target_source.read_text(encoding="utf-8"), "historical improvement")
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
                max_rejected=2,
            )
            self.assertEqual(start["phase"]["first_iteration"], 2)
            set_evaluation_profile(
                manifest,
                {
                    "manifest": str(manifest),
                    "job": str(root / "render_job.json"),
                    "reference": str(root / "reference"),
                    "output_root": str(candidate_dir / "evaluations"),
                    "backup_root": str(candidate_dir / "backups"),
                    "msbuild": "C:/tools/MSBuild.exe",
                    "renderer": "C:/tools/OverlayTrHarnessRenderer.exe",
                    "configuration": "Debug",
                    "platform": "x64",
                    "width": 1920,
                    "height": 1080,
                    "frame_start": 14,
                    "frame_end": 43,
                    "calibrate_progress": True,
                },
            )
            packet = build_next_iteration_packet(
                candidate_manifest_file=manifest,
                analysis_file=analysis,
                design_file=design,
                max_iterations=1,
                max_rejected=1,
                evaluate_after_edit=True,
            )
            self.assertEqual(packet["iteration"], 2)
            self.assertTrue(packet["evaluation_after_edit"])
            packet_data = json.loads(Path(packet["packet_file"]).read_text(encoding="utf-8"))
            self.assertEqual(packet_data["active_phase"]["name"], "optical_flow")
            self.assertEqual(packet_data["budgets"]["rejected_so_far"], 0)
            request = Path(packet["prompt_file"]).read_text(encoding="utf-8")
            self.assertIn("candidate-evaluate", request)
            self.assertIn("candidate-continue", request)
            self.assertIn("--continue-analysis", request)
            self.assertIn("`rejected`", request)
            self.assertIn("not a failure", request)
            self.assertIn("--iteration 2", request)
            self.assertIn("--calibrate-progress", request)

            candidate_source.write_text("rejected", encoding="utf-8")
            target_source.write_text("rejected", encoding="utf-8")
            (candidate_dir / "iteration_002_blur.json").write_text(
                json.dumps({"iteration": 2, "hypothesis_category": "blur", "status": "candidate_only"}),
                encoding="utf-8",
            )
            rejected_report = candidate_dir / "rejected_report.json"
            self._write_controller_report(rejected_report, mse=20.0, ssim=0.8, motion_similarity=0.5)
            outcome = record_candidate_evaluation(manifest, 2, rejected_report)
            self.assertEqual(outcome["status"], "rejected")

            continued = continue_candidate_refinement(
                candidate_manifest_file=manifest,
                analysis_file=analysis,
                design_file=design,
                max_iterations=1,
                max_rejected=1,
            )
            self.assertTrue(continued["restored_baseline"])
            self.assertEqual(continued["next_iteration"], 3)
            self.assertEqual(candidate_source.read_text(encoding="utf-8"), "baseline")
            next_request = Path(continued["prompt_file"]).read_text(encoding="utf-8")
            self.assertIn("--iteration 3", next_request)

            with self.assertRaisesRegex(ValueError, "phase name already exists"):
                start_refinement_phase(
                    candidate_manifest_file=manifest,
                    name="optical_flow",
                    baseline_iteration=1,
                    report_file=report,
                    max_iterations=2,
                    max_rejected=2,
                )

            restarted = start_refinement_phase(
                candidate_manifest_file=manifest,
                name="optical_flow_2",
                baseline_iteration=1,
                report_file=report,
                max_iterations=2,
                max_rejected=2,
            )
            state_data = json.loads((candidate_dir / "candidate_state.json").read_text(encoding="utf-8"))
            optical_phases = [phase for phase in state_data["phases"] if phase["name"] == "optical_flow_2"]
            self.assertEqual(len(optical_phases), 1)
            self.assertEqual(optical_phases[0]["status"], "active")
            self.assertEqual(restarted["phase"]["first_iteration"], 3)
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

    def test_prepare_sources_defaults_to_video_endpoints(self) -> None:
        root = Path(__file__).resolve().parents[1] / "work" / f"endpoint_sources_{uuid.uuid4().hex}"
        root.mkdir(parents=True)
        source_video = root / "sample.mp4"
        output_root = root / "sources"
        try:
            source_video.write_bytes(b"video")

            def extract(_ffmpeg: str, _video: Path, frame_index: int, output_file: Path, _width: int, _height: int) -> None:
                output_file.write_bytes(str(frame_index).encode("ascii"))

            with patch("agent_app.workflow._probe_video_frame_count", return_value=114):
                with patch("agent_app.workflow._extract_single_frame", side_effect=extract):
                    result = prepare_sources(
                        source_video=source_video,
                        output_root=output_root,
                        start_frame=None,
                        end_frame=None,
                        frame_count=2,
                        width=16,
                        height=16,
                        ffmpeg_path="ffmpeg.exe",
                    )

            self.assertEqual(result["source_a_frame"], 0)
            self.assertEqual(result["source_b_frame"], 113)
            self.assertEqual(result["selection"]["mode"], "video_endpoints")
            self.assertEqual((output_root / "source_a" / "frame_0000.png").read_bytes(), b"0")
            self.assertEqual((output_root / "source_b" / "frame_0000.png").read_bytes(), b"113")
        finally:
            for path in sorted(root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            root.rmdir()

    def test_resolve_source_boundaries_maps_prepared_reference_to_original_video(self) -> None:
        analysis = {
            "transition": {
                "stable_source_a_end_frame": 13,
                "stable_source_b_start_frame": 45,
            }
        }
        reference_manifest = {
            "frame_progress_mapping": [
                {"output_frame": 13, "normalized_clip_source_frame": 63},
                {"output_frame": 45, "normalized_clip_source_frame": 95},
            ]
        }
        with patch("agent_app.workflow.load_json", side_effect=[analysis, reference_manifest]):
            result = resolve_source_boundaries(Path("analysis.json"), Path("reference_transition_manifest.json"))

        self.assertEqual(result["mode"], "prepared_reference_mapping")
        self.assertEqual(result["source_a_frame"], 63)
        self.assertEqual(result["source_b_frame"], 95)

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
