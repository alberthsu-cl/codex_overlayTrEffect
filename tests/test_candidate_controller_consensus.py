from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
import unittest


AGENT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(AGENT_SRC))

from agent_app.candidate_controller import (
    _consensus_annotation,
    _load_cross_sample_consensus,
    _refinement_priority_instruction,
)


def _consensus(convergent: dict, divergent: dict, sample_count: int = 4) -> dict:
    return {
        "artifact_type": "cross_sample_consensus",
        "artifact_version": 1,
        "sample_count": sample_count,
        "samples": [],
        "convergent": convergent,
        "divergent": divergent,
        "limitations_by_sample": {},
    }


class LoadConsensusTests(unittest.TestCase):
    def test_returns_none_for_missing_analysis_file(self) -> None:
        self.assertIsNone(_load_cross_sample_consensus(None))

    def test_returns_none_when_no_sibling_consensus_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            analysis_file = Path(tmp) / "transition_structure.json"
            analysis_file.write_text("{}", encoding="utf-8")
            self.assertIsNone(_load_cross_sample_consensus(analysis_file))

    def test_returns_none_for_wrong_artifact_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            analysis_file = Path(tmp) / "transition_structure.json"
            analysis_file.write_text("{}", encoding="utf-8")
            (Path(tmp) / "cross_sample_consensus.json").write_text(
                json.dumps({"artifact_type": "something_else"}), encoding="utf-8"
            )
            self.assertIsNone(_load_cross_sample_consensus(analysis_file))

    def test_loads_valid_sibling_consensus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            analysis_file = Path(tmp) / "transition_structure.json"
            analysis_file.write_text("{}", encoding="utf-8")
            payload = _consensus({"transition.structure_type": {"value": "masked split"}}, {})
            (Path(tmp) / "cross_sample_consensus.json").write_text(json.dumps(payload), encoding="utf-8")
            loaded = _load_cross_sample_consensus(analysis_file)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["sample_count"], 4)


class ConsensusAnnotationTests(unittest.TestCase):
    def test_no_annotation_without_consensus(self) -> None:
        priority = {"level": "high", "focus": "dense_rgb_slices"}
        self.assertEqual(_consensus_annotation(priority, None), "")

    def test_no_annotation_for_unmapped_focus(self) -> None:
        consensus = _consensus({}, {"transition.motion_axes": {"vertical": {"support": 1, "total": 4}}})
        priority = {"level": "high", "focus": "transform_rotation"}
        self.assertEqual(_consensus_annotation(priority, consensus), "")

    def test_flags_divergent_field_as_low_payoff(self) -> None:
        consensus = _consensus(
            {},
            {"transition.motion_axes": {"vertical": {"support": 1, "total": 4, "samples": ["grid_art"]}}},
        )
        priority = {"level": "high", "focus": "signed_direction"}
        note = _consensus_annotation(priority, consensus)
        self.assertIn("sample-specific", note)
        self.assertIn("transition.motion_axes", note)
        self.assertIn("4", note)

    def test_flags_convergent_field_as_worth_investing(self) -> None:
        consensus = _consensus(
            {"visual_signals.rgb_split": {"support": 4, "total": 4, "value": True}},
            {},
        )
        priority = {"level": "high", "focus": "dense_rgb_slices"}
        note = _consensus_annotation(priority, consensus)
        self.assertIn("convergent", note)
        self.assertIn("visual_signals.rgb_split", note)

    def test_no_annotation_when_no_keyword_matches(self) -> None:
        consensus = _consensus({"planner_hints.family_status": {"value": "known"}}, {})
        priority = {"level": "high", "focus": "dense_rgb_slices"}
        self.assertEqual(_consensus_annotation(priority, consensus), "")


class RefinementPriorityIntegrationTests(unittest.TestCase):
    def test_base_text_preserved_and_annotation_appended(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            analysis_file = Path(tmp) / "transition_structure.json"
            analysis_file.write_text("{}", encoding="utf-8")
            payload = _consensus(
                {},
                {"transition.motion_axes": {"vertical": {"support": 1, "total": 4, "samples": ["a"]}}},
            )
            (Path(tmp) / "cross_sample_consensus.json").write_text(json.dumps(payload), encoding="utf-8")

            priority = {
                "level": "high",
                "focus": "signed_direction",
                "topology": {
                    "evidence_pair_count": 4,
                    "candidate_region_match_rate": 0.5,
                    "direction_match_rate": 0.0,
                },
            }
            text = _refinement_priority_instruction(priority, analysis_file)
            self.assertIn("Current refinement priority: high signed direction.", text)
            self.assertIn("Cross-sample consensus note:", text)

    def test_no_consensus_file_leaves_text_unchanged(self) -> None:
        priority = {
            "level": "high",
            "focus": "signed_direction",
            "topology": {
                "evidence_pair_count": 4,
                "candidate_region_match_rate": 0.5,
                "direction_match_rate": 0.0,
            },
        }
        text_without_path = _refinement_priority_instruction(priority, None)
        self.assertNotIn("Cross-sample consensus note:", text_without_path)


if __name__ == "__main__":
    unittest.main()
