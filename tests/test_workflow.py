from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch


AGENT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(AGENT_SRC))

from agent_app.workflow import build_report, score_candidate


class WorkflowTests(unittest.TestCase):
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

