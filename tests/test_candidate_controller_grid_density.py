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
    _secondary_grid_density_diagnostics,
    _secondary_grid_density_instruction,
)


def _make_workspace(tmp: str, fx_name: str = "GridReveal_01"):
    samples_root = Path(tmp) / "samples"
    primary = samples_root / "sample_primary"
    (primary / "analysis").mkdir(parents=True)
    analysis_file = primary / "analysis" / "transition_structure.json"
    analysis_file.write_text("{}", encoding="utf-8")
    candidate_manifest_file = primary / "candidates" / fx_name / "candidate_manifest.json"
    candidate_manifest_file.parent.mkdir(parents=True)
    candidate_manifest_file.write_text("{}", encoding="utf-8")
    return samples_root, analysis_file, candidate_manifest_file


def _write_secondary(
    samples_root: Path,
    name: str,
    fx_name: str = "GridReveal_01",
    diagnostics: dict | None = None,
    with_regression_job: bool = True,
) -> None:
    secondary = samples_root / name
    if with_regression_job:
        (secondary / "jobs").mkdir(parents=True)
        (secondary / "jobs" / f"{fx_name}_regression_job.json").write_text("{}", encoding="utf-8")
    if diagnostics is not None:
        (secondary / "diagnostics").mkdir(parents=True, exist_ok=True)
        (secondary / "diagnostics" / "grid_density_diagnostics.json").write_text(
            json.dumps(diagnostics), encoding="utf-8"
        )


class SecondaryGridDensityDiagnosticsTests(unittest.TestCase):
    def test_no_secondaries_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            samples_root, analysis_file, candidate_manifest_file = _make_workspace(tmp)
            result = _secondary_grid_density_diagnostics(analysis_file, candidate_manifest_file)
            self.assertEqual(result, [])

    def test_finds_secondary_with_estimated_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            samples_root, analysis_file, candidate_manifest_file = _make_workspace(tmp)
            _write_secondary(
                samples_root,
                "sample_secondary",
                diagnostics={
                    "status": "estimated",
                    "estimated_columns": 5.05,
                    "estimated_rows": 2.81,
                    "confidence": 0.1821,
                },
            )
            result = _secondary_grid_density_diagnostics(analysis_file, candidate_manifest_file)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["sample"], "sample_secondary")
            self.assertEqual(result[0]["estimated_columns"], 5.05)
            self.assertEqual(result[0]["estimated_rows"], 2.81)

    def test_skips_secondary_without_regression_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            samples_root, analysis_file, candidate_manifest_file = _make_workspace(tmp)
            _write_secondary(
                samples_root,
                "sample_unrelated",
                with_regression_job=False,
                diagnostics={"status": "estimated", "estimated_columns": 5.0, "estimated_rows": 3.0},
            )
            result = _secondary_grid_density_diagnostics(analysis_file, candidate_manifest_file)
            self.assertEqual(result, [])

    def test_skips_secondary_with_regression_job_for_different_fx(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            samples_root, analysis_file, candidate_manifest_file = _make_workspace(tmp)
            _write_secondary(
                samples_root,
                "sample_other_fx",
                fx_name="OtherEffect_02",
                diagnostics={"status": "estimated", "estimated_columns": 5.0, "estimated_rows": 3.0},
            )
            result = _secondary_grid_density_diagnostics(analysis_file, candidate_manifest_file)
            self.assertEqual(result, [])

    def test_skips_secondary_with_low_confidence_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            samples_root, analysis_file, candidate_manifest_file = _make_workspace(tmp)
            _write_secondary(
                samples_root,
                "sample_secondary",
                diagnostics={"status": "low_confidence", "estimated_columns": 96.0, "estimated_rows": 54.0},
            )
            result = _secondary_grid_density_diagnostics(analysis_file, candidate_manifest_file)
            self.assertEqual(result, [])

    def test_skips_secondary_missing_diagnostics_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            samples_root, analysis_file, candidate_manifest_file = _make_workspace(tmp)
            _write_secondary(samples_root, "sample_secondary", diagnostics=None)
            result = _secondary_grid_density_diagnostics(analysis_file, candidate_manifest_file)
            self.assertEqual(result, [])


class SecondaryGridDensityInstructionTests(unittest.TestCase):
    def test_empty_without_findings(self) -> None:
        self.assertEqual(_secondary_grid_density_instruction({}), "")
        self.assertEqual(_secondary_grid_density_instruction({"secondary_grid_density_diagnostics": None}), "")
        self.assertEqual(_secondary_grid_density_instruction({"secondary_grid_density_diagnostics": []}), "")

    def test_includes_sample_name_and_measurement(self) -> None:
        packet = {
            "secondary_grid_density_diagnostics": [
                {
                    "sample": "sample_grid_art",
                    "status": "estimated",
                    "estimated_columns": 5.05,
                    "estimated_rows": 2.81,
                    "confidence": 0.1821,
                }
            ]
        }
        text = _secondary_grid_density_instruction(packet)
        self.assertIn("sample_grid_art", text)
        self.assertIn("5.0x2.8", text)
        self.assertIn("not as this sample's actual grid count", text)


if __name__ == "__main__":
    unittest.main()
