from __future__ import annotations

import sys
from pathlib import Path


def load_harness_modules(workspace_root: Path):
    """Make the sibling harness package importable without duplicating it."""
    harness_src = (workspace_root / "harness" / "src").resolve()
    if not harness_src.exists():
        raise FileNotFoundError(f"harness source directory does not exist: {harness_src}")
    harness_src_text = str(harness_src)
    if harness_src_text not in sys.path:
        sys.path.insert(0, harness_src_text)

    from overlay_harness.evaluator import (
        analyze_reference_motion,
        create_motion_visualizations,
        score_frame_sequences,
        score_motion,
    )
    from overlay_harness.effect_catalog import build_effect_catalog
    from overlay_harness.effect_catalog import select_effect_candidate
    from overlay_harness.models import load_render_job
    from overlay_harness.renderer import prepare_render_invocation
    from overlay_harness.video_prep import prepare_reference_transition
    from overlay_harness.workspace import JobWorkspace

    return {
        "JobWorkspace": JobWorkspace,
        "load_render_job": load_render_job,
        "prepare_reference_transition": prepare_reference_transition,
        "prepare_render_invocation": prepare_render_invocation,
        "score_frame_sequences": score_frame_sequences,
        "score_motion": score_motion,
        "create_motion_visualizations": create_motion_visualizations,
        "analyze_reference_motion": analyze_reference_motion,
        "build_effect_catalog": build_effect_catalog,
        "select_effect_candidate": select_effect_candidate,
    }
