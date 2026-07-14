A codex-driven 2-pass solution to analyze the sample video in order to generate the HLSL/C++ codes for new A-B transition effect:
Pass 1: sample video -> Codex analysis summary
This should be Codex-driven, not harness-driven.
The reusable helper here is prepare_reference_transition, because it already:
normalizes the sample video
detects the likely transition window
produces aligned reference frames plus a manifest

Pass 2: generate/apply effect -> render -> score
This can reuse the current renderer and scoring stack almost directly.
