# Effect Refinement Prompt

Use this prompt after an effect has been generated, registered, rendered, and
scored. Refinement happens in an isolated candidate workspace and keeps the
existing FX ID unchanged.

```text
/goal

You are refining one candidate transition effect in the local workspace.

Read:
- the transition analysis JSON
- the current effect-design JSON
- the current candidate iteration packet JSON
- the latest render report
- the latest score report
- `motion_metrics` and worst-motion-pair diagnostics when the score report includes them
- the latest candidate review MP4 when available
- the latest optical-flow diagnostic MP4 when available
- all source files under the candidate workspace

Rules:
- edit only the candidate workspace files explicitly provided for this iteration
- do not edit existing production effects or registration tables
- preserve the existing FX class name, shader symbol, and FX ID
- compile-oriented C++ and HLSL must remain compatible with the existing project
- make the smallest source change that addresses the observed mismatch
- do not claim success without a later build, render, and score
- choose exactly one hypothesis category: timing, regions, displacement, blur,
  blend, shader_structure, or other
- do not repeat a rejected hypothesis category without new visual evidence

The transition may contain an arbitrary number of spatial regions. Do not assume
that the image has only two or four regions. Infer region boundaries and motion
from the reference frames when the evidence supports them.

Treat MSE and SSIM as image-similarity signals, not a complete description of a
transition. For horizontal banded motion, compare the candidate and reference
motion diagnostics before changing displacement, direction, or region layout.

After editing, summarize the intended change in a small JSON object containing:
- iteration
- hypothesis_category
- changed_files
- visual_hypothesis
- expected_effect_change
- unresolved_risks
```

Codex edits the candidate source. The local agent commands perform backup,
promotion, build, rendering, scoring, and reporting.
