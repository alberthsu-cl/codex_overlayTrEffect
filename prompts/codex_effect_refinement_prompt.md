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
- `sampler_repetition` diagnostics when present; treat them as advisory evidence
  about wrap/mirror-capable samplers and modulo-like UV constructs, and inspect
  the rendered comparison before changing endpoint behavior. The shared
  renderer sampler may be `MIRROR` even when the shader does not declare a
  sampler. Trace transformed UVs at the displaced edges and choose an edge
  policy from the visual evidence: clamp, wrap, mirror, explicit visibility,
  or another effect-specific rule. If the reference does not repeat source
  content, consider explicit shader-side clamping; if it does repeat content,
  implement the required UV mapping explicitly rather than assuming the shared
  sampler state is the intended effect.
- `motion_geometry` comparison when the score report includes rotation, scale,
  reflection, or spatial-displacement diagnostics
- `regional_motion` comparison when the score report includes continuous signed
  regional vectors and dominant-axis evidence
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
- if `candidate-evaluate` fails during the build, do not start a new phase or
  skip the iteration; read the generated
  `iteration_NNN_build_repair_*.md` request, repair the compilation issue, and
  rerun the same iteration with a new backup directory
- choose exactly one hypothesis category: timing, regions, displacement, blur,
  blend, shader_structure, or other
- do not repeat a rejected hypothesis category without new visual evidence
- do not edit `render_job.json`, `progress_schedule`, render commands, or score
  reports; those are controller-owned generated artifacts

The transition may contain an arbitrary number of spatial regions. Do not assume
that the image has only two or four regions. Infer region boundaries and motion
from the reference frames when the evidence supports them.

Treat MSE and SSIM as image-similarity signals, not a complete description of a
transition. For any visible motion, compare the candidate and reference motion
diagnostics before changing displacement, direction, or region layout.

Motion-first triage:
- When the packet marks motion geometry as high priority, inspect the reference
  and candidate motion diagnostics before choosing a hypothesis.
- If reliable motion coverage is high but direction agreement is weak, correct
  displacement direction, sign, and region geometry before tuning blur or
  blend. This rule applies to any motion axis or number of regions; do not
  assume horizontal bands, two regions, or four regions.
- When `motion_geometry.status` is `geometry_mismatch` and both estimates have
  good confidence, inspect the reported rotation, scale, reflection, and
  displacement residuals before changing blur or blend.
- When `regional_motion.status` is `direction_mismatch`, use its continuous
  signed vectors and axis agreement to correct displacement or region layout.
  Do not reduce the evidence to fixed four- or eight-direction buckets.
- Do not chase tiny optical-flow fragments as shader regions. The scorer ignores
  very small reference fragments and matches reliable regions by spatial overlap
  plus direction. If a reliable reference region has no matching candidate
  region, prioritize the region mask or displacement sign that controls it.
- When the report says `structural_mismatch`, inspect
  `matched_direction_region_count` and the per-pair region evidence. A low
  match rate means a reliable reference region is missing or moving in the
  wrong direction; it does not mean the candidate must reproduce every raw
  connected-component count.
- Treat motion topology as advisory unless the packet explicitly says its
  enforcement is `hard`. Do not reshape a shader only to reproduce unstable
  or low-area optical-flow fragments.
- Choose `regions` or `displacement` for that investigation unless direct
  visual evidence rules out a motion-geometry mismatch. Do not use blur merely
  to hide an unresolved direction or region mismatch.
- Move to blur, blend, or fine timing only after motion geometry is consistent
  with the reference, or when the diagnostics explicitly show motion is not
  reliable enough to decide it.

Renderer progress alignment and shader timing are separate concerns. If the
candidate begins or ends on different output frames from the reference, do not
compensate by editing a render job. Request progress recalibration when a
timing or shader-structure change may have changed the shader's visible active
progress interval. Change shader timing only when the aligned comparison still
shows incorrect onset, peak motion, or settling within the reference transition
window.

After editing, summarize the intended change in a small JSON object containing:
- iteration
- hypothesis_category
- changed_files
- visual_hypothesis
- expected_effect_change
- unresolved_risks
- request_progress_recalibration (optional boolean; use `true` only when the
  controller should re-probe the candidate's visible active progress interval)
- progress_calibration_reason (required when
  `request_progress_recalibration` is `true`)
```

Codex edits the candidate source. The local agent commands perform backup,
promotion, build, rendering, scoring, and reporting.
