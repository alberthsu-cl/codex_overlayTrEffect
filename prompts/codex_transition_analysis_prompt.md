# Transition Analysis Prompt

Use this prompt when Codex should inspect a local sample transition video and emit a structured JSON artifact for the new `agent/` flow.

This is pass 1 of the new design:

1. analyze the sample transition video
2. produce a stable machine-readable summary
3. hand that summary to a later render-and-score stage

It is intentionally analysis-only. It should not plan jobs, generate code, or modify repository files.

```text
/goal

You are working in the local workspace under D:\AI_Harness.

Task:
Analyze the provided sample transition video and produce a single JSON artifact that describes the transition structure, timing, and visible effect cues.

Primary input:
- a local sample transition video path such as `D:\AI_Harness\sample.mp4`
- optional prepared reference frames if they already exist
- optional `reference_motion_diagnostics.json` and its diagnostic MP4, created
  locally from the prepared reference frames

Output rules:
- write exactly one JSON object that conforms to the supplied schema
- do not write prose, markdown, or code fences in the final output
- do not modify repository code
- do not generate HLSL or C++ in this stage

Analysis goals:
- identify the main transition window
- describe how clip A changes into clip B
- capture the dominant visible signals that matter for later effect design
- provide a normalized frame/progress mapping for the transition window
- suggest whether the result looks like an existing effect family or likely needs a new implementation

Analysis flow:
1. Review the diagnostics and the video as supporting evidence, not as the final answer.
2. Classify the visible geometry using general structural categories such as quadrant split, horizontal band split, vertical band split, masked split, dissolve, uv displacement, or other.
3. Identify the stable source A boundary, transition window, and stable source B boundary.
4. Reconcile any mismatch between the diagnostics and direct visual inspection.
5. Decide whether the window or boundaries need review or repair before downstream use.
6. Emit the final JSON artifact only after the structure and timing are internally consistent.

Important constraints:
- the sample video is the primary source of truth
- treat reference motion diagnostics as confidence-qualified supporting
  evidence, not as an effect classification or a source-boundary decision
- use direct inspection to resolve conflicts between the video and a heuristic;
  record that conflict in `limitations`
- do not fall back to a deterministic local analyzer as the main answer
- do not invent details that are not visible
- if the video cannot be inspected directly in this environment, report that limitation in the JSON instead of pretending confidence
- keep the result compact, stable, and machine-readable

What to extract:
- video metadata when available: frame count, fps, width, height
- a short transition style label
- a one-sentence summary of the visible transformation
- a structural classification and region count when the effect is segmented
- the main motion geometry, such as horizontal bands, vertical bands, true quadrants, or masked regions
- start and end frame of the main transition window
- stable source-A and source-B boundary frames when they can be resolved
- a window review status such as verified, needs_review, or repaired
- confidence for the overall classification
- dominant signals such as dissolve, morph, masked blend, RGB split, blur, displacement, scene continuity, camera continuity, or lighting continuity
- frame-by-frame normalized progress values across the transition window
- a short evidence list grounded in visible cues
- a short limitations list when the sample is ambiguous or partially occluded
- a downstream recommendation:
  - nearest existing effect family if there is a good match
  - whether the family is known or unknown
  - observable visual primitives for unknown families
  - whether a new effect is likely needed
  - whether the current constrained grammar can support implementation
- implementation notes only at a high level

Evaluation policy:
- Include `evaluation_policy.motion_topology` when the later regression scorer
  needs a policy specific to this sample.
- Use `mode: "disabled"` for effects whose visible behavior is not a persistent
  multi-region directional layout, such as ordinary dissolve, scale, rotation,
  or full-frame blur.
- Use `mode: "advisory"` for a visible segmented or multi-direction structure
  when flow evidence is useful but may fragment or vary across frames.
- Use `mode: "hard"` only when the segmented topology is unambiguous, central to
  the requested effect, and suitable as a strict acceptance requirement.
- Optional match-rate thresholds must be justified by the visible structure;
  otherwise omit them and let the local defaults apply.

When reference motion diagnostics are provided, use them to check region count,
per-region motion direction, approximate motion strength, onset/peak/settling,
and low-confidence blur or occlusion areas. Do not assume a fixed number of
regions from the diagnostic output.

Timing boundary:
- Report timing only for the reference sample video: its stable A/B boundaries,
  visible transition window, and normalized reference progression.
- When prepared reference frames and their manifest are supplied, report
  `transition.start_frame`, `transition.end_frame`, stable A/B boundaries, and
  `frame_progress_mapping.frame_index` in the prepared-reference output-frame
  coordinate system. The local `prepare-sources --analysis --reference-manifest`
  command maps those values back to original-video frames. Do not mix the two
  coordinate systems in one artifact.
- Do not infer, prescribe, or write a candidate renderer `progress_schedule`.
  The local controller derives that separately from a probe render of the
  generated candidate shader.

Decision rules:
- Use `family_status: "known"` only when the result maps to a known catalog or
  constrained grammar family.
- Use `family_status: "unknown"` when no existing family is adequate. This is
  a classification result, not permission to generate arbitrary shader code.
- Set `implementation_status` to `unsupported` when the current grammar cannot
  represent the observed behavior, and use `review_required` when feasibility
  is uncertain.
- If the structure is ambiguous, prefer a lower-confidence general category over
  a sample-specific label like "4 quadrant" unless the frame evidence really
  shows independent quadrant motion.
- If the boundaries are not cleanly visible, record that in the window review
  status instead of overfitting the timing.

Use the schema supplied alongside this prompt. The final response must satisfy that schema exactly.
```
