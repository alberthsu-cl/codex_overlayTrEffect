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

Important constraints:
- the sample video is the primary source of truth
- do not fall back to a deterministic local analyzer as the main answer
- do not invent details that are not visible
- if the video cannot be inspected directly in this environment, report that limitation in the JSON instead of pretending confidence
- keep the result compact, stable, and machine-readable

What to extract:
- video metadata when available: frame count, fps, width, height
- a short transition style label
- a one-sentence summary of the visible transformation
- start and end frame of the main transition window
- confidence for the overall classification
- dominant signals such as dissolve, morph, masked blend, RGB split, blur, displacement, scene continuity, camera continuity, or lighting continuity
- frame-by-frame normalized progress values across the transition window
- a short evidence list grounded in visible cues
- a short limitations list when the sample is ambiguous or partially occluded
- a downstream recommendation:
  - nearest existing effect family if there is a good match
  - whether a new effect is likely needed
  - implementation notes only at a high level

Use the schema supplied alongside this prompt. The final response must satisfy that schema exactly.
```
