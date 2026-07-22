# New Sample Sequence

Run every command from `D:\AI_Harness` and use the `harness` Conda
environment:

```powershell
conda run -n harness python agent/src/main.py <command>
```

Use one workspace per input video. Replace `<sample-id>` with a lowercase,
unique ID, for example `boss_20260721`.

## 1. Create the Sample Workspace

```powershell
conda run -n harness python agent/src/main.py sample-init `
  --sample-id <sample-id> `
  --source-video "D:\input\sample.mp4"
```

This creates:

```text
agent/work/samples/<sample-id>/
  reference/
  sources/
  analysis/
  design/
  jobs/
  effects/
  candidates/
  reports/
```

## 2. Prepare Reference Frames

`--target-frame-count` is the maximum normalized count, not a request for the
first frames of the video. Preparation analyzes the full video, detects a
high-change transition window, then samples up to this many frames across that
window. Read `reference/reference_transition_manifest.json` afterward and use
its actual `frame_count` for the repeated A/B source sequences. If the detected
window contains fewer frames, preparation outputs fewer than the requested
count.

```powershell
conda run -n harness python agent/src/main.py prepare `
  --source-video "D:\input\sample.mp4" `
  --output-dir agent/work/samples/<sample-id>/reference `
  --target-frame-count 60
```

## 3. Ask Codex to Analyze the Sample

Use:

- `agent/prompts/codex_transition_analysis_prompt.md`
- `agent/prompts/codex_transition_analysis_schema.json`
- the original video
- `agent/work/samples/<sample-id>/reference`

Save the returned JSON as:

```text
agent/work/samples/<sample-id>/analysis/transition_structure.json
```

The analysis should identify the stable A/B source boundaries and the
transition window.

## 4. Prepare Source A/B Frames

Use stable source-frame boundaries found in the analysis. The command repeats
the selected A and B frames to create equal-length source sequences.

```powershell
conda run -n harness python agent/src/main.py prepare-sources `
  --source-video "D:\input\sample.mp4" `
  --output-root agent/work/samples/<sample-id>/sources `
  --start-frame <stable-A-frame> `
  --end-frame <stable-B-frame> `
  --frame-count <reference-manifest-frame-count>
```

## 5. Ask Codex for the Effect Design

Use:

- `agent/prompts/effect_design_prompt.md`
- `agent/prompts/effect_design_schema.json`
- `analysis/transition_structure.json`

For a sample that must produce a new shader, add this policy to the Codex
request before asking for the JSON:

```text
Delivery policy: This sample must produce a new shader effect. Do not select
reuse_existing_effect as the final result. Assign a new unique
ModelGenerated\<Family>_XX target_effect.effect_id after inspecting existing
registered ModelGenerated IDs. If an existing pure-HLSL effect is the closest
starting point, select tune_existing_effect and provide source_variant so it
can be cloned into the new ID. Do not modify the source effect.
```

In this workflow, `tune_existing_effect` means “clone and adapt a base effect
into a new ID,” not “edit the existing effect.”

Save the returned JSON as:

```text
agent/work/samples/<sample-id>/design/effect_design.json
```

## 6. Build a Render Job

Create the job from the analysis, design, source A/B frames, and prepared
reference frames:

```powershell
conda run -n harness python agent/src/main.py build-job `
  --analysis agent/work/samples/<sample-id>/analysis/transition_structure.json `
  --design agent/work/samples/<sample-id>/design/effect_design.json `
  --source-a agent/work/samples/<sample-id>/sources/source_a `
  --source-b agent/work/samples/<sample-id>/sources/source_b `
  --reference-transition agent/work/samples/<sample-id>/reference `
  --output agent/work/samples/<sample-id>/jobs/render_job.json
```

## 7. Generate, Register, and Refine When Needed

The FX ID is global to `OverlayTrEngine`, so it must be unique across all
sample workspaces, for example `ModelGenerated\\SeamlessSliding_03`.

`<effect-name>` becomes available after step 5. Read
`target_effect.effect_id` from `design/effect_design.json` and use the part
after `ModelGenerated\\` as the folder name. For example,
`ModelGenerated\\SeamlessSliding_03` uses `SeamlessSliding_03`.

Run this once before the commands below. It derives the folder name directly
from the design artifact, so it cannot drift from the selected FX ID:

```powershell
$sampleId = "<sample-id>"
$designPath = "agent/work/samples/$sampleId/design/effect_design.json"
$design = Get-Content -Raw $designPath | ConvertFrom-Json
$effectId = [string]$design.target_effect.effect_id
$effectName = $effectId -replace '^ModelGenerated\\', ''

if ([string]::IsNullOrWhiteSpace($effectName) -or $effectName -eq $effectId) {
  throw "effect_design.json must contain target_effect.effect_id as ModelGenerated\\Family_XX"
}
```

### Reuse an Existing Effect

For `reuse_existing_effect`, do not generate or register code. Ensure the
design's `target_effect.effect_id` is the existing runtime FX ID, build the
job in step 6, then render and score it:

```powershell
conda run -n harness python agent/src/main.py render `
  --job agent/work/samples/<sample-id>/jobs/render_job.json `
  --output-root agent/work/samples/<sample-id>/reports/reuse_render `
  --renderer harness/native_renderer/build/x64/Debug/OverlayTrHarnessRenderer.exe
```

### Generate and Register a New Model Effect

For `implement_new_effect` or `tune_existing_effect`, the design must contain
a new `ModelGenerated\\...` target ID. Generate the initial package, then
register it in `OverlayTrPlugInFx`:

```powershell
conda run -n harness python agent/src/main.py generate `
  --design $designPath `
  --output-dir agent/work/samples/$sampleId/effects/$effectName `
  --manifest agent/work/samples/$sampleId/effects/$effectName/manifest.json `
  --template-root overlaytrengine/OverlayTrPlugInFx

conda run -n harness python agent/src/main.py register `
  --manifest agent/work/samples/$sampleId/effects/$effectName/manifest.json `
  --target-root overlaytrengine
```

Current generator limits:

- `implement_new_effect` supports the dedicated
  `ModelGenerated\\Dissolve_XX` template.
- `tune_existing_effect` requires a `source_variant` section in the design and
  clones its declared pure-HLSL base sources into a new ModelGenerated ID.
- Resource-backed effects are not a generation target.

### Create and Evaluate the Candidate

Initialize the isolated candidate after registration. Codex edits only this
candidate directory during refinement:

```powershell
conda run -n harness python agent/src/main.py candidate-init `
  --manifest agent/work/samples/$sampleId/effects/$effectName/manifest.json `
  --output-dir agent/work/samples/$sampleId/candidates/$effectName

conda run -n harness python agent/src/main.py candidate-evaluate `
  --manifest agent/work/samples/$sampleId/candidates/$effectName/candidate_manifest.json `
  --job agent/work/samples/$sampleId/jobs/render_job.json `
  --reference agent/work/samples/$sampleId/reference `
  --output-root agent/work/samples/$sampleId/candidates/$effectName/evaluations `
  --backup-dir agent/work/samples/$sampleId/candidates/$effectName/backups/evaluation_001 `
  --msbuild "C:\Program Files\Microsoft Visual Studio\2022\Professional\MSBuild\Current\Bin\amd64\MSBuild.exe" `
  --renderer harness/native_renderer/build/x64/Debug/OverlayTrHarnessRenderer.exe `
  --width 1920 --height 1080
```

The first evaluation stages the candidate into the registered target, builds
the plugin, renders it, and creates review MP4s. Keep packets, backups, and
evaluations inside this candidate directory. Do not add `--restore` if the
result should remain available in `OverlayTrTool` for visual inspection.

## 8. Review and Decide

Review these generated artifacts after each candidate evaluation:

- `rendered_transition.mp4`
- `comparison_transition_window.mp4`
- `motion_diagnostics.mp4`
- `reports/candidate_iteration_report.json`

Use automated scores as diagnostics. When the visual result is acceptable,
restore the selected baseline and record `candidate-human-accept`; promotion to
the registered target remains a separate explicit action.

### Possible Wrong Cases Should be Handled
#### Correct a Wrong Transition Window

Do not refine the shader against a visibly wrong reference window. First find
the correct start and end frames in the original video at the preparation FPS
(normally 30 fps). Then recreate the reference with an explicit manual window:

```powershell
conda run -n harness python agent/src/main.py prepare `
  --source-video "D:\input\sample.mp4" `
  --output-dir agent/work/samples/<sample-id>/reference `
  --target-frame-count 60 `
  --start-frame <correct-start-frame> `
  --end-frame <correct-end-frame>
```

This replaces the prior `reference/` frames and writes a manifest with
`mode: manual_transition_window`. Read its actual `frame_count`, regenerate
the A/B source sequences for that count, update the analysis JSON's timing and
frame-progress mapping, and rebuild the render job. If the new window changes
the visible design materially, update the effect-design JSON but preserve the
already registered new FX ID.

Do not treat evaluations using the wrong window as a controller baseline. Run
a new baseline evaluation in a new output/backup directory after rebuilding;
because the corrected reference contains only the transition, score it from
frame `0` through `frame_count - 1`. Do not register the FX ID again.

### Decide Whether an Iteration Is Required

Do not create a shader iteration for an input or contract problem. Correct the
reference window, A/B sources, frame count, render job, effect ID, build error,
or scoring configuration first. Those changes invalidate prior comparisons, so
run a new baseline evaluation after they are corrected.

Create an iteration only when all of these are true:

- reference frames and A/B sources are visually correct;
- the render job targets the registered new FX ID;
- the baseline candidate compiles and renders successfully;
- visual review identifies a shader-specific mismatch to improve.

Examples of shader-specific mismatches are incorrect region count, movement
direction, displacement magnitude, blur, blend timing, or shader structure.

### Start a Valid Refinement Loop

After the initial candidate evaluation is valid, select it as iteration `0`
and start a bounded phase. Replace `<baseline-run>` with the evaluation folder
created by the valid baseline command:

```powershell
$candidateRoot = "agent/work/samples/<sample-id>/candidates/<effect-name>"
$baselineReport = "$candidateRoot/evaluations/<baseline-run>/reports/candidate_iteration_report.json"

conda run -n harness python agent/src/main.py candidate-start-phase `
  --manifest "$candidateRoot/candidate_manifest.json" `
  --name shader_refinement `
  --baseline-iteration 0 `
  --report $baselineReport `
  --max-iterations 5 `
  --max-rejected 3
```

Prepare one bounded iteration request:

```powershell
conda run -n harness python agent/src/main.py candidate-next `
  --manifest "$candidateRoot/candidate_manifest.json" `
  --analysis agent/work/samples/<sample-id>/analysis/transition_structure.json `
  --design agent/work/samples/<sample-id>/design/effect_design.json `
  --max-iterations 5 `
  --max-rejected 3
```

Read the emitted `packets/iteration_001_codex_request.md`, ask Codex to edit
only the candidate workspace, then evaluate that exact iteration. Use a unique
backup directory each time and the corrected scoring window:

```powershell
conda run -n harness python agent/src/main.py candidate-evaluate `
  --manifest "$candidateRoot/candidate_manifest.json" `
  --job agent/work/samples/<sample-id>/jobs/render_job.json `
  --reference agent/work/samples/<sample-id>/reference `
  --output-root "$candidateRoot/evaluations" `
  --backup-dir "$candidateRoot/backups/evaluation_003" `
  --msbuild "C:\Program Files\Microsoft Visual Studio\2022\Professional\MSBuild\Current\Bin\amd64\MSBuild.exe" `
  --renderer harness/native_renderer/build/x64/Debug/OverlayTrHarnessRenderer.exe `
  --width 1920 --height 1080 `
  --frame-start 0 --frame-end <reference-frame-count-minus-one> `
  --iteration 1
```

For an `accepted` outcome, the controller snapshots the new baseline
automatically. For a `rejected` or `tradeoff` outcome, restore the selected
baseline before asking Codex for another hypothesis:

```powershell
conda run -n harness python agent/src/main.py candidate-restore-baseline `
  --manifest "$candidateRoot/candidate_manifest.json"
```

Then run `candidate-next` again. Do not create a new FX ID or register again
for ordinary shader iterations. If visual review accepts the current baseline
despite imperfect diagnostics, record `candidate-human-accept` and close the
phase instead of continuing iterations.

## Automation Target

The current workflow deliberately has manual Codex handoffs for analysis,
effect design, and shader refinement. Local commands already automate frame
preparation, build, render, scoring, artifact creation, and controller state.

The intended future flow is:

```text
sample video
  -> local provisional transition-window detection
  -> Codex verifies or repairs the window and stable A/B source choices
  -> local manual-window preparation and frame-mapping validation
  -> Codex effect design with a new ModelGenerated FX ID
  -> local generation, registration, build, render, and baseline scoring
  -> bounded Codex shader refinement iterations
  -> local evaluation and controller decision after every iteration
  -> human acceptance or explicit promotion
```

Automation requirements that are not complete yet:

- Codex must be invoked by an explicit controller integration rather than a
  manual chat request.
- The analysis contract needs structured stable source-A and source-B frame
  selections, plus a window-review status, rather than keeping those choices
  only in evidence text.
- The controller needs to stop on invalid preparation inputs, repair the
  reference before scoring, and start a new baseline instead of treating the
  correction as a shader iteration.
- Candidate packets can eventually request an edit-and-evaluate cycle, while
  retaining bounded iteration and rejection budgets.

Until those items are implemented, use the manual Codex review points in this
memo and treat the local commands as the deterministic execution boundary.
