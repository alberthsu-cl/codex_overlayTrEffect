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
  diagnostics/
  jobs/
  effects/
  candidates/
  reports/
```

During `sample-init`, the harness synchronizes
`harness/configs/effect_catalog_sources.json` and
`harness/configs/effect_catalog.json` from the current `overlaytrengine`.
Therefore roll back the target project before creating the new sample
workspace. Existing sample workspaces are not resynchronized automatically.

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

## 3. Generate Reference Motion Evidence

Create a deterministic, confidence-qualified flow/region diagnostic before
asking Codex to analyze the sample. It is evidence only; it does not replace
Codex visual inspection or decide the transition window.

```powershell
conda run -n harness python agent/src/main.py reference-diagnostics `
  --reference agent/work/samples/<sample-id>/reference `
  --output-dir agent/work/samples/<sample-id>/diagnostics
```

Always use the explicit `--output-dir` above. Diagnostics belong beside the
prepared reference directory, at `agent/work/samples/<sample-id>/diagnostics`.
Do not point it under the sample's `analysis/` directory.

This produces:

```text
diagnostics/reference_motion_diagnostics.json
diagnostics/reference_motion_diagnostics.mp4
diagnostics/reference_motion_frames/
```

The diagnostic also records continuous signed regional vectors, inferred
dominant motion axis, and confidence-qualified transformation geometry. These
measurements are generic evidence for all transition types; a non-motion effect
may report them as unavailable or low confidence.

## 4. Ask Codex to Analyze the Sample

Use:

- `agent/prompts/codex_transition_analysis_prompt.md`
- `agent/prompts/codex_transition_analysis_schema.json`
- the original video
- `agent/work/samples/<sample-id>/reference`
- `agent/work/samples/<sample-id>/diagnostics/reference_motion_diagnostics.json`
- `agent/work/samples/<sample-id>/diagnostics/reference_motion_diagnostics.mp4`

The analysis pass should be a small structured loop:

1. classify the visible transition geometry;
2. identify stable source A, transition, and stable source B boundaries;
3. reconcile any mismatch between diagnostics and direct visual inspection;
4. assign a confidence level and a window review status;
5. emit the final JSON only after the structure and timing are consistent.

Save the returned JSON as:

```text
agent/work/samples/<sample-id>/analysis/transition_structure.json
```

The analysis should identify the stable A/B source boundaries, the transition
window, and the structural geometry of the transition so step 5 can prepare
accurate source repetitions.

## 5. Prepare Source A/B Frames

For generated two-still sample videos, use the default endpoint mode. It
extracts original video frame `0` as A and the final decoded frame as B, then
repeats those stills into equal-length source sequences.

```powershell
conda run -n harness python agent/src/main.py prepare-sources `
  --source-video "D:\input\sample.mp4" `
  --output-root agent/work/samples/<sample-id>/sources `
  --frame-count <reference-manifest-frame-count>
```

For an external, trimmed, or non-endpoint-stable video, use `--analysis` and
`--reference-manifest` to map prepared-reference stable boundaries back to
original-video frames. Use `--start-frame` and `--end-frame` only when their
values are known original source-video indices.

## 6. Ask Codex for the Effect Design

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

For a new generated effect, the design must also include `implementation_seed`.
Its `family` identifies the structural shader family, `template_effect_id`
identifies the actual clone source, and `required_shader_capabilities` lists
the behavior the new shader must support. These fields must agree with
`target_effect`; do not label a spatial multi-region transition as `dissolve`
solely because it contains a blend.

## 7. Build a Render Job

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

## 8. Generate, Register, and Refine When Needed

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
  --width 1920 --height 1080 `
  --frame-start <transition-output-start> `
  --frame-end <transition-output-end> `
  --calibrate-progress
```

The first evaluation stages the candidate into the registered target, builds
the plugin, renders it, and creates review MP4s. Keep packets, backups, and
evaluations inside this candidate directory. Do not add `--restore` if the
result should remain available in `OverlayTrTool` for visual inspection.

## 9. Review and Decide

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

### From Baseline Evaluation Through the Refinement Loop

Use this order after `candidate-init`. Set the two transition-output values
from the prepared reference and verified analysis. For example, the reference
window for a 60-frame sample might be `14` through `43`.

1. Create the calibrated baseline evaluation. Do not supply `--iteration` for
   this first run. Use a new backup directory and `--calibrate-progress`:

```powershell
$candidateRoot = "agent/work/samples/<sample-id>/candidates/<effect-name>"

conda run -n harness python agent/src/main.py candidate-evaluate `
  --manifest "$candidateRoot/candidate_manifest.json" `
  --job agent/work/samples/<sample-id>/jobs/render_job.json `
  --reference agent/work/samples/<sample-id>/reference `
  --output-root "$candidateRoot/evaluations" `
  --backup-dir "$candidateRoot/backups/evaluation_001" `
  --msbuild "C:\Program Files\Microsoft Visual Studio\2022\Professional\MSBuild\Current\Bin\amd64\MSBuild.exe" `
  --renderer harness/native_renderer/build/x64/Debug/OverlayTrHarnessRenderer.exe `
  --width 1920 --height 1080 `
  --frame-start <transition-output-start> `
  --frame-end <transition-output-end> `
  --calibrate-progress
```

Review the emitted comparison video. If the reference and candidate timing is
still visibly misaligned, correct calibration or preparation before starting a
shader iteration.

When `--iteration` is supplied, new evaluation folders are named from the
matching iteration record and effect job, for example
`iteration_030_displacement_band_magnitude_20260727T074106Z`.
The candidate directory already identifies the effect, so the run folder only
adds the refinement hypothesis and timestamp.
Evaluations without a unique matching iteration record retain the render job's
original folder prefix.

2. Start a bounded phase from this valid baseline. Replace `<baseline-run>`
with the new evaluation folder. This creates baseline iteration `0` and its
source snapshot:

```powershell
$baselineReport = "$candidateRoot/evaluations/<baseline-run>/reports/candidate_iteration_report.json"

conda run -n harness python agent/src/main.py candidate-start-phase `
  --manifest "$candidateRoot/candidate_manifest.json" `
  --name shader_refinement `
  --baseline-iteration 0 `
  --report $baselineReport `
  --max-iterations 8 `
  --max-rejected 4
```

3. Store the evaluation profile once for this candidate. It captures the
deterministic evaluation inputs that otherwise would have to be appended to
every Codex request:

```powershell
conda run -n harness python agent/src/main.py candidate-set-evaluation-profile `
  --manifest "$candidateRoot/candidate_manifest.json" `
  --job agent/work/samples/<sample-id>/jobs/render_job.json `
  --reference agent/work/samples/<sample-id>/reference `
  --output-root "$candidateRoot/evaluations" `
  --backup-root "$candidateRoot/backups" `
  --msbuild "C:\Program Files\Microsoft Visual Studio\2022\Professional\MSBuild\Current\Bin\amd64\MSBuild.exe" `
  --renderer harness/native_renderer/build/x64/Debug/OverlayTrHarnessRenderer.exe `
  --width 1920 --height 1080 `
  --frame-start <transition-output-start> `
  --frame-end <transition-output-end>
```

4. Generate the next packet *after* the phase starts. The `--evaluate-after-edit`
option writes an evaluation command and a continuation command into the generated
Codex request. Give `packets/iteration_001_codex_request.md` to Codex. Codex
edits the candidate workspace, runs the supplied evaluation, and asks the local
controller to prepare the next request:

```powershell
conda run -n harness python agent/src/main.py candidate-next `
  --manifest "$candidateRoot/candidate_manifest.json" `
  --analysis agent/work/samples/<sample-id>/analysis/transition_structure.json `
  --design agent/work/samples/<sample-id>/design/effect_design.json `
  --evaluate-after-edit
```

For a later refinement pass, use `candidate-resume` instead of manually
repeating the diagnostic, baseline, phase, and packet setup. It reads the
stored evaluation profile, regenerates missing or legacy reference diagnostics,
restores the selected baseline, closes the previous active phase, starts the
new bounded phase, and writes the first request with evaluation enabled:

```powershell
conda run -n harness python agent/src/main.py candidate-resume `
  --manifest "$candidateRoot/candidate_manifest.json" `
  --analysis agent/work/samples/<sample-id>/analysis/transition_structure.json `
  --design agent/work/samples/<sample-id>/design/effect_design.json `
  --phase shader_refinement_2 `
  --max-iterations 6 `
  --max-rejected 3
```

Use a new phase name for a separate pass. The command does not create a new FX
ID; it continues refining the same candidate workspace.

Motion-topology scoring is policy-driven. The effect design policy takes
priority over the transition analysis policy; otherwise the scorer infers it
from segmented structure such as bands, quadrants, or multiple regions. Use
`disabled` for non-segmented effects, `advisory` for useful but uncertain flow
evidence, and `hard` only for an explicit strict deliverable requirement.
Advisory topology remains available to guide Codex but does not reject a
candidate.

5. When the scored outcome is `accepted`, `rejected`, or `tradeoff`, the request
invokes `candidate-continue`. It restores the selected baseline for `rejected`
or `tradeoff` and returns the next `prompt_file`. Give that new request to Codex.
Codex stops after each continuation; it does not edit more than one candidate
iteration per request.

- visually acceptable but not diagnostically accepted: use
  `candidate-human-accept` to record the decision and close the active phase.

Do not create another FX ID or register again during ordinary shader
iterations. The same generated FX ID is refined until the phase is accepted,
closed by human review, or reaches its configured budget.

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
  -> bounded Codex shader refinement iterations with one local evaluation each
  -> controller decision after every iteration
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
- A future controller can invoke Codex itself, rather than requiring the user
  to submit each generated packet. The current bounded mode still requires the
  user to provide the generated request to Codex.

### Correct Renderer Progress Alignment

If the reference and candidate begin or end their visible transitions on
different output frames, treat this first as a renderer-progress calibration
problem. It is not automatically a shader iteration or a reference-window
problem.

The intended controller behavior is:

```text
candidate shader
  -> linear probe render
  -> local active-interval detection
  -> derive render.progress_schedule
  -> aligned evaluation and scoring
```

The controller, not Codex or the user, owns `render_job.json`. It may update
the generated `render.progress_schedule` after a probe, and it must record the
detected candidate interval, confidence, and fallback reason with the
evaluation artifacts. Codex may request recalibration in its iteration record
after a timing or shader-structure edit, but must not edit the job directly.

Retain the most recent calibrated schedule for region, displacement, blur, and
blend edits. Re-probe after timing or shader-structure edits, or whenever a
new probe indicates a material interval change. If detection is low confidence,
fall back to linear progress and mark the evaluation for review.

Only change shader timing when the aligned comparison still has incorrect
onset, peak motion, or settling *within* the reference transition window.

Until automatic probe calibration is implemented, use the following temporary
manual fallback. Ask Codex to review the side-by-side comparison and report the
reference transition frame range plus the candidate shader's visible active
progress range.

Rebuild the job with the reviewed mapping. The example below holds source A
through reference frame 13, stretches the shader's existing active range from
`24/59` through `40/59` across reference frames 14 through 43, then holds
source B from frame 44 onward:

```powershell
conda run -n harness python agent/src/main.py build-job `
  --analysis agent/work/samples/<sample-id>/analysis/transition_structure.json `
  --design agent/work/samples/<sample-id>/design/effect_design.json `
  --source-a agent/work/samples/<sample-id>/sources/source_a `
  --source-b agent/work/samples/<sample-id>/sources/source_b `
  --reference-transition agent/work/samples/<sample-id>/reference `
  --output agent/work/samples/<sample-id>/jobs/render_job.json `
  --progress-frame-start 14 `
  --progress-frame-end 43 `
  --progress-value-start 0.40677966 `
  --progress-value-end 0.67796610
```

The job stores a per-frame `render.progress_schedule`; the native renderer
uses it instead of a linear `0 -> 1` progression. Rebuild the native renderer
after this feature changes, then evaluate again in a new backup directory.

### Automatic Progress Calibration

`candidate-evaluate` can now calibrate progress without editing the sample job.
Use `--calibrate-progress` for an evaluation that first renders a temporary
linear probe, measures the candidate's visible interval against stable source
A and B, derives an evaluation-local schedule, and then runs the aligned render
and score:

```powershell
conda run -n harness python agent/src/main.py candidate-evaluate `
  --manifest "$candidateRoot/candidate_manifest.json" `
  --job agent/work/samples/<sample-id>/jobs/render_job.json `
  --reference agent/work/samples/<sample-id>/reference `
  --output-root "$candidateRoot/evaluations" `
  --backup-dir "$candidateRoot/backups/evaluation_<unique-number>" `
  --msbuild "C:\Program Files\Microsoft Visual Studio\2022\Professional\MSBuild\Current\Bin\amd64\MSBuild.exe" `
  --renderer harness/native_renderer/build/x64/Debug/OverlayTrHarnessRenderer.exe `
  --width 1920 --height 1080 `
  --frame-start <reference-transition-start> `
  --frame-end <reference-transition-end> `
  --calibrate-progress
```

The normal evaluation report contains `progress_calibration`, and its reports
folder contains `progress_calibration.json`. The original sample
`jobs/render_job.json` remains unchanged. This option is currently opt-in while
the detector is validated across more samples.
