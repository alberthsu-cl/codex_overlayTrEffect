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
  --design agent/work/samples/<sample-id>/design/effect_design.json `
  --output-dir agent/work/samples/<sample-id>/effects/<effect-name> `
  --manifest agent/work/samples/<sample-id>/effects/<effect-name>/manifest.json `
  --template-root overlaytrengine/OverlayTrPlugInFx

conda run -n harness python agent/src/main.py register `
  --manifest agent/work/samples/<sample-id>/effects/<effect-name>/manifest.json `
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
  --manifest agent/work/samples/<sample-id>/effects/<effect-name>/manifest.json `
  --output-dir agent/work/samples/<sample-id>/candidates/<effect-name>

conda run -n harness python agent/src/main.py candidate-evaluate `
  --manifest agent/work/samples/<sample-id>/candidates/<effect-name>/candidate_manifest.json `
  --job agent/work/samples/<sample-id>/jobs/render_job.json `
  --reference agent/work/samples/<sample-id>/reference `
  --output-root agent/work/samples/<sample-id>/candidates/<effect-name>/evaluations `
  --backup-dir agent/work/samples/<sample-id>/candidates/<effect-name>/backups/evaluation_001 `
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
