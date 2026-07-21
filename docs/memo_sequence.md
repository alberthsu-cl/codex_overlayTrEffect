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

Choose the normalized frame count for this experiment. Use the same count for
the repeated A/B source sequences and the render job.

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
  --frame-count 60
```

## 5. Ask Codex for the Effect Design

Use:

- `agent/prompts/effect_design_prompt.md`
- `agent/prompts/effect_design_schema.json`
- `analysis/transition_structure.json`

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
  --output agent/work/samples/<sample-id>/jobs/render_job.json `
  --frame-count 60
```

## 7. Generate, Register, and Refine When Needed

For `implement_new_effect`, generate sources under `effects/`, register the
effect, then initialize its candidate under `candidates/`. The FX ID is global
to `OverlayTrEngine`, so it must be unique across all sample workspaces, for
example `ModelGenerated\\SeamlessSliding_03`.

For `reuse_existing_effect` or `tune_existing_effect`, use the registered FX
chosen by the design and create a candidate only when shader refinement is
needed.

Evaluate candidates against only this sample's `reference/` frames. Keep every
candidate's packets, backups, and evaluations inside its own directory under
`agent/work/samples/<sample-id>/candidates/`.

## 8. Review and Decide

Review these generated artifacts after each candidate evaluation:

- `rendered_transition.mp4`
- `comparison_transition_window.mp4`
- `motion_diagnostics.mp4`
- `reports/candidate_iteration_report.json`

Use automated scores as diagnostics. When the visual result is acceptable,
restore the selected baseline and record `candidate-human-accept`; promotion to
the registered target remains a separate explicit action.
