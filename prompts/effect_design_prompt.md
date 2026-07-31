# Effect Design Prompt

Use this prompt after transition analysis is complete.

This is the bridge between pass 1 and pass 2:

1. read the transition analysis artifact
2. decide whether we should reuse, tune, or newly implement an effect
3. emit a compact machine-readable design decision

This prompt is intentionally not a code-generation prompt. It is a design-decision prompt.

```text
/goal

You are working in the local workspace under D:\AI_Harness.

Task:
Read the provided transition analysis artifact and decide the best downstream effect strategy for local rendering and regression.

Primary inputs:
- a transition analysis JSON artifact
- local repo context from `overlaytrengine`
- optional knowledge of existing built-in effect IDs

Output rules:
- write exactly one JSON object that conforms to the supplied effect-design schema
- do not write prose, markdown, or code fences in the final output
- do not modify repository files in this stage
- do not generate C++ or HLSL in this stage

Decision goals:
- decide whether to reuse an existing effect, tune one, or implement a new effect
- identify the nearest existing effect family when possible
- state what visible cues must be preserved in later rendering
- state what approximations are acceptable
- state the main implementation risks

Important constraints:
- prefer the smallest viable local runtime path
- do not claim a built-in effect is a strong fit unless the analysis artifact supports it
- if the transition likely exceeds the current single-pass effect model, say so clearly
- keep the result compact and machine-readable

New-effect delivery policy:
- When the request says that this sample must produce a new shader deliverable,
  do not choose `reuse_existing_effect` as the final strategy. Existing effects
  may still be named as the closest visual reference or code base.
- Use `tune_existing_effect` when the best implementation path is to clone and
  adapt an existing pure-HLSL effect. In this project, that action means create
  a new source variant; it does not mean changing the existing effect.
- For `tune_existing_effect` and `implement_new_effect`, set
  `target_effect.effect_id` to a new unique ID in the form
  `ModelGenerated\\<Family>_XX`. Inspect the registered ModelGenerated IDs
  before selecting the next index.
- For `tune_existing_effect`, include `source_variant` with the base source
  stem, source file names relative to `overlaytrengine/OverlayTrPlugInFx`, and
  only exact effect-specific initial replacements. Do not add replacements for
  the base class/stem, shader symbol, or base FX ID; generation handles those
  names automatically.
- `reuse_existing_effect` is permitted only when the request explicitly says
  that no new shader deliverable is required, such as a benchmark-only run.
- For every new deliverable, include `implementation_seed`:
  `family` names the structural implementation family used in the new FX ID,
  `template_effect_id` identifies the cloned seed when applicable, and
  `required_shader_capabilities` lists the capabilities the shader must expose.
  Do not use a visual primitive such as `dissolve` as the implementation family
  when the seed and required behavior are a spatial displacement or multi-region
  effect. For a source variant, `template_effect_id` must equal
  `target_effect.closest_existing_effect_id`.

Evaluation policy:
- Carry forward `evaluation_policy.selection` from the transition analysis
  unless the implementation changes which evidence is meaningful. In addition
  to pixel metrics, transform effects may use
  `foreground_body_rotation_error`, `foreground_body_scale_error`,
  `foreground_body_translation_error`, `foreground_body_pivot_error`,
  `foreground_body_rotation_direction_agreement`, and `geometry_similarity`.
- For rotation, scale, perspective-card, reflection, or flip effects, make the
  relevant signed transform errors primary. Keep MSE/MAE/peak metrics advisory
  unless pixel fidelity is the main deliverable; endpoints remain hard checks.
- Endpoint correctness is always enforced locally and must not be weakened by
  this policy.
- Include `evaluation_policy.motion_topology` when the design needs to override
  the analysis policy for regression.
- Set `mode` to `disabled` for non-segmented effects, `advisory` when motion
  topology is useful diagnostic evidence but should not block acceptance, and
  `hard` only when matching a clear multi-region directional structure is an
  explicit deliverable requirement.
- Do not apply a split or region topology requirement merely because dense flow
  found transient fragments in the reference.

Use the supplied effect-design schema. The final response must satisfy that schema exactly.
```
