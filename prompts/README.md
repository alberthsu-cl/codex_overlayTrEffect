# Agent Prompts

This folder now tracks only the prompts that fit the new `agent/` direction.

The old end-to-end goal prompts were removed because they were tied to the previous harness shape:

- deterministic analyzer
- rule-based planner
- direct codegen-first workflow
- single large `/goal` prompt that mixed analysis, implementation, rendering, and repair

The new direction is narrower and easier to control:

1. Codex analyzes the sample transition video
2. local Python prepares frames, invokes rendering, and scores results
3. implementation or effect generation happens only after the analysis artifact is stable

## Files

- [codex_transition_analysis_prompt.md](./codex_transition_analysis_prompt.md): pass-1 prompt for video analysis only
- [codex_transition_analysis_schema.json](./codex_transition_analysis_schema.json): strict schema for the analysis artifact

## How To Use

Use the analysis prompt and schema when you want Codex to inspect a sample transition video and return a machine-readable `transition_analysis` artifact.

That artifact is intended to drive later steps such as:

- selecting an existing built-in effect for regression
- deciding whether a new effect is needed
- preparing a later implementation prompt

The analysis artifact distinguishes `family_status: "known"` from
`family_status: "unknown"`. Unknown means that no existing family is adequate;
it does not authorize unrestricted shader generation. `visual_primitives`,
`new_effect_needed`, and `implementation_status` record whether the observed
behavior can proceed through the current constrained grammar or requires review.

## Refinement Prompt Modules

Candidate iteration requests use a small hierarchical prompt set. The
controller selects the family module from `transition_structure.json`, so a
rotation sample does not load sliding or sampler-specific guidance unless it
is relevant.

- [codex_effect_refinement_prompt.md](./codex_effect_refinement_prompt.md): stable refinement prompt index
- [base/refinement_contract.md](./base/refinement_contract.md): rules that apply to every candidate
- [diagnostics/motion_geometry.md](./diagnostics/motion_geometry.md): signed 2D geometry interpretation
- [diagnostics/optical_flow.md](./diagnostics/optical_flow.md): confidence and flow limitations
- [diagnostics/edge_content_policy.md](./diagnostics/edge_content_policy.md): optional sampler/edge guidance
- [families/affine_transform.md](./families/affine_transform.md): rotation, scale, and translation candidates
- [families/segmented_motion.md](./families/segmented_motion.md): split-line and multi-region candidates
- [families/general_motion.md](./families/general_motion.md): fallback for other motion types
