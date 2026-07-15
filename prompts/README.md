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

This folder does not yet contain the pass-2 implementation prompt set. That should be drafted after the new analysis contract is stable and the reusable render-and-score path has been extracted from `harness/`.
