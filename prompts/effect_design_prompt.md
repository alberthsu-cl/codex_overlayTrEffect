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

Use the supplied effect-design schema. The final response must satisfy that schema exactly.
```
