# Refinement Contract

Edit only the candidate workspace named by the request. Preserve the existing
FX ID, class names, shader symbols, endpoint behavior, and C++/HLSL build
compatibility. Do not edit production effects, registration tables, render
jobs, progress schedules, reports, or controller-owned files.

Choose exactly one hypothesis category from the request. Make the smallest
source change that addresses the reported mismatch. Record one iteration JSON
file containing `hypothesis_category`, `visual_hypothesis`, `changed_files`,
and expected outcome. Do not claim improvement without build, render, and
score evidence. Treat low-confidence diagnostics as advisory.

If the build fails, repair the same iteration from the generated build-repair
request and rerun evaluation with a new backup directory. Do not start a new
FX or phase to bypass a build failure.
