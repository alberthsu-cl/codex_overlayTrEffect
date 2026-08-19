# Goal run rules

Shared rules for the /goal prompts. The prompts stay short because `/goal`
caps its argument at 4000 characters; the reasoning lives here.

## Choosing the starting point

Produce a new shader deliverable under a unique `ModelGenerated\Family_XX`
FX ID, choosing between the two available actions explicitly and saying which
in `decision.reason`:

- **`implement_new_effect`** starts from the neutral scaffold,
  `ModelGenerated/TrGeneratedDissolve`: a plain linear crossfade whose wrapper
  passes progress through unmodified, carrying no inherited geometry or timing.
  This is the default when no registered effect is genuinely close.
- **`tune_existing_effect`** clones a registered effect, builtin or generated,
  as a starting point only; never modify the effect it clones. Use it when a
  registered effect already implements most of the required motion, and state
  what it actually supplies.

A cloned seed carries its seed's timing as well as its pixel logic. Before
relying on one, check that its `UpdatePSParam` passes progress through
linearly. Some effects remap it to a narrow band around the transition
midpoint: a clone of one such effect rendered byte-identical to source A for
progress 0.00-0.41, and diagnosing that cost a full refinement iteration.
Every generated effect to date is a clone of a clone, five generations deep,
which is how that remap propagated.

## Pinned progress schedule

Refine against a fixed progress schedule, not a recalibrated one.

`--calibrate-progress` re-probes the candidate on every run for the frames
distinct from both endpoints, then maps the reference window onto that band.
Any change to an effect's temporal profile - a longer tail, a slower
relaxation, an earlier onset - moves what the probe finds and silently
re-times the content. The score then reports content misalignment rather than
the change that was made, and the iteration is spent. One measured instance:
extending a stretch tail moved the probe's band end from 0.85 to 0.98, and the
iteration read as a regression for that reason alone.

Use `--calibrate-progress` to establish the baseline. Once it is accepted, pin
it:

1. Read `reports/progress_calibration.json` from the baseline run and copy the
   `aligned_evaluation_job.json` it names into the sample's `jobs/` folder.
2. Shorten that copy's `job_name` field. The name is embedded in every
   evaluation output path, and Windows caps paths at 260 characters - a
   generated `job_name` has already exceeded it, failing with "request file
   could not be opened".
3. Run every refinement evaluation with `--job` pointing at the pinned file and
   **without** `--calibrate-progress`.
4. Verify the pin once by re-evaluating the unchanged baseline under it. The
   metrics must reproduce the baseline's. If they do not, the pin is wrong -
   fix it before spending an iteration.

A measurement fitted under one schedule is invalid under another; re-derive any
curve against the pinned one.

## Interpreting outcomes

`accepted`, `rejected` and `tradeoff` are all completed evaluations, not
failures. Progress calibration and its linear probe are normal. Do not stop,
restore sources manually, or start a new phase after those outcomes; run the
embedded `candidate-continue` command and follow the next request. Treat an
evaluation as failed only when `candidate-evaluate` itself fails without
producing a controller outcome.

A similarity score is a regression signal, not proof of visual equivalence.
When a change is measurably closer to the reference but scores worse, say so
with the measurement rather than silently reverting to the score's preference.

## What to fix first

**Motion first.** When reliable motion coverage is high but direction agreement
is weak, investigate regions or displacement before blur or blend. Keep the
approach general; do not assume a fixed number of regions or a horizontal-only
transition.

**Geometry before timing.** For transform-like effects, read `motion_geometry`
in every iteration report. Treat translation, pivot, rotation-sign, scale and
reflection mismatches as source-code geometry problems, and prioritise
`displacement` or `shader_structure`. Timing, blur and blend are appropriate
only once transform position and direction agree, or when geometry confidence
is explicitly low.

## Paths

Resolve every placeholder to an actual path before running anything; never pass
a literal documentation placeholder such as `<candidate-root>`. Sample IDs must
match `[a-z0-9][a-z0-9_-]*` - an uppercase or space-separated ID is rejected by
`sample-init`.
