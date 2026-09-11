# Staged diversification — revision 21

## Status and purpose

**Current follow-up:** all 19 nominal variants completed with successful task and
physical checks after the three serialization-error replacements. The baseline
is not broad robustness qualification. See [the geometry and tracking follow-up](DIVERSIFICATION_PROGRESS.md)
for the current bounded run and stronger-margin checks.

The staged implementation is available; diversified physical qualification is
**in progress**, not established by manifest generation or unit tests. The initial
19-variant baseline ran in `output/diversification_v21/baseline` with 12 CPU worker
processes and captured RGB/tactile video. No later physical stage auto-starts.
The initial pushing recordings exposed an existing post-rollout metadata bug:
the writer requested a surface-only `working_end` on hook/pushing worlds. The
writer now limits that attribute to surface tasks. Three hook/push reruns were
completed in `baseline_recording_fix`; original serialization errors and partial
artifacts remain unqualified and preserved. This consumed three repair slots.

This work targets reusable contact-rich interactions, uncertainty and eventual
real matching—not larger tactile amplitudes for their own sake. Keep curved
gels, force-only 7×9×3 model observations, 2 mm marker spacing, FP32, impedance
control and diagnostic-only wrench comparison. Physics runs on CPU; rendering
uses the GPU. No training or large-scale data collection is part of this stage.

The active revision-21 roster has eight surface/following variants, one
jaw-aligned closed-loop hook, five insertions, three turning modes and two
tool-mediated pushing variants. Two insert–turn–hold composites follow their
prerequisite checks. Revision-18/19/20 recipes and results remain historical.
The `transfer` recipe is the old mechanics-repair selection; use the new driver
below for diversification, not the old `diversity` or `randomized` recipes.

Spatula serving is deferred and excluded. A future revisit should use a convex
pancake underside and an opposing support wall, not further tune the failed
square-table-edge scooping trajectory. Closed-hook mirroring, convex scraping,
flat peeling and material removal are not active tasks.

## Stages

| Stage | Candidate episodes | Changes |
|---|---:|---|
| baseline | 19 | One full nominal per isolated variant |
| prerequisites | 6 | Two additional repeats each: key insertion, friction turning, spring turning |
| composites | 2 | Insert–turn–hold, friction and spring; existing three-pass prerequisite gate retained |
| geometry | 48 | Eight representatives × handle ±10%, working geometry ±10%, roll +5°, pitch −5° |
| physics | 64 | Same eight × external friction 0.75/1.25, load 0.75/1.25, grip 25/30 N, gel 150/250 kPa |
| alignment | 12 | Four representatives × marginal/near-miss/incorrect contact candidates |
| combined | up to 76 | Four combinations per qualified isolated variant; conditional geometry/physics holdouts |

The nominal 227-candidate manifest leaves 29 attempts in the existing 256-attempt
campaign budget. Solver sensitivity and repairs consume that same budget.
These are candidate counts, not a promise every variant qualifies.

Two explicit corrections to the conversational plan:

- The existing safety cap is 35 N **per simulated finger**. Keep it; sweep 25/30 N
  against the 35 N baseline instead of silently allowing 40 N.
- Representative checks do not qualify untested variants. Combined execution
  requires **all six geometry and all eight physics conditions** reviewed and
  passing for that exact variant. Thus the first combined run can cover at most
  the eight representatives. `--all-variants` expands factor checks if later
  requested/budgeted; do not claim 19-variant robustness from eight representatives.

Representatives: flat scraping, cylinder peeling, curved slot, friction hook,
round insertion, key insertion, friction turning and pose pushing.

Working size means scraper/drawing/pusher width, peeler cylinder radius, guide
tip diameter, hook reach or mating-profile scale. Handle deformation on imported
SCFields tools is smoothly blended above the working end; it changes the actual
collision/visual mesh. Peg/socket profiles and guide spacing stay compatible.
Native bodies recompute inertia from the changed mesh at the configured mass;
mass is a separate physics factor, not implicitly constant density.

External load means normal-force command for surfaces, object mass for hook/push,
resistance for turning and speed for insertion. Hook/push supports currently use
shared material factors, so that friction sweep changes object/support AND
tool/object friction; it is explicitly not an isolated contact-pair experiment.
The support actor factor changes by 0.75²/1.25², giving 0.75/1.25 of effective
tool/object friction under the engine's geometric-mean rule; object/support
friction changes by the squared factor because both support actors change.
Keep the recorded effective-pair provenance. These are synthetic sensitivity
ranges, not calibrated real GelSight or printed-fixture material properties.

Alignment offsets are candidate probes, not guaranteed outcomes: insertion
0.3/0.9/2 mm (nominal clearance 0.6 mm), hook 8/18/24 mm, straight slot
0.5/1.5/3 mm, pushing 5/15/25 mm. Measure actual outcomes and revise the envelope
if these are not genuinely marginal; never infer labels from their names.

Combined conditions add ±5° roll/pitch and ±2 mm grasp depth. Global translation,
new oval handles, new push-object shapes, larger hook angle/opening changes and
camera extrinsic variation remain subsequent extensions—not implemented or
qualified merely because they were listed as diversification targets.

## Commands and review gates

Run commands from the repository root, with `OPENBLAS_NUM_THREADS=1` and
`OMP_NUM_THREADS=1`. Every execution uses captured RGB/tactile video.

```bash
.venv/bin/python experiments/tool_use_pilot/diversification.py geometry --output experiments/tool_use_pilot/output/diversification_v21
```

Without `--run`, this only exports a reproducible JSON manifest. An identical
export can be reused; changed manifests or existing run directories are never
overwritten. `--families` permits a bounded subset; `--phase` names a new run.

After completed RGB videos have actually been viewed, audit using the complete
set of visually reviewed episode IDs (omit IDs not yet inspected):

```bash
.venv/bin/python experiments/tool_use_pilot/diversification_review.py experiments/tool_use_pilot/output/diversification_v21 --visual-reviewed 0
```

That writes `review.json` and `DIVERSIFICATION_REPORT.md`, independently checking
task/physics completion, tool/environment geometry, actual sensor housing
clearance and episode hashes. Open housing meshes are tested one-way against
closed tool/fixture meshes at 11 frames; this is not a swept-volume certificate.
The review command does not infer visual approval from a success flag.

```bash
.venv/bin/python experiments/tool_use_pilot/diversification.py geometry --output experiments/tool_use_pilot/output/diversification_v21 --phase geometry_verified --review experiments/tool_use_pilot/output/diversification_v21/review.json --run --workers 12
```

Unreviewed/failed variants are excluded and the blocked count is printed. Later
stages never auto-promote. Composites retain the independent three-pass gate.
Preserve valid task failures as diagnostic/world-model candidates; these are
not successful demonstrations. Numerical/safety failures remain rejected.

Review start/contact/peak/end and the full representative videos, grasp movement,
penetration, gel Jacobian and task outcomes. Use a matched 16-iteration,
0.0005-tolerance spec with the same seeds for a high-load boundary case before
expanding its envelope. The existing `benchmark.py --specs` runner supports this;
do not reinterpret `STOPPED` as numerical convergence. Stop after one hour of
unresolved repair per issue and report the remaining limitation.

## Reproducible observations and identities

Each config records stage/condition, parent-config ID, geometry ID, physics ID,
independent camera/tactile seeds, intended outcome and a geometry–physics group.
Raw HDF5 keeps geometry and per-body poses, not per-frame PCDs. Reconstruct
tool/environment clouds from those saved meshes and poses, with ray visibility.
All observation augmentations of an episode remain in its group.
Review metadata additionally hashes the actual recorded tool/environment meshes,
independent of task labels; use this to catch shared geometry across variants
before constructing a geometry-disjoint training split.

`geometry_holdout` and `physics_holdout` are **conditional** protocols: new 0.95
working scale or new 1.1 load factor, respectively. Do not randomly split frames,
and do not claim a universal geometry-disjoint split from the pair-group hash.
Only compare each holdout against the applicable baseline/factor conditions;
inspect effective identities before any future training split.

```bash
.venv/bin/python experiments/tool_use_pilot/diversification_observations.py path/to/completed_episode.h5 --output path/to/new_observation_review
```

Produces a plot and reproducibility metadata, not modified raw sensor files:
visible clean, 1/2 mm XYZ noise, 15/30% patch removal and matched combinations.
A single camera-plane mask is calibrated at the initial frame and then fixed;
the requested fraction applies initially to the combined visible cloud, not
independently to every object or subsequent frame. Tool/environment counts are
reported, including fully occluded streams. The preview is an X–Z projection of
ray-visible 3D points, not a captured RGB image or calibrated depth-camera model.
Point noise includes a persistent per-stream registration bias. No GT labels
or contact poses choose the mask. Existing tactile corruption helpers remain
available separately; no tactile-benefit result is claimed here.

The offline pipeline was exercised on the previously completed revision-20 hook
recording in `output/diversification_v21/observation_pipeline_check`; that check
is not a newly simulated revision-21 episode.
