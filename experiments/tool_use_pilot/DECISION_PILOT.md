# Contact-decision pilot (v2)

This local study designs tasks and tests sensing observability; it does **not**
train the main ACWM or establish transfer to another simulator or hardware.

## Tasks and stopping rules

`decisions.py` adds configurable grip effort, Young's modulus, both friction
interfaces and contact-driven hook/key fixtures. The existing `pilot.py` tasks
remain available. A `DecisionSpec` retains the underlying legacy task label for
compatibility; **`kind` / the `task_kind` HDF5 attribute identifies the new task**.
Hook progress is metres; key rotor progress is radians, not tool yaw.

The key has a tapered entry, keyed blade and spring-loaded revolute socket. The
100-degree tool command compensates slot play; success is measured from the
**socket rotor reaching 90 +/- 5 degrees**, with the blade seated. It is not
declared successful from the arm command. The gate/rotor are never actuated by
the tool controller. The spring/damper torque only resists their motion.

The original v2 campaign allowed two task revisions; the subsequent user-requested
hook repair is a separate, small verification campaign. Physically valid task failure is useful
data; numerical penetration or loss of the prepared grasp is not. The strict
insertion clearance limit remains 0.3 mm. A task which does not qualify is
reported as such and is not silently made easier by widening this limit.

Hook revision 2 uses a cantilevered crossbar with a free end, optionally mirrored.
The old two-support fixture obstructed lateral recovery. Loaded pulls are held
briefly and unloaded before lowering; long loaded holds caused gel/grasp creep.

### Hook repair (revision 6)

The J tool is now one watertight extrusion, including the missing 18 x 8 mm
bottom corner. The cantilever is also a single exact box-union surface: shared
internal faces and open/nonmanifold seams are removed without changing its
external dimensions or volume. Triangle edges on the tool and cantilever are refined to at most
8 mm for contact integration. The fixture contact transition is reduced from
the default 10 mm span to 0.4 mm (0.2 mm smoothing half-width, 0.1 mm threshold,
5e10 Pa/m penalty). Gel material and gel/tool friction remain unchanged.

A bounded integral tool-pose correction (12 mm maximum per axis, 6 mm/s maximum
correction rate) compensates free-grasp drift and arm tracking error in the
scripted collector. **This is privileged simulator feedback, not a learned
policy or proof of tactile benefit.** It tracks the tool trajectory, not the
fixture, so object offsets are not silently removed. Actual EE actions,
nominal tool targets and oracle corrections are recorded separately. The
controller state, including recovery clock and gates, is checkpointed.

Before lateral motion and re-engagement, recovery waits until the preceding
tool waypoint is reached within 1.5 mm in the relevant clearance dimensions
(x/z for lowering, xyz for alignment and raising); it aborts after a 4 s timeout
instead of moving through an uncleared fixture. Episodes allow 35 s for these
waits. The pull command is reduced to 50 mm because corrected tracking no longer
requires the old 64 mm overtravel to reach the 40 mm slider goal.
Neither the fixture joint nor the tool receives a hidden grasp weld or assist.

Run the bounded four-anchor / sixteen-branch audit (no main-model training):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python experiments/tool_use_pilot/hook_repair_campaign.py --workers 16 --output experiments/tool_use_pilot/output/my_hook_repair
.venv/bin/python experiments/tool_use_pilot/hook_repair_report.py experiments/tool_use_pilot/output/my_hook_repair --mesh-audit
```

Each independent branch repeats the prefix; all saved prefix observations and
controller-state hashes must be identical before results are merged. Raw physics
snapshot bytes are retained but are not used for cross-process equality: in a
diagnostic pair, 36 of 32,664 serialized bytes differed despite every prefix
observation and the full controller-state hash being identical. Independent
rendered replays are compared on trajectories, actions and tactile arrays.
Worker outputs remain in `_branches/` even if merging fails, so a failed
consistency check cannot discard completed simulations.
The original v2 results remain historical artifacts: the corrected tool mesh
changes the implementation, so do not expect old v2 episodes to replay exactly.

The final local audit is in `output/hook_repair_final/REPORT.md`: all 16 branches
passed physical checks (six intended successes, ten valid failures), both
mirrored near-miss recoveries succeeded, and their recorded RGB/tactile replays
match the non-rendered arrays exactly. There remains up to 9.12 mm of measured
grasp-relative drift; this is not a rigid grasp. The worst independent sampled
rigid overlap was 0.336 mm. These results qualify this small mechanics test,
not sim-to-real transfer, broader robustness, or a learned tactile advantage.

## Reproduction

### Separate key stages (revision 7)

`key_stages.py` implements independent `key_insertion` and `key_turning`
tasks; the legacy combined `KeyDecision` and its historical results are unchanged.

- Insertion starts above a fixed keyed pocket, uses translation only, and
  measures blade seating/depth (at least 14 mm with the complete blade contained
  for 0.3 s). It does not require or command rotor rotation.
- Turning starts with a **prepared seated grasp**, verifies the seat before
  acting, and uses rotation only at a fixed nominal tool position. Success is
  measured rotor rotation of 90 ±5 degrees while seated for 0.3 s, not tool yaw.
  The rotor is passive with torsional spring/damping resistance.
- Prepared turning is an isolated test, not a demonstrated insertion-to-turn
  handoff. Both stages release the initialization grasp constraint at 1 s.
- A 3 N insertion contact guard stops advancing and withdraws from the guarded
  position. The guard uses GT contact force and the collector uses bounded GT
  tool-position feedback: these are privileged safety/collection channels, not
  learned tactile policies. The 0.3 mm penetration validity limit is unchanged.
- Orientation feedback is bounded to 12 degrees at 2 degrees/s. Turning uses
  tool-root-pivot feedforward (rather than rotating corrections about the wrist
  161 mm above the tool); nominal insertion translation remains zero.
- HDF5 `task_kind` / `task_stage` distinguish the tasks; `task_progress_units`
  is metres for insertion and radians for turning. Saved geometry/pose camera
  reconstruction and dense tactile fields retain the existing observation interface.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python experiments/tool_use_pilot/key_stages.py --specs experiments/tool_use_pilot/key_stage_specs.json --output experiments/tool_use_pilot/output/my_key_stages --workers 4 --render
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/tool_use_pilot/key_stages_report.py experiments/tool_use_pilot/output/my_key_stages --mesh-audit
```

Reviewed local results: [separate-stage report](output/key_stages_split/REPORT.md).
Both nominal stages succeed; the 3 mm lateral-miss insertion and hold-only
turning controls fail without invalid physics. Four of four episodes pass the
mechanics checks. Turning uses a 16 mm maximum triangle edge (insertion 8 mm),
with unchanged solid geometry; this is not a mesh-convergence study. Corrected
review-copy seating labels use the actual chamfered blade vertices, with the
same 0.1 mm lateral tolerance; the original oversized-box labels are retained
under `metric_history` and raw physics/actions/tactile arrays are unchanged.
Nominal whole-episode tactile force errors narrowly miss the existing 0.2 N
absolute gate, so mechanics success does not imply all tactile-fidelity gates pass.

### Key tactile observability

[Key sensing report](output/key_tactile_observability/REPORT.md) compares frozen
Point-M2AE vision, dense tactile, integrated wrench, calibrated/axis-aware
wrench controls, precontact tactile and mismatched test tactile. It is not a
policy evaluation. Insertion predicts future seating; turning predicts measured
future resisting moment about the rotor axis under a common turn command.

The original 24-case manifest is `key_observability_specs.json`. Five signed
insertion conditions failed grasp-retention checks, so a bounded six-case
follow-up (`key_observability_insertion_followup_specs.json`) plus the six
original positive-offset cases form the explicitly one-sided insertion cohort.
No geometry, friction, stiffness-of-gel or validity threshold was changed.
Turning is audited uniformly through 18 s, before return motion: its 16–17 s
target is complete, but one original full rollout aborts during return at 18.9 s.
The report preserves these limits rather than presenting all full rollouts as safe.

Each fitted stage has 12 physical conditions in three held-out nuisance groups
(socket yaw for insertion, grip effort for turning), with four corruption draws
kept together per condition. Turning holds out grip effort, not spring values.
The two observation windows per stage and seven PCD corruption conditions are
specified in `key_observability.py`. Run the probes using `.venv_probe/bin/python`
with `--physics`, `--output`, `--stage insertion|turning` and `--window before|after`;
then run `key_observability_report.py` on their common output directory.

Main local result: tactile helps turning-resistance prediction even with clean
vision; insertion is mostly visually solvable after probing. A known-axis net
wrench predictor beats dense-field regression for future turning torque, so this
does not establish dense-field necessity. No positive fusion gain survives the
global 28-test correction; small-sample results remain exploratory.

From the repository root, use `.venv/bin/python` for physics:

```bash
.venv/bin/python experiments/tool_use_pilot/decision_campaign.py --stage grip --kind hook --workers 12 --output experiments/tool_use_pilot/output/my_hook_grip
.venv/bin/python experiments/tool_use_pilot/decision_campaign.py --stage decision_screen --kind hook --branches --workers 12 --output experiments/tool_use_pilot/output/my_hook_decisions
.venv/bin/python -m unittest discover -s experiments/tool_use_pilot -p 'test_*.py'
```

`evaluation` generates the larger 24-anchor manifest, but should only be run after
physical and action-diversity qualification. Existing manifests cannot be
overwritten by the campaign runner. Keep all branch files beside their prefixes.

The GPU environment is deliberately separate:

```bash
.venv/bin/python -m venv .venv_probe
.venv_probe/bin/python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv_probe/bin/python -m pip install -r experiments/tool_use_pilot/requirements-decision-probes.txt
git clone --depth 1 https://github.com/ZrrSkywalker/Point-M2AE.git experiments/tool_use_pilot/vendor/Point-M2AE
git -C experiments/tool_use_pilot/vendor/Point-M2AE fetch --depth 1 origin 1bdbe05bc13e83dc476660887258abf1e457fed0
git -C experiments/tool_use_pilot/vendor/Point-M2AE checkout --detach 1bdbe05bc13e83dc476660887258abf1e457fed0
mkdir -p experiments/tool_use_pilot/checkpoints
.venv/bin/gdown 1HyUEv04V2K6vMaR0P7WksuoiMtoXx1fM -O experiments/tool_use_pilot/checkpoints/pre-train.pth
.venv_probe/bin/python experiments/tool_use_pilot/point_m2ae_probe.py
.venv_probe/bin/python experiments/tool_use_pilot/decision_probes.py experiments/tool_use_pilot/output/my_hook_decisions --output experiments/tool_use_pilot/output/my_hook_probes
```

The public checkpoint SHA-256 is
`6dd8f7a5993f6965c76772c6049824d33fd3043473f71079b210ebec78a644a3`.
The encoder adapter loads the official encoder definitions and **all** encoder
weights strictly. It does not import the reconstruction losses or old CUDA
extensions. FPS/KNN use deterministic PyTorch implementations. No encoder
parameters or batch-normalization statistics are trained. Dependencies and
checkpoints are ignored by Git; retain upstream licensing in the vendor clone.

## Observation and HDF5 contract

`vt_acwm_superdex_decision_v2` stores local triangle meshes, root-to-COM offsets
and per-frame mesh-root poses. `body_root_poses` is ordered by numerically sorted
`geometry/<index>` groups. It includes the robot/housings as visual occluders.
There are **no stored per-frame PCD sequences**. The compatibility helper still
reads v1 stored clouds; old datasets are not rewritten.
Deformed pad-surface coordinates already present in each observation also
occlude camera rays. Pass those surfaces when reconstructing camera observations.

Ray casting returns only visible tool/environment surfaces with ideal simulator
segmentation. Invisible robot surfaces block rays but do not enter those clouds.
Each stream has 1,024 points; sparse streams repeat only observed points and
report their unique count. Empty streams have a missing-observation flag. An
ideal full-surface condition is a privileged control, not a sensor measurement.

The five observation conditions are full ideal surfaces, clean visible front,
clean visible side, 0.5 mm noise + 10% spatial removal, and 1 mm noise + 25%
spatial removal. Augmentation is seeded per history and shared between sensing
ablations. It is a robustness diagnostic, not calibrated camera noise.

Object/environment normalized Point-M2AE embeddings are accompanied by observed
metric centroid, extent, covariance and scale in the EE frame. This preserves
metric relationships discarded by independent unit-sphere normalization.
No GT tool/fixture pose or mechanical label enters a non-oracle visual input.

The aggregate-wrench control is computed from the same simulated gel fields,
shifted to the EE origin and expressed in the EE frame. It is **not** an
independently simulated wrist F/T sensor. Wrench-audit ground truth and dynamic
compensation remain oracle diagnostics, not learned observation inputs.

Prefixes contain the full causal observation history. Branches contain simulator
and controller snapshots and a relative HDF5 link to that same prefix. The
one-second observation window ends before candidate execution. Physical validity,
task success, tactile validity and imitation eligibility remain separate.

## Interpretation

The probe reports incomplete/invalid branch groups and refuses policy-benefit
claims if the task gate fails. Fewer than eight independent clean anchors means
no fitted comparison is reported. Constant-action success, group coverage,
prediction errors and regret must be examined together. A negative tactile or
dense-field result is an acceptable study outcome.

Each report ends with a project-relevance decision: retain, revise within the
bounded allowance, or defer. Large collection and architecture expansion are not
automatic next steps.

`case` identifies an intended setup, not a ground-truth contact-mode label:
prepared-grasp settling can change the actual tool/fixture relationship. Use
measured contacts and outcomes, not that string, to evaluate engagement.

Earlier bounded study: [report](output/decision_v2/REPORT.md). Both revised task
families failed complete decision-benchmark qualification. The qualified nominal
hook and individual valid recoveries remain useful controls; fitted tactile
comparisons were deliberately not reported from incomplete valid branch groups.

## Repaired-hook noisy/occluded PCD study

Latest sensing study: [report](output/hook_tactile_observability/REPORT.md).
Twelve physical starting states, including eight new offset variants, are tested
with seven observation corruptions and four independent corruption draws per
state. All draws and mirrored outcome pairs at one offset level stay in the
same held-out fold. This is a straight-pull outcome classifier, not a four-way
recovery policy; do not conflate its accuracy with task completion rate.

Use `hook_observability.py` for this revision-6 experiment. Unlike the earlier
candidate probe, it excludes future executed commands, which can carry state
information from the collector's privileged pose-feedback controller. It also
excludes GT poses and extrinsic wrenches as learned features. Histories end at
5 s or 8 s; the latter follows a shared gentle pull before success is reached.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python experiments/tool_use_pilot/decisions.py --specs experiments/tool_use_pilot/hook_observability_specs.json --output experiments/tool_use_pilot/output/hook_tactile_observability/physics --workers 8 --selected pull
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv_probe/bin/python experiments/tool_use_pilot/hook_observability.py --folders experiments/tool_use_pilot/output/hook_repair_final experiments/tool_use_pilot/output/hook_tactile_observability/physics --output experiments/tool_use_pilot/output/hook_tactile_observability/before_probe --observation-time 5
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv_probe/bin/python experiments/tool_use_pilot/hook_observability.py --folders experiments/tool_use_pilot/output/hook_repair_final experiments/tool_use_pilot/output/hook_tactile_observability/physics --output experiments/tool_use_pilot/output/hook_tactile_observability/after_probe --observation-time 8
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python experiments/tool_use_pilot/hook_observability_report.py
```

Tactile improves local corrupted-PCD predictions, but net tactile wrench already
matches dense tactile after probing. Clean vision and the large-mask control
have ceiling performance. This is evidence for contact observability, not yet
dense-field necessity, recovery-direction identification, or sim-to-real transfer.
