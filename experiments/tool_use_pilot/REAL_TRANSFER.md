# Real-transfer task candidates — revision 19

Historical revision-19 implementation and results. The active transfer recipe
has moved to [revision-20 mechanics repair](MECHANICS_V20.md): one jaw-aligned
hook layout, a straight spatula scoop/lift task, and turning with an angular stop.
The results below do not qualify those replacements.

This package changes simulation geometry and adds verification candidates. It
does **not** claim printable-part fit, calibrated hardware friction, real-camera
visibility or demonstrated sim-to-real transfer. The previously completed
revision-18 pilot remains separate in `BENCHMARK_STATUS.md`.

The [verification review](output/real_transfer_v19/VERIFICATION_REVIEW.md)
lists selected current-geometry episodes, independent geometry checks,
perturbations, negative controls and outstanding checks. The complete
[development report](output/real_transfer_v19/REPORT.md) also includes rejected
and time-censored attempts; use the review selection for demonstrations.
For model-data filtering, use the post-audit `dataset_index.json` masks, not an
unfiltered scan of development HDF5s or the recorder's original validity flag.

## Geometry and mechanics

| Family | Revision-19 implementation | Intended physical counterpart |
|---|---|---|
| Shaped insertion | 10 mm neck, 22 mm working section; double-sized round/square/hex/D/key profiles; unchanged 0.6 mm per-side socket clearance | Thick interchangeable peg/socket fixtures |
| Turning / insert–turn | Same enlarged key; shaft and clearance bushing on a broad base; native joint friction at 0.015 Nm | Keyed shaft in bushings with an adjustable friction washer |
| Hook, friction normal/mirrored | 80 × 60 × 6 mm sled deck, supported arch, 2 mm side clearance, end stops, 0.2 kg effective ballast mass | Weighted printed sled sliding on a base between loose guides |
| Wall/slot following | 10 mm circular follower and neck, 1 mm nominal side clearance | Substantial follower in interchangeable guide fixtures |
| Levering | Blunt lever, rounded fulcrum, 100 × 70 × 8 mm flap, 0.1 kg mass, gravity and 0.001 Nm hinge friction | Gravity-loaded hinged flap, no spring or latch |
| Tool pushing | Blunt grasped pusher and free 100 mm-long T-block, 15 mm thick, 0.15 kg mass | Tool-mediated table-top object placement |

The hook arch is 56 mm above its support plane, with two posts rather than the
former long cantilever. A 44 mm clear aperture accepts the 24 mm-thick hook.
The two mirrored variants offset the complete sled/guide assembly by ±6 mm
relative to the hook; guide clearance remains unchanged. No rail joint,
anti-yaw force or stabilizing weld is used. The old spring-slider hook remains
available with `load_mode="spring"`.

Free sleds and T-blocks have gravity and six unconstrained DOFs. Native rotary
joints represent the flap hinge and key bearing; the bearing internals are not
resolved. New transfer recipes exclude base–rotor collision inside that ideal
bearing; tool contact with both bodies remains enabled. The key's friction mode has no spring/detent term. Spring and detent
turning controls remain available. Turning ends at the seated 90-degree hold,
not a return stroke.
The enlarged-key transfer recipe commands 94 degrees at the wrist: the previous
100-degree command overcompensated the smaller profile clearance. The measured
rotor target remains 90 ±5 degrees for the final 0.3-second dwell; it is not
widened to accommodate an over-turn.

The nominal grasp section remains 18 × 24 mm. Independent handle dimensions,
shaft diameter, tip length/scale and follower diameter are serialized in each
specification for future diversity. They are not a declaration that arbitrary
combinations fit a real gripper. Revision-18 and earlier manifests retain their
old dimensions; revision 19 is the default for new specifications.
`grip_force_n` is the closing effort applied to each simulated finger DOF
(35 N nominal); it is not a calibrated mapping to a hardware gripper API's
force argument. Pad loads are recorded separately.

Pushing is along the jaw-backed direction (90-degree task heading, with the
handle counter-rotated to preserve its grasp orientation). The lower working
end has a nominal 2 mm table gap. The lever uses a 160 mm working reach and a
35-degree command to maintain overlap with the flap. The lever platform starts
84 mm ahead of the grasp origin, beneath the fulcrum, so its front edge does not
obstruct the descending effort arm. Hook support boxes use
exact BOX colliders; other transfer contact meshes remain explicit. Tool mass
is held at 80 g for these comparisons, not automatically scaled by print density.
In these new free-body/flap tasks, the nominal support and moved-object collider
coefficients are each 0.4, giving effective object/support friction 0.4. The
tool coefficient is 1.4, so effective tool/environment friction is
sqrt(1.4 × 0.4) ≈ 0.748, while gel/tool friction is 1.4. The support-friction
perturbation changes both support and moved-object materials. The HDF5 audited
friction metadata distinguishes these values from the older `friction` setting.

## Data and interpretation

Existing HDF5 fields, geometry meshes and body poses are retained. No per-frame
point clouds are necessary: sample each saved mesh and transform it with that
body's pose, including the movable environment object. This remains a geometry
observation reference, not a validated real RGB-D model.

Both pads still supply 7 × 9 × 3 force fields at 2 mm pitch. Cell moments and
contact ground truth are oracles, not model input. New per-environment pair
wrenches are world-axis values: `tool_environment_pair_wrenches` is **on the
tool about tool COM**; `object_support_pair_wrenches` is **on the moved object
about its COM**, excluding the tool. Never add the latter to the fingertip
reconstruction target. Array order is recorded in `contact_pair_order`.

New progress/goal oracles include flap angle, object tilt, planar position/yaw
errors, support escape and support penetration. Prepared grasps are released at
1 s. Small settled pre-motion tool-translation calibration is privileged
collector information; it may include initial fulcrum support and is not a
learned tactile controller. Physical grasp limits and
existing impedance control remain in place.
The composite collector also retains the existing measured-seat transition
gate. Collector state/phase guards are privileged verification machinery,
not a demonstrated deployable policy or additional model observations.

Task goals: hook travel ≥40 mm; flap lift ≥20 degrees; pushing planar position
error ≤10 mm and yaw error ≤5 degrees. Goals must be held for 0.3 s at completion.
Pushing has translation and translation-plus-15-degree-yaw variants. Tipping,
support escape, dropped tools, sustained >1 mm penetration and inverted gel
elements invalidate episodes. Wrench agreement is diagnostic, never a gate.

## Bounded verification commands

Use a fresh phase name for every run; the existing root-wide 256-attempt limit
includes failures. These recipes are small verification matrices, not a request
to run large-scale data collection.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  experiments/tool_use_pilot/benchmark.py \
  --output experiments/tool_use_pilot/output/real_transfer_v19 \
  --phase my_nominal --recipe transfer --workers 8 --render
```

`--families hook levering pushing` selects a subset; `--variants curved_slot`
can narrow it further. `--max-wall-s 5400` increases only the per-episode wall
allowance, not the physical speed, solver settings or success criteria.
`transfer_smoke` only runs
7 s construction/settling checks and must not be mistaken for task completion.
Other recipes:

- `transfer_prerequisites`: two additional repeats of key insertion and each
  spring/friction turning condition; combined with a passing nominal run these
  can satisfy the three-repeat composite gate.
- `transfer_composite`: spring and friction insert–turn, requiring matching
  revision, geometry and resistance-mode prerequisite evidence.
- `transfer_perturbations`: three small mass/friction/grasp-force conditions for
  the sled, flap, translation pushing and friction turning. These are combined
  robustness checks, not independently identified parameter effects.
- `transfer_negatives`: explicit missed engagement/contact, plus a seated
  no-turn control. Turning starts seated: a large lateral offset would initialize
  the key inside the pocket wall. The earlier 3 mm-offset turning negative is
  therefore not a valid geometry control; alignment failures belong to insertion.
- `transfer_solver`: the same four nominal mechanisms at 16 rather than 8
  nonlinear iterations and half the solver tolerance.

```bash
.venv/bin/python experiments/tool_use_pilot/benchmark_report.py \
  experiments/tool_use_pilot/output/real_transfer_v19 --mesh-audit
```

Reports include the new families; independent sampled geometry audits include
moving-object/support pairs as well as tool/environment pairs. A sampled audit
is not proof of zero intersection. Inspect captured RGB at approach, contact,
peak load and end. `benchmark_gallery.py --review --selection selection.json`
can package a development subset; failed/incomplete movies receive a
`NOT_PASSED` filename and must not be presented as successful demonstrations.

The tested 8/16-iteration runs log native solver status `STOPPED` (2), not
`CONVERGED` (1). The report exports status counts, residual summaries and
iteration counts. Similar task trajectories and contact wrenches across these
settings are useful sensitivity evidence, but not a numerical convergence
certificate; no residual units or termination cause are inferred here.

## Bounded results — 9 September 2026

All **18 changed/new isolated variants** have full nominal feasibility passes:
five shaped insertions, three turning modes, five enlarged-follower interactions,
two friction-sled layouts, gravity levering and two pushing variants. This is
not a full randomized qualification. Both combined insert–turn sequences also
pass (spring and friction), with maximum grasp drift of 0.94 and 0.91 mm,
respectively. Neither sequence returns after turning.

The [verification review](output/real_transfer_v19/VERIFICATION_REVIEW.md)
links the selected HDF5 recordings, wrench plots and controls. The
[representative video gallery](output/real_transfer_v19/representative_videos/README.md)
contains all 20 successful nominal variants as captured RGB + synchronized
tactile movies, with full-decode, duration and checksum verification. All 20
pass the sampled tool/fixture geometry audit; the complementary robot/fixture
check finds no sampled overlap. These checks are not swept-volume guarantees.

| Mechanism | Nominal endpoint | Small coupled variations |
|---|---|---|
| Loose-guided hook | 48.0 / 48.3 mm travel, normal / mirrored | 2/3 pass |
| Gravity lever | 24.95° flap lift; 0.53 mm maximum grasp drift | 3/3 pass |
| Tool pushing | Translation ≈0.46 mm error; pose 9.79 mm and 2.74° error | 3/3 translation checks pass |
| Friction turning | 91.28° measured rotor angle with 94° command | 3/3 pass |

The failed hook condition combines 0.15 kg sled mass, support coefficient 0.25
and 30 N per-finger effort. It reached only 36.1 mm and exceeded the grasp-angle
limit (18.76°). Opposing rail reactions peaked at 1.26 and 0.47 N, consistent
with binding plus weaker retention. These coupled changes do not isolate the
effect of friction. The heavier/high-friction condition passes but moves the
tool about 5.2 mm / 11.2° in the grasp; passing does not mean zero compliance.

The supported lever engagement control gives hold-mean net contact force
1.057 N when engaged versus 0.398 N when missing the flap; the force-field
reconstruction gives 1.068 versus 0.420 N. This is contact-dependent response
evidence with oracle kinematic compensation, not a learned tactile advantage.
The full nominal lever force/torque RMSEs remain 0.170 N / 0.00718 Nm.

All four matched 16-iteration nominal checks also pass. Compared with 8
iterations, maximum progress differences are 0.035 mm (hook), 0.022 mm
(pushing), 0.016° (lever) and 0.071° (turning). Contact-force differences are
0.033/0.038/0.049/0.006 N RMSE respectively. This supports the retained
8-iteration FP32 baseline for these feasibility checks, subject to the native
`STOPPED` status and calibration limitations above.

## Project relevance

Surface/guide following, insertion and engagement test local contact state;
levering adds multiple changing contacts and mechanical advantage; pushing
adds free-object translation/rotation dynamics. Tactile need not outperform
vision in every family. The next step after mechanics verification is a small
real observation/hardware matching check, then structured handle/tip diversity.
Actual peeling/material removal, printed-part exports and model training are
outside this implementation.

The same-geometry spring/friction turning pair is also a useful next latent-load
test: similar achieved poses can conceal different resistance. Hide load-mode
metadata and validate on held-out nuisance variations before claiming a tactile
advantage; the present nominal rollouts alone do not establish that advantage.

For later model work, preserve the moved target in the pushing observation:
uniformly sampling a combined table/target environment cloud can dilute the
small movable block. The saved per-body meshes and poses support a stratified
observation without recollecting the physics. Ideal ray visibility and synthetic
noise/patch removal remain observation checks, not evidence about a real RGB-D
camera. A held-out tactile benefit test should follow mechanics/observation
matching; it should not be inferred from task success alone.

The lever's nominal task starts with fulcrum support. The supported negative
keeps that fulcrum and offsets the flap/hinge/rests by 100 mm, testing missed
flap engagement. An earlier unsupported negative (fulcrum also moved away)
dropped the long tool; it remains rejected evidence. This package does not
qualify unsupported lever pickup/carrying. Small pre-motion calibration is
not a replacement for that missing retention test.

Development phases contain intermediate geometries and wall-time-censored
episodes. Do not pool them into a success rate or call them a frozen benchmark.
Deterministic prerequisite repeats test repeatability, not independent physical
diversity. The three coupled perturbations per new mechanism are only a small
robustness check, not a factorial sweep or a learned-policy evaluation.
The bounded free-sled checks cover straight pulls in the two mirrored layouts.
Left/right re-engagement commands are available but are not requalified recovery
policies for the free sled; earlier ideal-rail recovery results do not transfer
automatically to this mechanism.
