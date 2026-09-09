# Contact mechanics and impedance validation

## 9 September: turn-and-hold objective

Revision 16 defaults to `return_after_turn=False`: isolated turning ends while
holding the turned key; composite insertion-and-turning stays seated and ends
after the turned hold, without reverse rotation or withdrawal. Success still
requires the actual rotor to stay within 90 +/-5 degrees and remain seated
for the final 0.3 seconds. Merely finishing the command is not sufficient.
Standalone peg insertion retains its withdrawal check. Revision 15 and older
records retain their original return/withdrawal semantics; old failures have
not been retrospectively relabelled. `return_after_turn=True` remains available.

The model observation remains the two 7x9x3 **force** fields. Cell moments and
corrected wrenches are diagnostic/oracle channels only. Curved geometry remains
the default; a lower compact-wrench error from a flat one-node-per-marker mesh
would establish a useful numerical control, not superior real-sensor fidelity.
The historical results below precede the revised task endpoint.

This is a bounded diagnostic campaign, not a large-scale dataset or a real
GelSight calibration. The hardware target is the latest GelSight Mini Standard
Gel. Its elastic/friction parameters have not been measured here.

Results and actual simulator RGB recordings are under
[`output/mechanics_repair`](output/mechanics_repair/MECHANICS.md). Every attempted
condition stays in `attempts.json`; startup errors and rejected trajectories
are not hidden or counted as successes. No earlier dataset is overwritten.

## Arm control

The arm was already torque controlled by `BASIC_OSC_PD`:
`tau = J.T @ (K * pose_error - D * ee_velocity)`. The benchmark now exposes and
records translation stiffness/damping (N/m, Ns/m), rotation stiffness/damping
(Nm/rad, Nms/rad), actual EE targets and commanded joint efforts.

There are distinct controller layers:

* The inner Cartesian impedance controller, with no kinematic tool attachment.
* Optional outer ground-truth tool-pose integration (`tool_pose_feedback`).
  This is privileged collector feedback, not a policy observation. It can be
  disabled independently of impedance.
* Optional `calibrate_precontact_grasp`: a single translation calibration at
  2.5 s, blended before the task begins at 4 s; no subsequent tool-pose servo.
  This still uses a privileged initial estimate, and is recorded as such.
* Surface normal/guide force admittance. Its simulator force feedback remains
  explicitly privileged. It is not presented as an already validated tactile
  closed-loop policy.

The FR3 prefab has approximately 0.9–1.1 Nm arm-joint Coulomb friction and
unspecified effort limits. Gravity was disabled but motor friction was not
compensated in the old collector. This can produce substantial pose error.

`ideal_arm_friction_compensation` models zero residual arm motor friction,
while preserving finger and contact friction. This reflects ideal cancellation
by the internal low-level controller, not measured perfect hardware behaviour.
It also loads finite arm effort limits (87 Nm for joints 1–4, 12 Nm for joints
5–7) from the local FR3v2.1 URDF. Franka documents automatic gravity and motor
friction compensation for externally commanded torques:
[FCI overview](https://frankarobotics.github.io/docs/doc/libfranka/docs/overview.html).
Arm gravity remains ideally compensated; tool gravity remains enabled. Motor
dynamics, residual compensation error and the full hardware control stack are
not calibrated. The 35 N per-jaw command cap is a pilot cap, not a verified
mapping from the real gripper's `grasp(force)` argument.

## Supported scraping and surface contact

The custom fingers previously had independent force-controlled prismatic DOFs.
The stock URDF instead has `fr3v2_1_finger_joint2` mimic finger 1. The independent
DOFs allow the grasp centre to float along the closing axis, absorbing a wrist
stroke even when the gel/tool contact does not slide. The optional
`finger_coupling_stiffness_n_m` adds a native bilateral transmission penalizing
`q_left-q_right`, with 20 Ns/m damping. At 1e6 N/m it approximates the rigid
mechanical coupling while preserving the common opening/closing mode. It is
not a tool attachment. Finger coordinates and synchronization error are logged.

`surface_heading_deg=90` aligns the stroke with the jaw closing axis. The tool
mesh is counter-rotated in task coordinates, preserving its original physical
grasp orientation. Turning the broad tool into the jaws is incorrect; the
initial rejected setup tests demonstrated that failure and are retained.

For flat floors and straight guides, `surface_contact_model=analytic_box`
uses the same finite box dimensions with an analytic box collider and a simple
closed visual mesh. This avoids dense coplanar floor samples querying oblique
blade-side normals. Curved surfaces retain their curved mesh; a box is never
substituted for a curved floor. Contact normals and per-contact friction excess
are logged separately. Full-history sampled mesh overlap remains an independent
check, in addition to simulator contact depths.

Revision 14 defaults to a fixed attack angle during the scrape; the previous
5°/8° attack-angle sweeps remain explicit variants. Normal admittance can also
relieve the nominal plunge instead of saturating at zero correction. These
changes address contact continuity separately from grasp retention.

Revision 15 additionally enables ideal arm-friction compensation, coupled
fingers, and disables ongoing ground-truth tool-pose integration. Surface
strokes are 20 mm (25 mm for the peeler). These are operating-envelope changes,
not a claim that all randomized variants have been qualified. Flat scraping is
supported along the finger closing axis; wall/slot variants still require
controller development. The guide acquisition event is 0.1 N for 0.2 s, matching
the existing contact-event threshold; the independent 80% continuity and 80%
working-tip travel requirements are unchanged. Floor normal feedback excludes
wall friction, and the floor is unloaded during guide acquisition. Opposing
wall forces are not allowed to cancel the contact-event indicator.

## Why the original 7x9 field appeared better at torque

The legacy flat gel had exactly 63 exposed nodes spanning approximately
20.75 x 25.25 mm: one actual node and force per output location. The source-fit
gel has additional FEM surface nodes, but the requested marker centres span
12 x 16 mm at 2 mm pitch. Extra exposed-node forces are assigned to the nearest
marker cell. This preserves net mapped force, not its first spatial moment.

The contact-force query itself has not changed. Forces are cell-integrated
loads in newtons, not pressures or optical marker displacements. It is the
many-to-one spatial representation that discards lever arms. For example,
opposite 2 N shear forces separated by 4 mm in one cell have zero resultant
force but an 8 mNm couple. No single 3D force at a fixed marker can retain that
couple.

The 7x9x3 force field is unchanged. New optional FEM diagnostics provide:

* `tactile_cell_moment_left/right`: 7x9x3 intrinsic moments in Nm, on the gel,
  expressed in sensor axes about each **current** marker coordinate.
* `tactile_unmapped_wrench_left/right`: force and torque of unmapped FEM nodes,
  in sensor axes about the housing root. Never silently inserted into markers.
* `moment_corrected_gel_wrench` and `moment_corrected_inferred_extrinsic_wrench`:
  marker-position force moments plus intrinsic cell moments, with the same
  sign/reference-frame conventions and Newton–Euler compensation as before.
* Optional full world-space FEM positions/forces (`record_dense_field`).

Reconstruction is `sum((marker - origin) x force + cell_moment)`; rotate all
vectors consistently and negate gel loads when inferring loads on the tool.
Use `SoftGripper.get_surface_cell_moments(side)` with the existing force API.
These extra couples are FEM labels/diagnostics, **not new real sensor channels**.
The unchanged 7x9 forces alone still cannot generally reconstruct exact torque.
No corrected wrench is injected into the existing force-only probe input.

## Numerical and task diagnostics

Solver iteration cap, tolerance, convergence status and residual are now
explicit/recorded. The four-iteration friction-turning diagnostic stopped
without convergence throughout its post-initialization window. A smaller
timestep and a stricter Newton solve are separate comparisons. A mechanics
gate pass does not certify numerical convergence or real-world accuracy.

Surface success is independently re-audited using the working tip's net travel
between the first and last 0.3 s of the **follow** phase. The old maximum root
displacement could count motion after lifting off; it is not accepted as a
scrape. This stronger check can revoke an earlier recorded task-success flag,
which remains preserved as `recorded_task_success` in the derived report.

Hook tests separate matched mesh topology, grasp depth, outer pose feedback and
solver settings. `angular_slip_deg` remains a rigid tool/EE rotation metric;
it is not a direct stick/slip classifier and is not renamed in old data.

The coupled 50 mm hook command reached only 39.28 mm against a 40 mm slider
goal, despite retained grasp. The final control uses a modest 54 mm command
with the same smooth timing. Friction-loaded turning has a different failure:
the key returned to about 1.3 degrees while the rotor remained at 9.4 degrees.
The geometric key/socket clearance permits this backlash without grip slip.
A bounded 12-degree reverse overtravel takes up that clearance, then returns
the key to neutral. Spring-loaded variants retain their prior return path.
This remains an experimental repair, not a qualified turning controller: the
final episode 73 ends at +4.04 degrees but oscillates from -6.08 to +5.38 degrees
over its last second, failing the unchanged +/-5-degree return dwell. Tool
retention is good (0.96 mm relative displacement, 0.46-degree relative rotation).
The force-only wrench errors remain high: 0.347 N and 17.81 mNm. The failed
settling and unconverged contact solve must be addressed before this variant
is used as successful training data; a final angle alone is insufficient.

The optional key seating retry unloads after the 3 N insertion guard, executes
one bounded +/-8° yaw search with force-limited descent, then either confirms
seating or safely withdraws and reports failure. It does not read the rotor's
ground-truth angle for alignment. Seating verification still uses geometric
ground truth; the original hard 15 N/0.5 Nm abort remains unchanged. Retry
exhaustion is not labelled success.

## Commands

Use a new phase name/output folder; existing runs are never overwritten:

```bash
.venv/bin/python experiments/tool_use_pilot/benchmark.py \
  --output experiments/tool_use_pilot/output/my_validation \
  --phase diagnostic --specs path/to/manifest.json --workers 8 --render

.venv/bin/python experiments/tool_use_pilot/benchmark_report.py \
  experiments/tool_use_pilot/output/my_validation
```

For a diagnostic composite rerun, `--prerequisite-output` can point to an
existing independently qualified insertion/turning campaign. This is read-only
qualification evidence, not a claim that the new composite passes.

The study-specific `mechanics_validation.py` and `curved_gel_regression.py`
drivers were retired after this study; their source is archived with the
9 September cleanup evidence. Use explicit `--specs` manifests for new sweeps.

## Completed representative checks

The completed ledger contains 79 attempts: 73 recordings (including failed
and runtime-censored diagnostics) and six setup errors from the subsequently
fixed FR3 joint-name lookup. Nothing is pending. The full per-attempt result
table is in `output/mechanics_repair/MECHANICS.md`.

Errors below are vector RMSE over post-initialization frames, against the
simulator's summed tool/environment wrench in world axes about tool COM.
All estimates use the same Newton–Euler compensation. These are single-condition
diagnostics, not multi-seed success rates.

| Episode | Result | Force RMSE N | Torque RMSE: force-only / with cell moments, mNm |
|---:|---|---:|---:|
| 58 | Supported flat scrape: 18.33/20 mm tip travel, 100% follow contact, 0.39° relative rotation; pass | 0.141 | 3.51 / 1.84 |
| 56 | Concave drawing: contact/retention/task gates pass; 27.32 mm tip travel for 20 mm command, so path tracking is not exact | 0.139 | 7.85 / 1.93 |
| 54 | Coupled-jaw key insertion: pass, 1.04 mm relative displacement | 0.167 | 2.66 / 2.66 |
| 72 | Hook: 42.75 mm slider travel against 40 mm goal; retained, complete, 12.31° peak relative rotation | 0.133 | 3.89 / 2.32 |
| 74 | Peeler: retained, continuous contact; 19.55/25 mm tip travel misses the 20 mm minimum | 0.043 | 4.84 / 1.30 |
| 24 | Earlier normal composite reference: full sequence passes, prior independent-jaw controller | 0.147 | 9.66 / 2.76 |

The insertion torque is almost zero, so its small absolute error is **not**
evidence of accurate relative torque sensing. Episode 57, the low-resistance
composite with the repaired controller, seats, turns and returns the rotor,
but ends 1.46 mm inside the socket; it fails withdrawal clearance. This is not
reported as fixed by the successful earlier retry/reference episodes.

For straight guides, episodes 75/76 (stroke-supported) achieve about 17 mm
travel but only 58% guide-contact continuity. Episodes 77/78 (wall-load-supported)
maintain 100% contact but travel only 6.4 mm. Both configurations retain the
tool but fail the full task gates. Curved guide variants remain unqualified.
Do not use these as successful imitation examples. The controller/grasp
orientation trade-off remains, rather than evidence that increasing gel
hardness is the necessary fix.

On the matched 16 s friction-turning diagnostic, raising the Newton cap from
4 to 16 (tolerance 1e-3 to 1e-4) reduces force error 0.302 to 0.124 N and torque
error 15.52 to 4.47 mNm. Neither solve reports convergence after initialization.
FP32 is retained. Use the stricter solve for follow-up fidelity checks, but do
not call it converged or treat the iteration change as hardware calibration.

The repaired flat scrape was visually reviewed from RGB and side cameras.
Its full-history sampled tool/floor overlap peaks at 0.384 mm (below the 1 mm
rejection threshold, not zero); a separate 30-frame robot-vertex/floor check
found no penetration. Peeler RGB/side views also show a retained, correctly
oriented tool. These sampled checks are not an exhaustive intersection proof.
The final hook and turning snapshots also show retained tools; their sampled
tool/environment mesh-overlap peaks are 0.186 and 0.157 mm respectively.

Representative captured video:
[`0058 scraper RGB + tactile`](output/mechanics_repair/final_surfaces/0058_surface_flat_scrape_nominal_rgb_tactile.mp4).
The video wrench overlay remains the original force-only estimate; use
[`cell_moment_comparison.png`](output/mechanics_repair/cell_moment_comparison.png)
for the augmented-wrench comparison. Surface tasks are rigid contact/path
following, not simulated cutting or material removal.

Verification commands:

```bash
.venv/bin/python -m pytest experiments/gelsight_mini_contact_validation -q
.venv/bin/python -m unittest discover -s experiments/tool_use_pilot -p 'test_*.py'
```

The useful ACWM deliverable is a contact-consistent, explicitly controlled
operating envelope and honest failure labels—not parameters chosen solely to
maximize nominal task success. Material calibration against the latest real
Standard Gel remains a separate requirement.
