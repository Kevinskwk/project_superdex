# FR3 / GelSight Mini tool-use pilot

Small task-feasibility and tactile-consistency experiments using the existing
FR3, Franka gripper and two compliant HydroShear GelSight Mini assemblies.
No SCFields downloads or external assets are added by this package.

## Current entry points

This document describes the original four-task baseline. For the repaired hook,
separate key insertion/turning tasks, and noisy/occluded Point-M2AE tactile
comparisons, start with [DECISION_PILOT.md](DECISION_PILOT.md).
See [CLEANUP.md](CLEANUP.md) for retained results and retired development artifacts.

## Tasks and scope

| Task | Physical interaction | Intended evidence |
|---|---|---|
| `hook` | Insert a blunt J-hook around a supported crossbar and pull its spring-loaded carriage at least 40 mm for 0.3 s | Engagement versus miss, indirect load transmission, wrench consistency |
| `probe` | Pull, unload, disengage, pause, re-engage and pull again; hidden spring stiffness 30/60/90 N/m | Information from an earlier interaction; explicitly audit current-state leakage before claiming memory is necessary |
| `push` | A rectangular paddle moves a visible cylindrical puck toward a 50 mm goal | Positive contact control; friction/offset variation and visible misses |
| `calibration` | Plug/table preload, hold, shear and release | Low-load reference for force and moment comparisons |

The tools start in a prepared grasp. The temporary tool constraint is released
at 1 s; no weld, pose teleportation or kinematic tool attachment assists the
interaction. Pickup, real sensor calibration, real-world transfer and ACWM
training are **not** implemented or demonstrated here.

The hook and slider are original metric, parametric meshes. Multiple closed
box components retain the openings; the collision representation is a triangle
mesh, not a convex hull. A native reduced-coordinate prismatic joint provides
the rail's five locked degrees of freedom. Supports/carriage are modeled.
The ideal guide has a 60 mm travel limit, a linear spring and 0.8 N s/m damping;
it does not model a real bearing's friction, backlash or manufacturing tolerance.

FP32 is used. Gel material and friction come from `soft_gripper.py` (200 kPa,
Poisson ratio 0.49, friction 1.4). Grip commands are 35 N/finger for hook/probe
and 28 N/finger for push/calibration. All nominal trajectories are C2 and stay
at or below 20 mm/s, with no commanded wrist rotation. Approach-direction yaw
is perturbed; it is not a gripper-orientation sweep.

The rigid contact penalty is 5e10, raised after rejected attempts exceeded the
1 mm penetration threshold. The existing housing/tool collision exclusion is
preserved: only the compliant gel is the instrumented grasp surface. This is a
modeling limitation, not a validated real housing/contact calibration.

## Run

From the repository root:

```bash
.venv/bin/python experiments/tool_use_pilot/pilot.py \
  --task hook --render --branches \
  --output experiments/tool_use_pilot/output/my_hook

.venv/bin/python experiments/tool_use_pilot/campaign.py \
  --workers 4 --output experiments/tool_use_pilot/output/my_pilot

.venv/bin/python experiments/tool_use_pilot/review.py \
  experiments/tool_use_pilot/output/my_pilot

.venv/bin/python experiments/tool_use_pilot/finalize.py \
  experiments/tool_use_pilot/output/my_pilot
.venv/bin/python experiments/tool_use_pilot/analyze.py \
  experiments/tool_use_pilot/output/my_pilot
.venv/bin/python experiments/tool_use_pilot/probes.py \
  experiments/tool_use_pilot/output/my_pilot
.venv/bin/python experiments/tool_use_pilot/branch_probe.py \
  experiments/tool_use_pilot/output/my_pilot
```

`--qualification` selects 12 main episodes. `--task probe --limit 24` selects
a smaller targeted collection. Workers are independent CPU processes; EGL RGB
rendering uses the GPU and is timed separately. The collected campaign has 180
main episodes (48 hook, 24 push, 96 probe, 12 calibration), plus four continuations
from each of 12 anchors. Push/calibration take 12 s. The corrected hook takes
18.5 s to return and unload the spring before lowering the hook; lowering while
loaded caused excessive in-grip rotation. The corrected probe takes 14 s so its
**first** pull really makes contact without exceeding the motion-speed limit.
Data are sampled at 100 Hz and review videos at 20 fps.

Optional learned-probe dependencies are listed in `requirements-probes.txt`.
Install the CPU Torch wheel using its official CPU wheel index. Tests require
the existing project simulation/analysis environment:

```bash
.venv/bin/python -m unittest discover -s experiments/tool_use_pilot -p 'test_*.py'
.venv/bin/ruff check experiments/tool_use_pilot --select F,E9
```

## HDF5 contract

`dataset.h5` is a relative-external-link index into individual episode HDF5
files. Keep the index and episode files together when moving the dataset.
The schema is `vt_acwm_superdex_tool_pilot_v1`; it retains the previous
ACWM observation names/aliases and makes changed action semantics explicit.

- `observations/objectpointcloud` and `point_cloud`: T x 256 x 3, full ideal
  tool surface in world coordinates, not an occluded/noisy RGB-D reconstruction.
- `observations/env_point_cloud`: T x 256 x 3, actual table/puck or rail/slider
  mesh samples transformed with each body's current pose.
- `ee_pose`, `tool_pose`, `ee_vel`, `tool_vel`: world-frame poses/velocities;
  tool pose is about its COM; quaternions are XYZW.
- `tactile_force_field_{left,right}` and `tactile_data_{left,right}`:
  T x 7 x 9 x 3 nodal forces in each sensor frame, in newtons.
- `tactile_coord_*`: deformed surface coordinates in world metres.
  `tactile_displacement_*`: sensor-frame marker displacements in metres.
- `tactile_reference`: the two pre-interaction force fields at 1.45 s.
- Wrenches: direct gel, integrated dense gel, direct tool/environment contact,
  and gel-plus-dynamics inferred environment wrench. All are in world axes
  about the instantaneous tool COM, in N and N m.
- Root `actions`: absolute EE translation offsets in the **initial tool frame**
  and rotation offsets in radians. They are **not per-step deltas**. Read
  `action_semantics` and `action_frame_rotation_world` before adapting a loader.
- Root `timestamps`, `rewards`, `dones`, `contact_phase`, `episode_lengths`;
  fixture state, progress, contact events, initialization masks and physical
  diagnostics are also recorded.
- `labels`: hidden stiffness/friction and separate physical-validity,
  tactile-validity, grasp-retention, imitation-eligibility and task-success flags.
- Anchor `checkpoint`: actual native state bytes, controller commands, RNG,
  tactile reference, recent history and hashes. Identical-action replay checks
  are stored in continuation metrics. Configuration hashes are separate from
  captured-physics and complete-checkpoint hashes.

Always split on **`split_group_id`**, added by `finalize.py`, not on filenames or
episode IDs. It ties together branch siblings, repeated physical configurations,
and all hidden stiffness values for the same geometry/timeline. `branch_group_id`
is only an anchor-family identifier. Some configurations are deterministic
repeats; counts must not be interpreted as independent randomized trials.

The HDF5 contains oracle evaluation channels (direct contact wrench, exact
fixture state, etc.). Their presence does not authorize using them as inputs to
a tactile-only or visual-only baseline. `probes.py` explicitly selects its input
channels and never consumes hidden stiffness except in the named oracle model.

## Acceptance and interpretation

Physical validity requires finite values, <12 mm tool/EE translation drift and
<15 degrees relative angular drift,
no rigid penetration above 1 mm for three consecutive samples, positive gel
tetrahedron Jacobians and <1 mm guide drift. Safety aborts stop nonfinite poses,
>25 mm drift or >15 N tool/environment force. Intentional misses are physically
valid task failures; they are never relabeled as successful demonstrations.

Probe success additionally requires measured contact and >1 mm carriage motion
in **both** the initial probe and later pull. Early 12 s probe attempts failed
this semantic gate despite eventually moving the carriage, and are rejected.

The tactile screen is deliberately separate: force RMS <0.2 N **or** NRMSE <20%,
and torque RMS <0.01 N m. Always inspect absolute and relative errors together;
NRMSE is uninformative for zero/near-zero reference wrenches. Initialization is
excluded. Both raw and equal-time 50 ms causal-average errors are reported.
Contact-query ground truth and gel forces come from the same simulator: their
agreement demonstrates numerical/internal consistency, not independently
validated real sensor accuracy.

`probes.py` compares current tactile, 0.5 s history, compact LSTM, no-tactile
LSTM, integrated-wrench LSTM, causally past-resampled tactile, and a stiffness oracle
with three seeds and train-only normalization. H1/H5 mean 100/500 ms at the
10 Hz probe-model rate, not 1/5 of the 100 Hz raw frames. Exact pose features are
privileged visual proxies, not a learned RGB perception system. Pause-time
stiffness decodability is a leakage test. A positive tactile or memory claim is
not justified merely because a recurrent model can fit the training data.

`branch_probe.py` is a separate anchor-only, leave-configuration-out linear
ranking check for the four known action macros. Twelve anchors are insufficient
for a general policy-performance claim. All candidate continuations, including
physically invalid ones, are retained and scored with an invalidity penalty.

Generated data, attempts and RGB assets stay under ignored `output/`. The final
report lists rejected iterations and limitations instead of silently filtering
them out of a claimed success rate.
