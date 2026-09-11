# Rail-free loaded-box pulling and independently tilted wall following

Revision 22 verification package, 10 September 2026. This is a task-mechanics
repair, not a new large-scale dataset or a learned transfer result.

## Changes and real-world interpretation

- `hook / loaded_box`: one free rigid tray with an integral loop on a level table.
  There are no guide rails, spring loads, wheels, hidden planar constraints,
  object-pose servos, or anti-tipping forces. The tray is a closed printable solid,
  120 × 100 mm in footprint, with a 6 mm base and retaining walls. The fixed-fixture
  comparison uses one 55 mm loop-crossbar height for every grasp orientation.
- A mostly straight shank with a J-shaped end is grasped by the same compliant
  GelSight Mini pads. The prepared hook begins inside the loop aperture with
  clearance. It acquires contact by moving horizontally, then commands a gentle
  54 mm pull. There is no 18 mm lifting engagement stroke. Finding and entering
  the loop from outside remains a separate future stage.
- Actual wrist poses include top-down, 10° roll, and sideways (−90° pitch about
  the task Y axis, so the straight shank aligns with the pull).
  Table/fixture axes remain level while tool geometry is transformed into the
  actual prepared grasp. Fixture placement follows the prepared working height;
  matching one fixed physical table height will additionally require arm-position
  sampling/IK. This package does not claim arbitrary 6-DOF pose coverage.
  The saved tool mesh contains the prepared grasp rotation; its rigid root frame
  initially aligns with the level task frame. For cross-simulator pose adapters,
  recover authored tool orientation as `R_world_mesh @ R_task_initial.T @
  R_grasp_initial`, using `body_root_poses`, `action_frame_rotation_world`, and
  `prepared_grasp_rotation_world`. New exports additionally store this constant
  factor as `mesh_from_authored_tool_rotation`. Reconstructed PCD/rendering must
  apply the saved mesh root pose only, without applying the baked rotation twice.
- Fixed ballast is approximated by total rigid-body mass (0.2 and 0.4 kg), not
  loose objects moving inside the tray. The nominal sliding coefficient is 0.4,
  giving reference horizontal loads of 0.785 and 1.570 N before dynamics and
  vertical loading. This is not a calibration to a particular real material.
  The engine currently regularizes Coulomb friction over 10 mm/s, while these
  pulls are slower. Consequently measured resistance can be below `mu*m*g`;
  friction coefficient alone is not enough to match another simulator or real
  hardware. A force-versus-speed calibration is still needed before transfer
  collection (see `mochi_core/contact/contact_params.h`, `frictionFalloffVel`).
- A bounded collector-only vertical force-relief controller limits inadvertent
  lifting/pressing: 0.5 mm/s maximum and ±6 mm travel. Its force input is explicitly
  privileged, not supplied as a deployable tactile observation.
- The top-down/rolled completion repair additionally tracks horizontal motion
  of a known point on the tool's working end (1 mm/s and ±12 mm bounded integral
  correction). This compensates compliant-grasp rotation without increasing
  the box's load or lowering its 40 mm goal. It uses oracle tool pose, not box
  pose; the same controller cannot be claimed deployable on hardware unchanged.
- `surface / rounded_wall`: actual wrist leans 10°/20° toward the positive guide
  in the task frame; the floor and wall stay level. The true authored working rim
  is retained after mesh rotation, instead of reselecting only its lowest node.
  Curve-normal projection uses working-tip position in this mode. Existing
  impedance, floor-load and guide-load control remain active.

The curved gel and force-only 7 × 9 × 3 representation per pad remain unchanged.
Tactile/extrinsic wrench agreement is diagnostic, never an episode-acceptance gate.

## Reproduce bounded checks

From the repository root, use a new output/phase name:

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  experiments/tool_use_pilot/benchmark.py \
  --output experiments/tool_use_pilot/output/my_pose_verification \
  --phase verification --recipe pose_transfer --workers 5 --render
```

The `pose_transfer` entry point selects fixed-box top-down/rolled grasps, the two
aligned sideways loads, and the 20° wall lean. It is a small verification recipe,
not permission to run broad randomization or assume unreviewed cases passed.
Rounds 4/5 preserve initial prototypes (including invalid oblique world-axis
sideways grasps, now rejected by preflight); round 6 is the aligned sideways pair;
round 7 repairs upright/rolled working-end stroke completion.
Historical revision-21
recipes remain replayable and are not silently rewritten or automatically
qualified by these new tests. Each run stores its source snapshot and HDF5 meshes,
poses, actions, force fields, wrench labels, controller metadata and captured RGB.

## Validation status

**Completed and reviewed 11 September 2026:** #84/#85 sideways box pulls and #86
top-down pass; #87 rolled pull fails the unchanged 15° relative-rotation gate.
All four finish 3,067 steps with no abort, tipping or support escape. Full movie
decoding and sampled RGB/geometry/housing review completed. See the
[main-server handoff](../../SUPERDEX_BENCHMARK_HANDOFF.md) for consolidated status.

Initial attempts 68–73 were stopped after rendered
sideways-table/finger interference was identified. These incomplete diagnostics
are not accepted episodes. Corrected clearance trials 74–79 and fixed-fixture
trials 80–83 are separate runs; completion, retained grasp, box tipping, actual
working-tip tracking, sampled mesh/housing clearance and video review must all
be checked before selecting a recommended configuration. Oblique world-axis
sideways prototypes 76–77 and 82 have invalid initial geometry and lost grasps;
they are excluded. A pre-rollout geometry check now rejects loop/tool and actual
housing overlaps above 0.1 mm. Aligned task-axis sideways trials are 84–85.

### Completed fixed-box checks before working-end tracking

| Case | Pull achieved (goal ≥40 mm) | Maximum box tilt | p95 vertical force on box | Maximum in-grip rotation | Outcome |
|---|---:|---:|---:|---:|---|
| Top-down, #80 | 36.56 mm | 0.0020° | 0.150 N (7.6% of weight) | 12.15° | Stable, short of goal |
| 10° world roll, #81 | 35.37 mm | 0.0018° | 0.324 N (16.5% of weight) | 13.21° | Stable, short of goal |

Both finish all 3,067 steps, retain their grasps and stay on the support. Their
sampled maximum overlap is 0.053/0.039 mm; actual housing/robot-link containment
is zero in the 11 checked frames. Full video decoding and sampled RGB review
pass. No numerical abort or tipping occurred. They are valid world-model failure
examples, **not** successful imitation episodes.

A 54 mm wrist command does not produce the same working-end stroke because of
compliant-grasp rotation. Trials 86–87 test bounded working-end feedback without
lowering the goal. Trials 84–85 test the corrected axial sideways setup at 0.2
and 0.4 kg. At the 15:33 UTC checkpoint, all four are still running and unqualified.
That checkpoint is historical; final results follow below. No further simulation
or parameter tuning was performed for the handoff.

### Final hook completion results — 11 September

| Case | Final pull | Relative tool/gripper rotation | Force / torque RMSE | Outcome |
|---|---:|---:|---:|---|
| Sideways 0.2 kg, #84 | 50.45 mm | 6.11° | 0.1714 N / 0.003075 Nm | Pass |
| Sideways 0.4 kg, #85 | 49.52 mm | 6.30° | 0.1770 N / 0.003191 Nm | Pass |
| Top-down with tip tracking, #86 | 47.06 mm | 13.65° | 0.1547 N / 0.002652 Nm | Pass, modest angular margin |
| 10° world roll with tip tracking, #87 | 43.48 mm | 16.34° | 0.1568 N / 0.002636 Nm | Rejected: angular grasp validity |

Wrench errors are force-only compact-field plus dynamics versus direct contact,
excluding initialization, and never selection gates. #87 reaches its pull goal
without a visible drop but fails the existing physical/grasp-validity criterion;
the relative-rotation metric does not distinguish elastic deflection from slip.
Do not silently promote it or relax the threshold. Maximum box tilt across the
four is <0.006°. Sampled maximum overlap is 0.066/0.062/0.059/0.058 mm; actual
housing/robot containment is zero at the 11 audited frames. The new sideways
setup is preferred. Two loads are not broad qualification.

[Final hook RGB review sheet](output/diversification_v21/handoff_review_20260911/page_0.png).

[Top-down captured RGB + tactile video](output/diversification_v21/loaded_box_v22_fixed_fixture/0080_hook_loaded_box_top_down_rgb_tactile.mp4)
and [10° roll video](output/diversification_v21/loaded_box_v22_fixed_fixture/0081_hook_loaded_box_roll_10_rgb_tactile.mp4).

Final completion records are in
`output/diversification_v21/loaded_box_v22_axial/` and
`output/diversification_v21/loaded_box_v22_tip_tracking/`.

### Completed wall comparison

| Case | Guide contact during follow | Tip travel (20 ±2 mm) | Working-edge lateral p95 | Stronger completion margin |
|---|---:|---:|---:|---|
| Previous upright, #66 | 100% | 22.186 mm | 3.064 mm | Fail |
| 10° lean, #78 | 82.83% | 20.081 mm | 0.703 mm | Fail: contact coverage |
| **20° lean, #79** | **100%** | **21.176 mm** | **0.229 mm** | **Pass** |

Both new wall runs finish all 3,067 steps without abort, retain their grasps, and
pass sampled tool/environment and actual-housing/robot-link clearance audits.
The maximum sampled tool/environment overlap is 0.124/0.165 mm for 10°/20°;
housing/robot-link containment is zero at the 11 audited frames. Captured videos
decode fully, and start/contact/peak/end RGB has been visually reviewed. These
checks are not proof of zero penetration at every surface point and instant.

The 20° case is the preferred nominal wall configuration. It still reaches the
8 mm tangential and 10 mm lateral controller bounds and has 11.12° maximum
in-grip angular deflection; this is a nominal repair, not broad pose robustness.
Force/torque reconstruction RMSE is 0.124 N / 0.00559 Nm (diagnostic only).

[20° captured RGB + tactile video](output/diversification_v21/loaded_box_lean_v22_clearance/0079_surface_rounded_wall_lean_20_rgb_tactile.mp4)
and [wrench plot](output/diversification_v21/loaded_box_lean_v22_clearance/0079_surface_rounded_wall_lean_20_wrenches.png).

For ACWM this provides interpretable changes in horizontal resistance and contact
geometry across grasps. It does not establish tactile necessity, counterfactual
learning benefit, or sim-to-real performance. Those remain learning/transfer
experiments after the task and data contract are frozen.

Regression checks: 111 tool-pilot unit tests passed, 2 skipped; all 13 soft-gel
tests passed. Targeted Ruff F/E9 checks and `git diff --check` passed. Existing
worktree changes were preserved; no commit, asset deletion or large collection
was performed in this repair turn.
