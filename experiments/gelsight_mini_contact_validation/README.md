# Soft GelSight Mini contact validation

This experiment replaces the Robotiq 2F-85 in the earlier contact-wrench test
with a Franka parallel gripper carrying two compliant GelSight Mini gels. The
sensor adapter, housing, and gel geometry comes from HydroShear; each gel is a
Neo-Hookean tetrahedral actor attached to its rigid sensor housing through
SuperDex's soft-skinned articulation API.

The default mesh is a `7×9×2` regular tetrahedralization of the exact HydroShear
gel bounds. Its exposed face provides a dense `7×9×3` force-vector field in the
authored sensor frame (`sensor_x`, `sensor_y`, `xyz`). Regenerate the assets or
request the slower high-fidelity Gmsh mesh with:

```bash
.venv/bin/python assets/bots/grippers/franka_gelsight_mini/generate_assets.py
.venv/bin/python assets/bots/grippers/franka_gelsight_mini/generate_assets.py --method gmsh
```

Run one FP32 headless trial with video:

```bash
.venv/bin/python experiments/gelsight_mini_contact_validation/run.py
```

Run the uncalibrated modulus/damping selection and then the six orientation
cases:

```bash
.venv/bin/python experiments/gelsight_mini_contact_validation/run_material_sweep.py
.venv/bin/python experiments/gelsight_mini_contact_validation/run_orientation_sweep.py
```

Each case writes `contact_wrenches.csv`, `tactile_fields.npz`, `metrics.json`,
`config.json`, force/shear/wrench-comparison plots, and (unless `--no-video` is supplied)
`simulation.mp4`. The NPZ stores `left_force_field` and `right_force_field` as
dense `time×7×9×3` arrays, plus matching exposed-surface displacement vectors,
surface coordinates, and integrated wrench arrays at 100 Hz. The legacy
`*_force_grid` keys alias the same dense arrays. Video panels use HydroShear's
colored-arrow convention: `(Fx,Fy)` controls arrow direction/length and `|Fz|`
controls the green-to-blue normal-load color.

`field_gels_on_plug_wrench_world` is the equal-and-opposite tool wrench obtained
by summing the nodal forces and `r×F` moments about the instantaneous plug COM.
`field_inferred_table_on_plug_wrench_world` applies the quasistatic gravity
correction so it can be compared directly with
`extrinsic_table_on_plug_wrench_world`. Each wrench array stores
`[Fx,Fy,Fz,Tx,Ty,Tz]` in SI units.

The force/torque comparison validates numerical consistency and causal force
transmission through the gels and plug. Absolute physical accuracy remains
uncalibrated until real GelSight force–indentation measurements are supplied.

For direct online access after each simulation step:

```python
gel = gripper.gels["left"]
gel.register_query(physics.QueryType.NODE_CONTACT_FORCES)
scene.step(dt)
left_force_sensor = gripper.get_surface_force_field("left")  # (7, 9, 3), N
```

## Parallel VT-ACWM collection

Collect the requested 128 episodes, with 200 steps and 1,024 tool/environment
points per step, in headless FP32 mode:

```bash
.venv/bin/python experiments/gelsight_mini_contact_validation/collect_parallel.py
```

The benchmarked default uses 16 simultaneous environments per scene and eight
cohorts. Larger single-scene batches were slower on this workstation once all
soft contacts were active. Override the batch size with `--parallel-envs`.

The output `superdex_128x200.hdf5` follows the AC-VTWM structured-overlap
requirements. Every `episodes/episode_NNNNNN` group contains:

- `observations/objectpointcloud` and `env_point_cloud` (`T×1024×3`)
- `ee_pose`, `ee_vel`, `tool_pose`, and `tool_vel`
- left/right `7×9×3` tactile force fields and marker coordinates
- extrinsic tool/table wrench and the wrench inferred from the dense fields
- 6D actions, rewards, done flags, timestamps, and contact phase
- perturbation metadata, branch-group/initial-state identity, and separate
  physical, tactile, imitation-eligibility, and task-success labels

Compatibility hard links expose `point_cloud` and `tactile_data_left/right`
without duplicating storage. Root `index` datasets permit fast episode lookup.
`benchmark.json` and `benchmark_report.md` record collection/write timing,
integrity counts, compressed size, and linear projections to the existing
IsaacGym and MuJoCo episode counts.

## SCFields tool-use fidelity campaign

Prepare the checksummed official SCFields subset and run the complete balanced
campaign:

```bash
.venv/bin/python experiments/gelsight_mini_contact_validation/scfields_assets.py
.venv/bin/python experiments/gelsight_mini_contact_validation/run_fidelity_campaign.py
```

The three capsule meshes default to the sibling TACS-L/IsaacGymEnvs checkout.
For another location, pass `--capsule-root PATH` or set
`SCFIELDS_CAPSULE_ROOT`.

The runner detects CPU/SMT topology, reserves 25% of physical cores for the
desktop, benchmarks several Mochi process/thread layouts, pins each process to
disjoint physical cores and their SMT siblings, and selects the fastest valid
layout. Use `--skip-autotune --processes 4 --parallel-envs 4 --worker-threads 2`
to reproduce this workstation's selected layout. `--smoke-test` runs the same
multi-process and external-link HDF5 path with eight short episodes.

The 999 deterministic episodes comprise 486 standardized contacts, 216
friction/mass robustness contacts, 108 paired extreme tool/gripper orientation
contacts, and 189 shape-specific rolling, writing, scraping, peeling, corner,
and lever interactions. Open official meshes retain a surface copy for point
clouds and use a labeled watertight convex-hull collision proxy because Mochi
requires positive enclosed volume for dynamic rigid actors.

Each shard extends the VT-ACWM layout with direct and dense-field fingertip
wrenches, a Newton–Euler dynamic environment-wrench estimate, direct/inferred
contact centers, the complete two-sided `T×7×9×3` field, protocol metadata, and
per-episode fidelity labels. `scfields_fidelity_master.hdf5` exposes every
episode through relative HDF5 external links without copying shard data.
`campaign_report.json` is the machine-readable result and
`campaign_report.md` summarizes accuracy, validity, CPU/RSS, throughput,
repeatability, and projected time/storage at the existing dataset sizes.

Record a deterministic episode through the offscreen EGL camera, with simulator
RGB, both normal/shear fields, and the direct/inferred wrench traces:

```bash
.venv/bin/python experiments/gelsight_mini_contact_validation/record_rgb_fidelity_episode.py \
  --episode-id 813 --friction 1.4 \
  --output experiments/gelsight_mini_contact_validation/output/rgb_tip_stroke
```

The RGB recorder adds both 7×9 shear panels, direct/inferred force and torque
traces, and live tool/EE slip. It refuses to publish a video if task-induced
slip exceeds 12 mm or bilateral contact falls below 80%. Task trajectories use
zero-velocity smooth pulses; the peeler is capped at 3 mm and 8 degrees and the
tip stroke at 8 mm and 10 degrees. The default grip is 28 N per finger and the
benchmarked gel/tool friction coefficient is 1.4.

Re-run the friction/retention sweep with:

```bash
.venv/bin/python experiments/gelsight_mini_contact_validation/run_friction_benchmark.py
```
