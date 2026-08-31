# Contact-wrench validation

This experiment uses the FR3 v2 + Robotiq 2F-85 assembly to grasp a known rigid
cuboid, hold it clear of a constrained tabletop, descend through contact, and
regulate 2 N, 5 N, and 10 N normal-force plateaus. It records fingertip and
plug/table contact wrenches from both actor queries and contact-point integration.

Run both installed precision modes from the repository root:

```bash
.venv/bin/python experiments/contact_wrench_validation/run.py --precision both
```

Results are written to `output/fp32`, `output/fp64`, and `output/comparison`.
Use `--no-video` to disable headless EGL recording or `--output PATH` to select
another output directory.

Each precision directory contains the raw `contact_wrenches.csv`, `metrics.json`,
the resolved `config.json`, three analysis plots, and a headless `simulation.mp4`.
The comparison directory contains an FP32/FP64 overlay and numerical deltas.

Run the recommended FP32 orientation sweep with:

```bash
.venv/bin/python experiments/contact_wrench_validation/run_orientation_sweep.py
```

The sweep adds positive/negative roll, positive/negative pitch, and compound
roll/pitch cases for both the gripper command and initial plug presentation.
Complete cases are reused unless `--force` is supplied. Consolidated results are
written to `output/orientation_sweep/orientation_summary.json` and
`output/orientation_sweep/orientation_sweep.png`.

FP32 is recommended for routine contact-data collection because its measured
wrenches and force trajectories agree closely with FP64 while running more
efficiently. Use FP64 when auditing numerical residuals near machine precision.

Wrenches in the CSV are explicit about their receiver and reference. Fingertip
torques are about each fingertip center of mass. The plug/table wrench acts on
the plug and is expressed about the plug center of mass. World and receiver-local
versions are included.
