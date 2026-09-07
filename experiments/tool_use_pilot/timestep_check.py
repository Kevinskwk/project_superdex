#!/usr/bin/env python3
"""Compare same commanded task at 10 ms and 5 ms with equal-time filtering."""

import argparse
import json
from pathlib import Path
import h5py
import numpy as np


def load(path):
    with h5py.File(path) as f:
        d = {k: v[:] for k, v in f["observations"].items()}
        spec = json.loads(f.attrs["config_json"])
        metrics = json.loads(f.attrs["metrics_json"])
    width = round(0.05 / spec["dt"])
    active = ~d["initialization"]
    err = (
        d["dynamic_inferred_extrinsic_wrench"][active]
        - d["extrinsic_contact_wrench"][active]
    )
    smooth = np.column_stack(
        [np.convolve(err[:, j], np.ones(width) / width, "valid") for j in range(6)]
    )
    return d, dict(
        dt=spec["dt"],
        physical_valid=metrics["physical_valid"],
        max_progress_mm=metrics["max_progress_mm"],
        raw_force_rmse=metrics["force_rmse"],
        raw_torque_rmse=metrics["torque_rmse"],
        force_rmse_50ms=float(np.sqrt(np.mean(np.sum(smooth[:, :3] ** 2, axis=1)))),
        torque_rmse_50ms=float(np.sqrt(np.mean(np.sum(smooth[:, 3:] ** 2, axis=1)))),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("half_step", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    a, am = load(args.baseline)
    b, bm = load(args.half_step)
    bp = np.column_stack(
        [
            np.interp(a["timestamps"], b["timestamps"], b["tool_pose"][:, j])
            for j in range(3)
        ]
    )
    report = dict(
        baseline=am,
        half_step=bm,
        max_matched_time_tool_position_difference_mm=float(
            np.linalg.norm(bp - a["tool_pose"][:, :3], axis=1).max() * 1000
        ),
        interpretation="Task mechanics and raw wrench estimates must be assessed separately. Equal-time 50 ms causal averaging is reported in addition to raw samples, not as a replacement.",
    )
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
