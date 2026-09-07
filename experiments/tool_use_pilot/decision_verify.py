"""Replay a selected episode, compare timesteps, optionally render actual RGB."""

import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from decisions import worker


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--render", action="store_true")
    p.add_argument("--timestep", action="store_true")
    a = p.parse_args()
    with h5py.File(a.source) as f:
        spec = json.loads(f.attrs["config_json"])
        reference = {
            k: f["observations"][k][:]
            for k in (
                "tool_pose",
                "body_root_poses",
                "tactile_force_field_left",
                "tactile_force_field_right",
                "timestamps",
            )
        }
    result = worker(spec, str(a.output / "replay"), False, a.render, spec["branch"])
    path = a.output / "replay" / a.source.name
    errors = {}
    with h5py.File(path) as f:
        for k, v in reference.items():
            other = f["observations"][k][:]
            errors[k] = (
                float(np.max(np.abs(v - other))) if v.shape == other.shape else None
            )
    report = dict(
        source=str(a.source),
        replay_max_errors=errors,
        exact=all(v == 0 for v in errors.values()),
        replay=result,
    )
    if a.timestep:
        spec = {**spec, "dt": spec["dt"] / 2}
        half = worker(spec, str(a.output / "half_dt"), False, False, spec["branch"])
        with h5py.File(a.output / "half_dt" / a.source.name) as f:
            times = f["timestamps"][:]
            poses = f["observations/tool_pose"][:]
            interp = np.stack(
                [
                    np.interp(reference["timestamps"], times, poses[:, j])
                    for j in range(3)
                ],
                1,
            )
            report["half_dt_max_matched_position_difference_mm"] = float(
                1000
                * np.linalg.norm(interp - reference["tool_pose"][:, :3], axis=1).max()
            )
        report["half_dt"] = half
    (a.output / "verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
