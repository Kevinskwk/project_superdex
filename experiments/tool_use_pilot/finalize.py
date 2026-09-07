#!/usr/bin/env python3
"""Add explicit leakage-safe split metadata and verify generated HDF5 contracts."""

import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from probes import split_group


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    records = []
    for path in sorted(args.folder.glob("episode_*.h5")):
        with h5py.File(path, "r+") as f:
            spec = json.loads(f.attrs["config_json"])
            group = hashlib.sha256(split_group(spec).encode()).hexdigest()[:20]
            f.attrs["split_group_id"] = group
            f.attrs["split_group_semantics"] = (
                "same geometry/timeline/friction AND all hidden stiffness values and branch siblings; use this for dataset splits"
            )
            d = f["observations"]
            n = len(d["timestamps"])
            assert d["tactile_force_field_left"].shape == (n, 7, 9, 3)
            assert d["tactile_force_field_right"].shape == (n, 7, 9, 3)
            assert d["objectpointcloud"].shape == (n, 256, 3)
            assert d["env_point_cloud"].shape == (n, 256, 3)
            assert np.all(np.diff(d["timestamps"][:]) > 0)
            assert "stiffness" not in d
            parent = (
                path.name.split("_branch_")[0] + ".h5"
                if "_branch_" in path.name
                else None
            )
            if parent:
                with h5py.File(args.folder / parent) as source:
                    sd = source["observations"]
                    first = int(np.flatnonzero(~sd["initialization"][:])[0])
                    reference = Rotation.from_quat(
                        sd["ee_pose"][first, 3:]
                    ).inv() * Rotation.from_quat(sd["tool_pose"][first, 3:])
            else:
                first = int(np.flatnonzero(~d["initialization"][:])[0])
                reference = Rotation.from_quat(
                    d["ee_pose"][first, 3:]
                ).inv() * Rotation.from_quat(d["tool_pose"][first, 3:])
            relative = Rotation.from_quat(
                d["ee_pose"][:, 3:]
            ).inv() * Rotation.from_quat(d["tool_pose"][:, 3:])
            angles = np.rad2deg((relative * reference.inv()).magnitude())
            if "angular_slip_deg" not in d:
                d.create_dataset("angular_slip_deg", data=angles, compression="lzf")
            maximum = float(angles[~d["initialization"][:]].max())
            metric = json.loads(f.attrs["metrics_json"])
            metric["max_angular_slip_deg"] = maximum
            if maximum >= 15.0:
                for key in (
                    "physical_valid",
                    "grasp_retained",
                    "imitation_eligible",
                    "tactile_valid",
                ):
                    metric[key] = False
                    f["labels"][key][...] = False
            f.attrs["metrics_json"] = json.dumps(metric)
            if "checkpoint" in f:
                ck = f["checkpoint"]
                assert (
                    hashlib.sha256(ck["physics_bytes"][:].tobytes()).hexdigest()
                    == ck.attrs["physics_state_hash"]
                )
            records.append(
                dict(
                    file=path.name,
                    frames=n,
                    split_group_id=group,
                    task=spec["task"],
                    branch=spec["branch"],
                    max_angular_slip_deg=maximum,
                )
            )
    (args.folder / "schema_validation.json").write_text(
        json.dumps(dict(passed=True, episodes=len(records), records=records), indent=2)
    )
    print(f"Validated {len(records)} episodes; leakage-safe split_group_id added.")


if __name__ == "__main__":
    main()
