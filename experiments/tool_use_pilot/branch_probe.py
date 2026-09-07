#!/usr/bin/env python3
"""Exploratory anchor-only branch ranking with leave-configuration-out fits.

Only checkpoint observations and the candidate action macro are inputs. No
future continuation observations or hidden spring labels are exposed to a fit.
Twelve anchors are far too few to claim a learned general-purpose policy.
"""

import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from probes import split_group


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    folder = parser.parse_args().folder
    anchors = []
    for source in sorted(folder.glob("episode_[0-9][0-9][0-9][0-9].h5")):
        paths = [
            folder / f"{source.stem}_branch_{b}.h5"
            for b in ("nominal", "hold", "retreat", "alternate")
        ]
        if not all(p.exists() for p in paths):
            continue
        with h5py.File(source) as f:
            spec = json.loads(f.attrs["config_json"])
            d = f["observations"]
            i = round(spec["branch_time"] / spec["dt"]) - 1
            base = np.r_[
                d["ee_vel"][i],
                d["tool_vel"][i],
                d["tool_pose"][i, :3] - d["ee_pose"][i, :3],
                d["task_progress"][i],
            ]
            tactile = np.r_[
                d["tactile_force_field_left"][i].ravel(),
                d["tactile_force_field_right"][i].ravel(),
            ]
            wrench = d["field_gel_wrench"][i]
            initial = float(d["task_progress"][i])
        values = []
        for p in paths:
            with h5py.File(p) as f:
                valid = bool(f["labels/physical_valid"][()])
                values.append(
                    float(f["observations/task_progress"][:].max())
                    - initial
                    - (0 if valid else 0.1)
                )
        anchors.append(
            dict(
                task=spec["task"],
                group=split_group(spec),
                base=base,
                tactile=tactile,
                wrench=wrench,
                target=np.asarray(values),
                source=source.name,
            )
        )
    results = []
    for mode in ("action_prior", "proprio", "dense_tactile", "gel_wrench"):
        folds = []

        def features(a):
            x = np.zeros(1) if mode == "action_prior" else a["base"]
            if mode == "dense_tactile":
                x = np.r_[x, a["tactile"]]
            if mode == "gel_wrench":
                x = np.r_[x, a["wrench"]]
            return np.asarray(
                [np.r_[one, x, np.outer(one, x).ravel()] for one in np.eye(4)]
            )

        for test in anchors:
            train = [
                a
                for a in anchors
                if a["task"] == test["task"] and a["group"] != test["group"]
            ]
            if len(train) < 2:
                continue
            model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            model.fit(
                np.concatenate([features(a) for a in train]),
                np.concatenate([a["target"] for a in train]),
            )
            pred = model.predict(features(test))
            selected = int(np.argmax(pred))
            best = float(test["target"].max())
            folds.append(
                dict(
                    source=test["source"],
                    task=test["task"],
                    predicted_values=pred.tolist(),
                    actual_values=test["target"].tolist(),
                    selected=selected,
                    regret_m=best - float(test["target"][selected]),
                    optimal=bool(best - test["target"][selected] < 0.001),
                )
            )
        results.append(
            dict(
                model=mode,
                anchors=len(folds),
                mean_regret_mm=float(np.mean([f["regret_m"] for f in folds]) * 1000)
                if folds
                else None,
                optimal_fraction=float(np.mean([f["optimal"] for f in folds]))
                if folds
                else None,
                folds=folds,
            )
        )
    report = dict(
        status="exploratory only: twelve anchors, not a policy-performance claim",
        anchors=len(anchors),
        target="peak continuation fixture travel minus anchor travel, minus 0.1 m for physical invalidity",
        candidate_order=["nominal", "hold", "retreat", "alternate"],
        models=results,
    )
    (folder / "branch_probes.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
