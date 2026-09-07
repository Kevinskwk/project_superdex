#!/usr/bin/env python3
"""Summarize every attempted pilot episode, including physical/task failures."""

import argparse
import json
from pathlib import Path
import h5py
import numpy as np


def analyze(folder):
    records = []
    replay = []
    for path in sorted(folder.glob("episode_*.h5")):
        with h5py.File(path) as f:
            spec = json.loads(f.attrs["config_json"])
            m = json.loads(f.attrs["metrics_json"])
            m.update(
                path=path.name, branch=spec["branch"], group=spec["group"], config=spec
            )
            d = f["observations"]
            active = ~d["initialization"][:]
            direct = d["direct_gel_wrench"][:][active]
            field = d["field_gel_wrench"][:][active]
            m["dense_vs_direct_gel_force_rmse"] = float(
                np.sqrt(np.mean(np.sum((direct[:, :3] - field[:, :3]) ** 2, axis=1)))
            )
            m["dense_vs_direct_gel_torque_rmse"] = float(
                np.sqrt(np.mean(np.sum((direct[:, 3:] - field[:, 3:]) ** 2, axis=1)))
            )
            # Best correlation lag is diagnostic, not a causality proof. Same-step
            # forces are expected; initialized samples are explicitly excluded.
            direct_env = d["extrinsic_contact_wrench"][:][active, :3]
            inferred = d["dynamic_inferred_extrinsic_wrench"][:][active, :3]
            width = max(1, round(0.05 / spec["dt"]))
            delta = (
                d["dynamic_inferred_extrinsic_wrench"][:][active]
                - d["extrinsic_contact_wrench"][:][active]
            )
            filtered = np.column_stack(
                [
                    np.convolve(delta[:, j], np.ones(width) / width, "valid")
                    for j in range(6)
                ]
            )
            m["force_rmse_50ms_mean"] = float(
                np.sqrt(np.mean(np.sum(filtered[:, :3] ** 2, axis=1)))
            )
            m["torque_rmse_50ms_mean"] = float(
                np.sqrt(np.mean(np.sum(filtered[:, 3:] ** 2, axis=1)))
            )
            correlations = []
            for lag in range(-5, 6):
                a = direct_env[max(lag, 0) : len(direct_env) + min(lag, 0)]
                b = inferred[max(-lag, 0) : len(inferred) + min(-lag, 0)]
                correlations.append(
                    float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
                    if np.std(a) > 1e-8
                    else -1.0
                )
            m["best_force_lag_ms"] = (
                (int(np.argmax(correlations)) - 5) * spec["dt"] * 1000
                if max(correlations) > 0
                else None
            )
            m["is_continuation"] = "_branch_" in path.name
            if "identical_action_replay_max_pose_error" in m:
                replay.append(m["identical_action_replay_max_pose_error"])
            records.append(m)
    mains = [r for r in records if not r["is_continuation"]]
    summary = {}
    for task in ("hook", "push", "probe", "calibration"):
        rows = [r for r in mains if r["task"] == task]
        if not rows:
            continue
        nominal = [r for r in rows if abs(r["config"]["offset"]) < 0.020]
        valid = [r for r in rows if r["physical_valid"]]
        contact = [r for r in valid if r["contact_fraction"] > 0.02]
        summary[task] = dict(
            episodes=len(rows),
            physically_valid=sum(r["physical_valid"] for r in rows),
            task_successes=sum(r["task_success"] for r in rows),
            nominal_count=len(nominal),
            nominal_valid_successes=sum(
                r["physical_valid"] and r["task_success"] for r in nominal
            ),
            tactile_valid=sum(r["tactile_valid"] for r in rows),
            contact_valid_count=len(contact),
        )
        for key in (
            "force_rmse",
            "torque_rmse",
            "force_nrmse",
            "torque_nrmse",
            "max_slip_mm",
            "max_penetration_mm",
            "force_correlation",
            "torque_correlation",
            "force_rmse_50ms_mean",
            "torque_rmse_50ms_mean",
            "max_angular_slip_deg",
            "dense_vs_direct_gel_force_rmse",
            "dense_vs_direct_gel_torque_rmse",
        ):
            vals = [r[key] for r in contact if r.get(key) is not None]
            summary[task]["median_" + key] = float(np.median(vals)) if vals else None
        summary[task]["mechanical_gate_pass"] = bool(
            nominal and summary[task]["nominal_valid_successes"] / len(nominal) >= 0.9
        )
    report = dict(
        tasks=summary,
        episodes=len(records),
        main_episodes=len(mains),
        continuations=len(records) - len(mains),
        identical_replay_max_pose_error=max(replay) if replay else None,
        all_attempted_episode_metrics=records,
    )
    (folder / "analysis.json").write_text(json.dumps(report, indent=2))
    plot(folder, summary, records)
    lines = [
        "# SuperDex tool-use pilot",
        "",
        "Simulator consistency and task feasibility only—not real GelSight calibration or demonstrated sim-to-real transfer.",
        "",
        "| Task | Physical valid | Nominal valid success | Force RMS [N] | Torque RMS [Nm] |",
        "|---|---:|---:|---:|---:|",
    ]
    for task, s in summary.items():

        def fmt(v):
            return "n/a" if v is None else f"{v:.4g}"

        lines.append(
            f"| {task} | {s['physically_valid']}/{s['episodes']} | {s['nominal_valid_successes']}/{s['nominal_count']} | {fmt(s['median_force_rmse'])} | {fmt(s['median_torque_rmse'])} |"
        )
    lines += [
        "",
        "Errors are medians across physically valid episodes with contact; initialized grasp is excluded. Intentional misses remain in the data. No failed episode is removed.",
        "",
        f"Captured-state identical-action replay: maximum pose-array discrepancy {report['identical_replay_max_pose_error']}. Quaternion components are dimensionless; position components are metres.",
        "",
        "Physical validity: finite state, tool/EE translation drift <12 mm and angular drift <15 degrees, no >1 mm rigid penetration lasting three samples, positive gel tetrahedron Jacobians, <1 mm rail constraint drift.",
        "",
        "Tactile validity is a separate coarse screening threshold: force RMS <0.2 N OR normalized RMS <20%, AND torque RMS <0.01 Nm. It is not a universal sensor-accuracy specification.",
        "",
        "The housing/tool collision exclusion is inherited from the existing validated GelSight assembly. The gel remains a deformable physical contact surface; the tool is not welded to the gripper.",
        "",
        "See `probes.json` for grouped learned comparisons and pause-time leakage, `timing.json` for measured collection speed, and `review/` for RGB videos and orthogonal views.",
    ]
    (folder / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "all_attempted_episode_metrics"},
            indent=2,
        )
    )


def plot(folder, summary, records):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mains = [r for r in records if not r["is_continuation"]]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for i, (key, label) in enumerate(
        (
            ("force_rmse", "Force RMS error [N]"),
            ("torque_rmse", "Torque RMS error [Nm]"),
            ("max_slip_mm", "Tool/EE drift [mm]"),
        )
    ):
        for j, task in enumerate(summary):
            rows = [r for r in mains if r["task"] == task]
            vals = [r[key] for r in rows]
            axs[i].scatter(
                j + np.linspace(-0.15, 0.15, len(rows)),
                vals,
                s=10,
                alpha=0.6,
                c=["tab:blue" if r["physical_valid"] else "tab:red" for r in rows],
            )
        axs[i].set_xticks(range(len(summary)), summary.keys(), rotation=15)
        axs[i].set_ylabel(label)
        axs[i].grid(alpha=0.2)
    fig.suptitle("All main episodes: blue = physically valid, red = rejected")
    fig.tight_layout()
    fig.savefig(folder / "quality_summary.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    analyze(parser.parse_args().folder)
