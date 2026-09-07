"""Audit all attempted runs, plot qualification and state project-relevance gates."""

import argparse
import json
from pathlib import Path
import hashlib
import shutil
import h5py
import numpy as np
from scipy.ndimage import uniform_filter1d
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def episode_audit(path):
    with h5py.File(path) as f:
        s = json.loads(f.attrs["config_json"])
        m = json.loads(f.attrs["metrics_json"])
        d = f["observations"]
        active = ~d["initialization"][:]
        truth = d["extrinsic_contact_wrench"][:][active]
        estimate = d["dynamic_inferred_extrinsic_wrench"][:][active]
        count = max(1, round(0.05 / s["dt"]))
        smooth_truth = uniform_filter1d(truth, count, axis=0, mode="nearest")
        smooth_estimate = uniform_filter1d(estimate, count, axis=0, mode="nearest")
        error = {}
        for name, sl in (("force", slice(0, 3)), ("torque", slice(3, 6))):
            rms = float(
                np.sqrt(np.mean(np.sum((truth[:, sl] - estimate[:, sl]) ** 2, axis=1)))
            )
            ref = float(np.sqrt(np.mean(np.sum(truth[:, sl] ** 2, axis=1))))
            error[name] = dict(
                raw_rmse=rms,
                reference_rms=ref,
                nrmse=rms / ref if ref > 1e-6 else None,
                mean50ms_rmse=float(
                    np.sqrt(
                        np.mean(
                            np.sum(
                                (smooth_truth[:, sl] - smooth_estimate[:, sl]) ** 2,
                                axis=1,
                            )
                        )
                    )
                ),
            )
        ref = f["tactile_reference"][:]
        field = np.stack(
            [d[f"tactile_force_field_{side}"][:][active] for side in ("left", "right")],
            axis=1,
        )
        delta = field - ref[None]
        for i, g in enumerate(sorted(f["geometry"], key=int)):
            assert int(g) == i
            assert f["geometry"][g]["vertices"].shape[1] == 3
            assert f["geometry"][g]["faces"].shape[1] == 3
        assert d["body_root_poses"].shape[1] == len(f["geometry"])
        assert "objectpointcloud" not in d and "env_point_cloud" not in d
        for side in ("left", "right"):
            assert d[f"tactile_force_field_{side}"].shape[1:] == (7, 9, 3)
        return dict(
            path=str(path),
            spec=s,
            metrics=m,
            wrench=error,
            frames=len(d["timestamps"]),
            delta_field_rms_n=float(np.sqrt(np.mean(delta**2))),
            geometry_contract_valid=True,
            prefix=path.name.endswith("_prefix.h5"),
        )


def plot_sweep(rows, path, title):
    rows = [r for r in rows if not r["prefix"]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, key, label in zip(
        axes,
        ("max_slip_mm", "max_angular_slip_deg", "max_penetration_mm"),
        ("Grasp drift [mm]", "Angular drift [deg]", "Penetration [mm]"),
    ):
        grid = np.full((3, 3), np.nan)
        for r in rows:
            s = r["spec"]
            g = (15, 25, 35).index(int(s["grip_force_n"]))
            e = (100000, 200000, 400000).index(int(s["gel_modulus_pa"]))
            grid[g, e] = r["metrics"][key]
        im = ax.imshow(grid, cmap="magma", origin="lower")
        ax.set_xticks(range(3), [100, 200, 400])
        ax.set_yticks(range(3), [15, 25, 35])
        ax.set_xlabel("Gel modulus [kPa]")
        ax.set_ylabel("Command per finger [N]")
        ax.set_title(label)
        for (i, j), v in np.ndenumerate(grid):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white")
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pointcloud_review(prefix, path):
    from decision_vision import Reconstructor, Observation

    with h5py.File(prefix) as f:
        r = Reconstructor(f)
        poses = f["observations/body_root_poses"][-1]
        configs = [
            Observation(mode="ideal"),
            Observation(),
            Observation(noise_m=0.001, patch_fraction=0.25),
        ]
        fig = plt.figure(figsize=(12, 4))
        for j, (cfg, name) in enumerate(
            zip(
                configs,
                (
                    "Ideal full mesh (oracle)",
                    "Camera-visible",
                    "Visible + noise + patch removal",
                ),
            )
        ):
            ax = fig.add_subplot(1, 3, j + 1, projection="3d")
            points, counts = r.clouds(
                poses,
                cfg,
                [
                    f[f"observations/tactile_coord_{side}"][-1]
                    for side in ("left", "right")
                ],
            )
            for p, color in zip(points, ("tab:blue", "tab:orange")):
                ax.scatter(*((p - r.target) * 1000).T, s=1, c=color)
            ax.set_title(name + f"\navailable points {counts.tolist()}")
            ax.set_xlabel("X [mm]")
            ax.set_ylabel("Y [mm]")
            ax.set_zlabel("Z [mm]")
            ax.set_box_aspect([1, 1, 1])
            ax.set_xlim(-80, 160)
            ax.set_ylim(-100, 150)
            ax.set_zlim(-80, 80)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    a = p.parse_args()
    root = a.folder
    rows = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or folder.name in (
            "source_snapshot",
            "verification",
            "review",
        ):
            continue
        for path in sorted(folder.glob("episode_*.h5")):
            rows.append(episode_audit(path))
    (root / "audit.json").write_text(json.dumps(rows, indent=2))
    groups = {}
    for r in rows:
        groups.setdefault(Path(r["path"]).parent.name, []).append(r)
    table = []
    for name, rr in groups.items():
        branch = [r for r in rr if not r["prefix"]]
        valid = [r for r in branch if r["metrics"]["physical_valid"]]
        table.append(
            f"| {name} | {len(branch)} | {len(valid)} | {sum(r['metrics']['task_success'] for r in valid)} |"
        )
    for kind in ("hook", "key"):
        if kind + "_grip" in groups:
            plot_sweep(
                groups[kind + "_grip"],
                root / f"{kind}_grip_sweep.png",
                f"{kind}: qualification attempts, not cherry-picked successes",
            )
    if "hook_grip_qualified" in groups:
        plot_sweep(
            groups["hook_grip_qualified"],
            root / "hook_operating_sweep.png",
            "Hook at 30 N/m with unloading: full 3 x 3 operating screen",
        )
    final = root / "hook_cantilever"
    prefixes = sorted(final.glob("*_prefix.h5"))
    if prefixes:
        pointcloud_review(prefixes[0], root / "pointcloud_observations.png")
    timings = []
    for path in root.glob("*/campaign.json"):
        r = json.loads(path.read_text())
        timings.append(
            dict(
                folder=path.parent.name,
                seconds=r["seconds"],
                frames=r["raw_frames"],
                fps=r["frames_per_second"],
                workers=r["workers"],
            )
        )
    (root / "timing.json").write_text(json.dumps(timings, indent=2))
    snapshot = root / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    hashes = {}
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, snapshot / path.name)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (snapshot / "hashes.json").write_text(json.dumps(hashes, indent=2))
    model_path = root / "hook_probes/probes.json"
    model = json.loads(model_path.read_text()) if model_path.exists() else {}
    physical = []
    for name in ("hook_grip_qualified", "hook_cantilever"):
        selected = [
            r
            for r in groups.get(name, [])
            if not r["prefix"] and r["metrics"]["physical_valid"]
        ]
        if selected:
            physical.append(
                f"| {name} | {len(selected)} | "
                + " | ".join(
                    f"{np.median([r['wrench'][component][field] for r in selected]):.5f}"
                    for component, field in (
                        ("force", "raw_rmse"),
                        ("force", "mean50ms_rmse"),
                        ("torque", "raw_rmse"),
                        ("torque", "mean50ms_rmse"),
                    )
                )
                + " |"
            )
    lines = [
        "# SuperDex contact-decision pilot",
        "",
        "## Outcome and scope",
        "",
        "This is a bounded local task-design study, not ACWM training, calibrated hardware sensing or sim-to-real evidence. All development attempts are retained below. Earlier geometries and loaded-hold failures are not pooled with final task revisions.",
        "",
        "| Run | Continuations | Physically valid | Valid successes |",
        "|---|---:|---:|---:|",
        *table,
        "",
        "## Physical findings",
        "",
        "- The initial hook holds produced creep; brief holds followed by unloading qualified the 30 N/m nominal case. Higher resistance did not qualify in that screen.",
        "- The original hook supports obstructed some recovery paths. Revision 2 uses a mirrored cantilever with a free end; inspect its complete branch-group gate below.",
        "- Keyed insertion/turn is implemented, but did not qualify through the two permitted geometry/controller revisions and grip/material screen. Penetration, seating failure and force-limit stops remain; it must not enter main-model training as a verified task.",
        "- Static composite fixtures now constrain their COM pose correctly. Previously, using a mesh-root position as a COM boundary target shifted asymmetric fixtures.",
        "",
        "![Hook grip sweep](hook_grip_sweep.png)",
        "",
        "![Key grip sweep](key_grip_sweep.png)",
        "",
        "## Vision and sensing",
        "",
        "The frozen official Point-M2AE encoder runs on the RTX 5090 in a separate CUDA environment. Exact simulator poses are restricted to oracles/targets. Camera-visible 1,024-point tool/environment clouds are reconstructed from mesh-root poses; no per-frame clouds are stored. Normalized pretrained features retain a separate observed metric-geometry path. Ideal segmentation and synthetic noise/patch removal remain simulation conveniences, not a real-camera calibration.",
        "",
        "![Actual point-cloud observations](pointcloud_observations.png)",
        "",
        "```json",
        json.dumps(model.get("task_gates", {}), indent=2),
        "```",
        "",
        "Complete numerical results: [audit](audit.json), [timing](timing.json), [GPU/observation audit](hook_probes/features.json), [sensing probes](hook_probes/probes.json).",
        "",
        "## Project relevance and stopping decision",
        "",
        "Retain the data/geometry/point-encoder interfaces and physically valid hook controls. Do not scale keyed insertion or claim tactile necessity from rejected mechanics. A failed branch-diversity gate means the hook is still a sensing/control diagnostic rather than evidence that tactile-dependent action selection is necessary. The next main-server experiment should use only a qualified task and compare the same observation interface with/without aggregate and dense tactile information; cross-domain and policy/data-efficiency claims remain untested.",
        "",
        "## Reproducibility",
        "",
        "See [implementation guide](../../DECISION_PILOT.md). `source_snapshot` records the final implementation; historical run manifests and revision labels identify superseded attempts. Raw contact consistency shares the same physics engine and is not independent sensor calibration. Keep episode continuations next to their prefix HDF5 files.",
    ]
    lines += [
        "",
        "## Qualified-load grip/material result",
        "",
        "All nine nominal hook runs reached the travel goal, but only the three 35 N-per-finger settings passed the retention limits. 15 N and 25 N must not be reported as successful retained grasps. Within the tested 100–400 kPa range, modulus changes were secondary to closing effort; retain 35 N and 200 kPa as the tested default, not a calibrated hardware setting.",
        "",
        "![Operating sweep](hook_operating_sweep.png)",
        "",
        "## Wrench consistency on physically valid continuations",
        "",
        "Episode medians, including free-space portions; inspect reference RMS in audit.json before interpreting low error as accuracy under load. The 50 ms filter is an offline diagnostic, not a model input.",
        "",
        "| Run | Episodes | Force raw RMSE [N] | Force 50 ms [N] | Torque raw RMSE [Nm] | Torque 50 ms [Nm] |",
        "|---|---:|---:|---:|---:|---:|",
        *physical,
        "",
        "## Videos and deterministic checks",
        "",
        "- [Qualified hook control: actual RGB + absolute/delta tactile](verification/hook_control/replay/episode_0000_pull_rgb_tactile.mp4)",
        "- [One valid cantilever recovery](verification/hook_recovery/replay/episode_0002_left_rgb_tactile.mp4) — this does not qualify its other candidate branches.",
        "- [Rejected keyed turn](verification/key_rejected/replay/episode_0006_turn_rgb_tactile.mp4) — useful for reviewing the failure, not a success demonstration.",
        "",
        "[Hook control wrench plot](verification/hook_control/replay/episode_0000_pull_wrenches.png) · [Recovery wrench plot](verification/hook_recovery/replay/episode_0002_left_wrenches.png)",
        "",
        "Timing reflects different workloads and some overlapping jobs, not a controlled CPU-core scaling benchmark. GPU encoder timings include observation reconstruction; physics remains on CPU.",
    ]
    lines += [
        "",
        "![Labeled video contact sheet](video_contact_sheet.png)",
        "",
        "Visual review sampled front/side frames at the key stages; numerical checks cover full episodes. All three review videos are nonblank, 20 fps and have the expected frame counts. See [media audit](media_audit.json).",
        "",
        "### Replay and timestep checks",
        "",
        "| Case | Exact same-dt replay | Half-dt maximum matched position difference [mm] |",
        "|---|---:|---:|",
    ]
    for path in sorted(root.glob("verification/*/verification.json")):
        result = json.loads(path.read_text())
        difference = result.get("half_dt_max_matched_position_difference_mm")
        lines.append(
            f"| {path.parent.name} | {result['exact']} | {difference:.3f} |"
            if difference is not None
            else f"| {path.parent.name} | {result['exact']} | not run |"
        )
    features_path = root / "hook_probes/features.json"
    if features_path.exists():
        features = json.loads(features_path.read_text())
        lines += [
            "",
            f"Frozen Point-M2AE observation extraction: {features['seconds']:.2f} seconds for {len(features['records'])} anchors, five observation conditions and ten history frames each; peak allocated GPU memory {features['peak_allocated_vram_mb']:.1f} MiB, reserved {features['peak_reserved_vram_mb']:.1f} MiB. This is encoding/observation time, not fitted model-training time.",
            "",
            "No fitted sensing comparison is reported because there are no complete physically valid final branch groups. This is not evidence that tactile fails to help; the task mechanics/coverage gate must be fixed first.",
        ]
    with h5py.File(root / "review_index.h5", "w") as index:
        index.attrs["purpose"] = (
            "Review index only. Includes rejected attempts; inspect labels and protocol revision before training."
        )
        for r in rows:
            path = Path(r["path"])
            g = index.require_group(path.parent.name)
            g[path.stem] = h5py.ExternalLink(str(path.relative_to(root)), "/")
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            dict(
                files=len(rows),
                raw_frames=sum(r["frames"] for r in rows),
                groups=list(groups),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
