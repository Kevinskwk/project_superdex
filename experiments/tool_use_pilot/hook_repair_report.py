"""Small hook qualification report; no training or large-scale collection."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import trimesh
from decision_report import episode_audit
from pilot import tool_mesh


def sampled_overlap_audit(path):
    """Independent mesh check at extrema, recovery waypoints and uniform times.

    Surface sampling is not a proof of zero intersection; it checks for gross
    missed geometry independently of the simulator's contact sample distances.
    """
    with h5py.File(path) as f:
        d = f["observations"]
        frames = sorted(
            set(
                [
                    int(np.argmax(d["rigid_penetration_m"][:])),
                    int(np.argmax(d["task_progress"][:])),
                    int(
                        np.argmax(
                            np.linalg.norm(d["extrinsic_contact_wrench"][:, :3], axis=1)
                        )
                    ),
                ]
            )
        )
        frames = sorted(
            set(
                frames
                + np.linspace(0, len(d["timestamps"]) - 1, 12, dtype=int).tolist()
                + [
                    int(np.argmin(abs(d["trajectory_time"][:] - t)))
                    for t in (8, 10, 12, 19.5, 26.5, 28.5)
                ]
            )
        )
        meshes = []
        samples = []
        for k in ("0", "1", "2"):
            g = f["geometry"][k]
            m = trimesh.Trimesh(g["vertices"][:], g["faces"][:], process=False)
            assert m.is_watertight, f"Non-closed review geometry: {path}, body {k}"
            meshes.append(m)
            points, _ = trimesh.sample.sample_surface(m, 1000, seed=42)
            samples.append(np.concatenate([m.vertices, points]))
        results = []
        for frame in frames:
            poses = d["body_root_poses"][frame]
            rotations = [Rotation.from_quat(p[3:]) for p in poses[:3]]
            peak = 0.0
            for source, target in ((0, 1), (1, 0), (0, 2), (2, 0)):
                world = rotations[source].apply(samples[source]) + poses[source, :3]
                local = rotations[target].inv().apply(world - poses[target, :3])
                inside = meshes[target].contains(local)
                if inside.any():
                    _, dist, _ = trimesh.proximity.closest_point(
                        meshes[target], local[inside]
                    )
                    peak = max(peak, float(dist.max()))
            results.append(
                dict(
                    time_s=float(d["timestamps"][frame]),
                    sampled_penetration_mm=peak * 1000,
                )
            )
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--mesh-audit", action="store_true")
    args = parser.parse_args()
    snapshot = args.folder / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    hashes = {}
    for name in (
        "pilot.py",
        "decisions.py",
        "visuals.py",
        "hook_repair_campaign.py",
        "hook_repair_report.py",
        "hook_repair_specs.json",
        "test_decisions.py",
    ):
        source = Path(__file__).with_name(name)
        shutil.copy2(source, snapshot / name)
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    (snapshot / "sha256.json").write_text(json.dumps(hashes, indent=2))
    for replay in args.folder.glob("verification/*/replay/episode_*.json"):
        info = json.loads(replay.read_text())
        if not info["branches"]:
            continue
        spec = info["spec"]
        branch = info["branches"][0]["branch"]
        name = f"episode_{spec['episode_id']:04d}_{branch}.h5"
        source = args.folder / name
        if not source.exists():
            continue
        errors = {}
        with h5py.File(source) as reference, h5py.File(replay.parent / name) as other:
            for key in (
                "tool_pose",
                "body_root_poses",
                "tactile_force_field_left",
                "tactile_force_field_right",
                "timestamps",
                "actions",
                "scripted_pose_correction_m",
            ):
                x, y = reference["observations"][key][:], other["observations"][key][:]
                errors[key] = float(np.abs(x - y).max()) if x.shape == y.shape else None
        verification = dict(
            source=str(source),
            exact=all(v == 0 for v in errors.values()),
            replay_max_errors=errors,
            replay=info,
        )
        (replay.parent.parent / "verification.json").write_text(
            json.dumps(verification, indent=2)
        )
    rows = [episode_audit(p) for p in sorted(args.folder.glob("episode_*.h5"))]
    branches = [r for r in rows if not r["prefix"]]
    prefixes = [r for r in rows if r["prefix"]]
    for r in branches:
        with h5py.File(r["path"]) as f:
            d = f["observations"]
            truth = d["extrinsic_contact_wrench"][:]
            estimate = d["dynamic_inferred_extrinsic_wrench"][:]
            contact = np.linalg.norm(truth[:, :3], axis=1) > 0.2
            r["loaded_contact_frames"] = int(contact.sum())
            r["loaded_wrench"] = {}
            if contact.any():
                for channel, sl in [("force", slice(0, 3)), ("torque", slice(3, 6))]:
                    error = estimate[contact, sl] - truth[contact, sl]
                    rms = float(np.sqrt(np.mean(np.sum(error**2, axis=1))))
                    ref = float(
                        np.sqrt(np.mean(np.sum(truth[contact, sl] ** 2, axis=1)))
                    )
                    r["loaded_wrench"][channel] = dict(
                        rmse=rms, nrmse=rms / ref if ref > 1e-6 else None
                    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    mesh = tool_mesh("hook")
    axes[0].triplot(
        mesh.vertices[:, 0] * 1000, mesh.vertices[:, 2] * 1000, mesh.faces, lw=0.5
    )
    axes[0].set(
        title="Repaired solid hook (24 mm thick)",
        xlabel="x [mm]",
        ylabel="z [mm]",
        aspect="equal",
    )
    for r in prefixes:
        with h5py.File(r["path"]) as f:
            d = f["observations"]
            rot = Rotation.from_matrix(
                f.attrs["action_frame_rotation_world"].reshape(3, 3)
            )
            root = d["body_root_poses"][:, 0, :3]
            actual = rot.inv().apply(root - root[0])
            nominal = d["nominal_tool_target_m"][:]
            error = (actual - nominal) * 1000
            r["anchor_tracking_error_mm"] = error[-1].tolist()
            axes[1].plot(
                d["timestamps"][:], error[:, 1], label=str(r["spec"]["episode_id"])
            )
    axes[1].set(
        title="Prefix lateral tracking error",
        xlabel="Time [s]",
        ylabel="Tool y error [mm]",
    )
    axes[1].legend(title="Episode")
    for r in branches:
        m = r["metrics"]
        axes[2].scatter(
            m["max_slip_mm"],
            m["max_penetration_mm"],
            marker="o" if m["task_success"] else "x",
            c="tab:green" if m["physical_valid"] else "tab:red",
        )
    axes[2].set(
        title="Green=valid; circle=success, x=failure",
        xlabel="Max grasp drift [mm]",
        ylabel="Max penetration [mm]",
    )
    fig.tight_layout()
    fig.savefig(args.folder / "qualification.png", dpi=160)
    plt.close(fig)
    lines = [
        "# Hook alignment and recovery verification",
        "",
        "The scripted collector uses bounded simulator tool-pose feedback. This is an oracle controller diagnostic, not evidence of a learned tactile policy or a tactile advantage. Nominal targets and actual EE commands are both recorded; corrections are oracle channels.",
        "",
        "The hook is a single watertight J solid with its formerly missing bottom corner filled. The same mesh is used for collision, RGB and point-cloud reconstruction.",
        "",
        "| Episode | Intended state | Branch | Physically valid | Success | Drift mm | Rotation deg | Penetration mm |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for r in branches:
        s, m = r["spec"], r["metrics"]
        lines.append(
            f"| {s['episode_id']} | {s['case']} | {s['branch']} | {m['physical_valid']} | {m['task_success']} | {m['max_slip_mm']:.2f} | {m['max_angular_slip_deg']:.2f} | {m['max_penetration_mm']:.3f} |"
        )
    good = [
        r
        for r in branches
        if r["metrics"]["physical_valid"] and r["metrics"]["task_success"]
    ]
    if args.mesh_audit:
        for r in branches:
            r["independent_mesh_overlap"] = sampled_overlap_audit(r["path"])
    lines += [
        "",
        f"Physically valid: {sum(r['metrics']['physical_valid'] for r in branches)}/{len(branches)}; valid successes: {len(good)}.",
        "",
        "Wrench comparison below includes only physically valid successful branches; world-frame force and torque about instantaneous tool COM, using the dynamics-corrected gel-field estimate.",
    ]
    groups = {}
    for r in branches:
        groups.setdefault(r["spec"]["group"], []).append(r)
    complete = [
        rs
        for rs in groups.values()
        if len(rs) == 4 and all(r["metrics"]["physical_valid"] for r in rs)
    ]
    lines += [
        "",
        f"Complete physically valid four-action groups: {len(complete)}/{len(groups)}.",
    ]
    if len(complete) == len(groups) and complete:
        constant = max(
            sum(
                any(
                    r["spec"]["branch"] == b and r["metrics"]["task_success"]
                    for r in rs
                )
                for rs in complete
            )
            for b in ("pull", "left", "right", "hold")
        ) / len(complete)
        lines.append(
            f"Best constant-action success across these starting states: {constant:.0%}. This small deterministic mechanics audit is not a statistical policy evaluation."
        )
    for r in prefixes:
        lines.append(
            f"- Episode {r['spec']['episode_id']} anchor xyz tracking error [mm]: {np.array(r['anchor_tracking_error_mm']).round(3).tolist()}."
        )
    if good:
        for channel, unit in [("force", "N"), ("torque", "Nm")]:
            for mode in ["raw_rmse", "mean50ms_rmse"]:
                value = np.median([r["wrench"][channel][mode] for r in good])
                lines.append(f"- Median {channel} {mode}: {value:.5f} {unit}.")
            loaded = [
                r["loaded_wrench"][channel]
                for r in good
                if channel in r["loaded_wrench"]
            ]
            if loaded:
                lines.append(
                    f"- Loaded frames only (extrinsic force >0.2 N), median {channel} RMSE: {np.median([v['rmse'] for v in loaded]):.5f} {unit}; median NRMSE: {np.median([v['nrmse'] for v in loaded]):.1%}."
                )
    if args.mesh_audit and branches:
        peak = max(
            x["sampled_penetration_mm"]
            for r in branches
            for x in r["independent_mesh_overlap"]
        )
        lines.append(
            f"\nIndependent bidirectional surface-sample overlap check across all branches at extrema, recovery waypoints and 12 uniform times: maximum {peak:.3f} mm. This is a sampled geometry audit, not a proof of zero intersection."
        )
    lines += [
        "",
        "![Mesh and qualification](qualification.png)",
        "",
        "Project relevance: repair repeatable engagement and valid recovery before fitting sensor probes. Preserve misses and wrong actions as physical failures; do not train on interpenetration or compensate away fixture offsets.",
    ]
    for clip in sorted(args.folder.glob("verification/*/replay/*_rgb_tactile.mp4")):
        lines.append(
            f"- [{clip.parent.parent.name}: actual RGB + tactile]({clip.relative_to(args.folder)})"
        )
        plot = clip.with_name(clip.name.replace("_rgb_tactile.mp4", "_wrenches.png"))
        if plot.exists():
            lines.append(
                f"- [{clip.parent.parent.name}: wrench comparison]({plot.relative_to(args.folder)})"
            )
    if (args.folder / "timing.json").exists():
        timing = json.loads((args.folder / "timing.json").read_text())
        lines.append(
            f"\nBranch campaign wall time: {timing['seconds']:.1f} s using {timing['workers']} CPU worker processes; render verification is separate."
        )
    (args.folder / "REPORT.md").write_text("\n".join(lines) + "\n")
    (args.folder / "audit.json").write_text(json.dumps(rows, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
