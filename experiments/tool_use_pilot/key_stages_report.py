"""Measured stage outcomes and independent geometry review; no fitted probes."""

import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hook_repair_report import sampled_overlap_audit
from visuals import plot_episode


def media_review(folder, rows):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    sheet = Image.new("RGB", (1600, 260 * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    audits = []
    for row_index, row in enumerate(rows):
        path = Path(row["path"])
        video = path.with_name(path.stem + "_rgb_tactile.mp4")
        if not video.exists():
            continue
        reader = imageio.get_reader(video)
        count = reader.count_frames()
        fps = reader.get_meta_data()["fps"]
        times = (0, 8, 10.5, 12, 16) if row["spec"]["stage"] == "insertion" else (0, 10.5, 16, 20, 25)
        deviations = []
        for col, t in enumerate(times):
            frame = reader.get_data(min(count-1, round(t*fps)))
            rgb = frame[8:608, :800]
            deviations.append(float(rgb.std()))
            sheet.paste(Image.fromarray(rgb).resize((320, 240)), (col*320, row_index*260+20))
            draw.text((col*320+5, row_index*260+3), f"{row['spec']['stage']} / {row['spec']['case']} t={t}s", fill="black")
            if row["spec"]["case"] == "aligned" and t == (10.5 if row["spec"]["stage"] == "insertion" else 16):
                Image.fromarray(frame).save(path.with_name(path.stem + "_peak_review.png"))
        reader.close()
        audits.append(dict(video=str(video.resolve()), fps=fps, frames=count, seconds=count/fps,
                           expected_frames=round(row["spec"]["duration"]*20), sampled_rgb_std=deviations))
    sheet.save(folder / "video_contact_sheet.png")
    (folder / "media_audit.json").write_text(json.dumps(audits, indent=2))


def refresh_metrics(path):
    """Recompute labels from saved geometry; preserve original labels and physics."""
    from key_stages import KeyStageSpec, measured_seat, stage_audit
    from scipy.spatial.transform import Rotation
    with h5py.File(path, "a") as f:
        if f.attrs.get("seat_metric_revision") == "exact_blade_vertices_v2":
            return
        spec = KeyStageSpec(**json.loads(f.attrs["config_json"]))
        d = {k: v[:] for k, v in f["observations"].items()}
        d["phase"] = d["phase"].astype(str)
        vertices = f["geometry/0/vertices"][:]
        blade = vertices[vertices[:, 2] <= -.061 + 1e-8]
        measured = []
        for poses in d["body_root_poses"]:
            rotations = [Rotation.from_quat(p[3:]) for p in poses[:3]]
            measured.append(measured_seat(blade, poses[0, :3], rotations[0],
                poses[1, :3], rotations[1], poses[2, :3], rotations[2]))
        history = f.require_group("metric_history/v1_bounding_box")
        history.attrs["metrics_json"] = f.attrs["metrics_json"]
        for key in ("seat_valid", "insertion_depth_m", "task_progress", "rewards"):
            history.create_dataset(key, data=d[key], compression="lzf")
        d["seat_valid"] = np.array([x[0] for x in measured])
        d["insertion_depth_m"] = np.array([x[1] for x in measured])
        if spec.stage == "insertion":
            d["task_progress"] = d["insertion_depth_m"].copy()
            d["rewards"] = d["task_progress"].copy()
        previous = json.loads(f.attrs["metrics_json"])
        result = stage_audit(d, spec, previous["abort_reason"])
        for key in ("seat_valid", "insertion_depth_m", "task_progress", "rewards"):
            f["observations"][key][:] = d[key]
        for key in f["labels"]:
            if key in result:
                f["labels"][key][...] = result[key]
        f.attrs["metrics_json"] = json.dumps(result)
        f.attrs["seat_metric_revision"] = "exact_blade_vertices_v2"
        f.attrs["metric_implementation_hash"] = hashlib.sha256(Path(__file__).with_name("key_stages.py").read_bytes()).hexdigest()
        f.attrs["metric_correction_note"] = "Actual chamfered blade vertices replace an oversized bounding box. Same tolerances. Physics, actions and tactile arrays unchanged; original labels preserved in metric_history."
    summary_path = path.with_suffix(".json")
    summary = json.loads(summary_path.read_text())
    summary["original_bounding_box_metrics"] = previous
    summary["metrics"] = result
    summary["seat_metric_revision"] = "exact_blade_vertices_v2"
    summary_path.write_text(json.dumps(summary, indent=2))


def main(folder, mesh_audit=False):
    paths = sorted(folder.glob("key_*.h5"))
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for path in paths:
        with h5py.File(path) as f:
            spec = json.loads(f.attrs["config_json"])
            result = json.loads(f.attrs["metrics_json"])
            d = {k: v[:] for k, v in f["observations"].items()}
        plot_episode(d, path.with_name(path.stem + "_wrenches.png"), "key", spec["stage"])
        row = dict(path=str(path.resolve()), spec=spec, metrics=result)
        if mesh_audit:
            row["mesh_overlap_samples"] = sampled_overlap_audit(path)
            row["sampled_overlap_max_mm"] = max(r["sampled_penetration_mm"] for r in row["mesh_overlap_samples"])
        rows.append(row)
        ax = axes[0 if spec["stage"] == "insertion" else 1]
        t = d["timestamps"]
        progress = d["insertion_depth_m"]*1000 if spec["stage"] == "insertion" else np.rad2deg(d["rotor_angle_rad"])
        name = f"{spec['case']} {spec['episode_id']}"
        ax[0].plot(t, progress, label=name)
        ax[1].plot(t, np.linalg.norm(d["extrinsic_contact_wrench"][:, :3], axis=1), label=name)
        ax[2].plot(t, np.linalg.norm(d["extrinsic_contact_wrench"][:, 3:], axis=1), label=name)
    for i, stage in enumerate(("Insertion", "Turning")):
        for j, unit in enumerate(("depth [mm]" if i == 0 else "rotor [deg]", "contact force [N]", "contact torque [Nm]")):
            axes[i, j].set_title(stage + ": " + unit)
            axes[i, j].set_xlabel("time [s]")
            axes[i, j].grid(alpha=.2)
        axes[i, 0].legend(fontsize=8)
        axes[i, 0].axhline(14 if i == 0 else 90, color="gray", ls=":")
    fig.tight_layout(); fig.savefig(folder / "stage_comparison.png", dpi=170); plt.close(fig)
    text = ["# Independent key insertion and turning", "",
            "These are two separately reset mechanics tasks, not two labels on the old combined rollout. No learned policy or tactile-benefit claim is made.", "",
            "| Stage | Initial condition | Action | Success |", "|---|---|---|---|",
            "| Insertion | Prepared grasp, blade above mouth; socket rotor fixed | Translate down, hold, withdraw; zero commanded rotation | ≥14 mm blade-bottom depth with full blade inside pocket bounds for ≥0.3 s |",
            "| Turning | Prepared grasp with blade seated; seat verified before action | Rotate held key, hold, unturn; zero nominal insertion translation | Contact-driven socket rotor 90 ±5°, blade still seated for ≥0.3 s |", "",
            "No hidden tool weld after 1 s; only the robot/gripper drives the tool. Turning resets to a prepared seated state: it does NOT establish successful insertion or a tested insertion-to-turn handoff. The rotor receives only spring/damping resistance, never a commanded target. Insertion intentionally locks the rotor to isolate seating mechanics.", "",
            "## Results", "",
            "| Stage / case | Physical valid | Success | Max depth (mm) | Max rotor (°) | Peak force (N) | Max contact penetration (mm) | Abort |",
            "|---|---|---|---:|---:|---:|---:|---|"]
    for row in rows:
        s, m = row["spec"], row["metrics"]
        text.append(f"| {s['stage']} / {s['case']} {s['episode_id']} | {m['physical_valid']} | {m['task_success']} | {m['max_insertion_depth_mm']:.2f} | {m['max_rotor_angle_deg']:.2f} | {m['peak_extrinsic_force_n']:.2f} | {m['max_penetration_mm']:.3f} | {m['abort_reason']} |")
    text += ["", f"{sum(r['metrics']['physical_valid'] for r in rows)}/{len(rows)} physically valid; {sum(r['metrics']['task_success'] for r in rows)} successful outcomes. Failed controls remain in the report and HDF5 files. This is a small local verification, not a measured general success rate."]
    text += ["", "![Stage comparison](stage_comparison.png)", "", "## Sensing and validity", "",
             "Dense 2 × 7 × 9 × 3 gel fields, surface coordinates, tactile-derived and direct contact wrenches, actual EE actions, poses and stage-specific labels are saved in HDF5. No PCD arrays are added: the existing camera/mesh/body-pose reconstruction interface is retained. `task_kind` is `key_insertion` or `key_turning`; progress units are respectively metres and radians. Legacy combined key data are unchanged. Seating uses the actual chamfered blade vertices with 0.1 mm lateral tolerance, not the oversized rectangular bounding box. Corrected review copies preserve original labels under `metric_history`; physics, actions and tactile data are unchanged.", "",
             "Stage collector uses bounded privileged tool-pose feedback and a GT contact-force insertion guard, explicitly saved and excluded from sensor-policy claims. Orientation correction is capped at 12 degrees and 2 degrees/s; translation correction at 12 mm and 3 mm/s. Turning uses tool-root-pivot compensation for the 161 mm wrist/tool offset. Rigid contact transition half-width is 0.05 mm, threshold 0.02 mm, penalty 5e10; strict reported penetration gate remains 0.3 mm. Independent sampled mesh intersections are diagnostics, not a proof of no intersection. Inspect both checks and videos before qualification.", "",
             "| Stage / case | Force RMSE (N) | Torque RMSE (Nm) | Grasp drift (mm / °) | Sampled overlap (mm) |",
             "|---|---:|---:|---|---:|"]
    for row in rows:
        s, m = row["spec"], row["metrics"]
        overlap = row.get("sampled_overlap_max_mm")
        text.append(f"| {s['stage']} / {s['case']} {s['episode_id']} | {m['force_rmse']:.4f} | {m['torque_rmse']:.5f} | {m['max_slip_mm']:.2f} / {m['max_angular_slip_deg']:.2f} | {overlap if overlap is not None else 'not run'} |")
    text += ["", "Wrench RMSE uses active frames and the existing gel+dynamics estimate about tool COM versus direct tool/environment contact. It is a simulator consistency diagnostic, not a deployable GT-free sensor estimator or real GelSight accuracy claim.", "", "## Review artifacts", ""]
    text += ["Nominal insertion and turning have whole-episode force RMSE around 0.205 N, narrowly above the existing 0.2 N absolute tactile gate; their `tactile_valid` flags remain false. Insertion/turn action-phase RMSE is lower; withdrawal/unturn transients contribute substantially. Mechanics qualification must not be confused with satisfying every tactile-fidelity threshold.", "",
             "Insertion uses 8 mm maximum collision-triangle edges; turning uses 16 mm, with identical external solid geometry. The bounded turning trials use the coarser tessellation to avoid excessive contact-processing cost; this is not a mesh-convergence proof. Raw simulation implementation hashes are preserved; `source_snapshot` contains the current reproducible implementation and corrected seating metric.", ""]
    for path in folder.rglob("*_rgb_tactile.mp4"):
        text.append(f"- [{path.stem}]({path.relative_to(folder)})")
    text += ["", "![Sampled video review](video_contact_sheet.png)", "", "Visual inspection samples the stages shown above; numerical checks use full recorded episodes. Camera images are captured simulator RGB with dense tactile/shear panels, not point-cloud-only renderings."]
    text += ["", "## Project relevance", "",
             "Separating these stages makes insertion alignment/contact uncertainty distinguishable from rotational load transmission. Useful next measurements are axial load/depth for insertion and torque/angle/slip for turning; hidden friction or torsional stiffness can test action-conditioned physical prediction. Do not scale collection or fit tactile-advantage probes on a stage that fails mechanics validation. Prepared turning alone is not evidence for a complete key-use skill."]
    (folder / "REPORT.md").write_text("\n".join(text)+"\n")
    (folder / "audit.json").write_text(json.dumps(rows, indent=2))
    media_review(folder, rows)
    print(folder / "REPORT.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--mesh-audit", action="store_true")
    parser.add_argument("--refresh-metrics", action="store_true", help="Correct review-copy labels from actual saved blade vertices; original labels are preserved")
    args = parser.parse_args()
    if args.refresh_metrics:
        for path in sorted(args.folder.glob("key_*.h5")):
            refresh_metrics(path)
    main(args.folder, args.mesh_audit)
