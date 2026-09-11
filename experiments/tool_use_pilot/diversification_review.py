"""Audit completed diversified episodes; explicit visual review controls promotion."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from benchmark_report import summarize, geometry_audit


def review_media(paths, output):
    """Decode actual movies and export start/contact/peak/end RGB contact sheets.

    This prepares evidence; it does not assert that a human/agent viewed it.
    """
    import imageio_ffmpeg
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    import io

    output.mkdir(parents=True, exist_ok=False)
    records = []
    for path in paths:
        completion = json.loads(path.with_suffix(".json").read_text())
        if completion.get("error"):
            continue
        with h5py.File(path) as f:
            d = f["observations"]
            t = d["timestamps"][:]
            w = d["extrinsic_contact_wrench"][:]
            active = np.flatnonzero(t >= 4)
            force = np.linalg.norm(w[:, :3], axis=1)
            contact = active[force[active] > 0.1]
            s = completion["spec"]
            load = (
                np.linalg.norm(w[:, 3:], axis=1) if s["family"] == "turning" else force
            )
            peak = active[np.argmax(load[active])] if len(active) else 0
            indices = [
                0,
                int(contact[0]) if len(contact) else len(t) // 2,
                int(peak),
                len(t) - 1,
            ]
            times = t[indices].tolist()
        movie = path.with_name(path.stem + "_rgb_tactile.mp4")
        stream = imageio_ffmpeg.read_frames(str(movie))
        try:
            meta = next(stream)
        finally:
            stream.close()
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-v",
                "error",
                "-xerror",
                "-i",
                str(movie),
                "-f",
                "null",
                "-",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if abs(meta["duration"] - float(t[-1])) > 0.15:
            raise ValueError(f"Video duration mismatch: {movie}")
        images = []
        for time in times:
            seek = min(time, meta["duration"] - 1 / meta["fps"])
            png = subprocess.check_output(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(),
                    "-v",
                    "error",
                    "-ss",
                    str(max(0, seek)),
                    "-i",
                    str(movie),
                    "-frames:v",
                    "1",
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "png",
                    "-",
                ]
            )
            images.append(np.asarray(Image.open(io.BytesIO(png)))[:, :800])
        records.append(
            dict(
                episode=str(path),
                spec=s,
                times=times,
                images=images,
                duration=meta["duration"],
            )
        )
        print(f"Decoded {path.stem}", flush=True)
    for start in range(0, len(records), 4):
        batch = records[start : start + 4]
        fig, axes = plt.subplots(
            len(batch), 4, figsize=(20, 4.4 * len(batch)), squeeze=False
        )
        for row, record in enumerate(batch):
            s = record["spec"]
            for col, (pixels, time) in enumerate(
                zip(record["images"], record["times"])
            ):
                axes[row, col].imshow(pixels)
                axes[row, col].set_title(
                    f"#{s['episode_id']} {s['variant']} / {s.get('diversification_condition')}\n{('start', 'contact', 'peak', 'end')[col]} {time:.2f}s",
                    fontsize=10,
                )
                axes[row, col].axis("off")
        fig.tight_layout()
        fig.savefig(output / f"page_{start // 4}.png", dpi=110)
        plt.close(fig)
    (output / "media.json").write_text(
        json.dumps(
            [{k: v for k, v in r.items() if k != "images"} for r in records], indent=2
        )
        + "\n"
    )


def repair_margin(spec, metrics):
    """Stronger forward-looking targets; never rewrite historical success flags."""
    if spec["family"] == "surface" and any(
        x in spec["variant"] for x in ("wall", "slot")
    ):
        requested = 1000 * (spec.get("surface_stroke_m") or 0.020)
        error = abs(metrics.get("follow_tip_travel_mm", float("inf")) - requested)
        contact = metrics.get("guide_contact_continuity", 0)
        return dict(
            passed=bool(
                metrics.get("imitation_eligible") and error <= 2.0 and contact >= 0.95
            ),
            endpoint_error_mm=error,
            guide_contact_fraction=contact,
            target="20 mm nominal stroke +/-2 mm; >=95% guide contact; otherwise full existing acceptance",
        )
    if spec["family"] == "pushing" and spec["variant"] == "pose":
        position = 1000 * metrics.get("final_position_error_m", float("inf"))
        yaw = abs(metrics.get("final_yaw_error_deg", float("inf")))
        return dict(
            passed=bool(
                metrics.get("imitation_eligible") and position <= 5.0 and yaw <= 3.0
            ),
            position_error_mm=position,
            yaw_error_deg=yaw,
            target="<=5 mm and <=3 degrees; otherwise full existing acceptance",
        )
    return None


def recorded_geometry_id(path):
    """Exact recorded tool/environment geometry, independent of task labels.

    Use this alongside manifest parameter IDs when forming future held-out splits.
    """
    fingerprint = hashlib.sha256()
    with h5py.File(path) as f:
        for key in sorted(f["geometry"], key=int):
            g = f["geometry"][key]
            if g.attrs["role"] not in ("object", "environment"):
                continue
            fingerprint.update(str(g.attrs["role"]).encode())
            for name in ("vertices", "faces"):
                array = g[name][:]
                fingerprint.update(str((array.shape, str(array.dtype))).encode())
                fingerprint.update(array.tobytes())
    return fingerprint.hexdigest()


def housing_clearance(path):
    """Actual housing and recorded robot links, not overlapping contact proxies.

    Eleven frames; one-way containment for the open housing mesh. Not swept CCD.
    """
    asset = (
        Path(__file__).resolve().parents[2]
        / "assets/bots/grippers/franka_gelsight_mini/generated/housing_visual.glb"
    )
    housing = trimesh.load(asset, force="mesh")
    maximum = 0.0
    robot_maximum = 0.0
    hits = []
    with h5py.File(path) as f:
        geometry = f["geometry"]
        targets = {
            int(k): trimesh.Trimesh(g["vertices"][:], g["faces"][:], process=False)
            for k, g in geometry.items()
            if g.attrs["role"] in ("object", "environment")
        }
        housings = [
            int(k)
            for k, g in geometry.items()
            if str(g.attrs["name"]).endswith("_gelsight_housing")
        ]
        if len(housings) != 2:
            raise ValueError("Missing housing poses")
        parts = {h: housing.vertices for h in housings}
        robot_parts = {
            int(k): g["vertices"][:]
            for k, g in geometry.items()
            if g.attrs["role"] == "occluder" and int(k) not in housings
        }
        if not robot_parts:
            raise ValueError("Missing recorded robot link geometry")
        parts.update(robot_parts)
        poses = f["observations/body_root_poses"]
        for frame in np.linspace(0, len(poses) - 1, 11, dtype=int):
            q = poses[frame]
            for h, vertices in parts.items():
                points = Rotation.from_quat(q[h, 3:]).apply(vertices) + q[h, :3]
                for i, mesh in targets.items():
                    if not mesh.is_watertight:
                        raise ValueError("Housing audit target is not closed")
                    local = Rotation.from_quat(q[i, 3:]).inv().apply(points - q[i, :3])
                    bounds = mesh.bounds
                    candidates = local[
                        np.all((local >= bounds[0]) & (local <= bounds[1]), axis=1)
                    ]
                    if not len(candidates):
                        continue
                    inside = candidates[mesh.contains(candidates)]
                    if len(inside):
                        _, distance, _ = trimesh.proximity.closest_point(mesh, inside)
                        overlap = float(distance.max() * 1000)
                        if h in housings:
                            maximum = max(maximum, overlap)
                        else:
                            robot_maximum = max(robot_maximum, overlap)
                        if overlap > 0.01:
                            hits.append(
                                dict(
                                    frame=int(frame),
                                    body=h,
                                    kind="housing" if h in housings else "robot",
                                    target=i,
                                    overlap_mm=overlap,
                                )
                            )
    return dict(
        max_overlap_mm=maximum,
        robot_max_overlap_mm=robot_maximum,
        passed=max(maximum, robot_maximum) < 0.1,
        hits=hits,
        asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
        note="Actual housing and recorded robot-link vertices against closed tools/fixtures at 11 frames; one-way containment, not a swept-volume guarantee.",
    )


def review(root, visual_ids):
    entries, rows, errors = [], [], []
    for completion_path in sorted(root.glob("*/*.json")):
        completion = json.loads(completion_path.read_text())
        if (
            isinstance(completion, dict)
            and "spec" in completion
            and completion.get("error")
        ):
            errors.append(
                dict(
                    episode=str(completion_path.with_suffix(".h5")),
                    error=completion["error"],
                )
            )
    for path in sorted(root.glob("*/*.h5")):
        if not path.with_suffix(".json").exists():
            continue  # Never read an in-progress HDF5.
        completion = json.loads(path.with_suffix(".json").read_text())
        if completion.get("error"):
            continue  # Failed serialization is not a complete accepted record.
        try:
            row = summarize(path)
            s, m = row["spec"], row["metrics"]
            geom = geometry_audit(path)
            housing = housing_clearance(path)
            entries.append(
                dict(
                    episode=str(path.relative_to(root)),
                    recorded_geometry_id=recorded_geometry_id(path),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    visual_review_passed=s["episode_id"] in visual_ids,
                    housing_clearance_passed=housing["passed"],
                    housing=housing,
                    geometry=geom,
                    repair_margin=repair_margin(s, m),
                )
            )
            rows.append(row)
        except Exception as exc:
            errors.append(dict(episode=str(path), error=str(exc)))
    ledger = (
        json.loads((root / "attempts.json").read_text())
        if (root / "attempts.json").exists()
        else []
    )
    result = dict(
        episodes=entries,
        errors=errors,
        note="Visual review is an explicit assertion after viewing RGB; never inferred from task success.",
    )
    (root / "review.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Diversification pilot — current evidence",
        "",
        f"Reserved: {len(ledger)}; audited: {len(rows)}; audit errors: {len(errors)}.",
        "",
        "Pending/reserved episodes are not failures or successes. Wrench agreement is not an acceptance gate.",
        "",
        "| ID | Stage / task / condition | Goal | Physical/full | Visual | Repair margin | Drift mm / deg | Force / torque RMSE |",
        "|---|---|---|---|---|---|---:|---:|",
    ]
    for row, entry in zip(rows, entries):
        s, m = row["spec"], row["metrics"]
        lines.append(
            f"| {s['episode_id']} | {s.get('diversification_stage')} / {s['family']}/{s['variant']} / {s.get('diversification_condition')} | {m['task_success']} | {m['physical_valid']} / {m['full_rollout_complete']} | {entry['visual_review_passed']} | {entry['repair_margin']['passed'] if entry['repair_margin'] else 'n/a'} | {m['max_slip_mm']:.2f} / {m['max_angular_slip_deg']:.2f} | {m['force_rmse']:.3f} N / {m['torque_rmse']:.5f} Nm |"
        )
    lines += [
        "",
        "Geometry and housing audit details: [review.json](review.json).",
        "This is a feasibility/sensitivity pilot, not calibrated sensor fidelity or learned sim-to-real evidence.",
    ]
    (root / "DIVERSIFICATION_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(audited=len(rows), errors=errors)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument(
        "--prepare-media",
        type=Path,
        help="Prepare RGB sheets in a new directory, without approving or changing review.json",
    )
    p.add_argument(
        "--media-ids",
        type=int,
        nargs="*",
        help="Limit media preparation to these completed episode IDs",
    )
    p.add_argument(
        "--visual-reviewed",
        type=int,
        nargs="*",
        default=[],
        help="Only IDs whose actual RGB has been inspected; pass the complete reviewed set",
    )
    args = p.parse_args()
    if args.prepare_media:
        paths = []
        for path in sorted(args.root.glob("*/*.h5")):
            if not path.with_suffix(".json").exists():
                continue
            completion = json.loads(path.with_suffix(".json").read_text())
            if completion.get("error"):
                continue
            if (
                args.media_ids is None
                or completion["spec"]["episode_id"] in args.media_ids
            ):
                paths.append(path)
        review_media(paths, args.prepare_media)
    else:
        review(args.root, set(args.visual_reviewed))
