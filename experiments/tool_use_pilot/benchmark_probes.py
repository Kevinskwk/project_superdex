"""Qualification-first Point-M2AE probe for the diversified turning fixture."""

from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
import time
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from key_observability import (
    qualify,
    fit,
    conditions,
    summary,
    regression_summary,
    paired_gain,
)
from hook_observability import (
    Reconstructor,
    FrozenPointM2AE,
    normalize_and_metric,
    corrupt,
    augmentation_seed,
    summarize_history,
    tactile_features,
    wrench_features,
    design,
    observed_relations,
)
from benchmark_observations import tactile_history, TACTILE_CONDITIONS
from benchmark_geometry import annotate_friction


def observation_preview(path, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with h5py.File(path) as f:
        spec = json.loads(f.attrs["config_json"])
        d = f["observations"]
        index = int(np.argmin(abs(d["timestamps"][:] - 12)))
        recon = Reconstructor(f)
        raw = recon.visible(
            d["body_root_poses"][index],
            0,
            [d["tactile_coord_" + s][index] for s in ("left", "right")],
        )
        forward = recon.target - recon.eyes[0]
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 0, 1])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharex=True, sharey=True)
        for ax, cfg in zip(axes, [conditions("turning")[i] for i in (0, 2, 3, 5)]):
            clouds, counts = corrupt(
                raw,
                recon.eyes[0],
                recon.target,
                cfg,
                augmentation_seed(spec["camera_seed"], 0),
                9,
            )
            for points, count, color in zip(clouds, counts, ("tab:blue", "gray")):
                if count:
                    rays = points - recon.eyes[0]
                    uv = np.c_[rays @ right, rays @ up] / (rays @ forward)[:, None]
                    ax.scatter(uv[:, 0], uv[:, 1], s=2, c=color, alpha=0.5)
            ax.set_title(cfg.name.replace("_", "\n"), fontsize=9)
            ax.set_aspect("equal")
            ax.set_xlabel(f"Visible object/env samples: {counts}")
        fig.suptitle(
            "Actual ray-visible point clouds at 12 s; blue tool, gray environment"
        )
        fig.tight_layout()
        fig.savefig(output, dpi=160)
        plt.close(fig)


def target_record(path):
    actual_friction = annotate_friction(path)
    with h5py.File(path) as f:
        s = json.loads(f.attrs["config_json"])
        m = json.loads(f.attrs["metrics_json"])
        d = f["observations"]
        nominal = d["trajectory_time"][:]
        selected = np.flatnonzero((nominal >= 16) & (nominal <= 17))
        complete = len(selected) >= round(0.9 / (s["dt"] * s["speed_scale"]))
        moment = None
        if complete:
            rotor_index = next(
                int(k)
                for k in f["geometry"]
                if f["geometry"][k].attrs["name"] == "env_0"
            )
            poses = d["body_root_poses"][:][selected, rotor_index]
            axis = Rotation.from_quat(poses[:, 3:]).apply(
                np.tile([0, 0, 1], (len(selected), 1))
            )
            w = d["extrinsic_contact_wrench"][:][selected]
            lever = d["tool_pose"][:][selected, :3] - poses[:, :3]
            moment = -float(
                np.mean(np.sum((w[:, 3:] + np.cross(lever, w[:, :3])) * axis, axis=1))
            )
        return dict(
            spec=s,
            metrics=m,
            actual_friction=actual_friction,
            path=str(path.resolve()),
            future_resistance_nm=moment,
            label=int(moment >= 0.027) if moment is not None else -1,
            group=s["grip_force_n"],
            target_complete=complete,
        )


def fit_sensor_stress(output):
    """Same held-out groups; dense field and net wrench share sensor corruption."""
    from sklearn.linear_model import LogisticRegression, Ridge

    manifest = json.loads((output / "features.json").read_text())
    if not manifest["gate"]["passed"]:
        return
    records = manifest["records"]
    data = [np.load(r["file"]) for r in records]
    ids = np.repeat(np.arange(len(records)), manifest["views"])
    anchor_y = np.array([r["label"] for r in records])
    y = anchor_y[ids]
    groups = np.array([r["group"] for r in records])[ids]
    target = np.array([r["future_resistance_nm"] for r in records])[ids]
    results = []
    comparisons = []
    for c, condition in enumerate(manifest["conditions"]):
        neural = np.stack([d["neural"][c] for d in data]).reshape(len(ids), -1)
        metric = np.stack([d["metric"][c] for d in data]).reshape(len(ids), -1)
        vision = [neural, metric, observed_relations(metric)]
        for noise in TACTILE_CONDITIONS:
            field = np.stack([d["stress_tactile_" + noise] for d in data])[ids]
            wrench = np.stack([d["stress_wrench_" + noise] for d in data])[ids]
            modes = dict(
                vision=vision,
                tactile=[field],
                wrench=[wrench],
                vision_tactile=[*vision, field],
                vision_wrench=[*vision, wrench],
            )
            current = {}
            for mode, blocks in modes.items():
                probabilities = np.zeros(len(ids))
                prediction = np.zeros(len(ids))
                for group in np.unique(groups):
                    train = np.flatnonzero(groups != group)
                    test = np.flatnonzero(groups == group)
                    assert not set(ids[train]) & set(ids[test])
                    x = design(blocks, train)
                    model = LogisticRegression(
                        C=1, solver="liblinear", max_iter=2000, random_state=0
                    ).fit(x[train], y[train])
                    probabilities[test] = model.predict_proba(x[test])[:, 1]
                    prediction[test] = (
                        Ridge(alpha=1).fit(x[train], target[train]).predict(x[test])
                    )
                row = dict(
                    condition=condition["name"],
                    sensor_condition=noise,
                    mode=mode,
                    **summary(y, probabilities, ids),
                    **regression_summary(target, prediction, ids),
                )
                results.append(row)
                current[mode] = row
            comparisons.append(
                dict(
                    condition=condition["name"],
                    sensor_condition=noise,
                    **paired_gain(
                        anchor_y, current["vision"], current["vision_tactile"]
                    ),
                )
            )
    result = dict(
        status="complete",
        results=results,
        paired_comparisons=comparisons,
        split="Leave one grip-force group out; views are replicates, not independent episodes.",
        note="Uncalibrated sensor stress. Both force-field and net-wrench inputs are derived from the SAME noisy/delayed/saturated surface readings, with preload subtraction. No extra physical trajectories.",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (output / "sensor_stress_probes.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    for d in data:
        d.close()


def extract(root, window, views=4):
    output = root / f"probe_{window}"
    output.mkdir(exist_ok=True)
    records = [target_record(p) for p in sorted((root / "sensing").glob("*.h5"))]
    gate = qualify(records)
    if any(not r["target_complete"] for r in records):
        gate["passed"] = False
        gate["reasons"].append("missing complete future torque window")
    speeds = {r["spec"]["speed_scale"] for r in records}
    if len(speeds) != 1:
        gate["passed"] = False
        gate["reasons"].append("common speed required for matched probe timing")
    speed = next(iter(speeds)) if speeds else 0.75
    observation_time = 4.0 if window == "before" else 4 + 6 / speed
    configs = [conditions("turning")[i] for i in (0, 2, 3, 5)]
    manifest = dict(
        stage="turning",
        window=window,
        views=views,
        records=records,
        gate=gate,
        observation_time_s=observation_time,
        conditions=[asdict(c) for c in configs],
        window_definition="pre-turn prepared seating"
        if window == "before"
        else "during turning",
        control_definition="The inherited precontact feature name means 1.05–1.45 s preload here; the key is already seated, so this is not guaranteed contact-free.",
        target_window_real_s=[4 + 12 / speed, 4 + 13 / speed],
        target_window_trajectory_s=[16, 17],
        note="All 12 planned physical conditions must pass full-rollout qualification. No post-hoc horizon truncation or replacement.",
    )
    if not gate["passed"]:
        (output / "features.json").write_text(json.dumps(manifest, indent=2) + "\n")
        fit(output)
        return
    model = FrozenPointM2AE().cuda().eval()
    torch.set_num_threads(2)
    observation_preview(records[0]["path"], output / "point_cloud_conditions.png")
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    start = time.perf_counter()
    cache = output / "features"
    cache.mkdir(exist_ok=True)
    for record in records:
        with h5py.File(record["path"]) as f:
            d = f["observations"]
            times = d["timestamps"][:]
            indices = [
                int(np.argmin(abs(times - t)))
                for t in np.linspace(observation_time - 0.9, observation_time, 10)
            ]
            before = [
                int(np.argmin(abs(times - t))) for t in np.linspace(1.05, 1.45, 10)
            ]
            recon = Reconstructor(f)
            raw = [
                recon.visible(
                    d["body_root_poses"][i],
                    0,
                    [d["tactile_coord_" + s][i] for s in ("left", "right")],
                )
                for i in indices
            ]
            neural = []
            metric = []
            coverage = []
            for cfg in configs:
                ns = []
                ms = []
                cs = []
                for view in range(views):
                    points = []
                    geom = []
                    counts = []
                    for frame, i in enumerate(indices):
                        cloud, n = corrupt(
                            raw[frame],
                            recon.eyes[0],
                            recon.target,
                            cfg,
                            augmentation_seed(record["spec"]["camera_seed"], view),
                            frame,
                        )
                        norm, m = normalize_and_metric(cloud, n, d["ee_pose"][i])
                        points.extend(norm)
                        geom.append(m)
                        counts.append(n)
                    with torch.no_grad():
                        encoded = np.concatenate(
                            [
                                model(
                                    torch.from_numpy(
                                        np.asarray(points[j : j + 8])
                                    ).cuda()
                                )
                                .cpu()
                                .numpy()
                                for j in range(0, len(points), 8)
                            ]
                        )
                    encoded = encoded.reshape(10, 2, -1)
                    encoded[np.array(counts) == 0] = 0
                    ns.append(summarize_history(encoded.reshape(10, -1)))
                    ms.append(summarize_history(np.array(geom)))
                    cs.append(counts)
                neural.append(ns)
                metric.append(ms)
                coverage.append(cs)
            stress = {}
            for name, cfg in TACTILE_CONDITIONS.items():
                field, wrench = tactile_history(
                    f, indices, cfg, record["spec"]["tactile_seed"]
                )
                stress["stress_tactile_" + name] = summarize_history(
                    field.reshape(10, -1)
                )
                stress["stress_wrench_" + name] = summarize_history(wrench)
            path = cache / (Path(record["path"]).stem + ".npz")
            record["file"] = str(path.resolve())
            np.savez_compressed(
                path,
                neural=neural,
                metric=metric,
                coverage=coverage,
                tactile=tactile_features(f, indices),
                precontact=tactile_features(f, before),
                wrench=wrench_features(f, indices),
                **stress,
            )
            record["raw_file_sha256"] = hashlib.sha256(
                Path(record["path"]).read_bytes()
            ).hexdigest()
        print("encoded " + Path(record["path"]).name, flush=True)
    assert all(torch.equal(v, initial[k]) for k, v in model.state_dict().items())
    manifest.update(
        seconds=time.perf_counter() - start,
        encoder_frozen=True,
        encoder_checkpoint=model.checkpoint_sha256,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        gpu=torch.cuda.get_device_name(),
        peak_allocated_vram_mb=torch.cuda.max_memory_allocated() / 2**20,
        peak_reserved_vram_mb=torch.cuda.max_memory_reserved() / 2**20,
    )
    (output / "features.json").write_text(json.dumps(manifest, indent=2) + "\n")
    result = fit(output)
    result["target"] = (
        f"Future measured resisting moment about rotor axis, real time {manifest['target_window_real_s']} s; threshold 0.027 Nm"
    )
    result["scope"] = (
        "Full-rollout qualified new fixture; no return-motion exclusions. Separate sensor_stress_probes.json compares corrupted fields and wrenches using the same held-out groups."
    )
    result["window_definition"] = manifest["window_definition"]
    result["control_definition"] = manifest["control_definition"]
    (output / "probes.json").write_text(json.dumps(result, indent=2) + "\n")
    fit_sensor_stress(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--window", choices=("before", "after"), required=True)
    a = parser.parse_args()
    extract(a.folder, a.window)
