"""Held-out local hook observability test; never consumes future executed actions.

The target is measured straight-pull success, not an authored contact label or
four-way policy success. Augmented camera views are replicates, not new episodes.
"""

import argparse
from dataclasses import dataclass, asdict
import hashlib
import json
import shutil
from pathlib import Path
import time

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import torch

from decision_vision import Reconstructor, normalize_and_metric
from decision_probes import summarize_history
from point_m2ae_probe import FrozenPointM2AE


@dataclass(frozen=True)
class Condition:
    name: str
    noise_m: float = 0.0
    patch_fraction: float = 0.0
    contact_window: bool = False
    window_size_m: tuple = (0.045, 0.028)
    window_center_z_m: float = 0.010


CONDITIONS = [
    Condition("visible_clean"),
    Condition("noise_0.5mm_patch10", 0.0005, 0.10),
    Condition("noise_1mm_patch25", 0.001, 0.25),
    Condition("noise_2mm_patch50", 0.002, 0.50),
    Condition("noise_1mm_contact_occlusion", 0.001, 0, True),
    Condition("noise_2mm_contact_occlusion_patch25", 0.002, 0.25, True),
    Condition(
        "noise_1mm_working_region_occluded", 0.001, 0, True, (0.14, 0.09), -0.015
    ),
]


def augmentation_seed(episode_id, view):
    # Unseen physical anchors also receive unseen camera corruption draws.
    return int(np.random.SeedSequence([8000, episode_id, view]).generate_state(1)[0])


def corrupt(clouds, eye, target, cfg, seed, frame):
    """No labels or object poses: fixed camera mask, patch deletion and noise."""
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    # Fixed image-plane window at the apparatus working region. It is NOT
    # repositioned using the episode's true engagement point or outcome.
    center = target + np.array([0, 0, cfg.window_center_z_m]) - eye
    center_uv = np.array([center @ right, center @ up]) / (center @ forward)
    half = np.array(cfg.window_size_m) / (2 * (center @ forward))
    result, counts = [], []
    for stream, original in enumerate(clouds):
        points = original.copy()
        spatial = np.random.default_rng(np.random.SeedSequence([seed, stream, 0]))
        sampling = np.random.default_rng(
            np.random.SeedSequence([seed, stream, frame + 1])
        )
        if len(points) and cfg.contact_window:
            rays = points - eye
            uv = np.c_[rays @ right, rays @ up] / (rays @ forward)[:, None]
            points = points[~np.all(abs(uv - center_uv) < half, axis=1)]
        direction = spatial.normal(size=2)
        direction /= np.linalg.norm(direction)
        if len(points) and cfg.patch_fraction:
            score = (points - target) @ (direction[0] * right + direction[1] * up)
            points = points[score >= np.quantile(score, cfg.patch_fraction)]
        counts.append(len(points))
        if not len(points):
            result.append(np.zeros((1024, 3), np.float32))
            continue
        indices = sampling.choice(len(points), 1024, replace=len(points) < 1024)
        sampled = points[indices].copy()
        # Independent XYZ point noise plus persistent per-stream registration
        # bias (sigma=noise/2), to avoid iid noise vanishing under pooling.
        bias = spatial.normal(0, cfg.noise_m / 2, (1, 3))
        sampled += bias + sampling.normal(0, cfg.noise_m, sampled.shape)
        result.append(sampled.astype(np.float32))
    return np.stack(result), np.array(counts)


def tactile_features(f, indices):
    d = f["observations"]
    ref = f["tactile_reference"][:].reshape(1, -1)
    force = np.concatenate(
        [
            d[f"tactile_force_field_{s}"][:][indices].reshape(len(indices), -1)
            for s in ("left", "right")
        ],
        axis=1,
    )
    return summarize_history(force - ref)


def wrench_features(f, indices):
    """Sum taxel forces about EE using robot/sensor poses, not tool GT or GT wrench."""
    d = f["observations"]
    names = {g.attrs["name"]: int(k) for k, g in f["geometry"].items()}
    series = []
    for i in indices:
        ee = d["ee_pose"][i]
        w = np.zeros(6)
        for s in ("left", "right"):
            housing = d["body_root_poses"][i, names[f"franka_{s}_gelsight_housing"]]
            forces = -Rotation.from_quat(housing[3:]).apply(
                d[f"tactile_force_field_{s}"][i].reshape(-1, 3)
            )
            positions = d[f"tactile_coord_{s}"][i].reshape(-1, 3) - ee[:3]
            w[:3] += forces.sum(0)
            w[3:] += np.cross(positions, forces).sum(0)
        inv = Rotation.from_quat(ee[3:]).inv()
        series.append(np.r_[inv.apply(w[:3]), inv.apply(w[3:])])
    return summarize_history(np.array(series))


def extract(folders, output, views=4, observation_time=5.0):
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, output / "extractor_source.py")
    cache = output / "features"
    cache.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    model = FrozenPointM2AE().cuda().eval()
    original = {k: v.clone() for k, v in model.state_dict().items()}
    protocol = dict(
        conditions=[asdict(c) for c in CONDITIONS],
        views=views,
        history_frames=10,
        history_seconds=0.9,
        observation_time_s=observation_time,
        encoder_checkpoint=model.checkpoint_sha256,
        extractor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    start = time.perf_counter()
    records = []
    for folder in folders:
        for path in sorted(folder.glob("episode_*_prefix.h5")):
            observed_path = (
                path
                if observation_time <= 5
                else path.with_name(path.name.replace("_prefix.h5", "_pull.h5"))
            )
            with h5py.File(path) as prefix, h5py.File(observed_path) as f:
                spec = json.loads(prefix.attrs["config_json"])
                key = f"episode_{spec['episode_id']:04d}"
                record = dict(
                    path=str(path.resolve()),
                    spec=spec,
                    prefix_valid=bool(
                        json.loads(prefix.attrs["metrics_json"])["physical_valid"]
                    ),
                )
                digest = hashlib.sha256(
                    json.dumps(protocol, sort_keys=True).encode()
                    + f["observations/body_root_poses"][:].tobytes()
                    + f["observations/tactile_force_field_left"][:].tobytes()
                ).hexdigest()
                record["feature_hash"] = digest
                record["file"] = str((cache / f"{key}.npz").resolve())
                meta = cache / f"{key}.json"
                if (
                    meta.exists()
                    and json.loads(meta.read_text()).get("feature_hash") == digest
                ):
                    records.append(json.loads(meta.read_text()))
                    print(f"Reused {key}", flush=True)
                    continue
                d = f["observations"]
                times = d["timestamps"][:]
                if times[-1] < observation_time - 1e-6:
                    raise ValueError(
                        f"Trajectory ended before observation time: {observed_path}"
                    )
                indices = np.array(
                    [
                        np.argmin(abs(times - t))
                        for t in np.linspace(
                            observation_time - 0.9, observation_time, 10
                        )
                    ]
                )
                before = np.array(
                    [
                        np.argmin(abs(prefix["observations/timestamps"][:] - t))
                        for t in np.linspace(1.05, 1.45, 10)
                    ]
                )
                recon = Reconstructor(f)
                raw = [
                    recon.visible(
                        d["body_root_poses"][i],
                        0,
                        [d[f"tactile_coord_{s}"][i] for s in ("left", "right")],
                    )
                    for i in indices
                ]
                neural, metric, coverage = [], [], []
                for cfg in CONDITIONS:
                    c_neural, c_metric, c_counts = [], [], []
                    for view in range(views):
                        pcs, geometry, counts = [], [], []
                        for frame, i in enumerate(indices):
                            # Modalities share identical inputs, but held-out
                            # physical anchors have unseen corruption seeds.
                            cloud, n = corrupt(
                                raw[frame],
                                recon.eyes[0],
                                recon.target,
                                cfg,
                                augmentation_seed(spec["episode_id"], view),
                                frame,
                            )
                            normalized, m = normalize_and_metric(
                                cloud, n, d["ee_pose"][i]
                            )
                            pcs.extend(normalized)
                            geometry.append(m)
                            counts.append(n)
                        encoded = []
                        for j in range(0, len(pcs), 8):
                            encoded.append(
                                model(
                                    torch.from_numpy(np.asarray(pcs[j : j + 8])).cuda()
                                )
                                .cpu()
                                .numpy()
                            )
                        e = np.concatenate(encoded).reshape(10, 2, -1)
                        e[np.array(counts) == 0] = 0
                        c_neural.append(summarize_history(e.reshape(10, -1)))
                        c_metric.append(summarize_history(np.array(geometry)))
                        c_counts.append(counts)
                    neural.append(c_neural)
                    metric.append(c_metric)
                    coverage.append(c_counts)
                np.savez_compressed(
                    cache / f"{key}.npz",
                    neural=neural,
                    metric=metric,
                    tactile=tactile_features(f, indices),
                    precontact=tactile_features(prefix, before),
                    wrench=wrench_features(f, indices),
                    coverage=coverage,
                )
                meta.write_text(json.dumps(record, indent=2))
                records.append(record)
                print(f"Encoded {key}", flush=True)
    assert all(torch.equal(v, original[k]) for k, v in model.state_dict().items())
    manifest = dict(
        protocol=protocol,
        records=records,
        seconds=time.perf_counter() - start,
        gpu=torch.cuda.get_device_name(),
        peak_allocated_vram_mb=torch.cuda.max_memory_allocated() / 2**20,
        peak_reserved_vram_mb=torch.cuda.max_memory_reserved() / 2**20,
        encoder_frozen=True,
    )
    (output / "features.json").write_text(json.dumps(manifest, indent=2))


def split_level(spec):
    base = 0.0375 if spec["case"] == "captured" else 0.0455
    return round((abs(spec["offset"]) - base) * 1e6)


def design(blocks, train):
    """Training-only scaling, equal block energy, no transductive normalization."""
    result = []
    for block in blocks:
        mean = block[train].mean(0)
        scale = np.maximum(block[train].std(0), 0.001)
        result.append(
            np.clip((block - mean) / scale, -20, 20) / np.sqrt(block.shape[1])
        )
    return np.concatenate(result, axis=1)


def observed_relations(metric):
    """Mirror-invariant relations from observed PCD centroids, never GT poses.

    Without these or a nonlinear head, a linear vision probe can fail merely
    because captured versus missed offsets form an XOR across mirrored sides.
    """
    chunks = []
    for start in (0, 34, 68):
        delta = metric[:, start : start + 3] - metric[:, start + 17 : start + 20]
        chunks.extend([delta, abs(delta), np.linalg.norm(delta, axis=1, keepdims=True)])
    return np.concatenate(chunks, axis=1)


def summarize(y, probability, anchor_ids):
    correct = (probability >= 0.5) == y
    anchors = np.unique(anchor_ids)
    per_anchor = np.array([correct[anchor_ids == a].mean() for a in anchors])
    rng = np.random.default_rng(701)
    boot = per_anchor[rng.integers(0, len(anchors), (4000, len(anchors)))].mean(1)
    return dict(
        balanced_accuracy=float(balanced_accuracy_score(y, probability >= 0.5)),
        auc=float(roc_auc_score(y, probability)),
        accuracy_ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
        per_anchor_accuracy=per_anchor.tolist(),
    )


def fit(output):
    manifest = json.loads((output / "features.json").read_text())
    records = manifest["records"]
    labels = []
    for r in records:
        path = Path(r["path"]).with_name(
            Path(r["path"]).name.replace("_prefix.h5", "_pull.h5")
        )
        with h5py.File(path) as f:
            m = json.loads(f.attrs["metrics_json"])
        r["pull_metrics"] = m
        labels.append(int(m["task_success"]))
    eligible = all(
        r["prefix_valid"] and r["pull_metrics"]["physical_valid"] for r in records
    )
    groups = np.array([split_level(r["spec"]) for r in records])
    if not eligible or len(records) != 12 or len(np.unique(groups)) != 3:
        result = dict(
            status="qualification_failed",
            reason="requires all 12 valid physical anchors in three offset levels",
            records=records,
        )
        (output / "probes.json").write_text(json.dumps(result, indent=2))
        return result
    y_anchor = np.array(labels)
    if any(set(y_anchor[groups == g]) != {0, 1} for g in np.unique(groups)):
        raise ValueError("Every held-out offset level must contain both outcomes")
    data = [np.load(r["file"]) for r in records]
    count = manifest["protocol"]["views"]
    anchor_ids = np.repeat(np.arange(len(records)), count)
    sample_groups = groups[anchor_ids]
    y = y_anchor[anchor_ids]
    outputs = []
    predictions = {}
    paired = []
    for c, cfg in enumerate(CONDITIONS):
        neural = np.stack([d["neural"][c] for d in data]).reshape(len(y), -1)
        metric = np.stack([d["metric"][c] for d in data]).reshape(len(y), -1)
        relations = observed_relations(metric)
        tactile = np.stack([d["tactile"] for d in data])[anchor_ids]
        before = np.stack([d["precontact"] for d in data])[anchor_ids]
        wrench = np.stack([d["wrench"] for d in data])[anchor_ids]
        modes = dict(
            vision=[neural, metric, relations],
            vision_metric=[metric, relations],
            tactile=[tactile],
            vision_tactile=[neural, metric, relations, tactile],
            vision_wrench=[neural, metric, relations, wrench],
            vision_precontact_tactile=[neural, metric, relations, before],
            vision_rbf=[neural, metric, relations],
            vision_tactile_rbf=[neural, metric, relations, tactile],
        )
        for mode, blocks in modes.items():
            p = np.zeros(len(y))
            shuffled = np.zeros((8, len(y)))
            for group in np.unique(groups):
                train = np.where(sample_groups != group)[0]
                test = np.where(sample_groups == group)[0]
                assert not set(anchor_ids[train]) & set(anchor_ids[test])
                x = design(blocks, train)
                model = (
                    SVC(C=1.0, kernel="rbf", gamma="scale")
                    if mode.endswith("_rbf")
                    else LogisticRegression(
                        C=1.0, solver="liblinear", max_iter=2000, random_state=0
                    )
                )
                model.fit(x[train], y[train])
                p[test] = (
                    (1 / (1 + np.exp(-model.decision_function(x[test]))))
                    if mode.endswith("_rbf")
                    else model.predict_proba(x[test])[:, 1]
                )
                if mode == "vision_tactile":
                    test_anchors = np.unique(anchor_ids[test])
                    for seed in range(8):
                        donor = np.random.default_rng(100 + seed).permutation(
                            test_anchors
                        )
                        # A cyclic shift of that order makes a derangement.
                        mapping = dict(zip(donor, np.roll(donor, 1)))
                        altered = tactile.copy()
                        for a in test_anchors:
                            altered[anchor_ids == a] = tactile[anchor_ids == mapping[a]]
                        sx = design([neural, metric, relations, altered], train)
                        shuffled[seed, test] = model.predict_proba(sx[test])[:, 1]
            result = dict(condition=cfg.name, mode=mode, **summarize(y, p, anchor_ids))
            outputs.append(result)
            predictions[f"{c}_{mode}"] = p
            if mode == "vision_tactile":
                # Report mean correctness across permutations, not accuracy
                # after averaging shuffled probabilities (different control).
                summaries = [summarize(y, sp, anchor_ids) for sp in shuffled]
                outputs.append(
                    dict(
                        condition=cfg.name,
                        mode="vision_tactile_shuffled_test",
                        balanced_accuracy=float(
                            np.mean([s["balanced_accuracy"] for s in summaries])
                        ),
                        per_anchor_accuracy=np.mean(
                            [s["per_anchor_accuracy"] for s in summaries], axis=0
                        ).tolist(),
                        permutation_count=8,
                    )
                )
        vision = next(
            r for r in outputs if r["condition"] == cfg.name and r["mode"] == "vision"
        )
        vt = next(
            r
            for r in outputs
            if r["condition"] == cfg.name and r["mode"] == "vision_tactile"
        )
        delta = np.array(vt["per_anchor_accuracy"]) - vision["per_anchor_accuracy"]
        rng = np.random.default_rng(801)
        boot = delta[rng.integers(0, len(delta), (4000, len(delta)))].mean(1)
        signs = (
            2 * ((np.arange(2 ** len(delta))[:, None] >> np.arange(len(delta))) & 1) - 1
        )
        pvalue = float(
            np.mean(abs((signs * delta).mean(1)) >= abs(delta.mean()) - 1e-12)
        )
        paired.append(
            dict(
                condition=cfg.name,
                accuracy_gain=float(delta.mean()),
                ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
                paired_signflip_p=pvalue,
            )
        )
    order = np.argsort([p["paired_signflip_p"] for p in paired])
    previous = 0.0
    for rank, i in enumerate(order):
        previous = max(
            previous, min(1.0, (len(paired) - rank) * paired[i]["paired_signflip_p"])
        )
        paired[i]["holm_adjusted_p"] = previous
    np.savez_compressed(
        output / "predictions.npz",
        labels=y,
        anchor_ids=anchor_ids,
        groups=sample_groups,
        **predictions,
    )
    summary = dict(
        status="complete",
        target="measured straight-pull success; not full recovery-policy success",
        independent_anchors=len(records),
        augmentation_views=count,
        held_out_offset_levels_um=np.unique(groups).tolist(),
        split="leave-one-offset-level-out: all four state types and every augmentation stay together",
        head="fixed C=1 L2 logistic regression, plus C=1 RBF-SVM sensitivity check; training-only block scaling; no test tuning",
        forbidden_inputs=[
            "tool/environment GT poses",
            "case/offset labels",
            "extrinsic contact wrench",
            "future executed commands",
            "privileged controller corrections",
        ],
        uncertainty="Approximate anchor-bootstrap intervals and paired sign-flip tests; only 12 local physical anchors, not 96 independent episodes",
        records=records,
        results=outputs,
        paired_comparisons=paired,
    )
    (output / "probes.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folders", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--observation-time", type=float, default=5.0)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--fit-only", action="store_true")
    args = parser.parse_args()
    if not args.fit_only:
        extract(args.folders, args.output, args.views, args.observation_time)
    if not args.extract_only:
        print(json.dumps(fit(args.output), indent=2))


if __name__ == "__main__":
    main()
