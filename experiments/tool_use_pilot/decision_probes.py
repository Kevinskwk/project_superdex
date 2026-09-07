"""GPU frozen-Point-M2AE observation and candidate-consequence diagnostics.

This is not ACWM training. Refuses to report task-policy benefits when the
physical/constant-action gates fail. Feature extraction can still be audited.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch
from torch import nn
from scipy.spatial.transform import Rotation
from sklearn.model_selection import GroupKFold

from decision_vision import Observation, Reconstructor, normalize_and_metric
from point_m2ae_probe import FrozenPointM2AE


def summarize_history(x):
    return np.r_[x[-1], x.mean(0), x[-1] - x[0]].astype(np.float32)


def extract(folder, output, limit=None):
    torch.set_num_threads(2)
    model = FrozenPointM2AE().cuda()
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    cfgs = [
        Observation(mode="ideal"),
        Observation(),
        Observation(camera=1),
        Observation(noise_m=0.0005, patch_fraction=0.1),
        Observation(noise_m=0.001, patch_fraction=0.25),
    ]
    records = []
    visual = []
    tactile = []
    wrench = []
    proprio = []
    oracle = []
    coverage = []
    start = time.perf_counter()
    paths = sorted(Path(folder).glob("episode_*_prefix.h5"))
    if limit:
        paths = paths[:limit]
    for path in paths:
        with h5py.File(path) as f:
            spec = json.loads(f.attrs["config_json"])
            metric = json.loads(f.attrs["metrics_json"])
            d = f["observations"]
            times = d["timestamps"][:]
            indices = np.array(
                [
                    np.argmin(abs(times - t))
                    for t in np.linspace(times[-1] - 0.9, times[-1], 10)
                ]
            )
            recon = Reconstructor(f)
            views = []
            view_counts = []
            for c in cfgs:
                # Same geometry perturbation seed and frozen features across
                # every sensing ablation. No selection by tactile model result.
                c = Observation(**{**asdict(c), "seed": spec["seed"]})
                pcs = []
                metric_geometry = []
                counts = []
                for i in indices:
                    raw, n = recon.clouds(
                        d["body_root_poses"][i],
                        c,
                        [d[f"tactile_coord_{side}"][i] for side in ("left", "right")],
                    )
                    normalized, m = normalize_and_metric(raw, n, d["ee_pose"][i])
                    pcs.extend(normalized)
                    metric_geometry.append(m)
                    counts.append(n)
                embeddings = []
                for batch in range(0, len(pcs), 8):
                    embeddings.append(
                        model(
                            torch.from_numpy(np.asarray(pcs[batch : batch + 8])).cuda()
                        )
                        .cpu()
                        .numpy()
                    )
                embedding = np.concatenate(embeddings).reshape(10, 2, -1)
                embedding[np.asarray(counts) == 0] = 0
                embedding = embedding.reshape(10, -1)
                views.append(summarize_history(np.c_[embedding, metric_geometry]))
                view_counts.append(np.asarray(counts).tolist())
            # Metric raw path is computed from visible geometry, not GT poses.
            visual.append(np.stack(views))
            coverage.append(view_counts)
            ref = f["tactile_reference"][:]
            force = np.concatenate(
                [
                    d[f"tactile_force_field_{s}"][:][indices].reshape(10, -1)
                    for s in ("left", "right")
                ],
                axis=1,
            )
            delta = force - ref.ravel()[None]
            tactile.append(np.r_[summarize_history(delta), ref.ravel()])
            # Frame both pad force fields around the measured EE, avoiding a
            # privileged instantaneous tool-COM origin in the wrench baseline.
            ee = d["ee_pose"][:][indices]
            # field_gel_wrench is on tool; convert its COM-origin torque to EE.
            w = d["field_gel_wrench"][:][indices].copy()
            w[:, 3:] += np.cross(d["tool_pose"][:][indices, :3] - ee[:, :3], w[:, :3])
            for j in range(10):
                r = Rotation.from_quat(ee[j, 3:]).inv()
                w[j, :3] = r.apply(w[j, :3])
                w[j, 3:] = r.apply(w[j, 3:])
            wrench.append(summarize_history(w))
            proprio.append(
                summarize_history(
                    np.c_[
                        ee,
                        d["ee_vel"][:][indices],
                        d["actions"][:][indices],
                        d["grip_command_n"][:][indices],
                    ]
                )
            )
            oracle.append(
                summarize_history(
                    np.c_[
                        d["tool_pose"][:][indices],
                        d["fixture_state"][:][indices],
                        d["task_progress"][:][indices],
                    ]
                )
            )
            siblings = []
            for sibling in sorted(
                path.parent.glob(path.name.replace("_prefix.h5", "_*.h5"))
            ):
                if sibling == path:
                    continue
                with h5py.File(sibling) as b:
                    bm = json.loads(b.attrs["metrics_json"])
                    bd = b["observations"]
                    y = []
                    for h in (0.1, 0.5):
                        j = int(np.argmin(abs(bd["timestamps"][:] - times[-1] - h)))
                        pose = bd["tool_pose"][j]
                        rotation = (
                            Rotation.from_quat(pose[3:])
                            * Rotation.from_quat(d["tool_pose"][-1, 3:]).inv()
                        ).as_rotvec()
                        y.extend(
                            np.r_[
                                pose[:3] - d["tool_pose"][-1, :3],
                                rotation,
                                bd["task_progress"][j] - d["task_progress"][-1],
                                bd["contact_event"][j],
                            ]
                        )
                    progress = float(np.max(bd["task_progress"][:])) / (
                        0.040 if spec["kind"] == "hook" else np.pi / 2
                    )
                    value = float(bm["task_success"]) + 0.25 * np.clip(progress, 0, 1)
                    action_times = bd["timestamps"][:]
                    ai = [
                        int(np.argmin(abs(action_times - t)))
                        for t in np.linspace(action_times[0], action_times[-1], 8)
                    ]
                    siblings.append(
                        dict(
                            name=bm["branch"],
                            valid=bool(bm["physical_valid"]),
                            success=bool(bm["task_success"]),
                            value=value,
                            target=y,
                            actions=bd["actions"][:][ai].ravel().tolist(),
                        )
                    )
            records.append(
                dict(
                    path=str(path),
                    spec=spec,
                    prefix_valid=bool(metric["physical_valid"]),
                    siblings=siblings,
                )
            )
        print(f"Encoded {path.name}: {len(siblings)} branches", flush=True)
    assert all(torch.equal(v, initial[k]) for k, v in model.state_dict().items())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "features.npz",
        visual=visual,
        tactile=tactile,
        wrench=wrench,
        proprio=proprio,
        oracle=oracle,
    )
    summary = dict(
        records=records,
        observation_conditions=[asdict(c) for c in cfgs],
        coverage=coverage,
        checkpoint_sha256=model.checkpoint_sha256,
        encoder_source_hash=model.source_hash,
        encoder_frozen=True,
        gpu=torch.cuda.get_device_name(),
        seconds=time.perf_counter() - start,
        peak_allocated_vram_mb=torch.cuda.max_memory_allocated() / 2**20,
        peak_reserved_vram_mb=torch.cuda.max_memory_reserved() / 2**20,
    )
    (output / "features.json").write_text(json.dumps(summary, indent=2))
    return summary


def task_gate(records):
    clean = [
        r
        for r in records
        if r["prefix_valid"]
        and r["siblings"]
        and all(b["valid"] for b in r["siblings"])
    ]
    if not clean:
        return dict(
            passed=False,
            clean_anchors=0,
            reason="no complete physically valid branch groups",
        )
    names = [b["name"] for b in clean[0]["siblings"]]
    clean = [r for r in clean if [b["name"] for b in r["siblings"]] == names]
    successes = np.array([[b["success"] for b in r["siblings"]] for r in clean])
    unique = successes.sum(1) == 1
    counts = (successes & unique[:, None]).sum(0)
    constant = float(successes.mean(0).max())
    passed = bool((counts >= 0.2 * len(clean)).sum() >= 2 and constant <= 0.8)
    return dict(
        passed=passed,
        clean_anchors=len(clean),
        candidate_names=names,
        unique_success_counts=counts.tolist(),
        best_constant_action_success=constant,
        reason="passed" if passed else "insufficient state-dependent action diversity",
    )


def fit(output):
    torch.set_num_threads(2)
    output = Path(output)
    manifest = json.loads((output / "features.json").read_text())
    records = manifest["records"]
    data = np.load(output / "features.npz")
    results = []
    gates = {}
    for kind in sorted({r["spec"]["kind"] for r in records}):
        relevant = [(i, r) for i, r in enumerate(records) if r["spec"]["kind"] == kind]
        gate = task_gate([r for _, r in relevant])
        gates[kind] = gate
        clean = [
            (i, r)
            for i, r in relevant
            if r["prefix_valid"]
            and r["siblings"]
            and all(b["valid"] for b in r["siblings"])
        ]
        if len(clean) < 8:
            results.append(
                dict(
                    kind=kind,
                    status="insufficient independent valid anchors for fitted probes",
                    anchors=len(clean),
                )
            )
            continue
        for condition in range(data["visual"].shape[1]):
            for mode in (
                "action_prior",
                "vision",
                "vision_wrench",
                "vision_tactile",
                "pose_oracle",
            ):
                x = []
                y = []
                groups = []
                anchor_ids = []
                for idx, r in clean:
                    obs = (
                        np.zeros(1)
                        if mode == "action_prior"
                        else np.r_[data["visual"][idx, condition], data["proprio"][idx]]
                    )
                    if mode == "vision_wrench":
                        obs = np.r_[obs, data["wrench"][idx]]
                    if mode == "vision_tactile":
                        obs = np.r_[obs, data["tactile"][idx]]
                    if mode == "pose_oracle":
                        obs = np.r_[data["oracle"][idx], data["proprio"][idx]]
                    spec = r["spec"]
                    # Tie repeats and material/resistance variants of one geometry.
                    key = json.dumps(
                        {
                            k: spec[k]
                            for k in ("kind", "case", "offset", "lateral_m", "yaw_deg")
                        },
                        sort_keys=True,
                    )
                    for b in r["siblings"]:
                        x.append(np.r_[obs, b["actions"]])
                        y.append(np.r_[b["target"], b["success"], b["value"]])
                        groups.append(key)
                        anchor_ids.append(idx)
                x = np.asarray(x, np.float32)
                y = np.asarray(y, np.float32)
                groups = np.asarray(groups)
                anchor_ids = np.asarray(anchor_ids)
                if len(set(groups)) < 4:
                    continue
                folds = GroupKFold(n_splits=min(4, len(set(groups))))
                for seed in (11, 22, 33):
                    prediction = np.zeros_like(y)
                    start = time.perf_counter()
                    for train, test in folds.split(x, y, groups):
                        val_group = sorted(set(groups[train]))[-1]
                        val = train[groups[train] == val_group]
                        train = train[groups[train] != val_group]
                        mean = x[train].mean(0)
                        std = np.maximum(x[train].std(0), 0.001)
                        ym = y[train].mean(0)
                        ys = np.maximum(y[train].std(0), 0.001)
                        xx = torch.from_numpy(np.clip((x - mean) / std, -20, 20)).cuda()
                        yy = torch.from_numpy((y - ym) / ys).cuda()
                        torch.manual_seed(seed)
                        model = nn.Sequential(
                            nn.Linear(x.shape[1], 128),
                            nn.GELU(),
                            nn.Linear(128, 64),
                            nn.GELU(),
                            nn.Linear(64, y.shape[1]),
                        ).cuda()
                        optimizer = torch.optim.AdamW(
                            model.parameters(), lr=0.001, weight_decay=0.01
                        )
                        best = float("inf")
                        saved = None
                        for epoch in range(300):
                            optimizer.zero_grad()
                            loss = (model(xx[train]) - yy[train]).square().mean()
                            loss.backward()
                            optimizer.step()
                            if epoch % 10 == 0:
                                with torch.no_grad():
                                    vl = float(
                                        (model(xx[val]) - yy[val]).square().mean()
                                    )
                                if vl < best:
                                    best = vl
                                    saved = {
                                        k: v.detach().clone()
                                        for k, v in model.state_dict().items()
                                    }
                        model.load_state_dict(saved)
                        with torch.no_grad():
                            prediction[test] = model(xx[test]).cpu().numpy() * ys + ym
                    regrets = []
                    success = []
                    for anchor in set(anchor_ids):
                        rows = np.where(anchor_ids == anchor)[0]
                        selected = rows[prediction[rows, -1].argmax()]
                        regrets.append(float(y[rows, -1].max() - y[selected, -1]))
                        success.append(float(y[selected, -2]))
                    results.append(
                        dict(
                            kind=kind,
                            condition=condition,
                            mode=mode,
                            seed=seed,
                            anchors=len(clean),
                            task_gate_passed=gate["passed"],
                            H1_position_rmse_mm=float(
                                1000
                                * np.sqrt(
                                    np.mean(
                                        np.sum((prediction[:, :3] - y[:, :3]) ** 2, 1)
                                    )
                                )
                            ),
                            H5_position_rmse_mm=float(
                                1000
                                * np.sqrt(
                                    np.mean(
                                        np.sum(
                                            (prediction[:, 8:11] - y[:, 8:11]) ** 2, 1
                                        )
                                    )
                                )
                            ),
                            candidate_success=float(np.mean(success)),
                            candidate_regret=float(np.mean(regrets)),
                            seconds=time.perf_counter() - start,
                        )
                    )
    summary = dict(
        task_gates=gates,
        results=results,
        interpretation="Exploratory local frozen-encoder probes, not trained ACWM or cross-domain evidence. Failed task gates preclude tactile-necessity/policy claims.",
    )
    (output / "probes.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--fit-only", action="store_true")
    a = p.parse_args()
    if not a.fit_only:
        extract(a.folder, a.output, a.limit)
    fit(a.output)


if __name__ == "__main__":
    main()
