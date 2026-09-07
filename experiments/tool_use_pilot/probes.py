#!/usr/bin/env python3
"""Small grouped CPU baselines, including a pause-time leakage audit.

Not an ACWM training pipeline. All modalities share the same train/test groups;
normalization is fitted on training episodes only. Simulator labels never enter
the non-oracle input features.
"""

import argparse
import json
from pathlib import Path
import h5py
import numpy as np


def split_group(spec):
    """Tie repeated geometry/timeline configurations and stiffness triplets together."""
    return json.dumps(
        {k: spec[k] for k in ("task", "offset", "angle_deg", "friction", "pause")},
        sort_keys=True,
    )


def past_indices(steps, rng):
    """Causal misalignment control: each time receives a random past sample."""
    return np.asarray([rng.integers(0, t + 1) for t in range(steps)])


def load(paths):
    episodes = []
    for path in paths:
        with h5py.File(path) as f:
            spec = json.loads(f.attrs["config_json"])
            spec["group"] = split_group(spec)
            metric = json.loads(f.attrs["metrics_json"])
            if (
                spec["task"] != "probe"
                or spec["branch"] != "nominal"
                or not metric["physical_valid"]
            ):
                continue
            d = {k: v[::10] for k, v in f["observations"].items()}
            n = len(d["timestamps"])
            if n != 140:
                continue
            ref = f["tactile_reference"][:]
            tactile = np.concatenate(
                [
                    d[f"tactile_force_field_{s}"].reshape(n, -1) - ref[i].ravel()
                    for i, s in enumerate(("left", "right"))
                ],
                axis=1,
            )
            tactile = np.c_[tactile, np.broadcast_to(ref.reshape(1, -1), (n, ref.size))]
            # Visible geometry is fixed across stiffness triplets. Use measured
            # current poses plus action and proprioception, not hidden spring state.
            base = np.concatenate(
                [
                    d["ee_pose"],
                    d["ee_vel"],
                    d["tool_pose"],
                    d["actions"],
                    d["fixture_state"][:, :7],
                ],
                axis=1,
            )
            targets = []
            for h in (1, 5):
                future = np.minimum(np.arange(n) + h, n - 1)
                targets.append(
                    np.c_[
                        d["tool_pose"][future, :3] - d["tool_pose"][:, :3],
                        d["task_progress"][future] - d["task_progress"],
                        d["contact_event"][future],
                    ]
                )
            episodes.append(
                dict(
                    path=str(path),
                    spec=spec,
                    base=base,
                    tactile=tactile,
                    wrench=d["field_gel_wrench"],
                    target=np.concatenate(targets, axis=1),
                    phase=d["phase"],
                    timestamps=d["timestamps"],
                    progress=d["task_progress"],
                )
            )
    return episodes


def leakage(episodes):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GroupKFold, cross_val_score

    if len({e["spec"]["group"] for e in episodes}) < 4:
        return {"status": "insufficient paired groups"}
    x = []
    y = []
    groups = []
    residual = []
    base_features = []
    tactile_features = []
    for e in episodes:
        pause = np.flatnonzero(e["phase"] == b"pause")
        if not len(pause):
            continue
        i = pause[-1]
        x.append(np.r_[e["base"][i], e["tactile"][i]])
        base_features.append(e["base"][i])
        tactile_features.append(e["tactile"][i])
        y.append(int(e["spec"]["stiffness"] // 30) - 1)
        groups.append(e["spec"]["group"])
        residual.append(abs(float(e["progress"][i])))
    accuracies = {}
    for name, features in (
        ("combined", x),
        ("no_tactile", base_features),
        ("tactile_only", tactile_features),
    ):
        scores = cross_val_score(
            make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)),
            np.asarray(features),
            y,
            groups=groups,
            cv=GroupKFold(4),
        )
        accuracies[name] = float(scores.mean())
    return dict(
        current_pause_stiffness_accuracy=accuracies,
        chance=1 / 3,
        max_pause_fixture_residual_mm=max(residual) * 1000,
        interpretation="Above-chance current-state decoding is leakage, not evidence that memory is necessary.",
    )


def evaluate(episodes, seed, epochs, output=None):
    import torch
    from torch import nn

    torch.set_num_threads(4)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    groups = sorted({e["spec"]["group"] for e in episodes})
    rng.shuffle(groups)
    train_groups = set(groups[: int(0.65 * len(groups))])
    val_groups = set(groups[int(0.65 * len(groups)) : int(0.8 * len(groups))])
    split = np.array(
        [
            0
            if e["spec"]["group"] in train_groups
            else 1
            if e["spec"]["group"] in val_groups
            else 2
            for e in episodes
        ]
    )
    Y = np.stack([e["target"] for e in episodes]).astype(np.float32)
    steps = Y.shape[1]
    end = steps - 5
    # Train only after prepared-grasp release; target tail lacking full H5 is masked.
    ym = Y[split == 0, 15:end].mean((0, 1))
    ys = Y[split == 0, 15:end].std((0, 1)).clip(0.001)
    yt = torch.tensor((Y - ym) / ys)
    report = []

    class Net(nn.Module):
        def __init__(self, n, recurrent):
            super().__init__()
            self.recurrent = recurrent
            self.encode = nn.Sequential(nn.Linear(n, 32), nn.Tanh())
            self.memory = (
                nn.LSTM(32, 32, batch_first=True) if recurrent else nn.Identity()
            )
            self.out = nn.Linear(32, 10)

        def forward(self, x):
            z = self.encode(x)
            if self.recurrent:
                z, _ = self.memory(z)
            return self.out(z)

    for mode in (
        "current_tactile",
        "fixed_0p5s",
        "lstm_tactile",
        "lstm_no_tactile",
        "lstm_wrench",
        "lstm_shuffled_tactile",
        "oracle_current",
    ):
        xs = []
        for e in episodes:
            b = e["base"]
            tactile = e["tactile"].copy()
            if mode == "lstm_shuffled_tactile":
                tactile = tactile[past_indices(steps, rng)]
            if mode == "lstm_no_tactile":
                x = b
            elif mode == "lstm_wrench":
                x = np.c_[b, e["wrench"]]
            elif mode == "oracle_current":
                x = np.c_[b, tactile, np.full(steps, e["spec"]["stiffness"])]
            else:
                x = np.c_[b, tactile]
            if mode == "fixed_0p5s":
                x = np.concatenate(
                    [x[np.maximum(np.arange(steps) - i, 0)] for i in range(5)], axis=1
                )
            xs.append(x)
        X = np.stack(xs).astype(np.float32)
        mean = X[split == 0].mean((0, 1))
        std = X[split == 0].std((0, 1)).clip(0.001)
        xt = torch.tensor(np.clip((X - mean) / std, -20, 20))
        net = Net(X.shape[-1], mode.startswith("lstm"))
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.003, weight_decay=0.01)
        best = None
        best_loss = float("inf")
        for epoch in range(epochs):
            net.train()
            optimizer.zero_grad()
            pred = net(xt[split == 0])
            loss = ((pred[:, 15:end] - yt[split == 0, 15:end]) ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            net.eval()
            with torch.no_grad():
                vl = float(
                    (
                        (net(xt[split == 1])[:, 15:end] - yt[split == 1, 15:end]) ** 2
                    ).mean()
                )
            if vl < best_loss:
                best_loss = vl
                best = {k: v.clone() for k, v in net.state_dict().items()}
        net.load_state_dict(best)
        if output is not None:
            models = output / "models"
            models.mkdir(exist_ok=True)
            torch.save(
                dict(
                    state_dict=best,
                    input_mean=mean.tolist(),
                    input_std=std.tolist(),
                    target_mean=ym.tolist(),
                    target_std=ys.tolist(),
                    mode=mode,
                    seed=seed,
                    train_groups=sorted(train_groups),
                    validation_groups=sorted(val_groups),
                    test_groups=sorted(set(groups) - train_groups - val_groups),
                ),
                models / f"{mode}_seed{seed}.pt",
            )
        with torch.no_grad():
            pred = net(xt[split == 2]).numpy() * ys + ym
        truth = Y[split == 2]
        row = dict(
            seed=seed,
            model=mode,
            train_episodes=int(sum(split == 0)),
            val_episodes=int(sum(split == 1)),
            test_episodes=int(sum(split == 2)),
        )
        row["parameters"] = sum(p.numel() for p in net.parameters())
        later = np.stack([e["phase"] == b"later_pull" for e in episodes])[split == 2]
        later[:, :15] = False
        later[:, end:] = False
        for horizon, start in ((1, 0), (5, 5)):
            error = (
                pred[:, 15:end, start : start + 4] - truth[:, 15:end, start : start + 4]
            )
            row[f"H{horizon}_motion_rmse_mm"] = float(np.sqrt(np.mean(error**2)) * 1000)
            row[f"H{horizon}_later_pull_motion_rmse_mm"] = float(
                np.sqrt(
                    np.mean(
                        (
                            pred[:, :, start : start + 4][later]
                            - truth[:, :, start : start + 4][later]
                        )
                        ** 2
                    )
                )
                * 1000
            )
            row[f"H{horizon}_zero_motion_baseline_rmse_mm"] = float(
                np.sqrt(np.mean(truth[:, 15:end, start : start + 4] ** 2)) * 1000
            )
            probability = np.clip(pred[:, 15:end, start + 4], 0, 1)
            event = truth[:, 15:end, start + 4]
            row[f"H{horizon}_event_brier"] = float(np.mean((probability - event) ** 2))
            bins = []
            for lo in np.arange(0, 1, 0.2):
                take = (probability >= lo) & (
                    (probability < lo + 0.2) if lo < 0.8 else (probability <= 1.0)
                )
                if take.any():
                    bins.append(
                        dict(
                            n=int(take.sum()),
                            confidence=float(probability[take].mean()),
                            frequency=float(event[take].mean()),
                        )
                    )
            row[f"H{horizon}_reliability_bins"] = bins
        report.append(row)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    episodes = load(sorted(args.data.glob("episode_*.h5")))
    result = dict(
        valid_probe_episodes=len(episodes),
        horizon_definition="H1=100ms, H5=500ms; observations downsampled to10Hz",
        leakage=leakage(episodes),
        shuffle_semantics="random resampling from current/past tactile only; no future samples",
        branch_action_ranking="see separate anchor-only leave-configuration-out branch_probes.json",
    )
    if len({e["spec"]["group"] for e in episodes}) >= 8:
        result["models"] = [
            r
            for seed in (11, 22, 33)
            for r in evaluate(episodes, seed, args.epochs, args.data)
        ]
    else:
        result["models"] = []
        result["status"] = (
            "insufficient independent valid groups for learned comparison"
        )
    (args.data / "probes.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
