"""Offline, reproducible RGB-D stress views; raw episode sensors remain untouched."""

import argparse
from dataclasses import dataclass, asdict
import json
from pathlib import Path

import h5py
import numpy as np

from decision_vision import Reconstructor


@dataclass(frozen=True)
class CloudCondition:
    name: str
    noise_m: float = 0.0
    patch_fraction: float = 0.0

    def __post_init__(self):
        if not np.isfinite(self.noise_m) or self.noise_m < 0:
            raise ValueError("invalid point noise")
        if not np.isfinite(self.patch_fraction) or not 0 <= self.patch_fraction < 1:
            raise ValueError("invalid patch fraction")


CONDITIONS = (
    CloudCondition("visible_clean"),
    CloudCondition("noise1mm", 0.001),
    CloudCondition("noise2mm", 0.002),
    CloudCondition("patch15", 0, 0.15),
    CloudCondition("patch30", 0, 0.30),
    CloudCondition("noise1mm_patch15", 0.001, 0.15),
    CloudCondition("noise2mm_patch30", 0.002, 0.30),
)


class CloudAugmenter:
    """Fixed camera-plane cut calibrated once, not tracking an object or label.

    Requested removal fraction holds at calibration only. Subsequent visibility
    counts vary naturally as objects cross the stationary occluder boundary.
    """

    def __init__(self, initial_clouds, eye, target, seed):
        self.seed = int(seed)
        forward = np.asarray(target) - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        angle = np.random.default_rng(seed).uniform(0, 2 * np.pi)
        self.direction = np.cos(angle) * right + np.sin(angle) * up
        self.eye = np.asarray(eye)
        self.forward = forward
        # One shared image-plane occluder for tool AND environment.
        merged = np.concatenate(initial_clouds)
        scores, valid = self.project(merged)
        self.initial_scores = scores[valid]

    def project(self, points):
        rays = np.asarray(points) - self.eye
        depth = rays @ self.forward
        return (rays @ self.direction) / np.maximum(depth, 1e-9), depth > 0

    def apply(self, clouds, condition, frame, count=1024):
        results, counts = [], []
        threshold = (
            -np.inf
            if not condition.patch_fraction or not len(self.initial_scores)
            else np.quantile(self.initial_scores, condition.patch_fraction)
        )
        for stream, cloud in enumerate(clouds):
            score, valid = self.project(cloud)
            points = np.asarray(cloud)[valid & (score >= threshold)]
            counts.append(len(points))
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, stream, int(frame)])
            )
            if not len(points):
                results.append(np.zeros((count, 3), np.float32))
                continue
            points = points[
                rng.choice(len(points), count, replace=len(points) < count)
            ].copy()
            # Same standard-normal draws across severity for paired comparisons.
            bias = np.random.default_rng(
                np.random.SeedSequence([self.seed, stream, 987])
            ).normal(size=(1, 3))
            points += condition.noise_m * (rng.normal(size=points.shape) + 0.5 * bias)
            results.append(points.astype(np.float32))
        return np.stack(results), np.asarray(counts)


def preview(episode, output):
    from diversification_review import recorded_geometry_id
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=False)
    with h5py.File(episode) as f:
        spec = json.loads(f.attrs["config_json"])
        d = f["observations"]
        recon = Reconstructor(f)
        indices = np.linspace(0, len(d["timestamps"]) - 1, 4, dtype=int)

        def visible(index):
            return recon.visible(
                d["body_root_poses"][index],
                0,
                [d["tactile_coord_" + side][index] for side in ("left", "right")],
            )

        first = visible(int(indices[0]))
        augmenter = CloudAugmenter(
            first, recon.eyes[0], recon.target, spec["camera_seed"]
        )
        fig, axes = plt.subplots(len(CONDITIONS), len(indices), figsize=(14, 19))
        records = []
        for col, index in enumerate(indices):
            clouds = first if col == 0 else visible(int(index))
            for row, cfg in enumerate(CONDITIONS):
                points, counts = augmenter.apply(clouds, cfg, int(index))
                ax = axes[row, col]
                for stream, color in enumerate(("tab:blue", "gray")):
                    if counts[stream]:
                        q = points[stream] - recon.target
                        ax.scatter(q[:, 0], q[:, 2], s=1, c=color)
                ax.set_aspect("equal")
                ax.set_xlim(-0.20, 0.20)
                ax.set_ylim(-0.15, 0.20)
                ax.set_title(
                    f"{cfg.name}; t={d['timestamps'][index]:.2f}s\nvisible tool/env={counts.tolist()}",
                    fontsize=8,
                )
                records.append(
                    dict(
                        condition=asdict(cfg),
                        frame=int(index),
                        visible_counts=counts.tolist(),
                    )
                )
        fig.tight_layout()
        fig.savefig(output / "point_cloud_conditions.png", dpi=130)
        plt.close(fig)
    (output / "observations.json").write_text(
        json.dumps(
            dict(
                episode=str(episode.resolve()),
                geometry_id=spec.get("geometry_id"),
                recorded_geometry_id=recorded_geometry_id(episode),
                split_group=spec.get("split_group"),
                camera_seed=spec["camera_seed"],
                calibration_frame=int(indices[0]),
                samples=records,
                note="Synthetic observation stress, not real RGB-D calibration. No PCD stored in raw HDF5; zero-count streams carry an explicit validity count.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    preview(args.episode, args.output)
