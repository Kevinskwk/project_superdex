"""Deterministic observation stress tests; never mutate the raw HDF5 sensors."""

from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class TactileCorruption:
    noise_n: float = 0.0
    bias_n: float = 0.0
    drift_n_per_s: float = 0.0
    delay_steps: int = 0
    saturation_n: float = 0.0

    def __post_init__(self):
        if (
            min(
                self.noise_n,
                self.bias_n,
                self.drift_n_per_s,
                self.delay_steps,
                self.saturation_n,
            )
            < 0
        ):
            raise ValueError("negative corruption magnitude")


TACTILE_CONDITIONS = {
    "ideal": TactileCorruption(),
    "noise_bias_delay": TactileCorruption(noise_n=0.005, bias_n=0.01, delay_steps=2),
    "drift_saturation": TactileCorruption(
        noise_n=0.01, bias_n=0.02, drift_n_per_s=0.0005, delay_steps=5, saturation_n=1.5
    ),
}


def corrupt_field(field, index, dt, seed, cfg):
    """Independent sensor seed; no geometry, material, outcome or future input."""
    constant = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    dynamic = np.random.default_rng(np.random.SeedSequence([seed, int(index) + 1]))
    result = np.asarray(field).copy() + constant.normal(0, cfg.bias_n, np.shape(field))
    result += constant.normal(0, cfg.drift_n_per_s, np.shape(field)) * index * dt
    result += dynamic.normal(0, cfg.noise_n, np.shape(field))
    if cfg.saturation_n:
        result = np.clip(result, -cfg.saturation_n, cfg.saturation_n)
    return result


def tactile_history(f, indices, cfg=TactileCorruption(), seed=0):
    """Return the SAME corrupted fields and EE-frame aggregate for fair baselines."""
    d = f["observations"]
    dt = float(np.median(np.diff(d["timestamps"][:])))
    names = {g.attrs["name"]: int(k) for k, g in f["geometry"].items()}
    reference_index = int(np.flatnonzero(d["reference_reset"][:])[0])

    def sample(index):
        index = max(0, int(index) - cfg.delay_steps)
        raw = np.stack(
            [d["tactile_force_field_" + s][index] for s in ("left", "right")]
        )
        field = corrupt_field(raw, index, dt, seed, cfg)
        ee = d["ee_pose"][index]
        wrench = np.zeros(6)
        for j, side in enumerate(("left", "right")):
            housing = d["body_root_poses"][
                index, names[f"franka_{side}_gelsight_housing"]
            ]
            force = -Rotation.from_quat(housing[3:]).apply(field[j].reshape(-1, 3))
            lever = d["tactile_coord_" + side][index].reshape(-1, 3) - ee[:3]
            wrench[:3] += force.sum(0)
            wrench[3:] += np.cross(lever, force).sum(0)
        rotation = Rotation.from_quat(ee[3:]).inv()
        return field, np.r_[rotation.apply(wrench[:3]), rotation.apply(wrench[3:])]

    # Preload calibration is a separate recorded reading, not a future sample.
    baseline, bw = sample(reference_index)
    readings = [sample(i) for i in indices]
    return np.array([r[0] - baseline for r in readings]), np.array(
        [r[1] - bw for r in readings]
    )
