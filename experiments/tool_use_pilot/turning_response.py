"""Compare physical turning loads and force-only tactile changes in saved trials."""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def response(path):
    with h5py.File(path) as f:
        d = f["observations"]
        spec = json.loads(f.attrs["config_json"])
        t = d["timestamps"][:]
        reference = (t >= 3) & (t < 4)
        if not reference.any():
            raise ValueError(f"No settled pre-turn reference: {path}")
        field = np.stack(
            [d[f"tactile_force_field_{s}"][:] for s in ("left", "right")], axis=1
        )
        delta = field - field[reference].mean(axis=0)
        rms = np.sqrt(np.mean(np.sum(delta**2, axis=-1), axis=(1, 2, 3)))
        truth = d["extrinsic_contact_wrench"][:]
        inferred = d["field_inferred_extrinsic_wrench"][:]
        q = np.rad2deg(d["rotor_angle_rad"][:])
        hold = t >= t[-1] - 1.0
        moving = (q > 20) & (q < 70)
        return dict(
            path=str(path),
            spec=spec,
            t=t,
            q=q,
            truth=truth,
            inferred=inferred,
            field_rms=rms,
            summary=dict(
                duration_s=float(t[-1]),
                final_angle_deg=float(q[-1]),
                hold_contact_torque_norm_nm=float(
                    np.linalg.norm(truth[hold, 3:], axis=1).mean()
                ),
                midturn_contact_torque_norm_nm=float(
                    np.linalg.norm(truth[moving, 3:], axis=1).mean()
                )
                if moving.any()
                else None,
                hold_inferred_torque_norm_nm=float(
                    np.linalg.norm(inferred[hold, 3:], axis=1).mean()
                ),
                hold_marker_delta_rms_n=float(rms[hold].mean()),
                hold_stop_overrun_deg=None
                if spec.get("rotor_stop_deg") is None
                else float(np.maximum(q[hold] - spec["rotor_stop_deg"], 0).mean()),
                max_grasp_drift_mm=float(d["slip_m"][:].max() * 1000),
            ),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [response(p) for p in args.episodes]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for row in rows:
        s = row["spec"]
        load = (
            f"k={s['torsion_nm_rad']:.3f} Nm/rad"
            if s["load_mode"] == "spring"
            else f"detent={s['detent_torque_nm']:.3f} Nm"
            if s["load_mode"] == "detent"
            else f"friction={s['friction_torque_nm']:.3f} Nm"
        )
        label = f"{s['load_mode']}, {load}, stop={s.get('rotor_stop_deg')}° (#{s['episode_id']})"
        (line,) = axes[0].plot(row["t"], row["q"], label=label)
        c = line.get_color()
        axes[1].plot(row["t"], np.linalg.norm(row["truth"][:, 3:], axis=1), color=c)
        axes[1].plot(
            row["t"],
            np.linalg.norm(row["inferred"][:, 3:], axis=1),
            color=c,
            linestyle="--",
            alpha=0.65,
        )
        axes[2].plot(row["t"], row["field_rms"], color=c)
    axes[0].axhline(90, color="gray", linestyle=":")
    axes[0].legend(fontsize=8)
    axes[0].set_ylabel("Rotor angle (deg)")
    axes[1].set_ylabel("Torque norm (Nm)\nsolid contact; dashed force-field")
    axes[2].set_ylabel("Marker Δforce RMS (N/cell)")
    axes[2].set_ylim(
        0, 1.1 * max(float(r["field_rms"][r["t"] >= 3].max()) for r in rows)
    )
    axes[2].set_xlabel("Simulation time (s)")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.set_xlim(left=3)
    fig.suptitle(
        "Turning load response — fixed physical units; reference = settled 3–4 s grasp"
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=160)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            [dict(episode=r["path"], summary=r["summary"]) for r in rows], indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
