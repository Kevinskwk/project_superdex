"""Report the bounded legacy-box / matched-box / source-surface comparison."""

from pathlib import Path
import argparse
import csv
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import trimesh

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parents[1] / "assets/bots/grippers/franka_gelsight_mini"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=HERE / "output/geometry_comparison"
    )
    args = parser.parse_args()
    out = args.output
    names = ["legacy_box", "matched_box", "source_surface"]
    source = trimesh.load(
        ASSETS / "source/gsmini_elastomer_transformed_enclosed.obj",
        force="mesh",
        process=False,
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    geometry = {}
    for name in names:
        path = (
            ASSETS
            / "generated"
            / (
                "gel_tet.mochi.json"
                if name == "legacy_box"
                else f"gel_{name}.mochi.json"
            )
        )
        payload = json.loads(path.read_text())
        vertices = np.asarray(payload["mesh"]["coordinates"]).reshape(-1, 3)
        if name == "legacy_box":
            markers = vertices[:63].reshape(9, 7, 3).transpose(1, 0, 2)
            top = vertices[:63].reshape(9, 7, 3)
        else:
            meta = json.loads(path.with_suffix(".metadata.json").read_text())
            markers = vertices[np.asarray(meta["marker_indices"])]
            top = vertices[:99].reshape(11, 9, 3)
        pitch = [
            np.diff(markers[..., 0], axis=0).mean() * 1000,
            np.diff(markers[..., 1], axis=1).mean() * 1000,
        ]
        geometry[name] = dict(
            marker_pitch_xy_mm=pitch,
            marker_span_xy_mm=np.ptp(markers.reshape(-1, 3)[:, :2], axis=0).tolist(),
            nodes=len(vertices),
        )
        geometry[name]["marker_span_xy_mm"] = (
            np.asarray(geometry[name]["marker_span_xy_mm"]) * 1000
        ).tolist()
        if name != "legacy_box":
            geometry[name]["marker_neighbor_3d_mm"] = [
                float(np.linalg.norm(np.diff(markers, axis=a), axis=-1).max() * 1000)
                for a in (0, 1)
            ]
        axes[0].plot(
            top[len(top) // 2, :, 0] * 1000,
            top[len(top) // 2, :, 2] * 1000,
            "o-",
            label=name,
            ms=3,
        )
        if name != "matched_box":
            axes[1].scatter(
                markers[..., 0] * 1000, markers[..., 1] * 1000, s=12, label=name
            )
    section = source.section(
        plane_origin=source.bounds.mean(axis=0), plane_normal=[0, 1, 0]
    )
    for line in section.discrete:
        axes[0].plot(line[:, 0] * 1000, line[:, 2] * 1000, "k:", lw=1, label="_source")
    axes[0].set(
        xlabel="Sensor X [mm]",
        ylabel="Sensor Z [mm]",
        title="Gel center section; dotted = source",
    )
    axes[1].set(
        xlabel="Sensor X [mm]",
        ylabel="Sensor Y [mm]",
        title="7 x 9 markers: projected layout",
    )
    axes[1].set_aspect("equal")
    shape = json.loads((ASSETS / "generated/gel_source_surface.mochi.json").read_text())
    top = np.asarray(shape["mesh"]["coordinates"]).reshape(-1, 3)[:99].reshape(11, 9, 3)
    axes[2].pcolormesh(
        top[..., 0] * 1000, top[..., 1] * 1000, top[..., 2] * 1000, shading="nearest"
    )
    axes[2].set(
        xlabel="Sensor X [mm]",
        ylabel="Sensor Y [mm]",
        title="Curved exposed height (source-fitted)",
    )
    for ax in axes[:2]:
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "geometry_and_markers.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    rows_out = []
    for name in names:
        metrics = json.loads((out / name / "metrics.json").read_text())
        with (out / name / "contact_wrenches.csv").open() as f:
            rows = list(csv.DictReader(f))
        data = lambda key: np.array([float(r[key]) for r in rows])
        t = data("time_since_incidence_s")
        axes[0].plot(t, data("table_on_plug_force_world_z"), label=name, lw=1)
        axes[1].plot(t, data("max_gel_indentation_m") * 1000, label=name, lw=1)
        npz = np.load(out / name / "tactile_fields.npz")
        active = np.count_nonzero(
            np.linalg.norm(npz["left_force_field"], axis=-1) > 0.01, axis=(1, 2)
        )
        axes[2].plot(t, active, label=name, lw=1)
        w = metrics["dense_field_wrench"]
        rows_out.append(
            dict(
                name=name,
                force_rmse_n=w["inferred_vs_extrinsic_force"]["vector_rmse"],
                marker_bin_torque_rmse_mnm=w["inferred_vs_extrinsic_torque"][
                    "vector_rmse"
                ]
                * 1000,
                peak_marker_indentation_mm=metrics["causality"]["max_indentation_m"]
                * 1000,
                contact_fraction=min(
                    v["both_gels_contact_fraction"]
                    for v in metrics["plateaus"].values()
                ),
            )
        )
    for ax, label in zip(
        axes,
        [
            "Table normal force [N]",
            "Peak marker indentation [mm]",
            "Left active force bins (>0.01 N)",
        ],
    ):
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
        ax.legend()
    axes[-1].set_xlabel("Time since first table contact [s]")
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=170)
    plt.close(fig)
    dense = {}
    for name in ["matched_dense", "source_review"]:
        path = out / name / "contact_wrenches.csv"
        if not path.exists():
            continue
        with path.open() as f:
            rows = list(csv.DictReader(f))
        rows = [r for r in rows if r["phase"].startswith("hold_")]

        def vectors(prefix, quantity):
            return np.array(
                [
                    [float(r[f"{prefix}_{quantity}_world_{a}"]) for a in "xyz"]
                    for r in rows
                ]
            )

        dense[name] = {}
        for q, unit in [("force", "n"), ("torque", "nm")]:
            unbinned = vectors("left_dense_on_plug", q) + vectors(
                "right_dense_on_plug", q
            )
            direct = vectors("gels_on_plug", q)
            binned = vectors("field_gels_on_plug", q)
            dense[name][f"dense_vs_direct_{q}_rmse_{unit}"] = float(
                np.sqrt(np.mean(np.sum((unbinned - direct) ** 2, axis=1)))
            )
            dense[name][f"binned_vs_dense_{q}_rmse_{unit}"] = float(
                np.sqrt(np.mean(np.sum((unbinned - binned) ** 2, axis=1)))
            )
    (out / "comparison.json").write_text(
        json.dumps(
            dict(geometry=geometry, comparison=rows_out, dense_wrench_checks=dense),
            indent=2,
        )
        + "\n"
    )
    text = [
        "# Gel geometry and marker-layout comparison",
        "",
        "Three matched nominal plug/table runs; 200 kPa, Poisson 0.49, friction 1.4, 18 N per jaw. Same controller and 2/5/10 N targets. No parameter retuning. This is a small mechanical check, not calibration or mesh convergence.",
        "",
        "| Geometry | Peak marker indentation (mm) | Field → table force RMSE (N) | Marker-bin → table torque RMSE (mNm) | Both gels contact during holds |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows_out:
        text.append(
            f"| {r['name']} | {r['peak_marker_indentation_mm']:.3f} | {r['force_rmse_n']:.3f} | {r['marker_bin_torque_rmse_mnm']:.3f} | {r['contact_fraction']:.0%} |"
        )
    text += [
        "",
        "Wrench errors use the existing runner loading-window/quasistatic convention. The torque column includes spatial binning error; it is not an isolated sensor-accuracy metric. Source-vs-matched-box isolates the sampled height change; legacy comparisons also change resolution and tetrahedral topology.",
        "",
        "## Marker verification",
        "",
        "- Original box: 3.458 x 3.156 mm pitch, spanning 20.75 x 25.25 mm. It did not satisfy the requested marker layout.",
        "- New source and matched box: centered 7 x 9 lattice, exactly 2 mm projected X/Y pitch and 12 x 16 mm span. Source curvature makes the largest 3D neighbor distance approximately 2.002 mm; this is not a geodesic-spacing constraint.",
        "- Markers are virtual displacement sample positions. Force bins contain summed nodal contact loads, not an optical marker-to-force reconstruction. Shoulder nodes map to the nearest marker; net force is conserved, but torque need not be. `get_dense_contact_field` exposes unbinned world-space positions and forces for wrench integration.",
        "",
        "## Geometry scope",
        "",
        "The large +Z backing is fixed; the narrower curved -Z side faces the object on both fingers. The new 297-node / 960-tet volume samples the original HydroShear front surface, retaining its rounded/narrowing profile. It is a coarse source-fitted approximation, not an exact or mesh-converged reproduction. Source-surface is now the default; select `--gel-geometry legacy_box` to reproduce the historical box.",
        "",
        "![Geometry and marker layout](geometry_and_markers.png)",
        "",
        "![Matched contact histories](comparison.png)",
        "",
        "Detailed unbinned-versus-binned wrench checks are in [comparison.json](comparison.json). [Source-shaped RGB review](source_review/simulation.mp4).",
        "",
    ]
    if "source_review" in dense:
        d = dense["source_review"]
        text += [
            "## Dense versus marker-bin wrench",
            "",
            f"During the source-shaped hold stages, dense nodal torque versus direct gel/tool contact torque has {d['dense_vs_direct_torque_rmse_nm'] * 1000:.3f} mNm RMSE. Collapsing that field into the 63 marker bins adds {d['binned_vs_dense_torque_rmse_nm'] * 1000:.3f} mNm RMS difference. Retain the dense nodal wrench for accurate torque accounting; the compact field is not lossless.",
            "",
        ]
    (out / "REPORT.md").write_text("\n".join(text))
    print(json.dumps(rows_out, indent=2))


if __name__ == "__main__":
    main()
