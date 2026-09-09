"""Coverage, full-episode validity, contact topology and wrench audit for the pilot."""

from collections import defaultdict
from pathlib import Path
import argparse
import json
import hashlib
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from benchmark_geometry import (
    sequence_completion,
    annotate_friction,
    profile,
    offset_profile,
    surface_stroke_audit,
)
from benchmark_geometry import (
    episode_eligibility,
    gel_benchmark_role,
    peeling_contact_audit,
    surface_task_pass,
)


def measured_surface_stroke(f):
    """Independent task check, also valid for older recordings without tip data."""
    s = json.loads(f.attrs["config_json"])
    d = f["observations"]
    v = f["geometry/0/vertices"][:]
    tip = (
        np.asarray(f.attrs["working_end_tool_m"])
        if "working_end_tool_m" in f.attrs
        else v[v[:, 2] <= v[:, 2].min() + 1e-6].mean(0)
    )
    poses = d["body_root_poses"][:, 0]
    world = (
        Rotation.from_quat(poses[:, 3:]).apply(np.tile(tip, (len(poses), 1)))
        + poses[:, :3]
    )
    task = Rotation.from_quat(poses[0, 3:]).inv().apply(world - poses[0, :3])
    length = s.get("surface_stroke_m") or (
        0.003
        if "peel" in s["variant"]
        else 0.006
        if "scrape" in s["variant"]
        else 0.020
    )
    return surface_stroke_audit(d["phase"].asstr()[:], task, s["dt"], length)


def surface_overlap(f):
    """Full-history sampled clearance against the saved extruded height surface.

    Uses recorded mesh geometry/poses, not simulator contact-point distances.
    Point-to-local-plane depths are sampled lower bounds, not exact global SDFs.
    """
    d = f["observations"]
    g = f["geometry"]
    tool = g["0"]
    surface = g["1"]["vertices"][:]
    spec = json.loads(f.attrs.get("config_json", "{}"))
    cylinder = spec.get("variant") == "cylindrical_peel"
    vertices, faces = trimesh.remesh.subdivide_to_size(
        tool["vertices"][:], tool["faces"][:], max_edge=0.002, max_iter=10
    )
    points = vertices
    xs = np.unique(surface[:, 0])
    top = np.array([surface[surface[:, 0] == x, 2].max() for x in xs])
    slopes = np.diff(top) / np.diff(xs)
    ymin, ymax = surface[:, 1].min(), surface[:, 1].max()
    poses = d["body_root_poses"][:, :2]
    depth = np.zeros(len(poses))
    for start in range(0, len(poses), 32):
        pp = poses[start : start + 32]
        rr = Rotation.from_quat(pp[:, 0, 3:]).as_matrix()
        inv = Rotation.from_quat(pp[:, 1, 3:]).inv().as_matrix()
        world = np.einsum("bij,nj->bni", rr, points) + pp[:, 0, None, :3]
        local = np.einsum("bij,bnj->bni", inv, world - pp[:, 1, None, :3])
        x, y, z = np.moveaxis(local, -1, 0)
        segment = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, len(slopes) - 1)
        height = top[segment] + slopes[segment] * (x - xs[segment])
        inside = (
            (x >= xs[0])
            & (x <= xs[-1])
            & (y >= ymin)
            & (y <= ymax)
            & (z >= surface[:, 2].min())
        )
        overlap = np.where(
            inside, np.maximum(0, height - z) / np.sqrt(1 + slopes[segment] ** 2), 0
        )
        if cylinder:
            center = (surface.min(0) + surface.max(0)) / 2
            radius = (surface[:, 1].max() - surface[:, 1].min()) / 2
            radial = radius - np.sqrt((y - center[1]) ** 2 + (z - center[2]) ** 2)
            overlap = np.maximum(
                0, np.minimum(radial, np.minimum(x - xs[0], xs[-1] - x))
            )
        depth[start : start + len(pp)] = overlap.max(axis=1)
    active = ~d["initialization"][:]
    bad = (depth > 0.001) & active
    sustained = bool(np.convolve(bad.astype(int), np.ones(3), "valid").max() >= 3)
    return dict(
        max_sampled_overlap_mm=float(depth[active].max() * 1000),
        sustained_over_1mm=sustained,
        failed=sustained,
        sampling_edge_mm=2.0,
        frames_checked=len(depth),
        note="Independent full-history height-surface sampling; wall/slot side contacts require a separate mesh audit.",
    )


def cached_surface_overlap(f):
    digest = hashlib.sha256(b"height_surface_full_history_v2_cylinder_2mm")
    for key in (
        "geometry/0/vertices",
        "geometry/0/faces",
        "geometry/1/vertices",
        "observations/body_root_poses",
        "observations/initialization",
    ):
        digest.update(f[key][:].tobytes())
    folder = Path(f.filename).parents[1] / "geometry_checks"
    folder.mkdir(exist_ok=True)
    path = folder / (Path(f.filename).stem + "_" + digest.hexdigest()[:16] + ".json")
    if path.exists():
        return json.loads(path.read_text())
    result = surface_overlap(f)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def geometry_audit(path):
    """Check every instrumented environment mesh, not fixed actor indices."""
    with h5py.File(path) as f:
        d = f["observations"]
        geom = f["geometry"]
        indices = [
            int(k) for k in geom if geom[k].attrs["role"] in ("object", "environment")
        ]
        meshes = {
            i: trimesh.Trimesh(
                geom[str(i)]["vertices"][:], geom[str(i)]["faces"][:], process=False
            )
            for i in indices
        }
        closed = {
            i: bool(m.is_watertight and m.is_winding_consistent and m.volume > 0)
            for i, m in meshes.items()
        }
        if not all(closed.values()):
            return dict(
                closed=closed, passed=False, reason="nonclosed_or_inverted_mesh"
            )
        frames = sorted(
            set(
                np.linspace(0, len(d["timestamps"]) - 1, 9, dtype=int).tolist()
                + [int(np.argmax(d["rigid_penetration_m"][:]))]
            )
        )
        samples = {
            i: np.vstack(
                [m.vertices, trimesh.sample.sample_surface(m, 300, seed=42)[0]]
            )
            for i, m in meshes.items()
        }
        results = []
        for frame in frames:
            poses = d["body_root_poses"][frame]
            peak = 0.0
            for j in indices[1:]:
                for source, target in ((0, j), (j, 0)):
                    points = (
                        Rotation.from_quat(poses[source, 3:]).apply(samples[source])
                        + poses[source, :3]
                    )
                    points = (
                        Rotation.from_quat(poses[target, 3:])
                        .inv()
                        .apply(points - poses[target, :3])
                    )
                    inside = meshes[target].contains(points)
                    if inside.any():
                        _, distance, _ = trimesh.proximity.closest_point(
                            meshes[target], points[inside]
                        )
                        peak = max(peak, float(distance.max()))
            results.append(
                dict(
                    time_s=float(d["timestamps"][frame]), sampled_overlap_mm=peak * 1000
                )
            )
        return dict(
            closed=closed,
            max_sampled_overlap_mm=max(r["sampled_overlap_mm"] for r in results),
            frames=results,
            note="Independent sampled diagnostic, not proof of zero intersection; initialization included.",
        )


def summarize(path):
    actual_friction = annotate_friction(path)
    with h5py.File(path) as f:
        spec = json.loads(f.attrs["config_json"])
        m = json.loads(f.attrs["metrics_json"])
        d = f["observations"]
        if spec["family"] in ("insertion", "turning", "composite"):
            completion = sequence_completion(
                dict(
                    phase=d["phase"].asstr()[:],
                    rotor_angle_rad=d["rotor_angle_rad"][:],
                    insertion_depth_m=d["insertion_depth_m"][:],
                    seat_valid=d["seat_valid"][:],
                ),
                spec["family"],
                spec["dt"],
                return_after_turn=spec.get("return_after_turn", True),
            )
            m["recorded_task_success"] = m["task_success"]
            m.update(completion)
            m["task_success"] &= completion["sequence_complete"]
            m["imitation_eligible"] &= m["task_success"]
            m.pop("max_progress_mm", None)
        independent = None
        if spec["family"] == "surface":
            independent = cached_surface_overlap(f)
            m["recorded_task_success"] = m["task_success"]
            m.update(measured_surface_stroke(f))
            m["task_success"] &= m["follow_stroke_pass"]
            if spec["variant"] == "cylindrical_peel":
                m.update(
                    peeling_contact_audit(
                        dict(
                            phase=d["phase"].asstr()[:],
                            initialization=d["initialization"][:],
                            blade_contact_force_n=d["blade_contact_force_n"][:],
                            holder_contact_force_n=d["holder_contact_force_n"][:],
                        )
                    )
                )
                m["task_success"] &= m["blade_contact_pass"]
            m["task_success"] = surface_task_pass(m, spec["variant"])
            m["recorded_physical_valid"] = m["physical_valid"]
            m["physical_valid"] &= not independent["failed"]
            m["imitation_eligible"] = bool(
                m["physical_valid"]
                and m["task_success"]
                and m.get("full_rollout_complete", False)
            )
            m["tactile_valid"] &= m["physical_valid"]
        contact = d["contact_event"][:]
        active = ~d["initialization"][:]
        truth = d["extrinsic_contact_wrench"][:]
        est = d["dynamic_inferred_extrinsic_wrench"][:]
        loaded = contact | (np.linalg.norm(truth[:, 3:], axis=1) > 0.005)
        planar = None
        if spec["family"] == "surface" and spec["variant"].startswith("flat_"):
            local = (
                Rotation.from_quat(d["body_root_poses"][:, 1, 3:])
                .inv()
                .apply(truth[:, :3])
            )
            selected = (d["phase"].asstr()[:] == "follow") & (local[:, 2] > 0.2)
            if selected.any():
                tangent = np.linalg.norm(local[selected, :2], axis=1)
                normal = local[selected, 2]
                mu = actual_friction["effective_environment_tool_friction"]
                fraction = float((tangent > mu * normal + 0.2).mean())
                planar = dict(
                    apparent_friction_ratio_median=float(np.median(tangent / normal)),
                    apparent_friction_ratio_p95=float(
                        np.quantile(tangent / normal, 0.95)
                    ),
                    configured_coulomb_mu=mu,
                    excess_fraction=fraction,
                    review_required=fraction > 0.05,
                    note="Top-plane force-direction diagnostic, not a per-contact friction-law proof. Edge/oblique manifold normals can invalidate the plane-only interpretation.",
                )
                if "oblique_contact_normal_load_fraction" in d:
                    planar["mean_oblique_normal_load_fraction"] = float(
                        d["oblique_contact_normal_load_fraction"][:][selected].mean()
                    )
                    planar["max_per_contact_coulomb_excess_n"] = float(
                        d["per_contact_coulomb_excess_n"][:][selected].max()
                    )
        m["contact_model_review_required"] = bool(
            (planar and planar["review_required"]) or spec["variant"] == "convex_scrape"
        )
        m["benchmark_variant_retired"] = (
            spec["family"] == "surface" and spec["variant"] == "convex_scrape"
        )
        if m["benchmark_variant_retired"]:
            m.update(episode_eligibility(m))
        dt = spec["dt"]
        phase = d["phase"].asstr()[:]
        transitions = []
        for i in np.flatnonzero(loaded[1:] != loaded[:-1]) + 1:
            if active[i]:
                transitions.append(
                    dict(
                        time_s=float(d["timestamps"][i]),
                        event="load_on" if loaded[i] else "load_off",
                    )
                )
        fields = np.stack(
            [d["tactile_force_field_" + s][:] for s in ("left", "right")], axis=1
        )
        data = dict(
            spec=spec,
            metrics=m,
            actual_friction=actual_friction,
            planar_force_direction=planar,
            independent_surface_geometry=independent,
            file=str(path),
            frames=len(contact),
            seconds_simulated=len(contact) * dt,
            events=transitions,
            zero_tactile_fraction=float(
                (np.linalg.norm(fields.reshape(len(fields), -1), axis=1) < 1e-10)[
                    active
                ].mean()
            ),
            force_rms=float(np.sqrt(np.mean(np.sum(truth[active, :3] ** 2, axis=1)))),
            torque_rms=float(np.sqrt(np.mean(np.sum(truth[active, 3:] ** 2, axis=1)))),
            contact_dwell_s=float(loaded[active].sum() * dt),
            force_only_contact_dwell_s=float(contact[active].sum() * dt),
            event_definition="Net force >0.1 N OR net moment about tool COM >0.005 Nm; includes near-pure couples.",
            phases=list(dict.fromkeys(phase)),
        )
        data["effect_vector"] = [
            data["force_rms"],
            data["torque_rms"],
            float(loaded[active].mean()),
            float(np.linalg.norm(truth[active, :2], axis=1).mean()),
            float(np.abs(truth[active, 2]).mean()),
            float(np.linalg.norm(truth[active, 3:5], axis=1).mean()),
            float(np.abs(truth[active, 5]).mean()),
        ]
        if spec["family"] == "surface" and "per_contact_coulomb_excess_n" in d:
            follow = phase == "follow"
            if follow.any():
                data["surface_contact_solver_diagnostics"] = dict(
                    mean_oblique_normal_load_fraction=float(
                        d["oblique_contact_normal_load_fraction"][:][follow].mean()
                    ),
                    max_per_contact_coulomb_excess_n=float(
                        d["per_contact_coulomb_excess_n"][:][follow].max()
                    ),
                    note="Normals are compared with the fixture Z axis, not the local curved-surface normal; obliqueness alone is not a failure. Coulomb excess is a solver diagnostic, not a tactile-wrench gate.",
                )
        if "dense_nodal_inferred_extrinsic_wrench" in d:
            dense = d["dense_nodal_inferred_extrinsic_wrench"][:]
            delta = dense[active] - truth[active]
            data["dense_nodal_force_rmse_n"] = float(
                np.sqrt(np.mean(np.sum(delta[:, :3] ** 2, axis=1)))
            )
            data["dense_nodal_torque_rmse_nm"] = float(
                np.sqrt(np.mean(np.sum(delta[:, 3:] ** 2, axis=1)))
            )
            data["marker_binning_torque_difference_rmse_nm"] = float(
                np.sqrt(
                    np.mean(np.sum((dense[active, 3:] - est[active, 3:]) ** 2, axis=1))
                )
            )
        if "moment_corrected_inferred_extrinsic_wrench" in d:
            corrected = d["moment_corrected_inferred_extrinsic_wrench"][:]
            delta = corrected[active] - truth[active]
            data["moment_corrected_force_rmse_n"] = float(
                np.sqrt(np.mean(np.sum(delta[:, :3] ** 2, axis=1)))
            )
            data["moment_corrected_torque_rmse_nm"] = float(
                np.sqrt(np.mean(np.sum(delta[:, 3:] ** 2, axis=1)))
            )
            data["moment_corrected_vs_dense_torque_rmse_nm"] = float(
                np.sqrt(
                    np.mean(
                        np.sum((corrected[active, 3:] - dense[active, 3:]) ** 2, axis=1)
                    )
                )
            )
            data["unmapped_force_peak_n"] = max(
                float(
                    np.linalg.norm(
                        d[f"tactile_unmapped_wrench_{s}"][:][active, :3], axis=1
                    ).max()
                )
                for s in ("left", "right")
            )
        if spec["family"] in ("turning", "composite"):
            rotor = (
                next(
                    int(k)
                    for k in f["geometry"]
                    if f["geometry"][k].attrs["name"] == "env_0"
                )
                if spec["profile_shape"] != "legacy"
                else 2
            )
            poses = d["body_root_poses"][:, rotor]
            axis = Rotation.from_quat(poses[:, 3:]).apply(
                np.tile([0, 0, 1], (len(poses), 1))
            )
            lever = d["tool_pose"][:, :3] - poses[:, :3]
            ground = np.sum(
                (truth[:, 3:] + np.cross(lever, truth[:, :3])) * axis, axis=1
            )
            inferred = np.sum((est[:, 3:] + np.cross(lever, est[:, :3])) * axis, axis=1)
            data["rotor_axis_torque_rmse_nm"] = float(
                np.sqrt(np.mean((ground[active] - inferred[active]) ** 2))
            )
        return data


def branch_audit(root):
    groups = defaultdict(list)
    for path in root.glob("branches*/*.h5"):
        with h5py.File(path) as f:
            groups[int(f.attrs["branch_anchor_id"])].append(path)
    result = []
    for anchor, paths in groups.items():
        arrays = []
        prefix = []
        fields = []
        n = None
        for path in sorted(paths):
            with h5py.File(path) as f:
                n = int(f.attrs["branch_frame"])
                d = f["observations"]
                arrays.append(d["actions"][:])
                fields.append(d["tactile_force_field_left"][:])
                prefix.append(str(f.attrs["common_prefix_hash"]))
        end = min(map(len, arrays))
        action_spread = (
            float(np.ptp(np.stack([a[n:end] for a in arrays]), axis=0).max())
            if end > n
            else 0.0
        )
        tactile_spread = (
            float(np.ptp(np.stack([a[n:end] for a in fields]), axis=0).max())
            if end > n
            else 0.0
        )
        result.append(
            dict(
                anchor_id=anchor,
                branches=len(paths),
                common_prefix=len(set(prefix)) == 1,
                distinct_executed_actions=action_spread > 1e-8,
                max_action_spread_m_or_rad=action_spread,
                max_left_taxel_spread_n=tactile_spread,
                note="Identical-action continuations are replay controls, not causal interventions. Invalid branches remain included.",
            )
        )
    return result


def wrench_plot(rows, path):
    selected = []
    for family in ("surface", "hook", "insertion", "turning", "composite"):
        good = [
            r
            for r in rows
            if r["spec"]["family"] == family
            and r["spec"]["block"] != "branches"
            and r["metrics"]["physical_valid"]
            and r["metrics"]["task_success"]
        ]
        if good:
            selected.append(
                sorted(
                    good,
                    key=lambda r: (
                        r["spec"]["profile_shape"] == "legacy",
                        r["spec"]["episode_id"],
                    ),
                )[0]
            )
    if not selected:
        return []
    fig, axes = plt.subplots(
        2, len(selected), figsize=(4 * len(selected), 6), squeeze=False
    )
    for col, r in enumerate(selected):
        with h5py.File(r["file"]) as f:
            d = f["observations"]
            active = ~d["initialization"][:]
            t = d["timestamps"][:][active]
            for row, offset in enumerate((0, 3)):
                for component, color in enumerate(("tab:red", "tab:green", "tab:blue")):
                    for key, style in (
                        ("extrinsic_contact_wrench", "-"),
                        ("dynamic_inferred_extrinsic_wrench", "--"),
                    ):
                        axes[row, col].plot(
                            t,
                            d[key][:][active, offset + component],
                            style,
                            color=color,
                            lw=0.8,
                            label=(
                                "xyz"[component]
                                + " "
                                + ("contact" if style == "-" else "tactile")
                            ),
                        )
                axes[row, col].set_ylabel("Force [N]" if row == 0 else "Torque [Nm]")
                axes[row, col].set_xlabel("Time [s]")
                axes[row, col].grid(alpha=0.2)
            axes[0, col].set_title(
                f"{r['spec']['family']}/{r['spec']['variant']} #{r['spec']['episode_id']}"
            )
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(
        "Contact vs force-field inferred wrench\nWorld axes; moment about tool COM",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return [r["file"] for r in selected]


def profile_plot(path):
    fig, axes = plt.subplots(1, 5, figsize=(12, 3))
    for ax, shape in zip(axes, ("round", "square", "hex", "d", "key")):
        p = profile(shape) * 1000
        q = offset_profile(profile(shape), 0.0006) * 1000
        ax.fill(p[:, 0], p[:, 1], color="tab:blue", alpha=0.5, label="Peg")
        q = np.vstack([q, q[:1]])
        ax.plot(q[:, 0], q[:, 1], "--", color="tab:orange", label="Pocket")
        ax.set_aspect("equal")
        ax.set_xlim(-7, 7)
        ax.set_ylim(-7, 7)
        ax.set_title(shape)
        ax.set_xlabel("mm")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Implemented cross-sections; nominal 0.6 mm edge-normal clearance")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def report(root, mesh=False):
    rows = []
    errors = []
    for phase in sorted(root.iterdir()):
        if not phase.is_dir() or phase.name in (
            "media",
            "physics_media",
            "source_snapshot",
        ):
            continue
        for path in sorted(phase.glob("*.h5")):
            if path.with_suffix(".json").exists():
                rows.append(summarize(path))
        for path in sorted(phase.glob("*.json")):
            r = json.loads(path.read_text())
            if isinstance(r, dict) and r.get("error"):
                errors.append(r)
    grouped = defaultdict(list)
    for row in rows:
        s = row["spec"]
        grouped[s["family"], s["variant"], s["block"]].append(row)
    text = [
        "# Contact-rich tool-use diversity pilot",
        "",
        "Local mechanics/observability screening, not a frozen benchmark, learned policy, real GelSight calibration or sim-to-real result. All attempts remain in the ledger; repeated nominal runs are not independent physical conditions.",
        "",
        "Active scraping is flat-only. Convex scraping was retired by user decision; its historical rows remain for provenance but are excluded from active benchmark eligibility.",
        "",
        "## Coverage and full-episode qualification",
        "",
        "| Family / variant / block | Episodes | Stability/clearance pass | Task successes among those | Force RMSE median (N) | Torque RMSE median (Nm) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    cells = []
    for key, rr in grouped.items():
        valid = [r for r in rr if r["metrics"]["physical_valid"]]
        nominal = [r for r in rr if r["spec"]["case"] == "nominal"]
        qualified = len(nominal) >= 3 and all(
            r["metrics"]["physical_valid"] and r["metrics"]["task_success"]
            for r in nominal
        )
        qualified &= not any(
            r["metrics"].get("contact_model_review_required")
            or r["metrics"].get("benchmark_variant_retired")
            for r in nominal
        )
        if "wall" in key[1] or "slot" in key[1]:
            qualified &= all(
                r["metrics"].get("guide_contact_continuity", 0) >= 0.8 for r in nominal
            )
        cells.append(
            dict(
                family=key[0],
                variant=key[1],
                block=key[2],
                episodes=len(rr),
                valid=len(valid),
                successful=sum(r["metrics"]["task_success"] for r in valid),
                nominal_qualified=qualified,
            )
        )
        text.append(
            f"| {' / '.join(key)} | {len(rr)} | {len(valid)}/{len(rr)} | {sum(r['metrics']['task_success'] for r in valid)} | {np.median([r['metrics']['force_rmse'] for r in rr]):.4f} | {np.median([r['metrics']['torque_rmse'] for r in rr]):.5f} |"
        )
    text += [
        "",
        "Task success and mechanics validity are separate. Clearance/grasp failures are not recast as ordinary task failures. Full recorded endpoints are included: historical return/withdrawal or revision-16 turn-and-hold. Wrench matching is diagnostic only, never an episode rejection gate.",
        "Surface validity additionally includes an independent, full-history 2 mm mesh-sampling check against the saved height surface. This can revoke a recorder pass; raw HDF5 metrics are preserved and `recorded_physical_valid` exposes the difference.",
        "",
        "## Interpretation and limits",
        "",
        "- Surface variants are rigid contact/path tasks: no ink deposition, cutting or material removal. SCFields convex collision proxies are identified in each episode, never claimed to preserve a peeler aperture.",
        "- Round peg yaw is symmetric; square/hexagonal/D/key profiles supply different rotational constraints. Peg/socket geometry shares a profile; seating uses actual working-end vertices.",
        "- Collector tool-pose feedback and surface force control are privileged and recorded. Tactile usefulness must be tested with these channels excluded.",
        "- Wrenches use world axes about the instantaneous tool COM with Newton–Euler compensation. Rotor-axis diagnostics explicitly transport moments; same-engine agreement is not independent sensor calibration.",
        "- Derived load events include force >0.1 N OR moment >0.005 Nm. The legacy recorder contact/false-contact fractions are force-only and must not label a near-pure torque couple as free space.",
        "- Flat-contact force-direction review flags sustained tangential force above the plane-only Coulomb bound plus 0.2 N tolerance. Such episodes can pass stability/geometry yet remain excluded from the default training index pending contact-manifold review; this is not by itself proof that individual contact points violate Coulomb friction.",
        "- Three nominal passes qualify only the local variant; the candidate randomization range remains unqualified unless its own trials pass. No timestep-invariant impact-peak claim is made.",
        "",
        "Friction provenance: an intermediate PegWorld helper overrode both tool and socket coefficients. Affected episodes retain valid measured signals but are not independent friction sweeps. Derived HDF5 coefficients are corrected from source snapshots, original reported values are preserved, and `audited_friction_json` records the correction. The current helper only transforms the environment coefficient.",
        "",
        "## Invalid episodes / runtime failures",
        "",
    ]
    for r in rows:
        m = r["metrics"]
        s = r["spec"]
        if not m["physical_valid"]:
            text.append(
                f"- {s['episode_id']:04d} {s['family']}/{s['variant']}: abort={m.get('abort_reason')}, penetration={m['max_penetration_mm']:.3f} mm, drift={m['max_slip_mm']:.2f} mm / {m['max_angular_slip_deg']:.2f}°."
            )
            if (
                r.get("independent_surface_geometry", {})
                and r["independent_surface_geometry"]["failed"]
            ):
                text.append(
                    f"  Independent surface overlap: {r['independent_surface_geometry']['max_sampled_overlap_mm']:.3f} mm, sustained above 1 mm."
                )
    for r in errors:
        text.append(f"- {r['spec']['episode_id']:04d}: {r['error']}")
    text += ["", "## Contact-model review flags", ""]
    for r in rows:
        if r["metrics"].get("contact_model_review_required"):
            p = r["planar_force_direction"]
            if p:
                text.append(
                    f"- {r['spec']['episode_id']:04d} {r['spec']['variant']}: apparent tangential/normal ratio median {p['apparent_friction_ratio_median']:.2f}, p95 {p['apparent_friction_ratio_p95']:.2f}; configured Coulomb coefficient {p['configured_coulomb_mu']:.2f}. Requires manifold/edge-contact interpretation; not calibrated frictional-shear evidence."
                )
            else:
                text.append(
                    f"- {r['spec']['episode_id']:04d} {r['spec']['variant']}: same broad-blade representation is conservatively held for contact-manifold review; the flat-plane diagnostic is not directly applicable to a curved surface."
                )
    times = []
    for path in root.glob("*/summary.json"):
        s = json.loads(path.read_text())
        if "seconds" in s:
            times.append(
                dict(
                    phase=path.parent.name,
                    wall_s=s["seconds"],
                    workers=s["workers"],
                    attempts=len(s["episodes"]),
                )
            )
    attempts = (
        json.loads((root / "attempts.json").read_text())
        if (root / "attempts.json").exists()
        else []
    )
    resolved = {r["spec"]["episode_id"] for r in rows} | {
        r["spec"]["episode_id"] for r in errors
    }
    pending = [a["episode_id"] for a in attempts if a["episode_id"] not in resolved]
    censored = sum(
        r["metrics"].get("abort_reason") == "runtime_budget_exceeded" for r in rows
    )
    text += [
        "",
        "## Runtime",
        "",
        f"Reserved: {len(attempts)}/256; recorded: {len(rows)}; runtime errors: {len(errors)}; unresolved reservations: {len(pending)}; wall-time-censored recordings: {censored}.",
        "Phases ran concurrently on shared CPUs. These wall times are not isolated throughput benchmarks and must not be added. Runtime censoring is not itself evidence of a physics defect.",
        "",
    ]
    for t in times:
        text.append(
            f"- {t['phase']}: {t['attempts']} attempts, {t['wall_s']:.1f} s wall, {t['workers']} CPU workers; {t['attempts'] / t['wall_s'] * 3600:.1f} attempted episodes/hour. Includes worker-side rendering when enabled; excludes this post-run report."
        )
    audits = {}
    if mesh:
        for family in ("surface", "hook", "insertion", "turning", "composite"):
            variants = sorted(
                {r["spec"]["variant"] for r in rows if r["spec"]["family"] == family}
            )
            for variant in variants:
                candidates = [
                    r
                    for r in rows
                    if r["spec"]["family"] == family
                    and r["spec"]["variant"] == variant
                    and r["metrics"]["physical_valid"]
                    and r["metrics"]["task_success"]
                    and r["spec"]["block"] != "branches"
                ]
                if not candidates:
                    continue
                path = Path(candidates[0]["file"])
                audits[path.name] = geometry_audit(path)
        (root / "mesh_audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    if rows:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        families = sorted({r["spec"]["family"] for r in rows})
        for ax, metric, label in zip(
            axes,
            ("force_rmse", "torque_rmse", "max_slip_mm"),
            ("Force RMSE [N]", "Torque RMSE [Nm]", "Grasp drift [mm]"),
        ):
            ax.boxplot(
                [
                    [
                        r["metrics"][metric]
                        for r in rows
                        if r["spec"]["family"] == family
                    ]
                    for family in families
                ],
                tick_labels=families,
            )
            ax.set_ylabel(label)
            ax.tick_params(axis="x", rotation=25)
            ax.grid(alpha=0.2)
        fig.suptitle(
            "All recorded outcomes, including invalid trials — not a qualified-only accuracy claim"
        )
        fig.tight_layout()
        fig.savefig(root / "quality.png", dpi=160)
        plt.close(fig)
        text += ["", "![All-outcome quality](quality.png)"]
    text += [
        "",
        "## Retention",
        "",
        "Keep final data, failed-case evidence, manifests, reports and representative RGB/tactile videos. No automatic deletion or replacement of earlier studies.",
    ]
    branch_results = branch_audit(root)
    if branch_results:
        text += ["", "## Same-state action branches", ""]
        for b in branch_results:
            text.append(
                f"- Anchor {b['anchor_id']}: {b['branches']} branches; common prefix={b['common_prefix']}; distinct executed actions={b['distinct_executed_actions']}; maximum left taxel spread={b['max_left_taxel_spread_n']:.4f} N."
            )
        text += [
            "Identical-action groups are replay controls, not evidence of action-to-tactile causality. Different actions and fields establish local responsiveness, not learned policy benefit."
        ]
    examples = wrench_plot(rows, root / "representative_wrenches.png")
    if examples:
        text += [
            "",
            "![Representative full-rollout wrenches](representative_wrenches.png)",
        ]
    profile_plot(root / "peg_profiles.png")
    text += ["", "![Peg and pocket profiles](peg_profiles.png)"]
    text += [
        "",
        "## Targeted tactile observability",
        "",
        "Pre-turn is already seated, not guaranteed contact-free. These are held-out grip-group prediction probes, not a learned closed-loop policy or cross-domain transfer result.",
        "",
        "| Window / camera | Vision BA | Vision + field BA | Net wrench BA | Vision + field torque MAE (mNm) |",
        "|---|---:|---:|---:|---:|",
    ]
    probe_notes = []
    for window in ("before", "after"):
        path = root / f"probe_{window}/probes.json"
        if not path.exists():
            probe_notes.append(f"- {window}: pending.")
            continue
        probe = json.loads(path.read_text())
        if probe["status"] != "complete":
            probe_notes.append(
                f"- {window}: qualification failed: {probe.get('gate', {}).get('reasons', [])}"
            )
            continue
        for camera in dict.fromkeys(r["condition"] for r in probe["results"]):
            rows_by_mode = {
                r["mode"]: r for r in probe["results"] if r["condition"] == camera
            }
            regression = next(
                r
                for r in probe["regression"]
                if r["condition"] == camera and r["mode"] == "vision_tactile"
            )
            values = [
                rows_by_mode[k]["balanced_accuracy"] * 100
                for k in ("vision", "vision_tactile", "wrench")
            ]
            text.append(
                f"| {window} / {camera} | {values[0]:.1f}% | {values[1]:.1f}% | {values[2]:.1f}% | {regression['mae_nm'] * 1000:.2f} |"
            )
        probe_notes.append(
            f"- {window}: [full estimates, condition-bootstrap intervals and paired comparisons](probe_{window}/probes.json); [same-corruption sensor stress comparison](probe_{window}/sensor_stress_probes.json)."
        )
    text += ["", *probe_notes]
    text += [
        "",
        "Twelve physical conditions and three held-out grip groups are a local pilot, not broad statistical evidence. Camera augmentations are replicates. Sensor stress magnitudes are not real-device calibrations.",
    ]
    (root / "REPORT.md").write_text("\n".join(text) + "\n")
    record = dict(
        episodes=rows,
        errors=errors,
        cells=cells,
        timing=times,
        reserved=len(attempts),
        unresolved_ids=pending,
        runtime_censored=censored,
        branches=branch_results,
        wrench_examples=examples,
        report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (root / "report.json").write_text(json.dumps(record, indent=2) + "\n")
    index = []
    for r in rows:
        s, m = r["spec"], r["metrics"]
        diagnostic = s["block"] in (
            "repair",
            "posture_diagnostic",
            "posture_repair",
            "planar_contact_check",
            "hook_flat_control",
        )
        masks = episode_eligibility(m, diagnostic)
        index.append(
            dict(
                file=str(Path(r["file"]).relative_to(root)),
                episode_id=s["episode_id"],
                family=s["family"],
                variant=s["variant"],
                physical_valid=m["physical_valid"],
                task_success=m["task_success"],
                gel_geometry=s.get("gel_geometry", "legacy_box"),
                gel_role=gel_benchmark_role(s.get("gel_geometry", "legacy_box")),
                compact_wrench_within_reference=bool(
                    (m["force_rmse"] < 0.2 or m["force_nrmse"] < 0.2)
                    and m["torque_rmse"] < 0.01
                ),
                dense_wrench_within_reference="dense_nodal_force_rmse_n" in r
                and (
                    r.get("dense_nodal_force_rmse_n", float("inf")) < 0.2
                    or r.get("dense_nodal_force_rmse_n", float("inf"))
                    / max(r["force_rms"], 1e-9)
                    < 0.2
                )
                and r.get("dense_nodal_torque_rmse_nm", float("inf")) < 0.01,
                **masks,
                setup_diagnostic=diagnostic,
                contact_model_review_required=m.get(
                    "contact_model_review_required", False
                ),
                benchmark_variant_retired=m.get("benchmark_variant_retired", False),
                actual_friction=r["actual_friction"],
            )
        )
    (root / "dataset_index.json").write_text(
        json.dumps(
            dict(
                schema="superdex_pilot_quality_index_v2",
                episodes=index,
                note="Use post-audit episode_accepted/world_model_eligible and imitation_eligible masks. Wrench agreement never rejects an episode; *_wrench_within_reference are diagnostics only, not input to any acceptance mask. Legacy tactile_valid is diagnostic only. Physically valid task failures may be world-model examples. Flat gels are reference domains, not implicitly pooled with the primary curved domain. No train/test split is frozen.",
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            dict(
                recorded=len(rows),
                errors=len(errors),
                valid=sum(r["metrics"]["physical_valid"] for r in rows),
                qualified_cells=sum(c["nominal_qualified"] for c in cells),
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--mesh-audit", action="store_true")
    a = parser.parse_args()
    report(a.folder, a.mesh_audit)
