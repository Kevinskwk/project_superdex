"""Closed procedural solids for the contact-rich diversity pilot (metres)."""

import numpy as np
import trimesh


def profile(shape, scale=1.0, count=48):
    theta = np.arange(count) * 2 * np.pi / count
    u = np.c_[np.cos(theta), np.sin(theta)]
    if shape == "key":
        radius = 1 / np.maximum(abs(u[:, 0]) / 0.005, abs(u[:, 1]) / 0.003)
    elif shape == "square":
        radius = 0.004 / np.maximum(abs(u[:, 0]), abs(u[:, 1]))
    elif shape == "hex":
        normals = np.c_[
            np.cos(np.arange(6) * np.pi / 3), np.sin(np.arange(6) * np.pi / 3)
        ]
        radius = 0.004 / np.max(u @ normals.T, axis=1)
    elif shape == "d":
        radius = np.minimum(0.0045, 0.0025 / np.maximum(u[:, 0], 1e-9))
    elif shape in ("round", "pen"):
        radius = np.full(count, 0.0045 if shape == "round" else 0.0025)
    else:
        raise ValueError(shape)
    return u * radius[:, None] * scale


def offset_profile(points, clearance):
    """Convex polygon offset by perpendicular edge distance, not radial scaling."""
    normals, distances = halfspaces(points)
    directions = points / np.linalg.norm(points, axis=1)[:, None]
    dots = directions @ normals.T
    radii = np.min(
        np.where(
            dots > 1e-10, (distances + clearance) / np.maximum(dots, 1e-10), np.inf
        ),
        axis=1,
    )
    return directions * radii[:, None]


def friction_provenance(spec, implementation):
    """Recover actual coefficients, including the retained pilot helper bug."""
    environment = (
        spec["friction"] ** 2 / spec["gel_friction"]
        if spec.get("effective_friction")
        else spec["friction"]
    )
    peg_header = implementation.split("class PegWorld", 1)[-1].split(
        "def make_tool_mesh", 1
    )[0]
    affected = (
        spec["family"] in ("insertion", "turning", "composite")
        and "def contact_params" in peg_header
    )
    tool = environment if affected else spec["gel_friction"]
    return dict(
        tool_actor_coefficient=tool,
        environment_actor_coefficient=environment,
        effective_gel_tool_friction=float(np.sqrt(spec["gel_friction"] * tool)),
        effective_environment_tool_friction=float(np.sqrt(environment * tool)),
        legacy_tool_friction_override=affected,
        independent_friction_sweep_valid=not affected,
        note="Actual construction from the phase source snapshot; requested config is preserved.",
    )


def annotate_friction(path):
    """Correct derived HDF5 metadata only; never change the recorded signals."""
    from pathlib import Path
    import json
    import hashlib
    import h5py

    path = Path(path)
    source = path.parent / "source_snapshot/benchmark.py"
    if not source.exists():
        return None
    implementation = source.read_text()
    with h5py.File(path) as f:
        result = friction_provenance(json.loads(f.attrs["config_json"]), implementation)
        result["source_sha256"] = hashlib.sha256(implementation.encode()).hexdigest()
        serialized = json.dumps(result, sort_keys=True)
        oracle = str(f.attrs.get("oracle_channels", ""))
        missing_oracles = [
            k
            for k in (
                "oblique_contact_normal_load_fraction",
                "per_contact_coulomb_excess_n",
            )
            if k in f["observations"] and k not in oracle.split(",")
        ]
        if f.attrs.get("audited_friction_json") == serialized and not missing_oracles:
            return result
    with h5py.File(path, "a") as f:
        for key in (
            "effective_gel_tool_friction",
            "effective_environment_tool_friction",
        ):
            if key in f.attrs and "original_reported_" + key not in f.attrs:
                f.attrs["original_reported_" + key] = f.attrs[key]
            f.attrs[key] = result[key]
        f.attrs["audited_friction_json"] = serialized
        if missing_oracles:
            f.attrs["oracle_channels"] = ",".join([oracle, *missing_oracles])
    return result


def episode_eligibility(metrics, setup_diagnostic=False):
    """Task/data masks never depend on tactile-to-contact wrench agreement.

    Valid task failures are useful world-model outcomes, not imitation targets.
    Contact-model review concerns invalid geometry/manifolds, not wrench RMSE.
    Retired variants remain replayable evidence, outside active benchmark data.
    """
    accepted = bool(
        metrics["physical_valid"]
        and metrics.get("full_rollout_complete", False)
        and not setup_diagnostic
        and not metrics.get("contact_model_review_required", False)
        and not metrics.get("benchmark_variant_retired", False)
    )
    return dict(
        episode_accepted=accepted,
        world_model_eligible=accepted,
        imitation_eligible=accepted and bool(metrics["task_success"]),
        wrench_matching_is_gate=False,
    )


def gel_benchmark_role(geometry):
    return {
        "source_surface": "primary",
        "matched_box": "flat_reference",
        "legacy_box": "legacy_reference",
    }[geometry]


def sequence_completion(data, family, dt, return_after_turn=True):
    """Measured final-state dwell; preserve return/withdrawal for old records."""
    n = max(1, round(0.3 / dt))
    enough = len(data["phase"]) >= n
    done = bool(enough and data["phase"][-1] == "done")
    returned = bool(
        enough and np.all(np.abs(data["rotor_angle_rad"][-n:]) <= np.deg2rad(5))
    )
    withdrawn = bool(enough and np.all(data["insertion_depth_m"][-n:] <= 0))
    complete = done and (
        returned
        if family == "turning"
        else withdrawn
        if family == "insertion"
        else returned and withdrawn
    )
    turned = bool(
        enough
        and np.all(np.abs(data["rotor_angle_rad"][-n:] - np.pi / 2) <= np.deg2rad(5))
    )
    seated = bool(enough and "seat_valid" in data and np.all(data["seat_valid"][-n:]))
    if not return_after_turn and family in ("turning", "composite"):
        complete = done and turned and seated
    return dict(
        sequence_complete=complete,
        turned_goal_held=turned,
        returned_to_zero=returned,
        withdrawn_clear=withdrawn,
        final_rotor_angle_deg=float(np.rad2deg(data["rotor_angle_rad"][-1])),
        final_insertion_depth_mm=float(data["insertion_depth_m"][-1] * 1000),
    )


def surface_stroke_audit(phase, tip_task, dt, required_length):
    """Require working-end travel DURING following, not root motion on release."""
    follow = np.asarray(phase) == "follow"
    tip = np.asarray(tip_task)[follow]
    n = max(1, round(0.3 / dt))
    progress = (
        float(np.median(tip[-n:, 0]) - np.median(tip[:n, 0]))
        if len(tip) >= 2 * n
        else 0.0
    )
    return dict(
        follow_tip_travel_mm=1000 * progress,
        follow_stroke_pass=bool(
            len(tip) >= 2 * n and progress >= 0.8 * required_length
        ),
        progress_semantics="Working-end net travel between first/last 0.3s of follow; excludes grasp and release",
    )


def halfspaces(points):
    edges = np.roll(points, -1, axis=0) - points
    normals = np.c_[edges[:, 1], -edges[:, 0]]
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    return normals, np.sum(normals * points, axis=1)


def contains_profile(points, query, clearance=0):
    normals, distances = halfspaces(points)
    return np.all(
        np.asarray(query) @ normals.T <= distances + clearance + 1e-10, axis=1
    )


def solid_rings(rings, cap=True):
    """Join equal-count CCW XY rings, ordered from top to bottom."""
    n = len(rings[0][1])
    vertices = np.concatenate([np.c_[xy, np.full(n, z)] for z, xy in rings])
    faces = []
    for layer in range(len(rings) - 1):
        for j in range(n):
            a, b = layer * n + j, layer * n + (j + 1) % n
            faces.extend([[a, a + n, b + n], [a, b + n, b]])
    if cap:
        for layer, reverse in ((0, False), (len(rings) - 1, True)):
            center = len(vertices)
            vertices = np.vstack(
                [vertices, vertices[layer * n : (layer + 1) * n].mean(0)]
            )
            for j in range(n):
                face = [center, layer * n + j, layer * n + (j + 1) % n]
                faces.append(face[::-1] if reverse else face)
    mesh = trimesh.Trimesh(vertices, faces, process=True)
    mesh.fix_normals()
    return mesh


def peg_mesh(shape="key", scale=1.0, edge=0.008):
    p = profile(shape, scale)
    angles = np.arange(len(p)) * 2 * np.pi / len(p)
    u = np.c_[np.cos(angles), np.sin(angles)]
    handle = u / np.maximum(abs(u[:, :1]) / 0.009, abs(u[:, 1:]) / 0.012)
    shaft = u * 0.0025
    rings = [(z, handle) for z in np.linspace(0.025, -0.025, 8)]
    rings += [(z, shaft) for z in np.linspace(-0.025, -0.060, 8)]
    rings += [(-0.061, p), (-0.068, p), (-0.069, p * 0.9)]
    mesh = solid_rings(rings)
    while mesh.edges_unique_length.max() > edge:
        mesh = mesh.subdivide()
    return mesh


def socket_mesh(shape="key", scale=1, clearance=0.0006, chamfer=0.003):
    """One closed solid with an open, blind profile-matched pocket."""
    p = offset_profile(profile(shape, scale), clearance)
    mouth = offset_profile(p, 0.002)
    angles = np.arange(len(p)) * 2 * np.pi / len(p)
    u = np.c_[np.cos(angles), np.sin(angles)]
    outer = 0.022 * u / np.maximum(abs(u[:, :1]), abs(u[:, 1:]))
    # Follow the solid boundary: exterior bottom -> exterior top -> cavity -> floor.
    rings = [(-0.019, outer), (0.0, outer), (0.0, mouth), (-chamfer, p), (-0.016, p)]
    mesh = solid_rings(rings)
    mesh.fix_normals()
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def peeling_contact_audit(data):
    active = ~np.asarray(data["initialization"], bool)
    follow = np.asarray(data["phase"]) == "follow"
    blade = np.asarray(data["blade_contact_force_n"]) > 0.1
    holder = np.asarray(data["holder_contact_force_n"]) > 0.1
    touching = np.flatnonzero(active & (blade | holder))
    first_blade = bool(len(touching) and blade[touching[0]] and not holder[touching[0]])
    bf = float(blade[follow].mean()) if follow.any() else 0.0
    hf = float(holder[follow].mean()) if follow.any() else 0.0
    return dict(
        blade_contact_continuity=bf,
        holder_contact_fraction=hf,
        blade_contacts_first=first_blade,
        blade_contact_pass=first_blade and bf >= 0.8 and hf <= 0.05,
    )


def surface_task_pass(metrics, variant):
    """Working-end task: root travel is not an extra gate during tool rotation."""
    passed = bool(
        metrics.get("contact_continuity", 0) >= 0.8
        and metrics.get("lateral_error_p95_m", float("inf")) < 0.003
        and metrics.get("follow_stroke_pass", False)
    )
    if "wall" in variant or "slot" in variant:
        passed &= metrics.get("guide_contact_continuity", 0) >= 0.8
    if variant == "cylindrical_peel":
        passed &= metrics.get("blade_contact_pass", False)
    return bool(passed)


def peeler_working_edge(mesh):
    """SCFields peeler_1 blade, excluding its lower holder ends (task yaw 90)."""
    v = np.asarray(mesh.vertices)
    blade = (np.abs(v[:, 1]) < 0.018) & (v[:, 2] < -0.080)
    if not blade.any():
        raise ValueError("Peeler blade region not found")
    z = v[blade, 2].min()
    return v[blade & (v[:, 2] < z + 0.0003)].mean(0)


def peeling_cylinder(radius=0.020):
    """Finite vegetable-shaped cylinder; axis along task X, crown at Z=0."""
    mesh = trimesh.creation.cylinder(radius=radius, height=0.120, sections=96)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    mesh.apply_translation([0.010, 0, -radius])
    return mesh


def height_surface(kind, radius=0.12, half_extent=None, lateral_samples=None):
    x = (
        np.linspace(-0.035, 0.055, 37)
        if half_extent is None
        else np.linspace(-half_extent, half_extent, round(2 * half_extent / 0.0025) + 1)
    )
    y = (
        np.linspace(-0.035, 0.035, 9)
        if half_extent is None
        else np.linspace(-half_extent, half_extent, round(2 * half_extent / 0.0025) + 1)
    )
    if lateral_samples is not None:
        y = np.linspace(y[0], y[-1], lateral_samples)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    sign = {"convex": -1, "concave": 1}.get(kind, 0)
    zz = sign * (radius - np.sqrt(radius**2 - xx**2)) if sign else np.zeros_like(xx)
    top = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    bottom = top.copy()
    bottom[:, 2] = -0.05
    vertices = np.vstack([top, bottom])
    n = len(top)
    faces = []
    for i in range(len(x) - 1):
        for j in range(len(y) - 1):
            a = i * len(y) + j
            b = a + len(y)
            faces += [
                [a, b, b + 1],
                [a, b + 1, a + 1],
                [a + n, b + 1 + n, b + n],
                [a + n, a + 1 + n, b + 1 + n],
            ]
    boundary = (
        [i * len(y) for i in range(len(x))]
        + [(len(x) - 1) * len(y) + j for j in range(1, len(y))]
        + [i * len(y) + len(y) - 1 for i in range(len(x) - 2, -1, -1)]
        + list(range(len(y) - 2, 0, -1))
    )
    for a, b in zip(boundary, boundary[1:] + boundary[:1]):
        faces += [[a, a + n, b + n], [a, b + n, b]]
    mesh = trimesh.Trimesh(vertices, faces)
    mesh.fix_normals()
    return mesh


def guide_wall(curved=False, side=1, half_width=0.0035):
    """Closed swept wall; changing normals without sharp controller corners."""
    x = np.linspace(-0.02, 0.055, 51)
    center = 0.004 * (1 - np.cos(np.pi * x / 0.04)) if curved else np.zeros_like(x)
    vertices = []
    for xx, yy in zip(x, center):
        inner, outer = yy + side * half_width, yy + side * (half_width + 0.008)
        vertices.extend(
            [
                [xx, inner, -0.010],
                [xx, outer, -0.010],
                [xx, outer, 0.014],
                [xx, inner, 0.014],
            ]
        )
    faces = [[0, 2, 1], [0, 3, 2]]
    for i in range(len(x) - 1):
        for j in range(4):
            a, b = i * 4 + j, i * 4 + (j + 1) % 4
            faces += [[a, b, b + 4], [a, b + 4, a + 4]]
    a = (len(x) - 1) * 4
    faces += [[a, a + 1, a + 2], [a, a + 2, a + 3]]
    mesh = trimesh.Trimesh(vertices, faces)
    mesh.fix_normals()
    return mesh
