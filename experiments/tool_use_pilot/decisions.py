"""Bounded, contact-driven decision tasks; legacy pilot remains independently usable."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from pilot import Spec, World, metrics, pose7, save, slider_mesh, smooth, stack


@dataclass(frozen=True)
class DecisionSpec(Spec):
    kind: str = "hook"
    case: str = "captured"
    lateral_m: float = 0.0
    yaw_deg: float = 0.0
    torsion_nm_rad: float = 0.02
    correction_m: float = 0.012
    correction_deg: float = 8.0
    revision: int = 0
    mirror_fixture: bool = False
    contact_edge_m: float = 0.008

    def __post_init__(self):
        if self.kind not in ("hook", "key"):
            raise ValueError(self.kind)
        object.__setattr__(
            self, "task", "hook" if self.kind == "hook" else "calibration"
        )
        if not self.duration:
            object.__setattr__(
                self,
                "duration",
                (
                    35.0
                    if self.revision >= 6
                    else 41.0
                    if self.revision >= 5
                    else 35.0
                    if self.revision >= 4
                    else 29.0
                )
                if self.kind == "hook"
                else 36.0,
            )
        object.__setattr__(self, "branch_time", 5.0 if self.kind == "hook" else 10.0)
        if not self.grip_force_n:
            object.__setattr__(self, "grip_force_n", 35.0)
        super().__post_init__()
        if min(self.grip_force_n, self.gel_modulus_pa) <= 0:
            raise ValueError("positive grasp force and gel modulus required")
        if (
            min(self.gel_friction, self.friction, self.stiffness, self.torsion_nm_rad)
            < 0
        ):
            raise ValueError("negative material/resistance parameter")


def candidates(kind):
    return (
        ("pull", "left", "right", "hold")
        if kind == "hook"
        else ("advance", "turn", "left", "right", "yaw_positive", "yaw_negative")
    )


def interpolate(knots, t):
    for (a, p, label), (b, q, _) in zip(knots, knots[1:]):
        if t < b:
            return np.asarray(p) + smooth((t - a) / (b - a)) * (
                np.asarray(q) - p
            ), label
    return np.asarray(knots[-1][1], dtype=float), knots[-1][2]


def target(spec, t, branch):
    zero = np.zeros(6)

    def v(x=0, y=0, z=0, yaw=0):
        return np.array([x, y, z, 0, 0, np.deg2rad(yaw)])

    if spec.kind == "hook":
        configured_pull=getattr(spec,'hook_pull_m',None)
        pull = -configured_pull if configured_pull is not None else (-0.050 if spec.revision >= 5 else -0.064)
        start = v(x=-0.006, z=0.018)
        knots = [
            (0, zero, "grasp"),
            (1.5, zero, "engage"),
            (3.3, v(z=0.018), "test_pull"),
            (4.2, start, "anchor"),
            (5.0, start, "decision"),
        ]
        if branch == "hold":
            knots += [(29, start, "hold")]
        elif branch in ("left", "right"):
            y = (-1 if branch == "left" else 1) * spec.correction_m
            knots += [
                (6, v(z=0.018), "unload"),
                (8, v(), "lower"),
                (10, v(y=y), "realign"),
                (12, v(y=y, z=0.018), "reengage"),
                (19, v(x=pull, y=y, z=0.018), "hold"),
                (19.5, v(x=pull, y=y, z=0.018), "unload"),
                (26.5, v(y=y, z=0.018), "lower"),
                (28.5, v(y=y), "done"),
                (29, v(y=y), "done"),
            ]
        else:
            knots += [
                (12, v(x=pull, z=0.018), "hold"),
                (12.5, v(x=pull, z=0.018), "unload"),
                (19.5, v(z=0.018), "lower"),
                (21.5, v(), "done"),
                (29, v(), "done"),
            ]
    else:
        depth = -0.004 if spec.case == "unseated" else -0.019
        start = v(z=depth)
        knots = [
            (0, zero, "grasp"),
            (1.5, zero, "approach"),
            (5, start, "insert"),
            (10, start, "decision"),
        ]
        if branch == "advance":
            knots += [(13, v(z=-0.019), "advance"), (36, v(z=-0.019), "hold")]
        elif branch == "turn":
            knots += [
                (24, v(z=depth, yaw=100), "turn"),
                (36, v(z=depth, yaw=100), "hold"),
            ]
        else:
            x = (
                spec.correction_m * (-1 if branch == "left" else 1)
                if branch in ("left", "right")
                else 0
            )
            yaw = (
                spec.correction_deg * (1 if branch == "yaw_positive" else -1)
                if branch.startswith("yaw_")
                else 0
            )
            knots += [
                (13, v(), "retract"),
                (15, v(x=x, yaw=yaw), "realign"),
                (19, v(x=x, z=-0.019, yaw=yaw), "insert"),
                (33, v(x=x, z=-0.019, yaw=yaw + 100), "turn"),
                (36, v(x=x, z=-0.019, yaw=yaw + 100), "hold"),
            ]
    return interpolate(knots, t)


def boxes(parts):
    meshes = []
    for dims, center in parts:
        m = trimesh.creation.box(dims)
        m.apply_translation(center)
        meshes.append(m)
    return trimesh.util.concatenate(meshes)


def box_union(parts):
    """Exact external surface of an axis-aligned box union, no internal seams."""
    bounds = [
        (np.array(c) - np.array(d) / 2, np.array(c) + np.array(d) / 2) for d, c in parts
    ]
    axes = [np.unique(np.round([v[j] for b in bounds for v in b], 9)) for j in range(3)]
    shape = tuple(len(a) - 1 for a in axes)
    occupied = np.zeros(shape, dtype=bool)
    for index in np.ndindex(shape):
        mid = np.array(
            [(axes[j][index[j]] + axes[j][index[j] + 1]) / 2 for j in range(3)]
        )
        occupied[index] = any(
            np.all(mid >= lo) and np.all(mid <= hi) for lo, hi in bounds
        )
    vertices, faces = [], []
    for index in np.ndindex(shape):
        if not occupied[index]:
            continue
        for axis in range(3):
            others = [j for j in range(3) if j != axis]
            for sign in (-1, 1):
                neighbor = list(index)
                neighbor[axis] += sign
                if 0 <= neighbor[axis] < shape[axis] and occupied[tuple(neighbor)]:
                    continue
                start = len(vertices)
                for a, b in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    corner = list(index)
                    corner[axis] += int(sign > 0)
                    corner[others[0]] += a
                    corner[others[1]] += b
                    vertices.append([axes[j][corner[j]] for j in range(3)])
                faces.extend(
                    [[start, start + 1, start + 2], [start, start + 2, start + 3]]
                )
    mesh = trimesh.Trimesh(vertices, faces)
    mesh.fix_normals()
    return mesh


def rim_mesh(width, height, depth, outer=0.040, floor=False):
    # Four disjoint walls; never convexify the opening.
    pieces = [
        (
            [outer, (outer - height) / 2, depth],
            [0, s * (outer + height) / 4, -depth / 2],
        )
        for s in (-1, 1)
    ]
    pieces += [
        ([(outer - width) / 2, height, depth], [s * (outer + width) / 4, 0, -depth / 2])
        for s in (-1, 1)
    ]
    if floor:
        pieces += [([outer, outer, 0.003], [0, 0, -depth - 0.0015])]
    return boxes(pieces)


def entrance_mesh():
    # A 2 mm lead-in per side, followed by the specified 0.6 mm blade clearance.
    parts = []
    for axis, half in ((0, 0.0056), (1, 0.0036)):
        for sign in (-1, 1):
            vertices = []
            for z, inner in ((0, half + 0.002), (-0.003, half)):
                for a in (inner, 0.020):
                    for b in (-0.020, 0.020):
                        point = [0.0, 0.0, z]
                        point[axis] = sign * a
                        point[1 - axis] = b
                        vertices.append(point)
            parts.append(trimesh.convex.convex_hull(np.array(vertices)))
    return trimesh.util.concatenate(parts)


class HookDecision(World):
    def make_tool_mesh(self):
        mesh = super().make_tool_mesh()
        return self.contact_mesh(mesh)

    def contact_mesh(self, mesh):
        if self.spec.kind == "hook" and self.spec.revision >= 3:
            vertices, faces = trimesh.remesh.subdivide_to_size(
                mesh.vertices, mesh.faces, max_edge=self.spec.contact_edge_m
            )
            return trimesh.Trimesh(vertices, faces)
        return mesh

    def make_slider_mesh(self):
        if self.spec.revision < 2:
            return slider_mesh()
        # One free crossbar end; the support is outside every recovery path.
        # Mirror the complete cantilever to test both recovery directions.
        mesh = (box_union if self.spec.revision >= 6 else boxes)(
            [
                ([0.010, 0.110, 0.008], [0, 0.025, 0]),
                ([0.010, 0.005, 0.044], [0, 0.0775, -0.026]),
                ([0.020, 0.110, 0.006], [0, 0.025, -0.051]),
            ]
        )
        if self.spec.mirror_fixture:
            mesh.apply_transform(np.diag([1.0, -1.0, 1.0, 1.0]))
        return self.contact_mesh(mesh)

    def __init__(self, spec):
        super().__init__(spec)
        self.tracking_correction = np.zeros(3)
        self.trajectory_time = 0.0
        self.recovery_gate = 0
        self.recovery_wait = 0.0
        self.controller_abort = ""
        self.nominal_tool_target = np.zeros(3)
        if spec.kind == "hook" and spec.task == "hook" and not hasattr(self, "environment_meshes"):
            self.environment_meshes = [
                self.make_slider_mesh(),
                trimesh.creation.box([0.16, 0.09, 0.012]),
            ]
            # Set the intended environmental friction on both tool-contact bodies.
        self.geometry = [("tool", self.tool, self.mesh, "object")]
        self.geometry += [
            (f"env_{i}", a, m, "environment")
            for i, (a, m) in enumerate(zip(self.env, self.environment_meshes))
        ]
        for name, a in self.gripper.links.items():
            try:
                mesh = self.p.get_shape_surface_mesh(a.get_reference_shape())
                vertices = np.asarray(mesh.coordinates).reshape(-1, 3).copy()
                faces = np.asarray(mesh.connectivity).reshape(-1, 3).copy()
                if len(vertices) and len(faces):
                    self.geometry.append(
                        (
                            name,
                            a,
                            trimesh.Trimesh(vertices, faces, process=False),
                            "occluder",
                        )
                    )
            except (AttributeError, ValueError, self.p.Error):
                continue
        self.root_to_com = []
        for _, a, _, _ in self.geometry:
            self.root_to_com.append(
                pose7(
                    a.get_root_transform().inverse() * a.get_center_of_mass_transform(),
                    np.zeros(3),
                )
            )

    def command_at(self, t, branch="pull", anchor_command=None):
        clock = t
        waiting = False
        if self.spec.kind == "hook" and self.spec.revision >= 4:
            actual = self.rot.inv().apply(
                np.asarray(self.tool.get_root_transform().translation) - self.origin
            )
            clock = (
                t if t < self.spec.branch_time else self.trajectory_time + self.spec.dt
            )
            if branch in ("left", "right") and self.recovery_gate < 3:
                gate = (8.0, 10.0, 12.0)[self.recovery_gate]
                if clock >= gate:
                    goal = target(self.spec, gate, branch)[0][:3]
                    error = goal - actual
                    if self.spec.revision >= 5 and self.recovery_gate == 0:
                        # Lateral alignment is the NEXT maneuver; clearing
                        # the crossbar first only requires safe x/z position.
                        error = error[[0, 2]]
                    if np.linalg.norm(error) > 0.0015:
                        clock = gate
                        waiting = True
                        self.recovery_wait += self.spec.dt
                        if self.recovery_wait > (
                            4.0 if self.spec.revision >= 5 else 2.0
                        ):
                            self.controller_abort = "recovery_alignment_timeout"
                    else:
                        self.recovery_gate += 1
                        self.recovery_wait = 0.0
            self.trajectory_time = clock
        value, phase = target(self.spec, clock, branch)
        self.nominal_tool_target = value[:3].copy()
        if waiting:
            phase = "wait_recovery_clearance"
        if self.spec.kind == "hook" and self.spec.revision >= 3 and t >= 1.5 and getattr(self.spec, 'tool_pose_feedback', True):
            # Privileged scripted-collector servo, NOT a policy observation.
            # Track the authored tool trajectory, not the fixture; offsets and
            # wrong recovery choices therefore remain genuine task variations.
            actual = self.rot.inv().apply(
                np.asarray(self.tool.get_root_transform().translation) - self.origin
            )
            gain, rate = (4.0, 0.006) if self.spec.revision >= 5 else (2.0, 0.003)
            increment = np.clip(gain * (value[:3] - actual), -rate, rate)
            self.tracking_correction = np.clip(
                self.tracking_correction + self.spec.dt * increment, -0.012, 0.012
            )
            value[:3] += self.tracking_correction
        return value[:3], phase, value[3:]

    def checkpoint(self):
        state, ctrl = super().checkpoint()
        ctrl["tracking_correction"] = self.tracking_correction.copy()
        for key in (
            "trajectory_time",
            "recovery_gate",
            "recovery_wait",
            "controller_abort",
        ):
            ctrl[key] = getattr(self, key)
        return state, ctrl

    def step(self, t, branch="pull", anchor_command=None):
        row = super().step(t, branch, anchor_command)
        row["scripted_pose_correction_m"] = self.tracking_correction.copy()
        row["nominal_tool_target_m"] = self.nominal_tool_target.copy()
        row["controller_abort"] = self.controller_abort
        row["trajectory_time"] = self.trajectory_time
        row.pop("objectpointcloud")
        row.pop("env_point_cloud")
        row["body_root_poses"] = np.stack(
            [pose7(a.get_root_transform(), np.zeros(3)) for _, a, _, _ in self.geometry]
        )
        row["grip_command_n"] = np.repeat(self.spec.grip_force_n * smooth(t / 0.9), 2)
        row["fingertip_load_n"] = np.array(
            [
                np.linalg.norm(row[f"tactile_force_field_{s}"].sum((0, 1)))
                for s in ("left", "right")
            ]
        )
        row["tactile_reference_valid"] = self.reference is not None
        row["reference_reset"] = abs(t - 1.45) < self.spec.dt / 2
        if self.spec.kind == "key":
            root = self.tool.get_root_transform()
            blade = self.rot.inv().apply(
                np.asarray(root.translation)
                + Rotation.from_rotvec(
                    np.asarray(root.rotation.to_rotation_vector())
                ).apply([0, 0, -0.065])
                - self.socket_origin
            )
            row["insertion_depth_m"] = -float(blade[2] + 0.004)
            row["rotor_angle_rad"] = self.rotor_angle()
            row["task_progress"] = row["rotor_angle_rad"]
            row["rewards"] = row["task_progress"]
            row["seat_valid"] = bool(
                -0.015 <= blade[2] <= -0.007 and np.linalg.norm(blade[:2]) < 0.002
            )
        else:
            row["insertion_depth_m"] = 0.0
            row["rotor_angle_rad"] = 0.0
            row["seat_valid"] = False
        return row


class KeyDecision(HookDecision):
    def make_tool_mesh(self):
        shaft = trimesh.creation.cylinder(radius=0.0025, height=0.036, sections=24)
        shaft.apply_translation([0, 0, -0.043])
        return trimesh.util.concatenate(
            [
                boxes(
                    [
                        ([0.018, 0.024, 0.050], [0, 0, 0]),
                        ([0.010, 0.006, 0.008], [0, 0, -0.065]),
                    ]
                ),
                shaft,
            ]
        )

    def build_environment(self):
        p = self.p
        self.socket_origin = self.origin + self.rot.apply(
            [self.spec.lateral_m, 0, -0.073]
        )
        socket_rot = self.rot * Rotation.from_euler(
            "z", self.spec.yaw_deg, degrees=True
        )
        self.socket_rot = socket_rot
        gate_mesh = entrance_mesh()
        rotor_mesh = rim_mesh(0.0112, 0.0072, 0.010, outer=0.024, floor=True)
        gate = self.actor(
            "keyed_gate",
            gate_mesh,
            self.socket_origin,
            socket_rot,
            1,
            self.spec.friction,
            static=True,
        )
        base_mesh = trimesh.creation.box([0.060, 0.060, 0.006])
        shape = p.create_tri_mesh_shape(
            np.asarray(rotor_mesh.vertices, dtype=np.float32).ravel(),
            np.asarray(rotor_mesh.faces, dtype=np.int32).ravel(),
        )
        base_shape = p.create_tri_mesh_shape(
            np.asarray(base_mesh.vertices, dtype=np.float32).ravel(),
            np.asarray(base_mesh.faces, dtype=np.int32).ravel(),
        )
        contact = p.ContactParams()
        contact.coulomb_friction_coefficient = self.spec.friction
        contact.penalty_coefficient = 5e12
        params = p.ArticulatedActorParams(name="key_socket")
        params.world_from_root = p.TransformRT(
            translation=self.socket_origin,
            rotation=p.Quaternion.from_rotation_vector(socket_rot.as_rotvec()),
        )
        params.joints = [
            p.ArticulatedJointParams(name="weld", type=p.ArticulatedJointType.HARD),
            p.ArticulatedJointParams(
                name="rotor",
                type=p.ArticulatedJointType.REVOLUTE,
                axis=[0, 0, 1],
                parent_link_from_joint=p.TransformRT(translation=[0, 0, 0.021]),
            ),
        ]
        params.links = [
            p.ArticulatedLinkParams(
                name="socket_base",
                parent_link=-1,
                parent_joint_from_link=p.TransformRT(translation=[0, 0, -0.025]),
                shape=base_shape,
                collider_type=p.ColliderType.MESH,
                density=1000,
            ),
            p.ArticulatedLinkParams(
                name="socket_rotor",
                parent_link=0,
                shape=shape,
                collider_type=p.ColliderType.MESH,
                density=0.06 / rotor_mesh.volume,
                contact=contact,
            ),
        ]
        self.fixture = self.scene.create_articulated_actor(params)
        self.base, self.rotor = [
            self.scene.get_actor(h) for h in self.fixture.get_nested_link_actors()
        ]
        self.env = [gate, self.rotor, self.base]
        self.environment_meshes = [gate_mesh, rotor_mesh, base_mesh]
        self.table_top = self.socket_origin[2]
        self.slider = None

    def rotor_angle(self):
        r = Rotation.from_rotvec(
            np.asarray(self.rotor.get_root_transform().rotation.to_rotation_vector())
        )
        return float((self.socket_rot.inv() * r).as_rotvec()[2])

    def apply_environment_force(self):
        q = self.rotor_angle()
        speed = float(
            np.asarray(self.rotor.get_angular_velocity()) @ self.rot.apply([0, 0, 1])
        )
        torque = -self.spec.torsion_nm_rad * q - 0.0001 * speed
        self.fixture.set_external_forces_on_dofs(
            np.array([0], dtype=np.int32), [torque]
        )


def audit(data, spec, abort=None):
    result = metrics(data, spec)
    active = ~data["initialization"]
    if spec.kind == "key":
        achieved = data["seat_valid"] & (
            np.abs(data["rotor_angle_rad"] - np.pi / 2) < np.deg2rad(5)
        )
        n = max(1, round(0.3 / spec.dt))
        result["task_success"] = bool(
            len(achieved) >= n
            and np.any(np.convolve(achieved.astype(int), np.ones(n), "valid") == n)
        )
        result["clearance_valid"] = bool(
            data["rigid_penetration_m"][active].max() < 0.0003
        )
        result["physical_valid"] &= result["clearance_valid"]
        result["max_rotor_angle_deg"] = float(np.rad2deg(data["rotor_angle_rad"].max()))
        result["max_insertion_depth_mm"] = float(1000 * data["insertion_depth_m"].max())
    result["abort_reason"] = abort
    result["physical_valid"] &= abort is None
    result["imitation_eligible"] = result["physical_valid"] and result["task_success"]
    result["tactile_valid"] &= result["physical_valid"]
    result["kind"] = spec.kind
    result["case"] = spec.case
    result["mean_fingertip_load_n"] = data["fingertip_load_n"][active].mean(0).tolist()
    result["peak_extrinsic_force_n"] = float(
        np.linalg.norm(data["extrinsic_contact_wrench"][active, :3], axis=1).max()
    )
    return result


def write_episode(path, data, spec, world, result, snapshot=None, prefix=None):
    save(path, data, spec, world, result, snapshot)
    with h5py.File(path, "a") as f:
        f.attrs["schema_version"] = "vt_acwm_superdex_decision_v2"
        f.attrs["task_kind"] = spec.kind
        f.attrs["task_progress_units"] = "m" if spec.kind == "hook" else "rad"
        f.attrs["implementation_hash"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        f.attrs["oracle_channels"] = (
            "body_root_poses,tool_pose,fixture_state,task_progress,extrinsic_contact_wrench,seat_valid,rotor_angle_rad,insertion_depth_m,scripted_pose_correction_m,nominal_tool_target_m"
        )
        f.attrs["collector_pose_feedback"] = bool(
            spec.kind == "hook" and spec.revision >= 3
        )
        if snapshot and "tracking_correction" in snapshot[1]:
            f["checkpoint"].create_dataset(
                "tracking_correction", data=snapshot[1]["tracking_correction"]
            )
            for key in (
                "trajectory_time",
                "recovery_gate",
                "recovery_wait",
                "controller_abort",
            ):
                f["checkpoint"].attrs[key] = snapshot[1][key]
        if prefix:
            f["prebranch_history"] = h5py.ExternalLink(prefix, "observations")
        geom = f.create_group("geometry")
        for i, (name, a, m, role) in enumerate(world.geometry):
            g = geom.create_group(str(i))
            g.attrs.update(name=name, role=role)
            g.create_dataset(
                "vertices",
                data=np.asarray(m.vertices, dtype=np.float32),
                compression="lzf",
            )
            g.create_dataset(
                "faces", data=np.asarray(m.faces, dtype=np.int32), compression="lzf"
            )
            g.create_dataset("root_to_com", data=world.root_to_com[i])
        camera = f.create_group("camera")
        target = world.origin + world.rot.apply(
            [0.015, 0, -0.025 if spec.kind == "hook" else -0.075]
        )
        target = getattr(world, "camera_target_world", target)
        camera["target_world"] = target
        camera["eyes_world"] = np.stack(
            [
                target + world.rot.apply([0.22, -0.25, 0.13]),
                target + world.rot.apply([0.18, 0.25, 0.10]),
            ]
        )
        camera.attrs.update(
            width=320,
            height=240,
            vertical_fov_deg=45.0,
            point_count=1024,
            segmentation="ideal body identity on visible rays; no learned segmentation",
        )


def stop_reason(row):
    if row.get("controller_abort"):
        return row["controller_abort"]
    if not np.isfinite(row["tool_pose"]).all():
        return "nonfinite"
    if row["slip_m"] > 0.025:
        return "drop"
    w = row["extrinsic_contact_wrench"]
    if np.linalg.norm(w[:3]) > 15 or np.linalg.norm(w[3:]) > 0.5:
        return "wrench_limit"
    if row["min_gel_jacobian"] <= 0:
        return "inverted_gel"
    return None


def run_decision(spec, output, branches=True, render=False, selected=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    world = (HookDecision if spec.kind == "hook" else KeyDecision)(spec)
    start = time.perf_counter()
    rows = []
    reports = []
    prefix_recorder = None
    default = "pull" if spec.kind == "hook" else "turn"
    try:
        if render:
            from visuals import Recorder

            prefix_recorder = Recorder(
                world, output / f"episode_{spec.episode_id:04d}_{selected or default}"
            )
        for i in range(round(spec.branch_time / spec.dt)):
            row = world.step(i * spec.dt, default)
            rows.append(row)
            if prefix_recorder and i % max(1, round(0.05 / spec.dt)) == 0:
                prefix_recorder.frame(row)
            abort = stop_reason(row) if i * spec.dt >= 1.5 else None
            if abort:
                break
        prefix = stack(rows)
        snapshot = world.checkpoint()
        world.initial_hash = hashlib.sha256(snapshot[0]).hexdigest()
        stem = f"episode_{spec.episode_id:04d}"
        pr = audit(prefix, spec, abort)
        write_episode(output / f"{stem}_prefix.h5", prefix, spec, world, pr, snapshot)
        names = (
            ([selected or default] if not branches else candidates(spec.kind))
            if not abort
            else []
        )
        for branch in names:
            world.restore(snapshot)
            tail = []
            recorder = None
            if render:
                from visuals import Recorder

                recorder = (
                    prefix_recorder
                    if branch == (selected or default)
                    else Recorder(world, output / f"{stem}_{branch}")
                )
            for i in range(
                round(spec.branch_time / spec.dt), round(spec.duration / spec.dt)
            ):
                row = world.step(i * spec.dt, branch)
                tail.append(row)
                if recorder and i % max(1, round(0.05 / spec.dt)) == 0:
                    recorder.frame(row)
                abort = stop_reason(row)
                if abort:
                    break
            d = stack(tail)
            bs = replace(spec, branch=branch)
            result = audit(d, bs, abort)
            result["branch"] = branch
            write_episode(
                output / f"{stem}_{branch}.h5",
                d,
                bs,
                world,
                result,
                snapshot,
                f"{stem}_prefix.h5",
            )
            if recorder:
                video_data = (
                    {k: np.concatenate([prefix[k], d[k]]) for k in d}
                    if recorder is prefix_recorder
                    else d
                )
                recorder.close(video_data)
                if recorder is prefix_recorder:
                    prefix_recorder = None
            reports.append(result)
        if prefix_recorder:
            prefix_recorder.close(prefix)
        summary = dict(
            spec=asdict(spec),
            prefix=pr,
            branches=reports,
            seconds=time.perf_counter() - start,
        )
        (output / f"{stem}.json").write_text(json.dumps(summary, indent=2))
        return summary
    finally:
        world.close()


def worker(spec, output, branches=False, render=False, selected=None):
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    import superdex.physics as p

    p.initialize(num_worker_threads=0)
    try:
        return run_decision(DecisionSpec(**spec), output, branches, render, selected)
    finally:
        p.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["hook", "key"], default="hook")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--specs", type=Path)
    parser.add_argument("--branches", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--selected")
    args = parser.parse_args()
    specs = (
        json.loads(args.specs.read_text())
        if args.specs
        else [asdict(DecisionSpec(kind=args.kind))]
    )
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing

    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        pending = [
            pool.submit(
                worker, s, str(args.output), args.branches, args.render, args.selected
            )
            for s in specs
        ]
        for future in as_completed(pending):
            r = future.result()
            print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
