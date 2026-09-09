#!/usr/bin/env python3
"""FR3/GelSight tool-use pilots. Run with the repository's .venv Python."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/gelsight_mini_contact_validation"))
os.environ["SUPERDEX_ASSETS_PATH"] = str(ROOT / "assets")
os.environ.pop("SUPERDEX_PRECISION", None)  # FP32, CPU simulation
from fidelity_campaign import (
    create_box_shape,
    sample_mesh_surface,
    transform_points,
    pose7,
)
from wrench_math import (
    aggregate_contact_points,
    integrate_surface_force_field,
    inertia_matrix,
)


@dataclass(frozen=True)
class Spec:
    episode_id: int = 0
    task: str = "hook"
    seed: int = 0
    stiffness: float = 30.0
    offset: float = 0.0
    angle_deg: float = 0.0
    friction: float = 0.5
    pause: float = 0.75
    dt: float = 0.01
    duration: float = 0.0
    branch: str = "nominal"
    branch_time: float = 5.0
    group: str = "nominal-0"
    grip_force_n: float = 0.0
    gel_modulus_pa: float = 200_000.0
    gel_friction: float = 1.4
    gel_geometry: str = 'source_surface'

    def __post_init__(self):
        if self.task not in ("hook", "probe", "push", "calibration"):
            raise ValueError(f"Unknown task: {self.task}")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.duration == 0:
            object.__setattr__(
                self,
                "duration",
                {"hook": 18.5, "probe": 14.0, "push": 12.0, "calibration": 12.0}[
                    self.task
                ],
            )
        if self.task == "probe" and self.branch_time == 5.0:
            object.__setattr__(self, "branch_time", 8.7 + self.pause)


def smooth(q):
    q = np.clip(q, 0.0, 1.0)
    return q**3 * (10 + q * (-15 + 6 * q))


def trajectory(spec, t):
    """Absolute tool-frame EE offsets; interpolation is C2 at waypoints."""
    if spec.task == "hook":
        knots = [
            (0, [0, 0, 0], "grasp"),
            (1, [0, 0, 0], "settle"),
            (1.5, [0, 0, 0], "engage"),
            (3.3, [0, 0, 0.018], "test_pull"),
            (4.2, [-0.006, 0, 0.018], "loaded_pull"),
            (9.8, [-0.064, 0, 0.018], "hold"),
            (10.3, [-0.064, 0, 0.018], "unload"),
            (16.5, [0, 0, 0.018], "disengage"),
            (18.3, [0, 0, 0], "done"),
            (18.5, [0, 0, 0], "done"),
        ]
    elif spec.task == "probe":
        end_pause = 8.7 + spec.pause
        knots = [
            (0, [0, 0, 0], "grasp"),
            (1, [0, 0, 0], "settle"),
            (1.5, [0, 0, 0], "engage"),
            (3.3, [0, 0, 0.018], "probe"),
            (5.1, [-0.018, 0, 0.018], "unload"),
            (6.9, [0, 0, 0.018], "disengage"),
            (8.7, [0, 0, 0], "pause"),
            (end_pause, [0, 0, 0], "reengage"),
            (end_pause + 1.8, [0, 0, 0.018], "later_pull"),
            (14.0, [-0.018, 0, 0.018], "hold"),
        ]
    elif spec.task == "push":
        knots = [
            (0, [0, 0, 0], "grasp"),
            (1, [0, 0, 0], "settle"),
            (1.5, [0, 0, 0], "approach"),
            (3.0, [0.010, 0, 0], "push"),
            (9.0, [0.068, 0, 0], "hold"),
            (10.0, [0.068, 0, 0], "retreat"),
            (12.0, [0.05, 0, 0], "done"),
        ]
    else:
        knots = [
            (0, [0, 0, 0], "grasp"),
            (1, [0, 0, 0], "settle"),
            (1.5, [0, 0, 0], "preload"),
            (4.0, [0, 0, -0.005], "hold"),
            (6.0, [0, 0, -0.005], "shear"),
            (9.0, [0.008, 0, -0.005], "release"),
            (12.0, [0.008, 0, 0.004], "done"),
        ]
    for (t0, p0, phase), (t1, p1, _) in zip(knots, knots[1:]):
        if t < t1:
            return np.asarray(p0) + (np.asarray(p1) - p0) * smooth(
                (t - t0) / (t1 - t0)
            ), phase
    return np.asarray(knots[-1][1], float), knots[-1][2]


def tool_mesh(task):
    if task in ("hook", "probe"):
        # One J-shaped solid. The former stem stopped at z=-48 mm while
        # the foot started at x=9 mm: their edge-only junction left a missing
        # lower corner. Explicit planar triangulation preserves the opening
        # without overlapping/internal faces or a convex-hull collision proxy.
        polygon = (
            np.array(
                [
                    [-9, 25],
                    [9, 25],
                    [9, -48],
                    [34, -48],
                    [34, -24],
                    [42, -24],
                    [42, -56],
                    [-9, -56],
                ],
                dtype=float,
            )
            * 0.001
        )
        triangles = np.array(
            [[0, 1, 2], [0, 2, 7], [2, 3, 7], [3, 6, 7], [3, 4, 5], [3, 5, 6]]
        )
        mesh = trimesh.creation.extrude_triangulation(polygon, triangles, 0.024)
        mesh.vertices = mesh.vertices[:, [0, 2, 1]]
        mesh.vertices[:, 1] -= 0.012
        mesh.fix_normals()
        return mesh
    if task == "push":
        return trimesh.creation.box([0.018, 0.024, 0.085])
    return trimesh.creation.box([0.018, 0.024, 0.070])


def slider_mesh():
    """Crossbar, two supports and carriage; clear central aperture for the hook."""
    parts = []
    for dims, center in [
        ([0.010, 0.060, 0.008], [0, 0, 0]),
        ([0.010, 0.005, 0.044], [0, -0.0275, -0.026]),
        ([0.010, 0.005, 0.044], [0, 0.0275, -0.026]),
        ([0.020, 0.060, 0.006], [0, 0, -0.051]),
    ]:
        part = trimesh.creation.box(dims)
        part.apply_translation(center)
        parts.append(part)
    return trimesh.util.concatenate(parts)


class World:
    def __init__(self, spec):
        import superdex.physics as p
        import superdex.robotics as r
        from soft_gripper import GelMaterial, build_soft_gripper, create_osc

        self.p, self.r, self.spec = p, r, spec
        self.scene = p.create_scene(f"pilot-{spec.episode_id}")
        self.scene.set_gravity([0, 0, -9.81])
        if hasattr(spec, 'solver_iterations'):
            solver = self.scene.get_solver_params()
            solver.non_linear_solver.max_iter = spec.solver_iterations
            solver.non_linear_solver.abs_tol = spec.solver_abs_tolerance
            self.scene.set_solver_params(solver)
        self.context = r.create_context()
        self.gripper = build_soft_gripper(
            self.scene,
            GelMaterial(
                geometry=spec.gel_geometry,
                youngs_modulus_pa=spec.gel_modulus_pa,
                friction_coefficient=spec.gel_friction,
            ),
            ideal_arm_friction_compensation=getattr(spec,'ideal_arm_friction_compensation',False),
            finger_coupling_stiffness_n_m=getattr(spec,'finger_coupling_stiffness_n_m',0.),
        )
        self.osc = create_osc(self.context, self.gripper)
        params = self.osc.get_params()
        params.kp_p = getattr(spec, 'arm_stiffness_n_m', 1100.0)
        params.kd_p = getattr(spec, 'arm_damping_ns_m', 85.0)
        params.kp_r = getattr(spec, 'arm_stiffness_nm_rad', 40.0)
        params.kd_r = getattr(spec, 'arm_damping_nms_rad', 4.0)
        params.max_translation_error = 0.05
        params.max_rotation_error = 0.5
        params.b_apply_max_osc_torque_normalization = True
        self.osc.set_params(params)
        if hasattr(self, "prepare_robot"):
            self.prepare_robot()
        obs = self.osc.get_current_observations_from_mochi()
        self.root = obs.world_from_root
        self.ee0 = np.asarray(obs.world_from_ee_link.translation).copy()
        self.ee_rot = Rotation.from_rotvec(
            np.asarray(obs.world_from_ee_link.rotation.to_rotation_vector())
        )
        centers = []
        for side in ("left", "right"):
            rest = transform_points(self.root, self.gripper.gel_rest_root[side])
            centers.append((rest.min(0) + rest.max(0)) / 2)
        y = centers[0] - centers[1]
        y /= np.linalg.norm(y)
        x = np.cross(y, [0, 0, 1])
        x /= np.linalg.norm(x)
        self.rot = Rotation.from_matrix(np.column_stack([x, y, [0, 0, 1]]))
        self.origin = (centers[0] + centers[1]) / 2 + [
            0,
            0,
            -0.009 if spec.task in ("hook", "probe") else -0.018,
        ]
        if hasattr(self, "prepared_frame"):
            self.rot, self.origin = self.prepared_frame(centers, self.rot, self.origin)
        self.mesh = self.make_tool_mesh()
        self.tool = self.actor(
            "tool", self.mesh, self.origin, self.rot, 0.08, spec.gel_friction
        )
        # Same collision policy as the validated GelSight assembly: only the gel
        # is an instrumented grasp surface. Housing collision meshes overlap it.
        for side in ("left", "right"):
            housing = self.gripper.links[f"franka_{side}_gelsight_housing"]
            self.scene.enable_actor_contact_symmetric(
                housing.get_handle(),
                self.tool.get_handle(),
                False,
                p.IncludeNestedActors.NO,
            )
        self.mass = 0.08
        self.inertia = inertia_matrix(self.tool.get_rigid_moment_of_inertia_local())
        tf = self.tool.get_center_of_mass_transform()
        self.tool.add_boundary_condition_dofs_world(
            np.arange(6, dtype=np.int32),
            np.r_[np.asarray(tf.translation), self.rot.as_rotvec()],
        )
        self.released = False
        self.finger_dofs = list(self.gripper.finger_dofs.values())
        self.dofs = np.arange(self.gripper.actor.get_num_dofs(), dtype=np.int32)
        self.env = []
        self.slider = None
        if hasattr(self, "build_environment"):
            self.build_environment()
        elif spec.task in ("hook", "probe"):
            self.slider0 = self.origin + self.rot.apply([0.024, spec.offset, -0.016])
            fixture_rotation = self.rot
            if hasattr(self, "hook_fixture_pose"):
                self.slider0, fixture_rotation = self.hook_fixture_pose(self.slider0)
            self.axis = fixture_rotation.apply([-1, 0, 0])
            # Native reduced-coordinate joint enforces all five locked DOFs;
            # a penalty-based rigid guide allowed crossbar rotation under load.
            params = p.ArticulatedActorParams(name="slider_fixture")
            slider_geometry = self.make_slider_mesh()
            slider_shape = p.create_tri_mesh_shape(
                np.asarray(slider_geometry.vertices, dtype=np.float32).ravel(),
                np.asarray(slider_geometry.faces, dtype=np.int32).ravel(),
            )
            params.world_from_root = p.TransformRT(
                translation=self.slider0,
                rotation=p.Quaternion.from_rotation_vector(fixture_rotation.as_rotvec()),
            )
            params.joints = [
                p.ArticulatedJointParams(
                    name="world_weld", type=p.ArticulatedJointType.HARD
                ),
                p.ArticulatedJointParams(
                    name="slider",
                    type=p.ArticulatedJointType.PRISMATIC,
                    axis=[-1, 0, 0],
                    min_limit=[0.002, 0, 0],
                    max_limit=[-0.060, 0, 0],
                    limit_stiffness=300.0,
                    limit_damping=2.0,
                    parent_link_from_joint=p.TransformRT(translation=[0, 0, 0.060]),
                ),
            ]
            fixture_contact = p.ContactParams()
            fixture_contact.coulomb_friction_coefficient = (
                self.environment_friction_coefficient()
                if hasattr(self, "environment_friction_coefficient") else spec.friction
            )
            if getattr(spec, "revision", 0) >= 4:
                # Millimetre-scale engagement needs a contact transition much
                # smaller than the 8 mm crossbar (default spans 10 mm).
                fixture_contact.penalty_smoothing_half_distance = 0.0002
                fixture_contact.penalty_threshold_default = 0.0001
                fixture_contact.penalty_coefficient = 5e10
            params.links = [
                p.ArticulatedLinkParams(
                    name="rail_base",
                    parent_link=-1,
                    parent_joint_from_link=p.TransformRT(translation=[0, 0, -0.060]),
                    shape=create_box_shape(p, [0.16, 0.09, 0.012]),
                    collider_type=p.ColliderType.BOX,
                    density=1000.0,
                ),
                p.ArticulatedLinkParams(
                    name="crossbar_slider",
                    parent_link=0,
                    shape=slider_shape,
                    collider_type=p.ColliderType.MESH,
                    density=0.12 / slider_geometry.volume,
                    **({"contact": fixture_contact} if hasattr(spec, "kind") else {}),
                ),
            ]
            self.fixture = self.scene.create_articulated_actor(params)
            base, self.slider = [
                self.scene.get_actor(h) for h in self.fixture.get_nested_link_actors()
            ]
            self.slider_com0 = np.asarray(
                self.slider.get_center_of_mass_transform().translation
            ).copy()
            self.env = [self.slider, base]
            self.table_top = self.slider0[2] - 0.054
        else:
            low = float((self.rot.apply(self.mesh.vertices) + self.origin)[:, 2].min())
            self.table_top = low - (0.004 if spec.task == "calibration" else 0.008)
            table = self.actor(
                "table",
                trimesh.creation.box([0.36, 0.24, 0.025]),
                self.origin + [0, 0, self.table_top - self.origin[2] - 0.0125],
                self.rot,
                1.0,
                spec.friction,
                static=True,
            )
            self.env = [table]
            if spec.task == "push":
                self.slider0 = self.origin + self.rot.apply([0.041, spec.offset, 0])
                self.slider0[2] = self.table_top + 0.012
                self.slider = self.actor(
                    "visible_puck",
                    trimesh.creation.cylinder(radius=0.020, height=0.024, sections=32),
                    self.slider0,
                    self.rot,
                    0.08,
                    spec.friction,
                )
                self.env.append(self.slider)
                self.axis = self.rot.apply([1, 0, 0])
        self.queries = []
        self.tets = {}
        self.rest_det = {}
        for actor in [self.tool, *self.gripper.gels.values()]:
            self.queries.append(actor.register_query(p.QueryType.CONTACT_POINTS))
        for side, gel in self.gripper.gels.items():
            for query in (p.QueryType.NODE_POSITIONS, p.QueryType.NODE_CONTACT_FORCES):
                self.queries.append(gel.register_query(query))
            mesh = p.get_shape_mesh(gel.get_reference_shape())
            self.tets[side] = (
                np.asarray(mesh.connectivity, dtype=np.int32).reshape(-1, 4).copy()
            )
            points = np.asarray(mesh.coordinates).reshape(-1, 3)[self.tets[side]]
            self.rest_det[side] = np.linalg.det(points[:, 1:] - points[:, :1])
        self.cloud = sample_mesh_surface(self.mesh, 256, spec.seed)
        if hasattr(self, "environment_meshes"):
            envmeshes = self.environment_meshes
        elif spec.task in ("hook", "probe"):
            envmeshes = [slider_mesh(), trimesh.creation.box([0.16, 0.09, 0.012])]
        else:
            envmeshes = [trimesh.creation.box([0.36, 0.24, 0.025])]
            if spec.task == "push":
                envmeshes.append(
                    trimesh.creation.cylinder(radius=0.020, height=0.024, sections=32)
                )
        self.envclouds = [
            sample_mesh_surface(m, 256 // len(envmeshes), spec.seed + 1 + j)
            for j, m in enumerate(envmeshes)
        ]
        self.rng = np.random.default_rng(spec.seed)
        self.history = []
        self.reference = None
        self.initial_relative = None
        self.initial_relative_rotation = None
        self.last_vel = np.zeros(6)
        self.last_ee = None
        self.command = np.zeros(3)

    def make_tool_mesh(self):
        return tool_mesh(self.spec.task)

    def make_slider_mesh(self):
        return slider_mesh()

    def actor(self, name, mesh, pos, rot, mass, friction, static=False, gravity=True):
        p = self.p
        shape = p.create_tri_mesh_shape(
            np.asarray(mesh.vertices, dtype=np.float32).ravel(),
            np.asarray(mesh.faces, dtype=np.int32).ravel(),
        )
        contact = p.ContactParams()
        contact.coulomb_friction_coefficient = friction
        contact.penalty_coefficient = 5e10
        a = self.scene.create_rigid_actor(
            name=name,
            shape=shape,
            mass=mass,
            contact=contact,
            collider_type=p.ColliderType.MESH,
            has_gravity=gravity and not static,
            world_from_local=p.TransformRT(
                translation=pos,
                rotation=p.Quaternion.from_rotation_vector(rot.as_rotvec()),
            ),
        )
        if static:
            fixed_com = a.get_center_of_mass_transform()
            a.add_boundary_condition_dofs_world(
                np.arange(6, dtype=np.int32),
                np.r_[np.asarray(fixed_com.translation), rot.as_rotvec()],
            )
        return a

    def checkpoint(self):
        buf = self.p.DynamicArrayUint8()
        self.scene.capture_state_to_bytes(buf)
        return bytes(buf), dict(
            released=self.released,
            last_vel=self.last_vel.copy(),
            last_ee=None if self.last_ee is None else self.last_ee.copy(),
            command=self.command.copy(),
            reference=self.reference,
            initial_relative=self.initial_relative,
            initial_relative_rotation=self.initial_relative_rotation,
            rng=self.rng.bit_generator.state,
            history=list(self.history),
        )

    def restore(self, checkpoint):
        import copy

        state, ctrl = checkpoint
        self.scene.restore_state_from_bytes(state)
        for k, v in copy.deepcopy(ctrl).items():
            if k == "rng":
                self.rng.bit_generator.state = v
            else:
                setattr(self, k, v)

    def command_at(self, t, branch="nominal", anchor_command=None):
        command, phase = trajectory(self.spec, t)
        if t >= self.spec.branch_time and branch != "nominal":
            q = smooth((t - self.spec.branch_time) / 1.8)
            delta = {
                "hold": [0, 0, 0],
                "retreat": [0, 0, -0.018],
                "alternate": [0.012, 0, 0],
            }[branch]
            if self.spec.task == "probe":
                delta = {
                    "hold": [0, 0, 0],
                    "retreat": [-0.012, 0, 0],
                    "alternate": [0, 0, 0.009],
                }[branch]
            command = anchor_command + q * np.asarray(delta)
            phase = "branch_" + branch
        command = Rotation.from_euler("z", self.spec.angle_deg, degrees=True).apply(
            command
        )
        return command, phase, np.zeros(3)

    def step(self, t, branch="nominal", anchor_command=None):
        p, r = self.p, self.r
        command, phase, rotation = self.command_at(t, branch, anchor_command)
        self.command = command.copy()
        if t >= 1.0 and not self.released:
            self.tool.clear_boundary_conditions()
            self.released = True
        target = p.TransformRT(
            translation=self.ee0 + self.rot.apply(command),
            rotation=p.Quaternion.from_rotation_vector(
                (
                    self.rot
                    * Rotation.from_rotvec(rotation)
                    * self.rot.inv()
                    * self.ee_rot
                ).as_rotvec()
            ),
        )
        obs = self.osc.get_current_observations_from_mochi()
        effort = np.asarray(
            self.osc.compute_output(
                obs,
                r.ControllerBasicOscPdTarget(
                    root_from_target_ee=self.root.inverse() * target
                ),
            ),
            dtype=np.float32,
        ).copy()
        effort[self.finger_dofs] -= (
            self.spec.grip_force_n
            or (35.0 if self.spec.task in ("hook", "probe") else 28.0)
        ) * smooth(t / 0.9)
        self.gripper.actor.set_external_forces_on_dofs(self.dofs, effort)
        if hasattr(self, "apply_environment_force"):
            self.apply_environment_force()
        elif self.spec.task in ("hook", "probe"):
            pos = np.asarray(self.slider.get_center_of_mass_transform().translation)
            q = float((pos - self.slider0) @ self.axis)
            v = float(np.asarray(self.slider.get_linear_velocity()) @ self.axis)
            # Spring returns to zero; soft stops limit travel to 60 mm.
            force = (
                -self.spec.stiffness * q
                - 0.8 * v
                + 300 * max(-q, 0)
                - 300 * max(q - 0.060, 0)
            )
            self.fixture.set_external_forces_on_dofs(
                np.array([0], dtype=np.int32), [force]
            )
        self.scene.step(self.spec.dt)
        solver_stats = self.scene.get_solver_stats()
        obs = self.osc.get_current_observations_from_mochi()
        ee = pose7(obs.world_from_ee_link, np.zeros(3))
        tf = self.tool.get_center_of_mass_transform()
        pose = pose7(tf, np.zeros(3))
        com = pose[:3]
        velocity = np.r_[
            np.asarray(self.tool.get_linear_velocity()),
            np.asarray(self.tool.get_angular_velocity()),
        ]
        contacts = list(self.tool.get_contact_points_world())
        envw = np.zeros(6)
        direct = np.zeros(6)
        fieldw = np.zeros(6)
        densew = np.zeros(6)
        moment_corrected_w = np.zeros(6)
        contact_count = 0
        rigid_pen = 0.0
        for actor in self.env:
            w = aggregate_contact_points(
                contacts, self.tool.get_handle(), actor.get_handle(), com
            )
            envw += np.r_[w.force, w.torque]
            contact_count += w.count
        for c in contacts:
            if c.actor_a in [a.get_handle() for a in self.env] or c.actor_b in [
                a.get_handle() for a in self.env
            ]:
                rigid_pen = max(rigid_pen, max(0.0, -float(c.distance)))
        row = {}
        counts = []
        min_jacobian = 1.0
        for side in ("left", "right"):
            gel = self.gripper.gels[side]
            housing = self.gripper.links[f"franka_{side}_gelsight_housing"]
            field = self.gripper.get_surface_force_field(side)
            positions = self.gripper.gel_rest_sensor_surface[side]
            sensor = housing.get_root_transform()
            from tactile_grid import surface_displacement

            displacement = surface_displacement(
                gel.get_node_positions_local(),
                self.gripper.gel_surface_grid_indices[side],
                sensor,
                self.root,
                positions,
            )
            nodes = np.asarray(gel.get_node_positions_local()).reshape(-1, 3)[
                self.tets[side]
            ]
            jac = np.linalg.det(nodes[:, 1:] - nodes[:, :1]) / self.rest_det[side]
            min_jacobian = min(min_jacobian, float(jac.min()))
            positions = positions + displacement
            w = integrate_surface_force_field(
                field, positions, sensor, com, negate=True
            )
            fieldw += np.r_[w.force, w.torque]
            dense_positions, dense_forces = self.gripper.get_dense_contact_field(side)
            if getattr(self.spec, 'record_dense_field', False):
                row[f'gel_node_positions_world_{side}'] = dense_positions.astype(np.float32)
                row[f'gel_node_forces_world_{side}'] = dense_forces.astype(np.float32)
            densew -= np.r_[dense_forces.sum(axis=0),np.cross(dense_positions-com,dense_forces).sum(axis=0)]
            from tactile_grid import surface_cell_moments
            marker_world = transform_points(sensor, positions)
            cell_moments, unmapped = surface_cell_moments(
                dense_positions, dense_forces, self.gripper.gel_surface_grid_indices[side], marker_world, sensor)
            sensor_rotation = Rotation.from_rotvec(np.asarray(sensor.rotation.to_rotation_vector()))
            # Forces are on the gel; negate the intrinsic couples for the tool.
            moment_corrected_w += np.r_[w.force, w.torque-sensor_rotation.apply(cell_moments.sum((0,1)))]
            row[f'tactile_cell_moment_{side}'] = cell_moments.astype(np.float32)
            row[f'tactile_unmapped_wrench_{side}'] = unmapped.astype(np.float32)
            w = aggregate_contact_points(
                contacts, self.tool.get_handle(), gel.get_handle(), com
            )
            direct += np.r_[w.force, w.torque]
            counts.append(w.count)
            row[f"tactile_force_field_{side}"] = field.copy()
            row[f"tactile_coord_{side}"] = transform_points(sensor, positions)
            row[f"tactile_displacement_{side}"] = displacement
        localrel = Rotation.from_quat(ee[3:]).inv().apply(pose[:3] - ee[:3])
        if t >= 1.5 and self.initial_relative is None:
            self.initial_relative = localrel.copy()
        relative_rotation = Rotation.from_quat(ee[3:]).inv() * Rotation.from_quat(
            pose[3:]
        )
        if t >= 1.5 and self.initial_relative_rotation is None:
            self.initial_relative_rotation = relative_rotation.as_quat()
        angular_slip = (
            0.0
            if self.initial_relative_rotation is None
            else np.rad2deg(
                (
                    relative_rotation
                    * Rotation.from_quat(self.initial_relative_rotation).inv()
                ).magnitude()
            )
        )
        slip = (
            0.0
            if self.initial_relative is None
            else np.linalg.norm(localrel - self.initial_relative)
        )
        acc = (velocity - self.last_vel) / self.spec.dt
        R = Rotation.from_quat(pose[3:]).as_matrix()
        I = R @ self.inertia @ R.T
        inertial = np.r_[
            self.mass * (acc[:3] - [0, 0, -9.81]),
            I @ acc[3:] + np.cross(velocity[3:], I @ velocity[3:]),
        ]
        eevel = (
            np.zeros(6)
            if self.last_ee is None
            else np.r_[
                (ee[:3] - self.last_ee[:3]) / self.spec.dt,
                (
                    Rotation.from_quat(ee[3:])
                    * Rotation.from_quat(self.last_ee[3:]).inv()
                ).as_rotvec()
                / self.spec.dt,
            ]
        )
        self.last_vel = velocity.copy()
        self.last_ee = ee.copy()
        progress = 0.0
        fixture = np.zeros(13)
        guide_drift = 0.0
        if self.slider is not None:
            fixture = np.r_[
                pose7(self.slider.get_center_of_mass_transform(), np.zeros(3)),
                np.asarray(self.slider.get_linear_velocity()),
                np.asarray(self.slider.get_angular_velocity()),
            ]
            progress = float((fixture[:3] - self.slider0) @ self.axis)
            if self.spec.task in ("hook", "probe"):
                guide_drift = float(
                    np.linalg.norm(
                        fixture[:3] - self.slider_com0 - progress * self.axis
                    )
                )
        ref = np.stack([row[f"tactile_force_field_{s}"] for s in ("left", "right")])
        if t >= 1.45 and self.reference is None:
            self.reference = ref.copy()
        row.update(
            ee_pose=ee,
            ee_vel=eevel,
            tool_pose=pose,
            tool_vel=velocity,
            objectpointcloud=transform_points(
                self.tool.get_root_transform(), self.cloud
            ),
            env_point_cloud=np.concatenate(
                [
                    transform_points(a.get_root_transform(), cloud)
                    for a, cloud in zip(self.env, self.envclouds)
                ]
            ),
            actions=np.r_[command, rotation],
            timestamps=t + self.spec.dt,
            direct_gel_wrench=direct,
            field_gel_wrench=fieldw,
            dense_nodal_gel_wrench=densew,
            moment_corrected_gel_wrench=moment_corrected_w,
            moment_corrected_inferred_extrinsic_wrench=inertial-moment_corrected_w,
            dense_nodal_inferred_extrinsic_wrench=inertial-densew,
            extrinsic_contact_wrench=envw,
            dynamic_inferred_extrinsic_wrench=inertial - fieldw,
            fixture_state=fixture,
            task_progress=progress,
            slip_m=slip,
            rigid_penetration_m=rigid_pen,
            angular_slip_deg=angular_slip,
            ee_target_pose=pose7(target, np.zeros(3)),
            commanded_joint_effort=effort.copy(),
            finger_opening_coordinates_m=np.asarray(obs.dof_positions)[self.finger_dofs].copy(),
            finger_synchronization_error_m=float(np.diff(np.asarray(obs.dof_positions)[self.finger_dofs])[0]),
            solver_convergence_status=int(solver_stats.convergence_status),
            solver_residual_norm=float(solver_stats.residual_norm),
            solver_nonlinear_iterations=int(solver_stats.max_non_linear_iters),
            min_gel_jacobian=min_jacobian,
            guide_drift_m=guide_drift,
            contact_counts=counts,
            environment_contact_count=contact_count,
            phase=phase,
            initialization=t < 1.5,
            rewards=progress,
            contact_event=np.linalg.norm(envw[:3]) > 0.1,
        )
        self.history.append(np.r_[ref.ravel(), ee, command].astype(np.float32))
        self.history = self.history[-50:]
        return row

    def close(self):
        self.p.destroy_scene(self.scene)


def stack(rows):
    return {k: np.asarray([r[k] for r in rows]) for k in rows[0]}


def metrics(data, spec):
    active = ~data["initialization"]
    truth = data["extrinsic_contact_wrench"][active]
    est = data["dynamic_inferred_extrinsic_wrench"][active]
    pen = data["rigid_penetration_m"][active] > 0.001
    sustained = bool(
        np.any(np.convolve(pen.astype(int), np.ones(3, dtype=int), "valid") == 3)
    )
    finite = all(np.isfinite(v).all() for v in data.values() if v.dtype.kind in "fc")
    retained = (
        float(data["slip_m"][active].max()) < 0.012
        and float(data["angular_slip_deg"][active].max()) < 15.0
    )
    progress = data["task_progress"]
    success = bool(np.max(progress) >= 0.040)
    if spec.task == "hook":
        above = progress >= 0.040
        hold_samples = max(1, round(0.3 / spec.dt))
        success = bool(
            len(above) >= hold_samples
            and np.any(
                np.convolve(
                    above.astype(int), np.ones(hold_samples, dtype=int), "valid"
                )
                == hold_samples
            )
        )
    if spec.task == "push":
        success = bool(abs(float(progress[-1]) - 0.050) <= 0.010)
    if spec.task == "probe":
        success = all(
            bool(
                np.any(
                    (data["phase"] == phase)
                    & data["contact_event"]
                    & (progress > 0.001)
                )
            )
            for phase in ("probe", "later_pull")
        )
    if spec.task == "calibration":
        success = bool(np.max(np.linalg.norm(truth[:, :3], axis=1)) > 0.2)
    healthy = bool(
        data["min_gel_jacobian"].min() > 0 and data["guide_drift_m"].max() < 0.001
    )
    result = dict(
        task=spec.task,
        episode_id=spec.episode_id,
        physical_valid=finite and retained and not sustained and healthy,
        grasp_retained=retained,
        task_success=success,
        max_slip_mm=float(data["slip_m"].max() * 1000),
        max_angular_slip_deg=float(data["angular_slip_deg"].max()),
        max_progress_mm=float(data["task_progress"].max() * 1000),
        max_penetration_mm=float(data["rigid_penetration_m"].max() * 1000),
        sustained_penetration=sustained,
        contact_fraction=float(data["contact_event"][active].mean()),
    )
    result.update(
        min_gel_jacobian=float(data["min_gel_jacobian"].min()),
        max_guide_drift_mm=float(data["guide_drift_m"].max() * 1000),
    )
    for name, sl in (("force", slice(0, 3)), ("torque", slice(3, 6))):
        err = np.linalg.norm(est[:, sl] - truth[:, sl], axis=1)
        scale = np.sqrt(np.mean(np.sum(truth[:, sl] ** 2, axis=1)))
        result[name + "_rmse"] = float(np.sqrt(np.mean(err**2)))
        result[name + "_nrmse"] = float(np.sqrt(np.mean(err**2)) / max(scale, 1e-6))
        result[name + "_p95_error"] = float(np.quantile(err, 0.95))
        flat_true = truth[:, sl].ravel()
        flat_est = est[:, sl].ravel()
        result[name + "_correlation"] = (
            float(np.corrcoef(flat_true, flat_est)[0, 1])
            if np.std(flat_true) > 1e-8
            else None
        )
    result["phase_metrics"] = {}
    for phase in np.unique(data["phase"][active]):
        mask = active & (data["phase"] == phase)
        error = (
            data["dynamic_inferred_extrinsic_wrench"][mask]
            - data["extrinsic_contact_wrench"][mask]
        )
        result["phase_metrics"][str(phase)] = {
            "frames": int(mask.sum()),
            "force_rmse": float(np.sqrt(np.mean(np.sum(error[:, :3] ** 2, axis=1)))),
            "torque_rmse": float(np.sqrt(np.mean(np.sum(error[:, 3:] ** 2, axis=1)))),
        }
    contact = data["contact_event"][active]
    inferred = np.linalg.norm(est[:, :3], axis=1) > 0.2
    result["free_space_false_contact_fraction"] = (
        float(inferred[~contact].mean()) if (~contact).any() else None
    )
    result["tactile_valid"] = (
        result["physical_valid"]
        and (result["force_nrmse"] < 0.2 or result["force_rmse"] < 0.2)
        and result["torque_rmse"] < 0.01
    )
    result["tactile_precision_note"] = (
        "valid means <0.2 N RMS or <20% relative force error, AND <0.01 Nm torque RMS; inspect both absolute and relative errors"
    )
    result["imitation_eligible"] = result["physical_valid"] and success
    return result


def save(path, data, spec, world, result, snapshot=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    config = json.dumps(asdict(spec), sort_keys=True)
    with h5py.File(path, "w") as f:
        f.attrs.update(
            schema_version="vt_acwm_superdex_tool_pilot_v1",
            config_json=config,
            config_hash=hashlib.sha256(config.encode()).hexdigest(),
            group_id=spec.group,
            branch_group_id=spec.group,
            metrics_json=json.dumps(result),
            precision="fp32",
            wrench_frame="world about instantaneous tool COM",
            action_semantics="absolute EE translation offset in initial tool frame; rotation offset rad",
            hidden_labels_excluded_from_observations=True,
            collision_policy="housing-tool disabled as in existing validated gripper; gel-tool enabled",
            initial_state_hash=getattr(world, "initial_hash", "unavailable"),
            implementation_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            action_frame_rotation_world=world.rot.as_matrix().ravel(),
            initial_ee_position_world=world.ee0,
        )
        g = f.create_group("observations")
        for k, v in data.items():
            if v.dtype.kind in "UO":
                v = v.astype(h5py.string_dtype())
            g.create_dataset(k, data=v, compression="lzf")
        for alias, key in (
            ("point_cloud", "objectpointcloud"),
            ("tactile_data_left", "tactile_force_field_left"),
            ("tactile_data_right", "tactile_force_field_right"),
            ("field_inferred_extrinsic_wrench", "dynamic_inferred_extrinsic_wrench"),
        ):
            if key in g:
                g[alias] = g[key]
        for key in ("actions", "rewards", "timestamps"):
            f[key] = g[key]
        dones = np.zeros(len(data["timestamps"]), dtype=bool)
        dones[-1] = True
        f.create_dataset("dones", data=dones)
        f["contact_phase"] = g["phase"]
        f["episode_lengths"] = np.int32(len(dones))
        f["episode_rewards"] = np.float32(data["rewards"].sum())
        f.create_dataset("tactile_reference", data=world.reference)
        labels = f.create_group("labels")
        labels.attrs.update(
            stiffness_n_per_m=spec.stiffness,
            friction=spec.friction,
            offset_m=spec.offset,
        )
        for key in (
            "physical_valid",
            "tactile_valid",
            "grasp_retained",
            "imitation_eligible",
            "task_success",
        ):
            labels[key] = np.bool_(result[key])
        if snapshot:
            state, ctrl = snapshot
            s = f.create_group("checkpoint")
            s.create_dataset(
                "physics_bytes",
                data=np.frombuffer(state, dtype=np.uint8),
                compression="lzf",
            )
            s.attrs["physics_state_hash"] = hashlib.sha256(state).hexdigest()
            for key in (
                "last_vel",
                "last_ee",
                "command",
                "reference",
                "initial_relative",
                "initial_relative_rotation",
                "history",
            ):
                if ctrl[key] is not None:
                    s.create_dataset(key, data=np.asarray(ctrl[key]), compression="lzf")
            s.attrs["released"] = ctrl["released"]
            s.attrs["rng_json"] = json.dumps(ctrl["rng"])
            controller_hash = hashlib.sha256(
                json.dumps(
                    ctrl, sort_keys=True, default=lambda a: np.asarray(a).tolist()
                ).encode()
            ).hexdigest()
            s.attrs["controller_state_hash"] = controller_hash
            s.attrs["snapshot_hash"] = hashlib.sha256(
                state + controller_hash.encode()
            ).hexdigest()


def run(spec, output, render=False, branches=False):
    from dataclasses import replace

    world = World(spec)
    rows = []
    snapshot = None
    abort = None
    recorder = None
    if render:
        from visuals import Recorder

        recorder = Recorder(world, output / f"episode_{spec.episode_id:04d}")
    start = time.perf_counter()
    try:
        for i in range(round(spec.duration / spec.dt)):
            t = i * spec.dt
            if abs(t - 1.5) < spec.dt / 2:
                initial = world.checkpoint()
                world.initial_hash = hashlib.sha256(initial[0]).hexdigest()
            if branches and abs(t - spec.branch_time) < spec.dt / 2:
                snapshot = world.checkpoint()
            row = world.step(t)
            rows.append(row)
            if recorder and i % max(1, round(0.05 / spec.dt)) == 0:
                recorder.frame(row)
            if t >= 1.5 and (
                not np.isfinite(row["tool_pose"]).all()
                or row["slip_m"] > 0.025
                or np.linalg.norm(row["extrinsic_contact_wrench"][:3]) > 15
            ):
                abort = "nonfinite/drop/force_limit"
                break
        data = stack(rows)
        result = metrics(data, spec)
        result.update(abort_reason=abort, rollout_seconds=time.perf_counter() - start)
        if abort:
            result.update(
                physical_valid=False, imitation_eligible=False, tactile_valid=False
            )
        save(
            output / f"episode_{spec.episode_id:04d}.h5",
            data,
            spec,
            world,
            result,
            snapshot,
        )
        if recorder:
            recorder.close(data)
        if branches and snapshot and not abort:
            anchor = trajectory(spec, spec.branch_time)[0]
            for b in ("nominal", "hold", "retreat", "alternate"):
                world.restore(snapshot)
                br = []
                branch_abort = None
                for i in range(
                    round(spec.branch_time / spec.dt), round(spec.duration / spec.dt)
                ):
                    row = world.step(i * spec.dt, b, anchor)
                    br.append(row)
                    if (
                        not np.isfinite(row["tool_pose"]).all()
                        or row["slip_m"] > 0.025
                        or np.linalg.norm(row["extrinsic_contact_wrench"][:3]) > 15
                    ):
                        branch_abort = "nonfinite/drop/force_limit"
                        break
                bd = stack(br)
                bs = replace(spec, branch=b, group=spec.group)
                bm = metrics(bd, bs)
                bm["abort_reason"] = branch_abort
                if branch_abort:
                    bm.update(
                        physical_valid=False,
                        imitation_eligible=False,
                        tactile_valid=False,
                    )
                if b == "nominal":
                    original = data["tool_pose"][round(spec.branch_time / spec.dt) :]
                    bm["identical_action_replay_max_pose_error"] = float(
                        np.max(np.abs(original - bd["tool_pose"]))
                    )
                save(
                    output / f"episode_{spec.episode_id:04d}_branch_{b}.h5",
                    bd,
                    bs,
                    world,
                    bm,
                    snapshot,
                )
        return result
    finally:
        world.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", choices=["hook", "probe", "push", "calibration"], default="hook"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--branches", action="store_true")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--stiffness", type=float, default=30)
    parser.add_argument("--offset", type=float, default=0)
    parser.add_argument("--specs", type=Path)
    args = parser.parse_args()
    import superdex.physics as p

    p.initialize(num_worker_threads=0)
    try:
        duration = {"hook": 18.5, "probe": 14.0, "push": 12.0, "calibration": 12.0}[
            args.task
        ]
        specs = (
            [Spec(**s) for s in json.loads(args.specs.read_text())]
            if args.specs
            else [
                Spec(
                    task=args.task,
                    dt=args.dt,
                    stiffness=args.stiffness,
                    offset=args.offset,
                    duration=duration,
                )
            ]
        )
        results = []
        for spec in specs:
            result = run(spec, args.output, args.render, args.branches)
            results.append(result)
            print(json.dumps(result), flush=True)
        (args.output / "metrics.json").write_text(json.dumps(results, indent=2))
    finally:
        p.shutdown()


if __name__ == "__main__":
    main()
