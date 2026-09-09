"""Bounded contact-rich benchmark pilot; legacy experiments remain unchanged.

Run --phase repair first, then canonical, diversity, branches and composite.
Every attempted trajectory consumes the persistent 256-trajectory budget.
"""

from dataclasses import asdict, dataclass, replace
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import hashlib
import json
import multiprocessing
import os
import time

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from decisions import (
    DecisionSpec,
    HookDecision,
    interpolate,
    stop_reason,
    write_episode,
)
from key_stages import KeyStageWorld, stage_audit
from pilot import metrics, smooth, stack
from benchmark_geometry import (
    peg_mesh,
    profile,
    socket_mesh,
    height_surface,
    guide_wall,
    contains_profile,
    sequence_completion,
    surface_stroke_audit,
)

HERE = Path(__file__).resolve().parent
SURFACES = (
    "flat_scrape",
    "flat_draw",
    "cylindrical_peel",
    "concave_draw",
    "straight_wall",
    "rounded_wall",
    "straight_slot",
    "curved_slot",
)
# Explicit old manifests remain replayable, but no active recipe emits these.
RETIRED_SURFACES = ("flat_peel", "convex_scrape")
PROFILES = ("round", "square", "hex", "d", "key")


@dataclass(frozen=True)
class BenchmarkSpec(DecisionSpec):
    family: str = "insertion"
    variant: str = "key"
    stage: str = "insertion"
    profile_shape: str = "key"
    block: str = "canonical"
    revision: int = 18
    gel_geometry: str | None = None
    scale: float = 1.0
    bar_scale: float = 1.0
    clearance_m: float = 0.0006
    chamfer_m: float = 0.003
    radius_m: float = 0.12
    fixture_x_m: float = 0.0
    fixture_y_m: float = 0.0
    fixture_roll_deg: float = 0.0
    fixture_pitch_deg: float = 0.0
    robot_roll_deg: float = 0.0
    robot_pitch_deg: float = 0.0
    grasp_depth_m: float | None = None
    speed_scale: float = 0.75
    turn_command_deg: float = 100.0
    load_mode: str = "spring"
    friction_torque_nm: float = 0.015
    detent_torque_nm: float = 0.015
    surface_force_n: float | None = None
    surface_tip_compensation: bool | None = None
    surface_half_extent_m: float | None = None
    camera_yaw_deg: float = 0.0
    camera_pitch_deg: float = 0.0
    camera_seed: int = 0
    tactile_seed: int = 0
    physical_seed: int = 0
    geometry_seed: int = 0
    max_wall_s: float = 1800.0
    effective_friction: bool = False
    arm_stiffness_n_m: float = 1100.0
    arm_damping_ns_m: float = 85.0
    arm_stiffness_nm_rad: float = 40.0
    arm_damping_nms_rad: float = 4.0
    surface_heading_deg: float | None = None
    tool_pose_feedback: bool | None = None
    record_dense_field: bool = False
    seat_alignment_search: bool = False
    surface_contact_model: str | None = None
    solver_iterations: int = 4
    solver_abs_tolerance: float = 0.001
    calibrate_precontact_grasp: bool | None = None
    surface_attack_sweep_deg: float | None = None
    surface_admittance_m_ns: float | None = None
    ideal_arm_friction_compensation: bool | None = None
    finger_coupling_stiffness_n_m: float | None = None
    guide_admittance_m_ns: float | None = None
    guide_max_offset_m: float | None = None
    guide_max_speed_m_s: float | None = None
    guide_normal_control: bool | None = None
    surface_stroke_m: float | None = None
    guide_acquisition_gate: bool | None = None
    guide_acquisition_timeout_s: float = 8.0
    hook_pull_m: float | None = None
    turning_return_overtravel_deg: float | None = None
    guide_acquisition_force_n: float | None = None
    return_after_turn: bool | None = None
    peeler_radius_m: float = 0.020
    surface_lateral_samples: int | None = None
    surface_working_tip_tracking: bool | None = None
    rotor_friction_model: str | None = None
    surface_sdf_voxel_m: float = 0.0005

    def __post_init__(self):
        modern = self.revision >= 15
        repaired = self.revision >= 18
        if self.rotor_friction_model is None:
            object.__setattr__(
                self,
                "rotor_friction_model",
                "joint" if repaired and self.load_mode == "friction" else "external",
            )
        if self.surface_working_tip_tracking is None:
            object.__setattr__(
                self,
                "surface_working_tip_tracking",
                repaired and self.variant in ("concave_draw", "convex_scrape"),
            )
        if (
            repaired
            and self.surface_lateral_samples is None
            and self.variant in ("concave_draw", "convex_scrape")
        ):
            object.__setattr__(self, "surface_lateral_samples", 2)
        if self.return_after_turn is None:
            object.__setattr__(self, "return_after_turn", self.revision < 16)
        guided = self.family == "surface" and (
            "wall" in self.variant or "slot" in self.variant
        )
        defaults = dict(
            ideal_arm_friction_compensation=modern,
            finger_coupling_stiffness_n_m=1e6 if modern else 0.0,
            grasp_depth_m=0.008 if modern and self.family == "hook" else 0.0,
            surface_force_n=1.0
            if repaired and self.variant == "cylindrical_peel"
            else 0.5
            if modern
            and (
                guided
                or self.variant == "concave_draw"
                or repaired
                and self.variant == "convex_scrape"
            )
            else 2.0,
            surface_contact_model="sdf"
            if repaired and self.variant == "cylindrical_peel"
            else "analytic_box"
            if modern
            and self.family == "surface"
            and not (
                "convex" in self.variant
                or "concave" in self.variant
                or self.variant == "cylindrical_peel"
            )
            else "mesh",
            calibrate_precontact_grasp=modern and self.family == "surface",
            surface_admittance_m_ns=0.0012 if modern else 0.0004,
            guide_normal_control=modern and guided,
            guide_acquisition_gate=modern and guided,
            guide_admittance_m_ns=0.004 if modern else 0.0004,
            guide_max_offset_m=0.010 if modern else 0.003,
            guide_max_speed_m_s=0.001 if modern else 0.0005,
        )
        defaults["hook_pull_m"] = 0.054 if modern else 0.050
        defaults["turning_return_overtravel_deg"] = (
            12.0 if modern and self.load_mode == "friction" else 0.0
        )
        # Acquisition uses the same 0.1 N contact threshold as the independent
        # guide-contact event. It does not relax the 80% continuity task gate.
        defaults["guide_acquisition_force_n"] = 0.1 if modern else 0.25
        for name, value in defaults.items():
            if getattr(self, name) is None:
                object.__setattr__(self, name, value)
        if modern and self.family == "surface":
            if self.surface_stroke_m is None:
                object.__setattr__(
                    self, "surface_stroke_m", 0.025 if "peel" in self.variant else 0.020
                )
            if self.surface_half_extent_m is None and (
                "scrape" in self.variant or "peel" in self.variant
            ):
                object.__setattr__(
                    self,
                    "surface_half_extent_m",
                    min(0.070, 0.9 * self.radius_m)
                    if "convex" in self.variant
                    else 0.070,
                )
        if self.surface_heading_deg is None:
            object.__setattr__(
                self,
                "surface_heading_deg",
                45.0
                if repaired and guided
                else 90.0
                if (modern and self.family == "surface")
                or (
                    self.revision >= 13
                    and ("scrape" in self.variant or "peel" in self.variant)
                )
                else 0.0,
            )
        if self.tool_pose_feedback is None:
            object.__setattr__(
                self,
                "tool_pose_feedback",
                not modern and not (self.revision >= 13 and self.family == "surface"),
            )
        if self.surface_attack_sweep_deg is None:
            object.__setattr__(
                self,
                "surface_attack_sweep_deg",
                0.0
                if self.revision >= 14
                else 8.0
                if "peel" in self.variant
                else 5.0
                if "scrape" in self.variant
                else 0.0,
            )
        for value in (
            self.arm_stiffness_n_m,
            self.arm_damping_ns_m,
            self.arm_stiffness_nm_rad,
            self.arm_damping_nms_rad,
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("finite positive arm impedance gains required")
        if self.surface_contact_model not in ("mesh", "analytic_box", "sdf"):
            raise ValueError("unknown surface contact model")
        if not 0.0001 <= self.surface_sdf_voxel_m <= 0.002:
            raise ValueError("surface SDF voxel size outside 0.1–2 mm bounds")
        if self.solver_iterations < 1 or self.solver_abs_tolerance <= 0:
            raise ValueError("positive solver iterations/tolerance required")
        if (
            not np.isfinite(self.finger_coupling_stiffness_n_m)
            or self.finger_coupling_stiffness_n_m < 0
        ):
            raise ValueError("nonnegative finite finger coupling stiffness required")
        if not np.isfinite(self.surface_heading_deg) or not np.isfinite(
            self.surface_attack_sweep_deg
        ):
            raise ValueError("finite surface angles required")
        if (
            not np.isfinite(self.surface_admittance_m_ns)
            or self.surface_admittance_m_ns <= 0
        ):
            raise ValueError("positive finite surface admittance required")
        if any(
            not np.isfinite(v) or v <= 0
            for v in (
                self.guide_admittance_m_ns,
                self.guide_max_offset_m,
                self.guide_max_speed_m_s,
            )
        ):
            raise ValueError("positive finite guide admittance bounds required")
        if self.surface_stroke_m is not None and not 0 < self.surface_stroke_m <= 0.04:
            raise ValueError("surface stroke must be in (0, 40 mm]")
        if (
            not 0.04 <= self.hook_pull_m <= 0.06
            or not 0 <= self.turning_return_overtravel_deg <= 15
        ):
            raise ValueError("bounded hook pull and return overtravel required")
        if not 0 < self.guide_acquisition_force_n <= 0.5:
            raise ValueError("guide acquisition force must be in (0, 0.5 N]")
        if self.variant == "cylindrical_peel" and (
            not 0.01 <= self.peeler_radius_m <= 0.025
            or self.surface_heading_deg != 90.0
        ):
            raise ValueError(
                "Blade-matched peeling currently supports 10–25 mm radius and 90-degree stroke heading"
            )
        if self.surface_contact_model == "analytic_box" and (
            "convex" in self.variant
            or "concave" in self.variant
            or self.variant == "cylindrical_peel"
        ):
            raise ValueError("analytic box is only exact for a flat floor")
        if self.gel_geometry is None:
            object.__setattr__(
                self,
                "gel_geometry",
                "source_surface" if self.revision >= 12 else "legacy_box",
            )
        if self.gel_geometry not in ("source_surface", "legacy_box", "matched_box"):
            raise ValueError("unknown gel geometry")
        if self.surface_tip_compensation is None:
            object.__setattr__(self, "surface_tip_compensation", self.revision >= 11)
        if self.family not in ("surface", "hook", "insertion", "turning", "composite"):
            raise ValueError(self.family)
        object.__setattr__(
            self,
            "kind",
            "key" if self.family in ("insertion", "turning", "composite") else "hook",
        )
        stage = {
            "surface": "surface_following",
            "hook": "capture_pull",
            "turning": "turning",
        }.get(self.family, "insertion")
        object.__setattr__(self, "stage", stage)
        duration = self.duration
        super().__post_init__()
        if self.family == "surface":
            object.__setattr__(self, "task", "calibration")
        if not 0 < self.speed_scale <= 1.25:
            raise ValueError("speed multiplier outside screened bounds")
        if not 0 < self.grip_force_n <= 35:
            raise ValueError(
                "pilot command cap: 35 N per simulated finger; hardware mapping uncalibrated"
            )
        if (
            min(
                self.scale,
                self.bar_scale,
                self.clearance_m,
                self.chamfer_m,
                self.radius_m,
            )
            <= 0
        ):
            raise ValueError("positive geometry dimensions required")
        if self.surface_half_extent_m is not None and self.surface_half_extent_m <= 0:
            raise ValueError("positive surface extent required")
        if (
            self.surface_lateral_samples is not None
            and self.surface_lateral_samples < 2
        ):
            raise ValueError(
                "at least two samples across the extruded surface required"
            )
        if self.family == "surface" and (
            "convex" in self.variant or "concave" in self.variant
        ):
            if self.radius_m <= (self.surface_half_extent_m or 0.055):
                raise ValueError("curvature radius must exceed the surface half-span")
        if self.effective_friction and self.gel_friction <= 0:
            raise ValueError(
                "positive tool coefficient required for independent friction pairs"
            )
        if self.profile_shape not in (*PROFILES, "legacy"):
            raise ValueError(self.profile_shape)
        if self.family == "surface" and self.variant not in (
            *SURFACES,
            *RETIRED_SURFACES,
        ):
            raise ValueError(self.variant)
        if self.load_mode not in ("spring", "friction", "detent"):
            raise ValueError(self.load_mode)
        if self.rotor_friction_model not in ("external", "joint"):
            raise ValueError("rotor friction model must be external or joint")
        end = dict(surface=18, hook=35, insertion=20, turning=30, composite=48)[
            self.family
        ]
        if self.family == "composite" and self.seat_alignment_search:
            end = 72
        if self.family == "turning" and self.turning_return_overtravel_deg:
            end = 36
        if self.family == "turning" and not self.return_after_turn:
            end = 20
        if self.family == "composite" and not self.return_after_turn:
            end = 56 if self.seat_alignment_search else 32
        if self.family == "surface" and self.guide_acquisition_gate:
            end += self.guide_acquisition_timeout_s * self.speed_scale
        object.__setattr__(
            self, "duration", duration or (4 + (end - 4) / self.speed_scale)
        )
        object.__setattr__(self, "branch_time", 5.0 if self.family == "hook" else 8.0)


class BenchmarkMixin:
    def environment_friction_coefficient(self):
        # Mochi combines collider coefficients by geometric mean. The tool's
        # coefficient is gel_friction, so compensate on the environment body.
        return (
            self.spec.friction**2 / self.spec.gel_friction
            if self.spec.effective_friction
            else self.spec.friction
        )

    def prepare_robot(self):
        """Rotate the actual wrist before building a prepared tool grasp."""
        s = self.spec
        if not (s.robot_roll_deg or s.robot_pitch_deg):
            return
        p, r = self.p, self.r
        observation = self.osc.get_current_observations_from_mochi()
        initial = observation.world_from_ee_link
        start = Rotation.from_rotvec(np.asarray(initial.rotation.to_rotation_vector()))
        rotvec = Rotation.from_euler(
            "xy", [s.robot_roll_deg, s.robot_pitch_deg], degrees=True
        ).as_rotvec()
        desired = Rotation.from_rotvec(rotvec) * start
        correction = np.zeros(3)
        dofs = np.arange(self.gripper.actor.get_num_dofs(), dtype=np.int32)
        for i in range(800):
            obs = self.osc.get_current_observations_from_mochi()
            actual = Rotation.from_rotvec(
                np.asarray(obs.world_from_ee_link.rotation.to_rotation_vector())
            )
            if i >= 300:
                error_vec = (desired * actual.inv()).as_rotvec()
                correction += np.clip(error_vec, -np.deg2rad(1), np.deg2rad(1)) * s.dt
                correction = np.clip(correction, -np.deg2rad(5), np.deg2rad(5))
            target = p.TransformRT(
                translation=initial.translation,
                rotation=p.Quaternion.from_rotation_vector(
                    (
                        Rotation.from_rotvec(correction)
                        * Rotation.from_rotvec(smooth(i / 300) * rotvec)
                        * start
                    ).as_rotvec()
                ),
            )
            force = np.asarray(
                self.osc.compute_output(
                    obs,
                    r.ControllerBasicOscPdTarget(
                        root_from_target_ee=obs.world_from_root.inverse() * target
                    ),
                ),
                dtype=np.float32,
            ).copy()
            self.gripper.actor.set_external_forces_on_dofs(dofs, force)
            self.scene.step(s.dt)
        final = self.osc.get_current_observations_from_mochi().world_from_ee_link
        error = (
            Rotation.from_rotvec(rotvec)
            * start
            * Rotation.from_rotvec(
                np.asarray(final.rotation.to_rotation_vector())
            ).inv()
        ).magnitude()
        if error > np.deg2rad(2):
            raise RuntimeError(
                f"robot preposition orientation failed: {np.rad2deg(error):.3f} deg"
            )

    def prepared_frame(self, centers, default_rot, default_origin):
        if not (self.spec.robot_roll_deg or self.spec.robot_pitch_deg):
            return default_rot, default_origin + default_rot.apply(
                [0, 0, self.spec.grasp_depth_m]
            )
        centers = []
        for side in ("left", "right"):
            housing = self.gripper.links[
                f"franka_{side}_gelsight_housing"
            ].get_root_transform()
            rest = self.gripper.gel_rest_sensor_surface[side].reshape(-1, 3)
            from fidelity_campaign import transform_points

            centers.append(transform_points(housing, rest).mean(0))
        y = centers[0] - centers[1]
        y /= np.linalg.norm(y)
        z = Rotation.from_euler(
            "xy", [self.spec.robot_roll_deg, self.spec.robot_pitch_deg], degrees=True
        ).apply([0, 0, 1])
        x = np.cross(y, z)
        x /= np.linalg.norm(x)
        z = np.cross(x, y)
        rot = Rotation.from_matrix(np.column_stack([x, y, z]))
        offset = -0.009 if self.spec.task == "hook" else -0.018
        return rot, np.mean(centers, axis=0) + rot.apply(
            [0, 0, offset + self.spec.grasp_depth_m]
        )

    def clock(self, t):
        return t if t < 4 else 4 + (t - 4) * self.spec.speed_scale

    def checkpoint(self):
        state, control = super().checkpoint()
        for name in (
            "rotation_correction",
            "last_force",
            "force_guard",
            "guard_z",
            "prepared_seat_verified",
            "seat_dwell",
            "turn_start",
            "normal_offset",
            "last_normal_force",
            "last_local_force",
            "path_progress",
            "branch_value",
        ):
            if hasattr(self, name):
                value = getattr(self, name)
                control[name] = value.copy() if isinstance(value, np.ndarray) else value
        for name in (
            "lateral_offset",
            "last_guide_force",
            "search_start",
            "search_z",
            "search_origin_z",
            "search_yaw",
            "turn_anchor_yaw",
            "grasp_translation_calibration",
            "surface_wait_s",
            "guide_acquired",
            "guide_acquisition_dwell",
        ):
            if hasattr(self, name):
                control[name] = getattr(self, name)
        return state, control


class HookWorld(BenchmarkMixin, HookDecision):
    def hook_fixture_pose(self, position):
        s = self.spec
        return (
            position + self.rot.apply([s.fixture_x_m, s.fixture_y_m, 0]),
            self.rot
            * Rotation.from_euler(
                "xyz",
                [s.fixture_roll_deg, s.fixture_pitch_deg, s.yaw_deg],
                degrees=True,
            ),
        )

    def make_tool_mesh(self):
        mesh = super().make_tool_mesh()
        # Preserve the grip section; change only the working end below the grip.
        low = mesh.vertices[:, 2] < -0.025
        mesh.vertices[low, 0] *= self.spec.scale
        return mesh

    def make_slider_mesh(self):
        mesh = super().make_slider_mesh()
        mesh.vertices[:, 2] *= self.spec.bar_scale
        return mesh

    def command_at(self, t, branch="pull", anchor_command=None):
        tau = self.clock(t)
        if tau >= self.spec.branch_time:
            self.trajectory_time += self.spec.dt * (self.spec.speed_scale - 1)
        return super().command_at(tau, branch, anchor_command)


class PegWorld(BenchmarkMixin, KeyStageWorld):
    def make_tool_mesh(self):
        if self.spec.profile_shape == "legacy":
            return super().make_tool_mesh()
        return peg_mesh(self.spec.profile_shape, self.spec.scale)

    def build_environment(self):
        if self.spec.profile_shape == "legacy":
            return super().build_environment()
        s, p = self.spec, self.p
        prepared = 0.019 if s.family == "turning" else 0
        self.socket_origin = self.origin + self.rot.apply(
            [s.lateral_m + s.fixture_x_m, s.fixture_y_m, -0.073 + prepared]
        )
        self.socket_rot = self.rot * Rotation.from_euler(
            "xyz", [s.fixture_roll_deg, s.fixture_pitch_deg, s.yaw_deg], degrees=True
        )
        pocket = socket_mesh(s.profile_shape, s.scale, s.clearance_m, s.chamfer_m)
        base = trimesh.creation.box([0.060, 0.060, 0.006])

        def shape(mesh):
            return p.create_tri_mesh_shape(
                np.asarray(mesh.vertices, np.float32).ravel(),
                np.asarray(mesh.faces, np.int32).ravel(),
            )

        params = p.ArticulatedActorParams(name="benchmark_socket")
        params.world_from_root = p.TransformRT(
            translation=self.socket_origin,
            rotation=p.Quaternion.from_rotation_vector(self.socket_rot.as_rotvec()),
        )
        params.joints = [
            p.ArticulatedJointParams(name="base", type=p.ArticulatedJointType.HARD),
            p.ArticulatedJointParams(
                name="rotor",
                type=p.ArticulatedJointType.REVOLUTE
                if s.family in ("turning", "composite")
                else p.ArticulatedJointType.HARD,
                axis=[0, 0, 1],
                parent_link_from_joint=p.TransformRT(translation=[0, 0, 0.025]),
            ),
        ]
        if s.rotor_friction_model == "joint":
            resistance = p.ArticulatedJointFrictionParams()
            resistance.viscous = 0.0005
            if s.load_mode == "friction":
                resistance.coulomb = s.friction_torque_nm
                resistance.falloff_vel = 0.02
            params.joints[1].friction = resistance
        params.links = [
            p.ArticulatedLinkParams(
                name="base",
                parent_link=-1,
                parent_joint_from_link=p.TransformRT(translation=[0, 0, -0.025]),
                shape=shape(base),
                collider_type=p.ColliderType.MESH,
                density=1000,
            ),
            p.ArticulatedLinkParams(
                name="rotor",
                parent_link=0,
                shape=shape(pocket),
                collider_type=p.ColliderType.MESH,
                density=0.06 / pocket.volume,
                contact=self.contact_params(self.environment_friction_coefficient()),
            ),
        ]
        self.fixture = self.scene.create_articulated_actor(params)
        self.base, self.rotor = [
            self.scene.get_actor(h) for h in self.fixture.get_nested_link_actors()
        ]
        self.env = [self.rotor, self.base]
        self.environment_meshes = [pocket, base]
        self.slider = None
        self.table_top = self.socket_origin[2]
        if (
            np.linalg.norm(
                np.asarray(self.rotor.get_root_transform().translation)
                - self.socket_origin
            )
            > 1e-6
        ):
            raise RuntimeError(
                "socket rotor root does not coincide with authored mouth"
            )

    def __init__(self, spec):
        super().__init__(spec)
        self.seat_dwell = 0.0
        self.turn_start = -1.0
        self.branch_value = None
        self.search_start = -1.0
        self.search_z = 0.0
        self.search_origin_z = 0.0
        self.search_yaw = 0.0
        self.turn_anchor_yaw = 0.0

    def apply_environment_force(self):
        if self.spec.family not in ("turning", "composite"):
            return
        s = self.spec
        q = self.rotor_angle()
        speed = float(
            np.asarray(self.rotor.get_angular_velocity())
            @ self.socket_rot.apply([0, 0, 1])
        )
        torque = -0.0005 * speed if s.rotor_friction_model == "external" else 0.0
        if s.load_mode == "spring":
            torque -= s.torsion_nm_rad * q
        elif s.load_mode == "friction":
            if s.rotor_friction_model == "external":
                torque -= s.friction_torque_nm * np.tanh(speed / 0.02)
        else:
            # Passive periodic potential with wells at 0 and 90 degrees.
            torque -= s.detent_torque_nm * np.sin(4 * q)
        self.fixture.set_external_forces_on_dofs(np.array([0], np.int32), [torque])

    def command_at(self, t, branch="nominal", anchor_command=None):
        tau = self.clock(t)
        if self.spec.profile_shape == "legacy":
            return super().command_at(tau, branch, anchor_command)
        s = self.spec

        def v(z=0, yaw=0):
            return np.array([0, 0, z, 0, 0, np.deg2rad(yaw)], float)

        if s.family == "turning":
            angle = 0 if s.case == "hold_control" else s.turn_command_deg
            knots = [
                (0, v(), "prepared_grasp"),
                (4, v(), "turn"),
                (16, v(yaw=angle), "turned_hold"),
                (18, v(yaw=angle), "unturn"),
                (28, v(), "done"),
                (30, v(), "done"),
            ]
            if s.turning_return_overtravel_deg and s.case != "hold_control":
                over = -s.turning_return_overtravel_deg
                knots = [
                    (0, v(), "prepared_grasp"),
                    (4, v(), "turn"),
                    (16, v(yaw=angle), "turned_hold"),
                    (18, v(yaw=angle), "unturn"),
                    (28, v(yaw=over), "backlash_hold"),
                    (30, v(yaw=over), "release_backlash"),
                    (34, v(), "done"),
                    (36, v(), "done"),
                ]
            if not s.return_after_turn:
                knots = [
                    (0, v(), "prepared_grasp"),
                    (4, v(), "turn"),
                    (16, v(yaw=angle), "turned_hold"),
                    (18, v(yaw=angle), "done"),
                    (20, v(yaw=angle), "done"),
                ]
        else:
            knots = [
                (0, v(), "prepared_grasp"),
                (4, v(), "insert"),
                (10, v(-0.019), "seated_hold"),
                (12, v(-0.019), "withdraw"),
                (18, v(), "done"),
                (20, v(), "done"),
            ]
            if s.family == "composite":
                knots = [
                    (0, v(), "prepared_grasp"),
                    (4, v(), "insert"),
                    (10, v(-0.019), "verify_seat"),
                    (48, v(-0.019), "verify_seat"),
                ]
                if self.turn_start >= 0:
                    a = self.turn_start
                    knots = [
                        (a, v(-0.019, self.turn_anchor_yaw), "turn"),
                        (a + 12, v(-0.019, s.turn_command_deg), "turned_hold"),
                        (a + 14, v(-0.019, s.turn_command_deg), "unturn"),
                        (a + 24, v(-0.019), "returned_hold"),
                        (a + 26, v(-0.019), "withdraw"),
                        (a + 32, v(), "done"),
                    ]
                    if not s.return_after_turn:
                        knots = [
                            (a, v(-0.019, self.turn_anchor_yaw), "turn"),
                            (a + 12, v(-0.019, s.turn_command_deg), "turned_hold"),
                            (a + 14, v(-0.019, s.turn_command_deg), "done"),
                            (a + 16, v(-0.019, s.turn_command_deg), "done"),
                        ]
                elif tau >= 14 and self.search_start < 0:
                    self.controller_abort = "seat_handoff_timeout"
        value, phase = interpolate(knots, tau)
        actual = self.rot.inv().apply(
            np.asarray(self.tool.get_root_transform().translation) - self.origin
        )
        if (
            s.family != "turning"
            and 4 <= tau < 12
            and self.last_force > 3
            and not self.force_guard
        ):
            if (
                s.family == "composite"
                and s.seat_alignment_search
                and self.search_start < 0
            ):
                self.search_start = tau
                self.search_origin_z = float(actual[2])
                self.search_z = float(actual[2])
            elif self.search_start < 0:
                self.force_guard = True
                self.guard_z = actual[2] + 0.0005
        if self.search_start >= 0 and self.turn_start < 0:
            elapsed = tau - self.search_start
            if elapsed < 2:
                self.search_z = self.search_origin_z + 0.002 * smooth(elapsed / 2)
                phase = "retry_unload"
            else:
                # A single bounded, force-limited reseating attempt. No socket
                # GT angle feedback: yaw explores +/-8 degrees while normal
                # force limits the descent. The original hard wrench stop stays.
                rate = np.clip(0.001 * (1.0 - self.last_force), -0.001, 0.001)
                self.search_z = float(
                    np.clip(
                        self.search_z - s.dt * rate,
                        -0.019,
                        self.search_origin_z + 0.003,
                    )
                )
                self.search_yaw = (
                    8.0
                    * np.sin(2 * np.pi * (elapsed - 2) / 8)
                    * smooth((elapsed - 2) / 2)
                )
                phase = (
                    "retry_alignment" if self.last_force <= 3 else "retry_force_retreat"
                )
            value = v(self.search_z, self.search_yaw)
            if elapsed >= 26:
                # Retry exhausted: lift to clear before reporting the failure.
                value = v(
                    self.search_z + 0.022 * smooth((elapsed - 26) / 6),
                    self.search_yaw * (1 - smooth((elapsed - 26) / 6)),
                )
                phase = "retry_safe_withdraw"
                if elapsed >= 32:
                    self.controller_abort = "seat_retry_exhausted"
        if self.force_guard:
            value[2] = self.guard_z * (1 - smooth((tau - 12) / 6))
            value[3:] = 0
            phase = "blocked_force_guard" if tau < 12 else "guarded_withdraw"
        if branch != "nominal" and t >= s.branch_time:
            if self.branch_value is None:
                self.branch_value = value.copy()
            a = smooth((t - s.branch_time) / 4)
            value = self.branch_value.copy()
            if branch == "retract":
                value[2] += 0.006 * a
            elif branch == "reverse":
                value[5] -= np.deg2rad(15) * a
            elif branch == "advance":
                value[2] -= 0.002 * a
            elif branch != "hold":
                raise ValueError(branch)
            phase = "branch_" + branch
        self.nominal_tool_target = value[:3].copy()
        self.trajectory_time = tau
        if t >= 1.5 and s.tool_pose_feedback:
            self.tracking_correction = np.clip(
                self.tracking_correction
                + s.dt * np.clip(4 * (value[:3] - actual), -0.002, 0.002),
                -0.012,
                0.012,
            )
            value[:3] += self.tracking_correction
            actual_r = self.rot.inv() * Rotation.from_rotvec(
                np.asarray(self.tool.get_root_transform().rotation.to_rotation_vector())
            )
            goal = Rotation.from_rotvec(value[3:])
            error = (goal * actual_r.inv()).as_rotvec()
            self.rotation_correction = np.clip(
                self.rotation_correction
                + s.dt * np.clip(4 * error, -np.deg2rad(1), np.deg2rad(1)),
                -np.deg2rad(12),
                np.deg2rad(12),
            )
            value[3:] = (
                Rotation.from_rotvec(self.rotation_correction) * goal
            ).as_rotvec()
        if t >= 1.5:
            # All insertion orientations also rotate about the tool root, not wrist.
            offset = self.rot.inv().apply(self.origin - self.ee0)
            value[:3] += offset - Rotation.from_rotvec(value[3:]).apply(offset)
        return value[:3], phase, value[3:]

    def step(self, t, branch="nominal", anchor_command=None):
        if self.spec.profile_shape == "legacy":
            return super().step(t, branch, anchor_command)
        row = HookDecision.step(self, t, branch, anchor_command)
        root = self.tool.get_root_transform()
        rot = Rotation.from_rotvec(np.asarray(root.rotation.to_rotation_vector()))
        rotor = self.rotor.get_root_transform()
        rr = Rotation.from_rotvec(np.asarray(rotor.rotation.to_rotation_vector()))
        vertices = rr.inv().apply(
            rot.apply(self.blade_vertices)
            + np.asarray(root.translation)
            - np.asarray(rotor.translation)
        )
        contained = bool(
            contains_profile(
                profile(self.spec.profile_shape, self.spec.scale),
                vertices[:, :2],
                self.spec.clearance_m + 0.0001,
            ).all()
        )
        depth = -float(vertices[:, 2].min())
        seated = bool(
            contained
            and vertices[:, 2].max() <= -self.spec.chamfer_m
            and depth <= 0.0161
            and depth >= 0.014
        )
        self.seat_dwell = self.seat_dwell + self.spec.dt if seated else 0.0
        tau = self.clock(t)
        if (
            self.spec.family == "composite"
            and self.seat_dwell >= 0.3
            and tau >= 10
            and self.turn_start < 0
            and not self.force_guard
        ):
            self.turn_start = tau
            self.turn_anchor_yaw = self.search_yaw if self.search_start >= 0 else 0.0
        if self.spec.family == "turning" and 3.5 <= t < 4:
            self.prepared_seat_verified |= seated
        row.update(
            seat_valid=seated,
            insertion_depth_m=depth,
            blade_center_socket_m=vertices.mean(0),
            force_guard=self.force_guard,
            prepared_seat_verified=self.prepared_seat_verified,
            scripted_rotation_correction_rad=self.rotation_correction.copy(),
            measured_seat_dwell_s=self.seat_dwell,
            composite_turn_start_s=self.turn_start,
        )
        row.update(
            seat_retry_active=self.search_start >= 0, seat_retry_yaw_deg=self.search_yaw
        )
        row["task_progress"] = (
            row["rotor_angle_rad"] if self.spec.family == "turning" else depth
        )
        row["rewards"] = row["task_progress"]
        self.last_force = float(np.linalg.norm(row["extrinsic_contact_wrench"][:3]))
        return row


class SurfaceWorld(BenchmarkMixin, HookDecision):
    def prepared_frame(self, centers, default_rot, default_origin):
        rot, origin = super().prepared_frame(centers, default_rot, default_origin)
        # Task X is the stroke direction. A quarter turn places it along the
        # physical jaw closing axis, with gel/housing backing the stroke load.
        return rot * Rotation.from_euler(
            "z", self.spec.surface_heading_deg, degrees=True
        ), origin

    def make_tool_mesh(self):
        s = self.spec
        self.asset_record = None
        if "wall" in s.variant or "slot" in s.variant or "concave" in s.variant:
            mesh = peg_mesh("pen")
            mesh.apply_transform(
                trimesh.transformations.rotation_matrix(
                    np.deg2rad(-s.surface_heading_deg), [0, 0, 1]
                )
            )
            return mesh
        family = (
            "peeler"
            if "peel" in s.variant
            else "scraper"
            if "scrape" in s.variant
            else "cylinder_pen"
        )
        manifest = json.loads(
            (HERE.parents[1] / "assets/scfields/manifest.json").read_text()
        )
        choices = [r for r in manifest["tools"] if r["family"] == family]
        self.asset_record = choices[len(choices) // 2]
        mesh = trimesh.load(
            self.asset_record["canonical_path"], force="mesh", process=False
        )
        # Change the STROKE direction, not the established physical grasp.
        # prepared_frame rotates task axes; counter-rotate the authored tool so
        # its broad blade/handle never gets turned sideways into the gel pads.
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(
                np.deg2rad(-s.surface_heading_deg), [0, 0, 1]
            )
        )
        if not mesh.is_watertight:
            raise ValueError("SCFields collision asset is not closed")
        # Broad SCFields faces otherwise undersample edge/surface contact and
        # can geometrically overlap while the reported contact-point depth is small.
        return self.contact_mesh(mesh)

    def contact_params(self):
        p = self.p.ContactParams()
        p.coulomb_friction_coefficient = self.environment_friction_coefficient()
        p.penalty_coefficient = 5e10
        p.penalty_smoothing_half_distance = 0.00005
        p.penalty_threshold_default = 0.00002
        return p

    def build_environment(self):
        s = self.spec
        from benchmark_geometry import peeler_working_edge, peeling_cylinder

        self.working_end = (
            peeler_working_edge(self.mesh)
            if s.variant == "cylindrical_peel"
            else self.mesh.vertices[
                self.mesh.vertices[:, 2] <= self.mesh.bounds[0, 2] + 1e-6
            ].mean(0)
        )
        self.surface_origin = self.origin + self.rot.apply(
            [s.fixture_x_m, s.fixture_y_m, self.working_end[2] - 0.004]
        )
        self.surface_rot = self.rot * Rotation.from_euler(
            "xyz", [s.fixture_roll_deg, s.fixture_pitch_deg, s.yaw_deg], degrees=True
        )
        self.camera_target_world = self.surface_origin + self.surface_rot.apply(
            [0.01, 0, 0.01]
        )
        self.environment_meshes = [
            height_surface(
                "convex"
                if "convex" in s.variant
                else "concave"
                if "concave" in s.variant
                else "flat",
                s.radius_m,
                s.surface_half_extent_m,
                s.surface_lateral_samples,
            )
        ]
        if s.variant == "cylindrical_peel":
            self.environment_meshes = [peeling_cylinder(s.peeler_radius_m)]
        if "wall" in s.variant or "slot" in s.variant:
            curved = "curved" in s.variant or "rounded" in s.variant
            self.environment_meshes += [guide_wall(curved, 1)]
            if "slot" in s.variant:
                self.environment_meshes += [guide_wall(curved, -1)]
        self.env = []
        for i, mesh in enumerate(self.environment_meshes):
            analytic = s.surface_contact_model == "analytic_box" and (
                i == 0 or "straight" in s.variant
            )
            if analytic:
                # Same finite solid, without spurious interior top-face sample
                # vertices querying the blade's side normals. BOX uses its OBB.
                bounds = mesh.bounds
                mesh = trimesh.creation.box(bounds[1] - bounds[0])
                mesh.apply_translation(bounds.mean(0))
                self.environment_meshes[i] = mesh
            shape = self.p.create_tri_mesh_shape(
                np.asarray(mesh.vertices, np.float32).ravel(),
                np.asarray(mesh.faces, np.int32).ravel(),
            )
            collider = self.p.ColliderType.BOX if analytic else self.p.ColliderType.MESH
            options = {}
            if s.surface_contact_model == "sdf":
                collider = self.p.ColliderType.SDF
                options["sdf"] = self.p.GridSdfParams(
                    resolution_mode=self.p.GridSdfResolutionMode.EXPLICIT,
                    resolution_delta=[s.surface_sdf_voxel_m] * 3,
                    boundary_padding_dist=0.003,
                )
            actor = self.scene.create_rigid_actor(
                name=f"surface_{i}",
                shape=shape,
                mass=1,
                contact=self.contact_params(),
                collider_type=collider,
                has_gravity=False,
                **options,
                world_from_local=self.p.TransformRT(
                    translation=self.surface_origin,
                    rotation=self.p.Quaternion.from_rotation_vector(
                        self.surface_rot.as_rotvec()
                    ),
                ),
            )
            com = actor.get_center_of_mass_transform()
            actor.add_boundary_condition_dofs_world(
                np.arange(6, dtype=np.int32),
                np.r_[np.asarray(com.translation), self.surface_rot.as_rotvec()],
            )
            self.env.append(actor)
        self.slider = None
        self.table_top = self.surface_origin[2]

    def __init__(self, spec):
        super().__init__(spec)
        self.normal_offset = 0.0
        self.last_normal_force = 0.0
        self.last_local_force = np.zeros(3)
        self.path_progress = 0.0
        self.rotation_correction = np.zeros(3)
        self.branch_value = None
        self.lateral_offset = 0.0
        self.last_guide_force = 0.0
        self.grasp_translation_calibration = None
        self.surface_wait_s = 0.0
        self.guide_acquired = False
        self.guide_acquisition_dwell = 0.0

    def command_at(self, t, branch="nominal", anchor_command=None):
        s = self.spec
        tau = self.clock(t) - self.surface_wait_s * s.speed_scale
        waiting = False
        if s.guide_acquisition_gate and tau >= 7 and not self.guide_acquired:
            self.guide_acquisition_dwell = (
                self.guide_acquisition_dwell + s.dt
                if self.last_guide_force >= s.guide_acquisition_force_n
                else 0.0
            )
            if self.guide_acquisition_dwell >= 0.2:
                self.guide_acquired = True
            else:
                self.surface_wait_s += s.dt
                tau = 7.0
                waiting = True
                if self.surface_wait_s >= s.guide_acquisition_timeout_s:
                    self.controller_abort = "guide_acquisition_timeout"
        progress = smooth((tau - 7) / 7)
        length = s.surface_stroke_m or (
            0.003 if "peel" in s.variant else 0.006 if "scrape" in s.variant else 0.020
        )
        x = length * progress
        curved = "curved" in s.variant or "rounded" in s.variant
        y = (
            0.004 * (1 - np.cos(np.pi * x / 0.04))
            if curved
            else 0.002 * np.sin(2 * np.pi * progress)
            if "draw" in s.variant
            else 0.0
        )
        sign = -1 if "convex" in s.variant else 1 if "concave" in s.variant else 0
        z = sign * (s.radius_m - np.sqrt(s.radius_m**2 - x * x))
        if 4 <= tau < 15:
            # GT force is an explicitly recorded collector/safety channel, never a probe input.
            rate = 0.002 if s.revision >= 14 else 0.001
            lower = -0.004 if s.revision >= 14 else 0.0
            # Do not try to acquire a side guide while the tip is stuck to a
            # loaded floor. Unload the floor during side acquisition, then
            # restore the task's normal preload once the side contact is held.
            normal_target = 0.0 if waiting else s.surface_force_n
            self.normal_offset = np.clip(
                self.normal_offset
                + s.dt
                * np.clip(
                    s.surface_admittance_m_ns
                    * (normal_target - self.last_normal_force),
                    -rate,
                    rate,
                ),
                lower,
                0.009,
            )
        release = smooth((tau - 15) / 3)
        approach = smooth((tau - 4) / 3)
        value = np.array(
            [
                x,
                y,
                z
                - (0.004 + self.normal_offset) * approach * (1 - release)
                + 0.004 * release,
            ]
        )
        if "wall" in s.variant or "slot" in s.variant:
            if 4 <= tau < 15:
                self.lateral_offset = np.clip(
                    self.lateral_offset
                    + s.dt
                    * np.clip(
                        s.guide_admittance_m_ns * (0.5 - self.last_guide_force),
                        -s.guide_max_speed_m_s,
                        s.guide_max_speed_m_s,
                    ),
                    -0.001,
                    s.guide_max_offset_m,
                )
            value[1] += (0.001 + self.lateral_offset) * approach * (1 - release)
        if s.case == "miss":
            value[2] += 0.012 * approach
        if branch != "nominal" and t >= s.branch_time:
            if self.branch_value is None:
                self.branch_value = value.copy()
            value = self.branch_value.copy()
            a = smooth((t - s.branch_time) / 3)
            if branch == "retract":
                value[2] += 0.006 * a
            elif branch == "reverse":
                value[0] -= 0.006 * a
            elif branch == "advance":
                value[0] += 0.006 * a
            elif branch != "hold":
                raise ValueError(branch)
        rotation = np.array(
            [0, -np.arctan2(sign * x, np.sqrt(s.radius_m**2 - x * x)), 0]
        )
        if "peel" in s.variant or "scrape" in s.variant:
            rotation[1] += (
                np.deg2rad(s.surface_attack_sweep_deg) * np.sin(np.pi * progress) ** 2
            )
        if s.surface_tip_compensation:
            # Preserve the modeled lowest working-edge height while changing
            # attack angle; rotating a broad blade otherwise adds millimetres
            # of unintended plunge/lift that a slow load controller cannot track.
            value[2] += (
                self.mesh.bounds[0, 2]
                - Rotation.from_rotvec(rotation).apply(self.mesh.vertices)[:, 2].min()
            )
        if s.surface_working_tip_tracking:
            # Rotating about the tool root otherwise adds working-tip XY travel
            # to the commanded stroke on curved surfaces. Z retains the existing
            # lowest-edge/load compensation; do not apply it twice.
            value[:2] += (
                self.working_end
                - Rotation.from_rotvec(rotation).apply(self.working_end)
            )[:2]
        self.nominal_tool_target = value.copy()
        self.trajectory_time = tau
        actual = self.rot.inv().apply(
            np.asarray(self.tool.get_root_transform().translation) - self.origin
        )
        if t >= 1.5 and s.tool_pose_feedback:
            # Tangential tool tracking; normal motion is controlled by load only.
            increment = np.clip(4 * (value - actual), -0.002, 0.002)
            increment[2] = 0
            self.tracking_correction = np.clip(
                self.tracking_correction + s.dt * increment, -0.012, 0.012
            )
            value += self.tracking_correction
        offset = self.rot.inv().apply(self.origin - self.ee0)
        value += offset - Rotation.from_rotvec(rotation).apply(offset)
        if s.calibrate_precontact_grasp:
            if self.grasp_translation_calibration is None and t >= 2.5:
                # Estimate the settled rigid grasp transform ONCE, before the
                # task moves. No ongoing tool-pose servo during interaction.
                ee = np.asarray(
                    self.osc.get_current_observations_from_mochi().world_from_ee_link.translation
                )
                tool = np.asarray(self.tool.get_root_transform().translation)
                self.grasp_translation_calibration = self.rot.inv().apply(
                    ee + self.origin - tool - self.ee0
                )
                if np.linalg.norm(self.grasp_translation_calibration) > 0.012:
                    self.controller_abort = "precontact_grasp_calibration_out_of_bounds"
            if self.grasp_translation_calibration is not None:
                value += Rotation.from_rotvec(rotation).apply(
                    self.grasp_translation_calibration
                ) * smooth((t - 2.5) / 1.5)
        phase = (
            "prepared_grasp"
            if tau < 4
            else "acquire"
            if tau < 7
            else "follow"
            if tau < 15
            else "release"
        )
        if waiting:
            phase = "acquire_guide"
        return (
            value,
            "branch_" + branch if branch != "nominal" and t >= s.branch_time else phase,
            rotation,
        )

    def step(self, t, branch="nominal", anchor_command=None):
        row = super().step(t, branch, anchor_command)
        root = self.tool.get_root_transform()
        local = self.rot.inv().apply(np.asarray(root.translation) - self.origin)
        working_end = self.working_end
        tip_world = Rotation.from_rotvec(
            np.asarray(root.rotation.to_rotation_vector())
        ).apply(working_end) + np.asarray(root.translation)
        row["working_tip_task_m"] = self.rot.inv().apply(tip_world - self.origin)
        force = self.surface_rot.inv().apply(row["extrinsic_contact_wrench"][:3])
        self.last_local_force = force
        self.last_normal_force = max(0.0, force[2])
        from wrench_math import aggregate_contact_points

        contacts = list(self.tool.get_contact_points_world())
        guide_force = np.zeros(3)
        guide_forces = []
        if self.spec.variant == "cylindrical_peel":
            blade_force = np.zeros(3)
            holder_force = np.zeros(3)
            for c in contacts:
                if self.env[0].get_handle() not in (c.actor_a, c.actor_b):
                    continue
                on_a = c.actor_a == self.tool.get_handle()
                point = np.asarray(c.pos_a if on_a else c.pos_b)
                loc = (
                    Rotation.from_rotvec(np.asarray(root.rotation.to_rotation_vector()))
                    .inv()
                    .apply(point - np.asarray(root.translation))
                )
                force_on_tool = np.asarray(c.force) * (1 if on_a else -1)
                if abs(loc[1]) < 0.023 and -0.098 < loc[2] < -0.085:
                    blade_force += force_on_tool
                else:
                    holder_force += force_on_tool
            row.update(
                blade_contact_force_n=float(np.linalg.norm(blade_force)),
                holder_contact_force_n=float(np.linalg.norm(holder_force)),
            )
        if self.spec.guide_normal_control:
            floor_force = aggregate_contact_points(
                contacts,
                self.tool.get_handle(),
                self.env[0].get_handle(),
                row["tool_pose"][:3],
            ).force
            self.last_normal_force = max(
                0.0, float(self.surface_rot.inv().apply(floor_force)[2])
            )
        for actor in self.env[1:]:
            component = aggregate_contact_points(
                contacts,
                self.tool.get_handle(),
                actor.get_handle(),
                row["tool_pose"][:3],
            ).force
            guide_force += component
            guide_forces.append(component)
        if self.spec.guide_normal_control:
            # Control the positive guide's normal load, not the net force of
            # both slot walls (which can cancel). Other wall contacts still
            # count as contact events and remain in the full extrinsic wrench.
            x = float(
                self.surface_rot.inv().apply(
                    np.asarray(root.translation) - self.surface_origin
                )[0]
            )
            slope = (
                0.004 * np.pi / 0.04 * np.sin(np.pi * x / 0.04)
                if ("curved" in self.spec.variant or "rounded" in self.spec.variant)
                else 0.0
            )
            normal = np.array([slope, -1.0, 0.0])
            normal /= np.linalg.norm(normal)
            self.last_guide_force = (
                max(0.0, float(self.surface_rot.inv().apply(guide_forces[0]) @ normal))
                if guide_forces
                else 0.0
            )
        else:
            self.last_guide_force = float(np.linalg.norm(guide_force))
        handles = {actor.get_handle() for actor in self.env}
        normal_load = 0.0
        oblique_load = 0.0
        friction_excess = 0.0
        up = self.surface_rot.apply([0, 0, 1])
        mu = np.sqrt(self.spec.gel_friction * self.environment_friction_coefficient())
        for c in contacts:
            if c.actor_a not in handles and c.actor_b not in handles:
                continue
            normal = np.asarray(c.normal)
            f = np.asarray(c.force)
            fn = max(0.0, float(f @ normal))
            normal_load += fn
            if abs(normal @ up) < 0.95:
                oblique_load += fn
            friction_excess = max(
                friction_excess,
                float(np.linalg.norm(f - (f @ normal) * normal) - mu * fn),
            )
        lateral_error = float(abs(local[1] - self.nominal_tool_target[1]))
        if self.spec.guide_normal_control and guide_forces:
            # Under force control the equilibrium target is deliberately beyond
            # the wall. Measure the working edge's wall gap, not that commanded
            # spring deflection. Guide contact is still independently required.
            tip_points = self.mesh.vertices[
                self.mesh.vertices[:, 2] <= self.mesh.bounds[0, 2] + 1e-6
            ]
            tip_world = Rotation.from_rotvec(
                np.asarray(root.rotation.to_rotation_vector())
            ).apply(tip_points) + np.asarray(root.translation)
            points = self.surface_rot.inv().apply(tip_world - self.surface_origin)
            guide_y = (
                0.004 * (1 - np.cos(np.pi * points[:, 0] / 0.04))
                if ("curved" in self.spec.variant or "rounded" in self.spec.variant)
                else np.zeros(len(points))
            )
            lateral_error = float(abs(np.min(guide_y + 0.0035 - points[:, 1])))
        row.update(
            surface_normal_force_n=self.last_normal_force,
            path_lateral_error_m=lateral_error,
            guide_acquired=self.guide_acquired,
            guide_acquisition_wait_s=self.surface_wait_s,
            precontact_grasp_translation_m=np.zeros(3)
            if self.grasp_translation_calibration is None
            else self.grasp_translation_calibration.copy(),
            scripted_normal_offset_m=self.normal_offset,
            scripted_lateral_offset_m=self.lateral_offset,
            oblique_contact_normal_load_fraction=oblique_load / max(normal_load, 1e-12),
            per_contact_coulomb_excess_n=friction_excess,
            guide_contact_force_n=self.last_guide_force,
            guide_contact_event=any(np.linalg.norm(f) > 0.1 for f in guide_forces)
            if self.spec.guide_normal_control
            else self.last_guide_force > 0.1,
            task_progress=float(local[0]),
            rewards=float(local[0]),
        )
        return row


def make_world(spec):
    return (
        SurfaceWorld
        if spec.family == "surface"
        else HookWorld
        if spec.family == "hook"
        else PegWorld
    )(spec)


def audit(data, spec, abort):
    result = metrics(data, spec)
    active = ~data["initialization"]
    result["physical_valid"] &= abort is None
    if spec.family in ("insertion", "turning", "composite"):
        result.update(stage_audit(data, spec, abort))
        result.pop("max_progress_mm", None)
        if spec.family == "composite":
            turned = data["seat_valid"] & (
                abs(data["rotor_angle_rad"] - np.pi / 2) <= np.deg2rad(5)
            )
            result["task_success"] = bool(
                np.convolve(turned.astype(int), np.ones(30), "valid").max() >= 30
                and data["phase"][-1] == "done"
            )
        result.update(
            sequence_completion(
                data, spec.family, spec.dt, return_after_turn=spec.return_after_turn
            )
        )
        result["task_success"] &= result["sequence_complete"]
    elif spec.family == "surface":
        follow = data["phase"] == "follow"
        length = spec.surface_stroke_m or (
            0.003
            if "peel" in spec.variant
            else 0.006
            if "scrape" in spec.variant
            else 0.020
        )
        continuity = (
            float(data["contact_event"][follow].mean()) if follow.any() else 0.0
        )
        lateral = (
            float(np.quantile(data["path_lateral_error_m"][follow], 0.95))
            if follow.any()
            else float("inf")
        )
        result.update(
            contact_continuity=continuity,
            lateral_error_p95_m=lateral,
            task_success=bool(
                continuity >= 0.8
                and data["task_progress"].max() >= 0.8 * length
                and lateral < 0.003
            ),
        )
        if "working_tip_task_m" in data:
            result.update(
                surface_stroke_audit(
                    data["phase"], data["working_tip_task_m"], spec.dt, length
                )
            )
            result["task_success"] &= result["follow_stroke_pass"]
        if "wall" in spec.variant or "slot" in spec.variant:
            guide_fraction = (
                float(data["guide_contact_event"][follow].mean())
                if follow.any() and "guide_contact_event" in data
                else 0.0
            )
            result["guide_contact_continuity"] = guide_fraction
            result["task_success"] &= guide_fraction >= 0.8
        if spec.variant == "cylindrical_peel":
            from benchmark_geometry import peeling_contact_audit

            result.update(peeling_contact_audit(data))
            result["task_success"] &= result["blade_contact_pass"]
        if "working_tip_task_m" in data:
            from benchmark_geometry import surface_task_pass

            result["task_success"] = surface_task_pass(result, spec.variant)
    result["tactile_valid"] &= result["physical_valid"]
    result["imitation_eligible"] = bool(
        result["physical_valid"] and result["task_success"]
    )
    result.update(
        abort_reason=abort,
        full_rollout_complete=bool(
            len(data["timestamps"]) == round(spec.duration / spec.dt)
        ),
        peak_force_n=float(
            np.linalg.norm(data["extrinsic_contact_wrench"][active, :3], axis=1).max()
        ),
        peak_torque_nm=float(
            np.linalg.norm(data["extrinsic_contact_wrench"][active, 3:], axis=1).max()
        ),
    )
    from benchmark_geometry import episode_eligibility

    result["benchmark_variant_retired"] = (
        spec.family == "surface" and spec.variant == "convex_scrape"
    )
    result.update(episode_eligibility(result))
    result["tactile_precision_note"] = (
        "Diagnostic only: legacy tactile_valid reports wrench agreement, never episode acceptance."
    )
    return result


def worker(payload, output, render=False):
    import superdex.physics as p

    p.initialize(num_worker_threads=0)
    spec = BenchmarkSpec(**payload)
    world = None
    recorder = None
    start = time.perf_counter()
    path = (
        Path(output) / f"{spec.episode_id:04d}_{spec.family}_{spec.variant}_{spec.case}"
    )
    try:
        world = make_world(spec)
        world.initial_hash = hashlib.sha256(world.checkpoint()[0]).hexdigest()
        if render:
            from visuals import Recorder

            recorder = Recorder(world, path)
        rows = []
        abort = None
        for i in range(round(spec.duration / spec.dt)):
            row = world.step(
                i * spec.dt,
                spec.branch
                if spec.family != "hook"
                else ("pull" if spec.branch == "nominal" else spec.branch),
            )
            rows.append(row)
            if i % 1000 == 0:
                print(
                    f"episode {spec.episode_id}: t={i * spec.dt:.1f}s phase={row['phase']}",
                    flush=True,
                )
            if recorder and i % 5 == 0:
                recorder.frame(row)
            if i * spec.dt >= 1.5:
                abort = stop_reason(row)
            if time.perf_counter() - start > spec.max_wall_s:
                abort = "runtime_budget_exceeded"
            if (
                spec.family == "turning"
                and i * spec.dt >= 4
                and not row["prepared_seat_verified"]
            ):
                abort = "prepared_seat_not_verified"
            if abort:
                break
        data = stack(rows)
        result = audit(data, spec, abort)
        write_episode(path.with_suffix(".h5"), data, spec, world, result)
        with h5py.File(path.with_suffix(".h5"), "a") as f:
            f.attrs.update(
                schema_version="vt_acwm_superdex_benchmark_v1",
                task_family=spec.family,
                task_variant=spec.variant,
                task_stage=spec.stage,
                task_kind=spec.family,
                task_progress_units="rad" if spec.family == "turning" else "m",
                collector_pose_feedback=spec.tool_pose_feedback,
                collector_force_feedback=spec.family == "surface",
                randomization_json=json.dumps(
                    {
                        k: getattr(spec, k)
                        for k in (
                            "geometry_seed",
                            "physical_seed",
                            "camera_seed",
                            "tactile_seed",
                        )
                    }
                ),
            )
            snapshot = Path(output) / "source_snapshot/benchmark.py"
            f.attrs["benchmark_implementation_hash"] = hashlib.sha256(
                (snapshot if snapshot.exists() else Path(__file__)).read_bytes()
            ).hexdigest()
            from soft_gripper import GELSIGHT_ROOT, LEGACY_GEL_SHAPE

            gel_path = (
                LEGACY_GEL_SHAPE
                if spec.gel_geometry == "legacy_box"
                else GELSIGHT_ROOT / "generated" / f"gel_{spec.gel_geometry}.mochi.json"
            )
            f.attrs["gel_geometry"] = spec.gel_geometry
            if spec.family == "surface":
                f.attrs["working_end_tool_m"] = world.working_end
            from benchmark_geometry import gel_benchmark_role

            f.attrs["gel_role"] = gel_benchmark_role(spec.gel_geometry)
            f.attrs["wrench_matching_is_gate"] = False
            f.attrs["gel_mesh_sha256"] = hashlib.sha256(
                gel_path.read_bytes()
            ).hexdigest()
            f.attrs["tactile_representation"] = (
                "legacy nodal grid"
                if spec.gel_geometry == "legacy_box"
                else "2mm XY marker grid; nearest-marker force bins; use dense_nodal wrench for unbinned moments"
            )
            f.attrs["marker_pitch_xy_m"] = (
                [0.02075 / 6, 0.02525 / 8]
                if spec.gel_geometry == "legacy_box"
                else [0.002, 0.002]
            )
            f.attrs["arm_controller"] = (
                "Cartesian impedance: J.T @ (K pose_error - D ee_velocity); joint effort normalized to prefab limits"
            )
            f.attrs["arm_gravity_model"] = (
                "Arm/gripper gravity disabled as ideal gravity compensation; tool gravity enabled. Not a full hardware motor/friction/controller model."
            )
            f.attrs["arm_friction_model"] = (
                "ideal motor-friction compensation (zero residual arm-joint friction)"
                if spec.ideal_arm_friction_compensation
                else "uncompensated FR3 prefab motor friction"
            )
            f.attrs["finger_coupling_model"] = (
                "bilateral q_left-q_right=0 transmission, compliant penalty, common closing mode remains free"
                if spec.finger_coupling_stiffness_n_m
                else "legacy independent force-controlled fingers (not Franka mimic kinematics)"
            )
            f.attrs["arm_effort_limit_source"] = (
                "assets/test/urdf/fr3v2_1_urdf/robots/fr3v2_1_franka_hand.urdf"
                if spec.ideal_arm_friction_compensation
                else "FR3 prefab: arm effort limits unspecified (-1), normalization does not impose a finite cap"
            )
            f.attrs["collector_precontact_grasp_calibration"] = (
                spec.calibrate_precontact_grasp
            )
            f.attrs["arm_impedance_gains_json"] = json.dumps(
                dict(
                    translation_n_m=spec.arm_stiffness_n_m,
                    translation_ns_m=spec.arm_damping_ns_m,
                    rotation_nm_rad=spec.arm_stiffness_nm_rad,
                    rotation_nms_rad=spec.arm_damping_nms_rad,
                )
            )
            f.attrs["tactile_cell_moment_semantics"] = (
                "Nm on gel in sensor axes about CURRENT marker coordinate; add to r cross force; not an optical measurement"
            )
            f.attrs["tactile_unmapped_wrench_semantics"] = (
                "Non-mapped FEM contacts: N,Nm on gel in sensor axes about housing root; excluded from compact corrected wrench"
            )
            f.attrs["relative_rotation_semantics"] = (
                "angular_slip_deg is tool/EE relative rotation, NOT a direct interfacial slip detector"
            )
            f.attrs["oracle_channels"] += (
                ",precontact_grasp_translation_m,seat_retry_active,seat_retry_yaw_deg,solver_convergence_status,solver_residual_norm,solver_nonlinear_iterations,tactile_cell_moment_left,tactile_cell_moment_right,tactile_unmapped_wrench_left,tactile_unmapped_wrench_right,moment_corrected_gel_wrench,moment_corrected_inferred_extrinsic_wrench"
            )
            if spec.record_dense_field:
                f.attrs["oracle_channels"] += (
                    ",gel_node_positions_world_left,gel_node_positions_world_right,gel_node_forces_world_left,gel_node_forces_world_right"
                )
            if spec.family == "surface":
                centers = [
                    np.asarray(
                        world.gripper.links[f"franka_{side}_gelsight_housing"]
                        .get_root_transform()
                        .translation
                    )
                    for side in ("left", "right")
                ]
                jaw = centers[0] - centers[1]
                jaw /= np.linalg.norm(jaw)
                f.attrs["stroke_jaw_alignment_abs_cos"] = float(
                    abs(world.rot.apply([1, 0, 0]) @ jaw)
                )
            f.attrs["effective_gel_tool_friction"] = spec.gel_friction
            f.attrs["effective_environment_tool_friction"] = (
                spec.friction
                if spec.effective_friction
                else float(np.sqrt(spec.gel_friction * spec.friction))
            )
            f.attrs["oracle_channels"] += (
                ",scripted_normal_offset_m,measured_seat_dwell_s,composite_turn_start_s"
            )
            if getattr(world, "asset_record", None):
                f.attrs["asset_provenance_json"] = json.dumps(world.asset_record)
            f.attrs["oracle_channels"] += (
                ",scripted_lateral_offset_m,guide_contact_force_n,guide_contact_event,path_lateral_error_m"
            )
            if spec.variant == "cylindrical_peel":
                f.attrs["oracle_channels"] += (
                    ",blade_contact_force_n,holder_contact_force_n"
                )
            f.attrs["oracle_channels"] += (
                ",oblique_contact_normal_load_fraction,per_contact_coulomb_excess_n"
            )
            camera = f["camera"]
            target = camera["target_world"][:]
            perturb = Rotation.from_euler(
                "zy", [spec.camera_yaw_deg, spec.camera_pitch_deg], degrees=True
            )
            camera["eyes_world"][:] = target + perturb.apply(
                camera["eyes_world"][:] - target
            )
        if recorder:
            recorder.close(data)
            recorder = None
        summary = dict(
            spec=asdict(spec),
            metrics=result,
            seconds=time.perf_counter() - start,
            file=path.with_suffix(".h5").name,
        )
    except Exception as error:
        import traceback

        summary = dict(
            spec=asdict(spec),
            error=repr(error),
            traceback=traceback.format_exc(),
            seconds=time.perf_counter() - start,
        )
    finally:
        if world:
            world.close()
        p.shutdown()
    path.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def specifications(phase):
    if phase == "screen":
        return [replace(s, block="screen") for s in specifications("canonical")[1::6]]
    if phase == "qualification":
        return [s for i, s in enumerate(specifications("canonical")) if i % 6 < 3]
    specs = []

    def add(**kw):
        i = len(specs)
        kw.setdefault("group", f"{phase}-{i:04d}")
        kw.setdefault("camera_seed", 100000 + i)
        kw.setdefault("tactile_seed", 200000 + i)
        specs.append(BenchmarkSpec(episode_id=i, block=phase, seed=i, **kw))

    if phase == "canonical":
        cells = (
            [("surface", v) for v in SURFACES]
            + [("hook", "normal"), ("hook", "mirrored")]
            + [("insertion", v) for v in PROFILES]
            + [("turning", v) for v in ("spring", "friction", "detent")]
        )
        for family, variant in cells:
            for trial in range(6):
                kwargs = dict(
                    family=family,
                    variant=variant,
                    case="nominal"
                    if trial < 3
                    else ("miss" if family == "surface" else "near_miss"),
                    group=f"{family}-{variant}-{trial}",
                    physical_seed=trial,
                    geometry_seed=trial,
                )
                if family == "insertion":
                    kwargs.update(
                        profile_shape=variant,
                        lateral_m=0
                        if trial < 3
                        else (-1 if trial == 3 else 1)
                        * (0.001 if trial < 5 else 0.003),
                    )
                if family == "turning":
                    kwargs.update(
                        load_mode=variant,
                        case="hold_control" if trial == 5 else "nominal",
                        torsion_nm_rad=(0.006, 0.02, 0.034)[trial % 3],
                    )
                if family == "hook":
                    kwargs.update(
                        mirror_fixture=variant == "mirrored",
                        offset=0
                        if trial < 3
                        else (-0.03 if variant == "mirrored" else 0.03),
                        branch="nominal"
                        if trial < 4
                        else ("left" if variant == "mirrored" else "right"),
                    )
                # Canonical trials use each task's configured nominal load.
                # Load sweeps belong to diversity/repair manifests; overriding
                # guide/peeler defaults here silently changes the task again.
                add(**kwargs)
    elif phase == "diversity":
        factors = (
            [dict(grip_force_n=x) for x in (25.0, 30.0)]
            + [dict(gel_modulus_pa=x) for x in (100000.0, 300000.0)]
            + [dict(gel_friction=x) for x in (1.0, 1.8)]
            + [dict(friction=x) for x in (0.2, 0.8)]
            + [
                dict(robot_roll_deg=-10.0, robot_pitch_deg=10.0),
                dict(robot_roll_deg=10.0, robot_pitch_deg=-10.0),
                dict(
                    scale=0.9,
                    bar_scale=1.1,
                    clearance_m=0.0003,
                    chamfer_m=0.0015,
                    radius_m=0.06,
                    fixture_x_m=-0.002,
                    fixture_y_m=0.001,
                    fixture_pitch_deg=-2.0,
                    yaw_deg=-10.0,
                    camera_yaw_deg=10.0,
                ),
                dict(
                    scale=1.1,
                    bar_scale=0.9,
                    clearance_m=0.001,
                    chamfer_m=0.004,
                    grasp_depth_m=0.002,
                    fixture_x_m=0.002,
                    fixture_y_m=-0.001,
                    fixture_roll_deg=2.0,
                    yaw_deg=10.0,
                    speed_scale=1.0,
                    camera_pitch_deg=-10.0,
                ),
            ]
        )
        for family, variant in (
            ("surface", "flat_scrape"),
            ("hook", "normal"),
            ("insertion", "d"),
            ("turning", "spring"),
        ):
            for i, kwargs in enumerate(factors):
                kwargs = dict(kwargs)
                kwargs.setdefault("friction", float(np.sqrt(0.7)))
                add(
                    family=family,
                    variant=variant,
                    profile_shape="d" if family == "insertion" else "key",
                    physical_seed=i,
                    effective_friction=True,
                    **kwargs,
                )
    elif phase == "composite":
        for i in range(8):
            add(
                family="composite",
                variant="key_sequence",
                lateral_m=(0, 0.00015, -0.00015, 0.001)[i % 4],
                torsion_nm_rad=0.006 if i < 4 else 0.02,
                max_wall_s=3600.0,
            )
    elif phase == "sensing":
        for grip in (25.0, 30.0, 35.0):
            for load in (0.006, 0.010, 0.026, 0.034):
                add(
                    family="turning",
                    variant="spring",
                    grip_force_n=grip,
                    torsion_nm_rad=load,
                    group=f"grip-{grip:g}-load-{load:g}",
                    case="resistance_probe",
                    max_wall_s=1800.0,
                )
    elif phase == "randomized":
        rng = np.random.default_rng(20260908)
        for family, variant in (
            ("surface", "flat_scrape"),
            ("hook", "normal"),
            ("insertion", "d"),
            ("turning", "spring"),
        ):
            for i in range(4):
                add(
                    family=family,
                    variant=variant,
                    profile_shape="d" if family == "insertion" else "key",
                    effective_friction=True,
                    max_wall_s=1800.0,
                    scale=float(rng.uniform(0.9, 1.1)),
                    bar_scale=float(rng.uniform(0.9, 1.1)),
                    clearance_m=float(rng.choice([0.0003, 0.0006, 0.001])),
                    chamfer_m=float(rng.choice([0.0015, 0.003, 0.004])),
                    robot_roll_deg=float(rng.uniform(-10, 10)),
                    robot_pitch_deg=float(rng.uniform(-10, 10)),
                    fixture_roll_deg=0.0
                    if family == "turning"
                    else float(rng.uniform(-2, 2)),
                    fixture_pitch_deg=0.0
                    if family == "turning"
                    else float(rng.uniform(-2, 2)),
                    fixture_x_m=0.0
                    if family == "turning"
                    else float(rng.uniform(-0.005, 0.005)),
                    fixture_y_m=0.0
                    if family == "turning"
                    else float(rng.uniform(-0.005, 0.005)),
                    yaw_deg=0.0 if family == "turning" else float(rng.uniform(-10, 10)),
                    grasp_depth_m=float(rng.uniform(-0.002, 0.002)),
                    torsion_nm_rad=float(rng.choice([0.006, 0.02, 0.034]))
                    if family == "turning"
                    else 0.02,
                    grip_force_n=float(rng.choice([25, 30, 35])),
                    gel_modulus_pa=float(rng.choice([100000, 200000, 300000])),
                    friction=float(rng.choice([0.2, 0.5, 0.8])),
                    gel_friction=float(rng.choice([1, 1.4, 1.8])),
                    speed_scale=float(
                        rng.choice(
                            [0.75, 1]
                            if family in ("hook", "turning")
                            else [0.75, 1, 1.25]
                        )
                    ),
                    camera_yaw_deg=float(rng.uniform(-10, 10)),
                    camera_pitch_deg=float(rng.uniform(-10, 10)),
                    physical_seed=300000 + i,
                    geometry_seed=400000 + i,
                )
    else:
        raise ValueError(phase)
    return specs


def composite_gate(records):
    """Nominal geometry, not a literal case-name test across older recipes."""
    evidence = {}
    for family in ("insertion", "turning"):
        valid = []
        for record in records:
            s, m = record["spec"], record["metrics"]
            nominal = (
                s["case"] in ("nominal", "aligned", "captured", "resistance_probe")
                and abs(s["lateral_m"]) <= 0.00015
                and all(
                    abs(s.get(k, 0)) < 1e-10
                    for k in (
                        "fixture_x_m",
                        "fixture_y_m",
                        "fixture_roll_deg",
                        "fixture_pitch_deg",
                        "robot_roll_deg",
                        "robot_pitch_deg",
                        "yaw_deg",
                    )
                )
            )
            if (
                s["family"] == family
                and s["profile_shape"] == "key"
                and s["load_mode"] == "spring"
                and nominal
                and m["physical_valid"]
                and m["task_success"]
                and m.get("full_rollout_complete")
                and m.get("sequence_complete")
            ):
                valid.append(s["episode_id"])
        evidence[family] = sorted(set(valid))
        if len(evidence[family]) < 3:
            raise ValueError(
                f"Composite gate: needs three full nominal {family} successes; found {len(evidence[family])}"
            )
    return evidence


def completion_from_file(path, spec):
    """Recheck the recorded endpoint, including the no-return seated hold."""
    with h5py.File(path) as f:
        d = f["observations"]
        n = max(1, round(0.3 / spec["dt"]))
        return sequence_completion(
            dict(
                phase=d["phase"].asstr()[-n:],
                rotor_angle_rad=d["rotor_angle_rad"][-n:],
                insertion_depth_m=d["insertion_depth_m"][-n:],
                seat_valid=d["seat_valid"][-n:]
                if "seat_valid" in d
                else np.zeros(n, bool),
            ),
            spec["family"],
            spec["dt"],
            return_after_turn=spec.get("return_after_turn", True),
        )


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    specs = (
        [BenchmarkSpec(**s) for s in json.loads(Path(args.specs).read_text())]
        if args.specs
        else specifications(args.recipe or args.phase)
    )
    if args.limit:
        specs = specs[: args.limit]
    gate_evidence = None
    if any(s.family == "composite" for s in specs):
        records = []
        prerequisite = (
            Path(args.prerequisite_output)
            if getattr(args, "prerequisite_output", None)
            else output
        )
        for path in prerequisite.glob("*/*.json"):
            if path.parent.name in ("media", "physics_media"):
                continue
            record = json.loads(path.read_text())
            if isinstance(record, dict) and "spec" in record and "metrics" in record:
                s = record["spec"]
                episode = path.with_suffix(".h5")
                if s["family"] in ("insertion", "turning") and episode.exists():
                    record["metrics"].update(completion_from_file(episode, s))
                records.append(record)
        gate_evidence = composite_gate(records)
        gate_evidence["prerequisite_output"] = str(prerequisite.resolve())
        gate_evidence["scope"] = (
            "Existing independent stage qualification; new composite mechanics still require their own checks."
        )
    ledger = output / "attempts.json"
    folder = output / args.phase
    if folder.exists():
        raise FileExistsError(
            "Use a new phase name with --specs; never overwrite a run"
        )
    import fcntl

    with (output / ".budget.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = json.loads(ledger.read_text()) if ledger.exists() else []
        if len(previous) + len(specs) > 256:
            raise ValueError("256 trajectory budget exceeded")
        folder.mkdir()
        specs = [
            replace(s, episode_id=len(previous) + i, block=args.phase)
            for i, s in enumerate(specs)
        ]
        previous.extend(
            dict(episode_id=s.episode_id, phase=args.phase, spec=asdict(s))
            for s in specs
        )
        temporary = ledger.with_suffix(".tmp")
        temporary.write_text(json.dumps(previous, indent=2) + "\n")
        temporary.replace(ledger)
    (folder / "specs.json").write_text(
        json.dumps([asdict(s) for s in specs], indent=2) + "\n"
    )
    if gate_evidence is not None:
        (folder / "gate_evidence.json").write_text(
            json.dumps(gate_evidence, indent=2) + "\n"
        )
    snapshot = folder / "source_snapshot"
    snapshot.mkdir()
    import shutil

    for name in (
        "benchmark.py",
        "benchmark_geometry.py",
        "benchmark_observations.py",
        "benchmark_report.py",
        "benchmark_branches.py",
        "benchmark_probes.py",
        "pilot.py",
        "decisions.py",
        "key_stages.py",
    ):
        shutil.copy2(HERE / name, snapshot / name)
    gel_code = HERE.parent / "gelsight_mini_contact_validation"
    for name in ("soft_gripper.py", "tactile_grid.py", "wrench_math.py"):
        shutil.copy2(gel_code / name, snapshot / name)
    gel_assets = HERE.parents[1] / "assets/bots/grippers/franka_gelsight_mini"
    for name in (
        "gel_tet.mochi.json",
        "gel_source_surface.mochi.json",
        "gel_source_surface.mochi.metadata.json",
    ):
        shutil.copy2(gel_assets / "generated" / name, snapshot / name)
    start = time.perf_counter()
    results = []
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        jobs = {
            pool.submit(worker, asdict(s), str(folder), args.render): s for s in specs
        }
        for future in as_completed(jobs):
            result = future.result()
            results.append(result)
            print(
                json.dumps(
                    {
                        "id": result["spec"]["episode_id"],
                        "family": result["spec"]["family"],
                        "variant": result["spec"]["variant"],
                        "metrics": result.get("metrics"),
                        "error": result.get("error"),
                    }
                ),
                flush=True,
            )
    (folder / "summary.json").write_text(
        json.dumps(
            dict(
                seconds=time.perf_counter() - start,
                workers=args.workers,
                episodes=results,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", default="screen")
    parser.add_argument("--specs")
    parser.add_argument(
        "--recipe",
        choices=(
            "screen",
            "qualification",
            "canonical",
            "diversity",
            "randomized",
            "composite",
            "sensing",
        ),
    )
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--prerequisite-output",
        help="Read-only existing stage-qualification output for a diagnostic composite rerun",
    )
    parser.add_argument(
        "--replay",
        type=Path,
        help="Render a recorded episode; verify arrays, do not count as a new physical condition",
    )
    args = parser.parse_args()
    if args.replay:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        with h5py.File(args.replay) as f:
            payload = json.loads(f.attrs["config_json"])
        expected = (
            out
            / f"{payload['episode_id']:04d}_{payload['family']}_{payload['variant']}_{payload['case']}.h5"
        )
        if expected.exists():
            raise FileExistsError(expected)
        result = worker(payload, str(out), True)
        if "error" in result:
            raise RuntimeError(result["traceback"])
        checks = {}
        with h5py.File(args.replay) as a, h5py.File(expected) as b:
            for name in (
                "tool_pose",
                "actions",
                "tactile_force_field_left",
                "tactile_force_field_right",
            ):
                x, y = a["observations"][name][:], b["observations"][name][:]
                checks[name] = dict(
                    shape_match=x.shape == y.shape,
                    max_abs_error=float(np.max(abs(x - y)))
                    if x.shape == y.shape
                    else None,
                )
        (expected.with_suffix(".replay.json")).write_text(
            json.dumps(checks, indent=2) + "\n"
        )
        if any(
            not r["shape_match"] or r["max_abs_error"] > 1e-6 for r in checks.values()
        ):
            raise RuntimeError("Rendered replay differs from recorded episode")
    else:
        run(args)
