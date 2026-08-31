#!/usr/bin/env python3
"""Run one HDF5 shard of the SCFields/GelSight tactile-fidelity campaign."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from campaign_specs import EpisodeSpec, read_specs
from scfields_assets import DEFAULT_ROOT, load_manifest


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
SCHEMA_VERSION = "vt_acwm_superdex_scfields_v2"
PHASE_NAMES = ("grasp", "settle", "approach", "interact", "retract")
GRAVITY = np.array([0.0, 0.0, -9.81], dtype=float)


def create_box_shape(physics: Any, dimensions: np.ndarray) -> Any:
    x, y, z = np.asarray(dimensions) / 2.0
    vertices = np.array(
        [[-x,-y,-z],[x,-y,-z],[x,y,-z],[-x,y,-z],
         [-x,-y,z],[x,-y,z],[x,y,z],[-x,y,z]], dtype=np.float32,
    ).ravel()
    faces = np.array(
        [[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,4,7],[0,7,3],
         [1,2,6],[1,6,5],[0,1,5],[0,5,4],[3,7,6],[3,6,2]], dtype=np.int32,
    ).ravel()
    return physics.create_tri_mesh_shape(vertices, faces)


def sample_mesh_surface(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    """Area-weighted deterministic triangle sampling without global RNG state."""
    rng = np.random.default_rng(seed)
    triangles = np.asarray(mesh.triangles, dtype=float)
    areas = np.asarray(mesh.area_faces, dtype=float)
    selected = rng.choice(len(triangles), count, p=areas / areas.sum())
    uv = rng.random((count, 2))
    reflected = uv.sum(axis=1) > 1.0
    uv[reflected] = 1.0 - uv[reflected]
    tri = triangles[selected]
    return (tri[:, 0] + uv[:, :1] * (tri[:, 1] - tri[:, 0])
            + uv[:, 1:] * (tri[:, 2] - tri[:, 0])).astype(np.float32)


def sample_box_surface(dimensions: np.ndarray, count: int, seed: int) -> np.ndarray:
    mesh = trimesh.creation.box(extents=dimensions)
    return sample_mesh_surface(mesh, count, seed)


def transform_points(transform: Any, points: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_rotvec(
        np.asarray(transform.rotation.to_rotation_vector(), dtype=float)
    )
    return (rotation.apply(points) + np.asarray(transform.translation)).astype(np.float32)


def inverse_transform_points(transform: Any, points: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_rotvec(
        np.asarray(transform.rotation.to_rotation_vector(), dtype=float)
    )
    return rotation.inv().apply(points - np.asarray(transform.translation))


def pose7(transform: Any, origin: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_rotvec(
        np.asarray(transform.rotation.to_rotation_vector(), dtype=float)
    )
    return np.concatenate(
        (np.asarray(transform.translation, dtype=float) - origin, rotation.as_quat())
    ).astype(np.float32)


def phase_for_step(step: int, steps: int) -> int:
    fraction = step / max(steps - 1, 1)
    if fraction < 0.18:
        return 0
    if fraction < 0.28:
        return 1
    if fraction < 0.48:
        return 2
    if fraction < 0.90:
        return 3
    return 4


def motion_command(spec: EpisodeSpec, progress: float) -> tuple[np.ndarray, np.ndarray]:
    """Return interaction-only XYZ translation and XYZ rotation offsets."""
    translation = np.zeros(3)
    rotation = np.zeros(3)
    if progress < 0.48 or progress >= 0.90:
        return translation, rotation
    q = np.clip((progress - 0.48) / 0.42, 0.0, 1.0)
    # Every task command begins and ends at zero with zero endpoint velocity.
    # The former -amplitude -> +amplitude ramp introduced an 18 mm and up to
    # 45 degree discontinuity at contact onset, which could eject the tool.
    pulse = np.sin(np.pi * q) ** 2
    wave = np.sin(2.0 * np.pi * q) * pulse
    protocol = spec.protocol
    if protocol in ("drag_x", "rolling", "tip_stroke", "scrape", "peel", "corner_push"):
        translation[0] = 0.008 * pulse
    if protocol == "drag_y":
        translation[1] = 0.008 * pulse
    if protocol == "torsion":
        rotation[2] = np.deg2rad(10.0) * pulse
    if protocol == "rocking":
        rotation[:2] = np.deg2rad([7.0 * wave, 5.0 * pulse])
    if protocol == "dynamic_tap":
        translation[2] = 0.003 * (np.sin(3.0 * np.pi * q) ** 2) * pulse
    if protocol.startswith("composite"):
        translation[0] = 0.006 * pulse
        translation[1] = 0.003 * wave
        rotation[2] = np.deg2rad(7.0) * pulse
        if spec.campaign_block == "orientation":
            translation *= 0.25
            rotation *= 0.25
    if protocol == "rolling":
        translation[0] = 0.002 * pulse
        rotation[1] = -translation[0] / 0.025
    if protocol == "tip_stroke":
        rotation[1] = np.deg2rad(10.0) * pulse
    if protocol == "scrape":
        translation[0] = 0.006 * pulse
        rotation[1] = np.deg2rad(min(spec.attack_angle_deg, 15.0)) * pulse
    if protocol == "peel":
        translation[0] = 0.003 * pulse
        rotation[1] = np.deg2rad(min(spec.attack_angle_deg, 8.0)) * pulse
    if protocol == "corner_push":
        rotation[0] = np.deg2rad(8.0) * wave
        rotation[1] = np.deg2rad(10.0) * pulse
    if protocol == "lever":
        rotation[1] = np.deg2rad(12.0) * wave
        translation[0] = 0.005 * pulse
    return translation, rotation


class Environment:
    pass


def make_environment(
    scene: Any,
    context: Any,
    spec: EpisodeSpec,
    record: dict[str, Any],
    local_index: int,
    cohort_size: int,
    material: Any,
    table_shape: Any,
    table_dimensions: np.ndarray,
    query_handles: list[Any],
    physics: Any,
    robotics: Any,
) -> Environment:
    from soft_gripper import build_soft_gripper, create_osc

    env = Environment()
    columns = int(np.ceil(np.sqrt(cohort_size)))
    row, column = divmod(local_index, columns)
    env.root_offset = np.array([0.75 * column, 0.75 * row, 0.0], dtype=float)
    suffix = f"_{spec.episode_id:06d}"
    env.spec, env.record = spec, record
    env.gripper = build_soft_gripper(
        scene, material, name_suffix=suffix,
        world_from_root=physics.TransformRT(translation=env.root_offset),
    )
    env.osc = create_osc(context, env.gripper, suffix)
    params = env.osc.get_params()
    params.kp_p, params.kd_p = 1100.0, 85.0
    params.kp_r, params.kd_r = 40.0, 4.0
    params.max_translation_error, params.max_rotation_error = 0.05, 0.5
    params.b_apply_max_osc_torque_normalization = True
    env.osc.set_params(params)
    observation = env.osc.get_current_observations_from_mochi()
    initial_ee = observation.world_from_ee_link
    env.world_from_root = observation.world_from_root
    env.initial_ee_position = np.asarray(initial_ee.translation, dtype=float)
    env.initial_ee_rotation = Rotation.from_rotvec(
        np.asarray(initial_ee.rotation.to_rotation_vector(), dtype=float)
    )

    gel_centers = {}
    for side in ("left", "right"):
        rest = transform_points(observation.world_from_root, env.gripper.gel_rest_root[side])
        gel_centers[side] = 0.5 * (rest.min(axis=0) + rest.max(axis=0))
    opening_axis = gel_centers["left"] - gel_centers["right"]
    opening_axis /= np.linalg.norm(opening_axis)
    local_z = np.array([0.0, 0.0, 1.0])
    local_x = np.cross(opening_axis, local_z)
    local_x /= np.linalg.norm(local_x)
    base_rotation = Rotation.from_matrix(np.column_stack((local_x, opening_axis, local_z)))
    env.tool_rotation = base_rotation * Rotation.from_euler(
        "xy", [spec.tool_roll_deg, spec.tool_pitch_deg], degrees=True
    )
    env.tool_position = 0.5 * (gel_centers["left"] + gel_centers["right"])
    grasp_z_offset = (
        -0.009
        if spec.tool_name in {"peeler_7", "peeler_10"}
        and spec.campaign_block == "orientation"
        else -0.018
    )
    env.tool_position += np.array([0.0, 0.0, grasp_z_offset])
    tool_shape = physics.load_shape_from_file(record["canonical_path"])
    contact = physics.ContactParams()
    contact.coulomb_friction_coefficient = float(record["friction"]) * spec.friction_scale
    env.mass = float(record["mass_kg"]) * spec.mass_scale
    env.inertia_local = np.asarray(record["inertia_local_kg_m2"], dtype=float) * spec.mass_scale
    env.tool = scene.create_rigid_actor(
        name=f"tool{suffix}", shape=tool_shape, mass=env.mass, contact=contact,
        world_from_local=physics.TransformRT(
            rotation=physics.Quaternion.from_rotation_vector(env.tool_rotation.as_rotvec()),
            translation=env.tool_position,
        ),
    )
    # Rigid-body translational DOFs constrain the COM, while world_from_local
    # places the authored mesh frame. SCFields tools are intentionally grasp-
    # framed and therefore not COM-centred.
    center_mass_local = np.asarray(record["center_mass_local_m"], dtype=float)
    tool_com_initial = env.tool_position + env.tool_rotation.apply(center_mass_local)
    env.tool.add_boundary_condition_dofs_world(
        np.arange(6, dtype=np.int32),
        [*tool_com_initial.tolist(), *env.tool_rotation.as_rotvec().tolist()],
    )
    vertices = np.asarray(trimesh.load(record["canonical_path"], force="mesh").vertices)
    lowest = float((env.tool_rotation.apply(vertices) + env.tool_position).min(axis=0)[2])
    env.table_top_z = lowest - 0.004
    env.table_center = np.array(
        [env.tool_position[0], env.tool_position[1], env.table_top_z - table_dimensions[2] / 2]
    )
    env.table = scene.create_rigid_actor(
        name=f"tabletop{suffix}", shape=table_shape, mass=100.0,
        has_gravity=False, contact=contact,
        world_from_local=physics.TransformRT(translation=env.table_center),
    )
    env.table.add_boundary_condition_dofs_world(
        np.arange(6, dtype=np.int32), [*env.table_center.tolist(), 0.0, 0.0, 0.0]
    )
    for housing in (
        env.gripper.links["franka_left_gelsight_housing"],
        env.gripper.links["franka_right_gelsight_housing"],
    ):
        scene.enable_actor_contact_symmetric(
            housing.get_handle(), env.tool.get_handle(), False,
            physics.IncludeNestedActors.NO,
        )
    for actor in (*env.gripper.gels.values(), env.tool, env.table):
        query_handles.append(actor.register_query(physics.QueryType.CONTACT_POINTS))
        if actor in env.gripper.gels.values():
            query_handles.append(actor.register_query(physics.QueryType.NODE_POSITIONS))
            query_handles.append(actor.register_query(physics.QueryType.NODE_CONTACT_FORCES))
    env.ee_z_command = float(env.initial_ee_position[2])
    env.finger_dofs = (env.gripper.finger_dofs["left"], env.gripper.finger_dofs["right"])
    env.all_dofs = np.arange(env.gripper.actor.get_num_dofs(), dtype=np.int32)
    env.released = False
    env.filtered_env_force = 0.0
    env.previous_ee_pose = None
    env.local_cloud = sample_mesh_surface(
        trimesh.load(record.get("surface_path", record["canonical_path"]), force="mesh"),
        256, spec.seed,
    )
    return env


ARRAY_SHAPES = {
    "objectpointcloud": (256, 3),
    "env_point_cloud": (256, 3),
    "ee_pose": (7,), "ee_vel": (6,), "tool_pose": (7,), "tool_vel": (6,),
    "tactile_force_field_left": (7, 9, 3),
    "tactile_force_field_right": (7, 9, 3),
    "tactile_coord_left": (7, 9, 3), "tactile_coord_right": (7, 9, 3),
    "direct_gel_wrench": (6,), "field_gel_wrench": (6,),
    "extrinsic_contact_wrench": (6,), "dynamic_inferred_extrinsic_wrench": (6,),
    "direct_gel_centroid_left": (3,), "direct_gel_centroid_right": (3,),
    "field_gel_centroid_left": (3,), "field_gel_centroid_right": (3,),
    "direct_environment_cop": (3,), "inferred_environment_cop": (3,),
    "actions": (6,), "rewards": (), "timestamps": (),
}


def allocate(count: int, steps: int) -> dict[str, np.ndarray]:
    result = {
        key: np.full((count, steps, *shape), np.nan if "cop" in key or "centroid" in key else 0.0,
                     dtype=np.float32)
        for key, shape in ARRAY_SHAPES.items()
    }
    for key in ("environment_contact_count", "left_contact_count", "right_contact_count"):
        result[key] = np.zeros((count, steps), dtype=np.int16)
    result["contact_phase"] = np.zeros((count, steps), dtype=np.int8)
    result["dones"] = np.zeros((count, steps), dtype=np.bool_)
    return result


def contact_centroid(
    contacts: list[Any], receiver: Any, other: Any, sensor_transform: Any
) -> np.ndarray:
    points, weights = [], []
    for point in contacts:
        if point.actor_a == receiver and point.actor_b == other:
            force, position = np.asarray(point.force), np.asarray(point.pos_a)
        elif point.actor_b == receiver and point.actor_a == other:
            force, position = -np.asarray(point.force), np.asarray(point.pos_b)
        else:
            continue
        points.append(position)
        weights.append(np.linalg.norm(force))
    if not points or sum(weights) <= 1e-12:
        return np.full(3, np.nan)
    local = inverse_transform_points(sensor_transform, np.asarray(points))
    return np.average(local, axis=0, weights=np.asarray(weights))


def field_centroid(field: np.ndarray, positions: np.ndarray) -> np.ndarray:
    weights = np.linalg.norm(field, axis=-1).ravel()
    if weights.sum() <= 1e-12:
        return np.full(3, np.nan)
    return np.average(positions.reshape(-1, 3), axis=0, weights=weights)


def environment_cop(
    contacts: list[Any], receiver: Any, other: Any, root_offset: np.ndarray
) -> np.ndarray:
    points, weights = [], []
    for point in contacts:
        if point.actor_a == receiver and point.actor_b == other:
            force, position = np.asarray(point.force), np.asarray(point.pos_a)
        elif point.actor_b == receiver and point.actor_a == other:
            force, position = -np.asarray(point.force), np.asarray(point.pos_b)
        else:
            continue
        points.append(position - root_offset)
        weights.append(max(abs(float(force[2])), 1e-9))
    if not points:
        return np.full(3, np.nan)
    return np.average(np.asarray(points), axis=0, weights=np.asarray(weights))


def wrench_cop(wrench: np.ndarray, com: np.ndarray, plane_z: float) -> np.ndarray:
    force, torque = wrench[:3], wrench[3:]
    if abs(force[2]) < 0.05:
        return np.full(3, np.nan)
    rz = plane_z - com[2]
    return np.array([
        com[0] + (rz * force[0] - torque[1]) / force[2],
        com[1] + (torque[0] + rz * force[1]) / force[2],
        plane_z,
    ])


def add_dynamic_inference(
    data: dict[str, np.ndarray], envs: list[Environment], dt: float
) -> None:
    steps = data["tool_vel"].shape[1]
    for row, env in enumerate(envs):
        velocity = data["tool_vel"][row].astype(float)
        # Mochi reports the post-step velocity together with that step's contact
        # forces. The causal backward difference is therefore the matching
        # discrete acceleration. Centered/Savitzky-Golay derivatives smear
        # impulses into free-space frames and create false-positive contacts.
        acceleration = np.vstack(
            (np.zeros((1, 6), dtype=float), np.diff(velocity, axis=0) / dt)
        )
        for step in range(steps):
            rotation = Rotation.from_quat(data["tool_pose"][row, step, 3:]).as_matrix()
            inertia_world = rotation @ env.inertia_local @ rotation.T
            omega = velocity[step, 3:]
            inertial_force = env.mass * (acceleration[step, :3] - GRAVITY)
            inertial_torque = inertia_world @ acceleration[step, 3:] + np.cross(
                omega, inertia_world @ omega
            )
            tactile = data["field_gel_wrench"][row, step]
            inferred = np.concatenate((inertial_force, inertial_torque)) - tactile
            data["dynamic_inferred_extrinsic_wrench"][row, step] = inferred
            com = data["tool_pose"][row, step, :3]
            data["inferred_environment_cop"][row, step] = wrench_cop(
                inferred, com, env.table_top_z
            )


def collect_cohort(
    specs: list[EpisodeSpec], records: dict[str, dict[str, Any]], steps: int, dt: float,
    material: Any, grip_force_n: float = 28.0, frame_hook: Any | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    import superdex.physics as physics
    import superdex.robotics as robotics
    from wrench_math import aggregate_contact_points, integrate_surface_force_field

    data = allocate(len(specs), steps)
    table_dimensions = np.array([0.36, 0.36, 0.04])
    table_cloud = sample_box_surface(table_dimensions, 256, specs[0].seed + 11)
    setup_start = time.perf_counter()
    scene = physics.create_scene(f"SCFields cohort {specs[0].episode_id}")
    scene.set_gravity(GRAVITY)
    context = robotics.create_context()
    query_handles: list[Any] = []
    table_shape = create_box_shape(physics, table_dimensions)
    envs = [
        make_environment(
            scene, context, spec, records[spec.tool_name], row, len(specs), material,
            table_shape, table_dimensions, query_handles, physics, robotics,
        )
        for row, spec in enumerate(specs)
    ]
    if frame_hook is not None and hasattr(frame_hook, "setup"):
        frame_hook.setup(scene, envs)
    setup_seconds = time.perf_counter() - setup_start
    rollout_start = time.perf_counter()
    for step in range(steps):
        phase = phase_for_step(step, steps)
        progress = step / max(steps - 1, 1)
        for row, env in enumerate(envs):
            if phase >= 1 and not env.released:
                env.tool.clear_boundary_conditions()
                env.released = True
            grasp_q = min(1.0, progress / 0.18)
            finger_force = -grip_force_n * grasp_q * grasp_q * (3.0 - 2.0 * grasp_q)
            delta_z = 0.0
            if phase == 2:
                delta_z = -0.00065 if env.spec.campaign_block == "orientation" else -0.00035
                env.ee_z_command += delta_z
            elif phase == 3:
                target_force_n = (
                    min(env.spec.target_force_n, 5.0)
                    if env.spec.campaign_block == "orientation"
                    else env.spec.target_force_n
                )
                error = env.filtered_env_force - target_force_n
                delta_z = float(np.clip(0.00010 * error, -0.00025, 0.00025))
                env.ee_z_command += delta_z
            elif phase == 4:
                delta_z = 0.0004
                env.ee_z_command += delta_z
            interaction_translation, interaction_rotation = motion_command(env.spec, progress)
            base_rotation = np.deg2rad(
                [env.spec.gripper_roll_deg, env.spec.gripper_pitch_deg, 0.0]
            )
            blend = min(1.0, max(0.0, (progress - 0.18) / 0.30))
            target_rotation = Rotation.from_rotvec(
                base_rotation * blend + interaction_rotation
            ) * env.initial_ee_rotation
            world_target = physics.TransformRT(
                translation=env.initial_ee_position
                + np.array([interaction_translation[0], interaction_translation[1],
                            env.ee_z_command - env.initial_ee_position[2] + interaction_translation[2]]),
                rotation=physics.Quaternion.from_rotation_vector(target_rotation.as_rotvec()),
            )
            effort = np.asarray(
                env.osc.compute_output(
                    env.osc.get_current_observations_from_mochi(),
                    robotics.ControllerBasicOscPdTarget(
                        root_from_target_ee=env.world_from_root.inverse() * world_target
                    ),
                ), dtype=np.float32,
            ).copy()
            effort[env.finger_dofs[0]] += finger_force
            effort[env.finger_dofs[1]] += finger_force
            env.gripper.actor.set_external_forces_on_dofs(env.all_dofs, effort)
            data["actions"][row, step, :3] = [*interaction_translation[:2], delta_z + interaction_translation[2]]
            data["actions"][row, step, 3:] = interaction_rotation
            data["contact_phase"][row, step] = phase
        scene.step(dt)

        now = float(scene.get_total_simulation_time())
        for row, env in enumerate(envs):
            obs = env.osc.get_current_observations_from_mochi()
            ee_pose = pose7(obs.world_from_ee_link, env.root_offset)
            data["ee_pose"][row, step] = ee_pose
            if env.previous_ee_pose is not None:
                data["ee_vel"][row, step, :3] = (ee_pose[:3] - env.previous_ee_pose[:3]) / dt
                data["ee_vel"][row, step, 3:] = (
                    Rotation.from_quat(ee_pose[3:])
                    * Rotation.from_quat(env.previous_ee_pose[3:]).inv()
                ).as_rotvec() / dt
            env.previous_ee_pose = ee_pose.copy()
            tool_transform = env.tool.get_center_of_mass_transform()
            tool_com_world = np.asarray(tool_transform.translation, dtype=float)
            tool_com = tool_com_world - env.root_offset
            data["tool_pose"][row, step] = pose7(tool_transform, env.root_offset)
            data["tool_vel"][row, step, :3] = env.tool.get_linear_velocity()
            data["tool_vel"][row, step, 3:] = env.tool.get_angular_velocity()
            data["objectpointcloud"][row, step] = transform_points(
                env.tool.get_root_transform(), env.local_cloud
            ) - env.root_offset
            data["env_point_cloud"][row, step] = transform_points(
                env.table.get_root_transform(), table_cloud
            ) - env.root_offset
            tool_contacts = list(env.tool.get_contact_points_world())
            environment_wrench = aggregate_contact_points(
                tool_contacts, env.tool.get_handle(), env.table.get_handle(), tool_com_world
            )
            data["extrinsic_contact_wrench"][row, step] = np.r_[
                environment_wrench.force, environment_wrench.torque
            ]
            data["environment_contact_count"][row, step] = environment_wrench.count
            data["direct_environment_cop"][row, step] = environment_cop(
                tool_contacts, env.tool.get_handle(), env.table.get_handle(), env.root_offset
            )
            env.filtered_env_force = 0.85 * env.filtered_env_force + 0.15 * max(
                0.0, float(environment_wrench.force[2])
            )
            direct_force, direct_torque = np.zeros(3), np.zeros(3)
            field_force, field_torque = np.zeros(3), np.zeros(3)
            for side in ("left", "right"):
                gel = env.gripper.gels[side]
                housing = env.gripper.links[f"franka_{side}_gelsight_housing"]
                sensor_transform = housing.get_root_transform()
                field = env.gripper.get_surface_force_field(side)
                positions = env.gripper.gel_rest_sensor_surface[side]
                data[f"tactile_force_field_{side}"][row, step] = field
                data[f"tactile_coord_{side}"][row, step] = (
                    transform_points(sensor_transform, positions) - env.root_offset
                )
                field_wrench = integrate_surface_force_field(
                    field, positions, sensor_transform, tool_com_world, negate=True
                )
                field_force += field_wrench.force
                field_torque += field_wrench.torque
                direct_wrench = aggregate_contact_points(
                    tool_contacts, env.tool.get_handle(), gel.get_handle(), tool_com_world
                )
                direct_force += direct_wrench.force
                direct_torque += direct_wrench.torque
                gel_contacts = list(gel.get_contact_points_world())
                data[f"{side}_contact_count"][row, step] = direct_wrench.count
                data[f"direct_gel_centroid_{side}"][row, step] = contact_centroid(
                    gel_contacts, gel.get_handle(), env.tool.get_handle(), sensor_transform
                )
                data[f"field_gel_centroid_{side}"][row, step] = field_centroid(field, positions)
            data["direct_gel_wrench"][row, step] = np.r_[direct_force, direct_torque]
            data["field_gel_wrench"][row, step] = np.r_[field_force, field_torque]
            force_error = np.linalg.norm(environment_wrench.force)
            data["rewards"][row, step] = -abs(env.filtered_env_force - env.spec.target_force_n) - 0.01 * force_error
            data["timestamps"][row, step] = now
            data["dones"][row, step] = step == steps - 1
        if frame_hook is not None:
            frame_hook(step, scene, envs, data)
    rollout_seconds = time.perf_counter() - rollout_start
    add_dynamic_inference(data, envs, dt)
    return data, {"setup_seconds": setup_seconds, "rollout_seconds": rollout_seconds}


def _nrmse(estimate: np.ndarray, truth: np.ndarray, floor: float) -> float:
    rmse = float(np.sqrt(np.mean((estimate - truth) ** 2)))
    scale = max(float(np.percentile(np.linalg.norm(truth, axis=1), 95)), floor)
    return rmse / scale


def _correlation(estimate: np.ndarray, truth: np.ndarray) -> float:
    a, b = np.linalg.norm(estimate, axis=1), np.linalg.norm(truth, axis=1)
    if len(a) < 3 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _lag(estimate: np.ndarray, truth: np.ndarray, maximum: int = 10) -> int:
    a, b = np.linalg.norm(estimate, axis=1), np.linalg.norm(truth, axis=1)
    a, b = a - a.mean(), b - b.mean()
    if np.linalg.norm(a) < 1e-9 or np.linalg.norm(b) < 1e-9:
        return 0
    lags = range(-maximum, maximum + 1)
    scores = [
        np.dot(a[max(0, lag):len(a) + min(0, lag)],
               b[max(0, -lag):len(b) - max(0, lag)])
        for lag in lags
    ]
    return int(list(lags)[int(np.argmax(scores))])


def relative_grasp_rotation_deg(
    ee_quaternions: np.ndarray,
    tool_quaternions: np.ndarray,
    reference_indices: np.ndarray,
) -> np.ndarray:
    """Return tool/EE rotation change without the rotation-vector pi discontinuity."""
    relative = Rotation.from_quat(ee_quaternions).inv() * Rotation.from_quat(
        tool_quaternions
    )
    reference = relative[reference_indices].mean()
    return np.rad2deg((reference.inv() * relative).magnitude())


def episode_metrics(
    data: dict[str, np.ndarray], row: int, spec: EpisodeSpec, dt: float = 0.01
) -> dict[str, Any]:
    direct_gel = data["direct_gel_wrench"][row]
    field_gel = data["field_gel_wrench"][row]
    tactile_mask = np.linalg.norm(direct_gel[:, :3], axis=1) > 0.05
    env_truth = data["extrinsic_contact_wrench"][row]
    env_estimate = data["dynamic_inferred_extrinsic_wrench"][row]
    env_mask = data["environment_contact_count"][row] > 0
    result: dict[str, Any] = {
        "tactile_contact_frames": int(tactile_mask.sum()),
        "environment_contact_frames": int(env_mask.sum()),
        "peak_environment_force_n": float(np.linalg.norm(env_truth[:, :3], axis=1).max()),
        "peak_inferred_environment_force_n": float(np.linalg.norm(env_estimate[:, :3], axis=1).max()),
        "peak_surface_node_force_n": float(max(
            np.linalg.norm(data["tactile_force_field_left"][row], axis=-1).max(),
            np.linalg.norm(data["tactile_force_field_right"][row], axis=-1).max(),
        )),
    }
    if all(key in data for key in ("ee_pose", "tool_pose", "left_contact_count", "right_contact_count")):
        ee_pose = data["ee_pose"][row]
        tool_pose = data["tool_pose"][row]
        relative_position = np.asarray([
            Rotation.from_quat(ee[3:]).inv().apply(tool[:3] - ee[:3])
            for ee, tool in zip(ee_pose, tool_pose)
        ])
        # Measure task-induced slip from the settled end of the approach, not
        # the harmless compliant seating that occurs just after release.
        approach_indices = np.flatnonzero(data["contact_phase"][row] == 2)
        reference_indices = approach_indices[-min(10, len(approach_indices)):]
        reference = np.median(relative_position[reference_indices], axis=0)
        active_motion = data["contact_phase"][row] == 3
        slip = np.linalg.norm(relative_position - reference, axis=1)
        # Rotation vectors are discontinuous at pi, so a componentwise median
        # can turn an unchanged near-pi grasp into a false 180-degree flip.
        rotation_slip_deg = relative_grasp_rotation_deg(
            ee_pose[:, 3:], tool_pose[:, 3:], reference_indices
        )
        bilateral = (
            (data["left_contact_count"][row] > 0)
            & (data["right_contact_count"][row] > 0)
        )
        result["grasp_slip_max_m"] = float(np.max(slip[active_motion]))
        result["grasp_slip_final_m"] = float(slip[np.flatnonzero(active_motion)[-1]])
        result["grasp_rotation_max_deg"] = float(np.max(rotation_slip_deg[active_motion]))
        result["grasp_rotation_final_deg"] = float(
            rotation_slip_deg[np.flatnonzero(active_motion)[-1]]
        )
        result["bilateral_grasp_fraction"] = float(np.mean(bilateral[active_motion]))
        result["grasp_retained"] = bool(
            result["grasp_slip_max_m"] <= 0.012
            and result["grasp_rotation_max_deg"] <= 25.0
            and result["bilateral_grasp_fraction"] >= 0.80
        )
    else:
        result.update(
            grasp_slip_max_m=float("nan"), grasp_slip_final_m=float("nan"),
            grasp_rotation_max_deg=float("nan"), grasp_rotation_final_deg=float("nan"),
            bilateral_grasp_fraction=float("nan"), grasp_retained=False,
        )
    if tactile_mask.any():
        force_error = np.linalg.norm(field_gel[tactile_mask, :3] - direct_gel[tactile_mask, :3], axis=1)
        force_truth = np.linalg.norm(direct_gel[tactile_mask, :3], axis=1)
        torque_error = np.linalg.norm(field_gel[tactile_mask, 3:] - direct_gel[tactile_mask, 3:], axis=1)
        dots = np.sum(field_gel[tactile_mask, :3] * direct_gel[tactile_mask, :3], axis=1)
        cosine = dots / np.maximum(
            np.linalg.norm(field_gel[tactile_mask, :3], axis=1) * force_truth, 1e-12
        )
        centroid_errors = []
        for side in ("left", "right"):
            a = data[f"field_gel_centroid_{side}"][row]
            b = data[f"direct_gel_centroid_{side}"][row]
            finite = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
            centroid_errors.extend(np.linalg.norm(a[finite, :2] - b[finite, :2], axis=1))
        result.update(
            dense_force_rel_p95=float(np.percentile(force_error / np.maximum(force_truth, 0.05), 95)),
            dense_force_abs_p95_n=float(np.percentile(force_error, 95)),
            dense_torque_abs_p95_nm=float(np.percentile(torque_error, 95)),
            dense_force_cosine_p05=float(np.percentile(cosine, 5)),
            dense_centroid_p95_m=float(np.percentile(centroid_errors, 95)) if centroid_errors else float("nan"),
        )
        active_indices = np.flatnonzero(tactile_mask)
        midpoint = len(active_indices) // 2
        gain = np.linalg.norm(field_gel[tactile_mask, :3], axis=1) / np.maximum(
            np.linalg.norm(direct_gel[tactile_mask, :3], axis=1), 0.05
        )
        result["hysteresis_gain_delta"] = float(
            abs(np.median(gain[:midpoint]) - np.median(gain[midpoint:]))
        ) if midpoint else float("nan")
    else:
        result.update({key: float("nan") for key in (
            "dense_force_rel_p95", "dense_force_abs_p95_n", "dense_torque_abs_p95_nm",
            "dense_force_cosine_p05", "dense_centroid_p95_m",
        )})
        result["hysteresis_gain_delta"] = float("nan")
    if env_mask.any():
        normal = np.abs(env_truth[env_mask, 2])
        lateral = np.linalg.norm(env_truth[env_mask, :2], axis=1)
        result.update(
            environment_force_nrmse=_nrmse(env_estimate[env_mask, :3], env_truth[env_mask, :3], 0.05),
            environment_torque_nrmse=_nrmse(env_estimate[env_mask, 3:], env_truth[env_mask, 3:], 0.002),
            environment_force_correlation=_correlation(env_estimate[env_mask, :3], env_truth[env_mask, :3]),
            environment_lag_steps=_lag(env_estimate[:, :3], env_truth[:, :3]),
            cross_axis_leakage=float(np.median(lateral / np.maximum(normal, 0.05))),
        )
        released = data["contact_phase"][row] >= 1
        truth_indices = np.flatnonzero(
            (np.linalg.norm(env_truth[:, :3], axis=1) > 0.05) & released
        )
        truth_onset = int(truth_indices[0]) if len(truth_indices) else int(np.flatnonzero(env_mask)[0])
        inferred_indices = np.flatnonzero(
            (np.linalg.norm(env_estimate[:, :3], axis=1) > 0.05) & released
        )
        result["contact_onset_delta_steps"] = (
            int(inferred_indices[0]) - truth_onset if len(inferred_indices) else 10_000
        )
        result["contact_detection_threshold_n"] = float(
            np.linalg.norm(env_truth[inferred_indices[0], :3])
        ) if len(inferred_indices) else float("nan")
        lateral_force = env_truth[env_mask, :2]
        shear_jerk = np.linalg.norm(np.diff(lateral_force, axis=0), axis=1) / dt
        result["stick_slip_shear_jerk_p95_n_s"] = float(
            np.percentile(shear_jerk, 95)
        ) if len(shear_jerk) else 0.0
        if len(shear_jerk) >= 3:
            threshold = np.percentile(shear_jerk, 90)
            result["stick_slip_event_count"] = int(np.count_nonzero(
                (shear_jerk[1:-1] > shear_jerk[:-2])
                & (shear_jerk[1:-1] >= shear_jerk[2:])
                & (shear_jerk[1:-1] >= threshold)
            ))
        else:
            result["stick_slip_event_count"] = 0
        a, b = data["inferred_environment_cop"][row], data["direct_environment_cop"][row]
        finite = env_mask & np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
        cop = np.linalg.norm(a[finite, :2] - b[finite, :2], axis=1)
        result["cop_median_m"] = float(np.median(cop)) if len(cop) else float("nan")
        result["cop_p95_m"] = float(np.percentile(cop, 95)) if len(cop) else float("nan")
    else:
        for key in ("environment_force_nrmse", "environment_torque_nrmse",
                    "environment_force_correlation", "cop_median_m", "cop_p95_m",
                    "cross_axis_leakage"):
            result[key] = float("nan")
        result["environment_lag_steps"] = 0
        result["contact_onset_delta_steps"] = 10_000
        result["contact_detection_threshold_n"] = float("nan")
        result["stick_slip_shear_jerk_p95_n_s"] = float("nan")
        result["stick_slip_event_count"] = 0
    # Boundary constraints intentionally hold the tool during the grasp phase;
    # their reaction is not an environment contact and is not observable by the
    # gel field, so exclude that phase from false-positive accounting.
    free = (~env_mask) & (data["contact_phase"][row] >= 1)
    result["free_space_force_p95_n"] = float(
        np.percentile(np.linalg.norm(env_estimate[free, :3], axis=1), 95)
    ) if free.any() else float("nan")
    total_nodes = np.prod(data["tactile_force_field_left"][row].shape[:-1]) * 2
    saturated_nodes = sum(
        np.count_nonzero(np.linalg.norm(data[f"tactile_force_field_{side}"][row], axis=-1) > 20.0)
        for side in ("left", "right")
    )
    result["surface_saturation_fraction"] = float(saturated_nodes / max(total_nodes, 1))
    dense_pass = bool(
        tactile_mask.any()
        and (result["dense_force_rel_p95"] <= 0.02 or result["dense_force_abs_p95_n"] <= 0.05)
        and result["dense_torque_abs_p95_nm"] <= 0.002
        and result["dense_force_cosine_p05"] >= 0.95
        and (not np.isfinite(result["dense_centroid_p95_m"]) or result["dense_centroid_p95_m"] <= 0.0031)
    )
    environment_pass = bool(
        env_mask.any()
        and result["environment_force_nrmse"] <= 0.10
        and result["environment_torque_nrmse"] <= 0.15
        and (not np.isfinite(result["environment_force_correlation"])
             or result["environment_force_correlation"] >= 0.95)
        and (not np.isfinite(result["cop_median_m"]) or result["cop_median_m"] <= 0.002)
        and (not np.isfinite(result["cop_p95_m"]) or result["cop_p95_m"] <= 0.005)
        and abs(result["environment_lag_steps"]) <= 2
    )
    result["dense_pass"] = dense_pass
    result["environment_pass"] = environment_pass
    result["physical_valid"] = bool(tactile_mask.any() and env_mask.any())
    result["normal_protocol_leakage_pass"] = bool(
        spec.protocol != "normal_load_unload"
        or (np.isfinite(result["cross_axis_leakage"]) and result["cross_axis_leakage"] <= 0.10)
    )
    return result


def write_shard(
    path: Path, specs: list[EpisodeSpec], chunks: list[tuple[list[EpisodeSpec], dict[str, np.ndarray]]],
    args: argparse.Namespace, metrics: dict[int, dict[str, Any]], benchmark: dict[str, Any],
) -> None:
    import h5py

    compression = None if args.compression == "none" else args.compression
    text = h5py.string_dtype("utf-8")
    partial = path.with_suffix(path.suffix + ".partial")
    with h5py.File(partial, "w", libver="latest") as stream:
        stream.attrs.update({
            "schema_version": SCHEMA_VERSION, "domain": "superdex",
            "simulator": "superdex_mochi", "interaction_family": "scfields_tool_use_fidelity",
            "episode_count": len(specs), "steps_per_episode": args.steps,
            "sample_period_s": args.dt, "point_count": 256,
            "tactile_grid_shape": np.asarray([7, 9, 3], dtype=np.int32),
            "force_units": "N", "torque_units": "N m", "position_units": "m",
            "quaternion_order": "xyzw", "precision": "fp32",
            "grip_force_per_finger_n": args.grip_force_n,
            "gel_friction_coefficient": args.gel_friction,
        })
        index = stream.create_group("index")
        index.create_dataset("episode_id", data=np.asarray([s.episode_id for s in specs], dtype=np.int32))
        for key in ("tool_name", "tool_family", "campaign_block", "protocol"):
            index.create_dataset(key, data=np.asarray([getattr(s, key) for s in specs], dtype=text))
        episodes = stream.create_group("episodes", track_order=True)
        for chunk_specs, data in chunks:
            for row, spec in enumerate(chunk_specs):
                group = episodes.create_group(f"episode_{spec.episode_id:06d}")
                for key, value in asdict(spec).items():
                    group.attrs[key] = value
                group.attrs["branch_group_id"] = spec.branch_group_id
                group.attrs["initial_state_hash"] = spec.initial_state_hash
                observations = group.create_group("observations")
                for key in ARRAY_SHAPES:
                    if key in ("actions", "rewards", "timestamps"):
                        continue
                    observations.create_dataset(key, data=data[key][row], compression=compression)
                for key in ("environment_contact_count", "left_contact_count", "right_contact_count"):
                    observations.create_dataset(key, data=data[key][row], compression=compression)
                observations["point_cloud"] = observations["objectpointcloud"]
                observations["tactile_data_left"] = observations["tactile_force_field_left"]
                observations["tactile_data_right"] = observations["tactile_force_field_right"]
                observations["field_inferred_extrinsic_wrench"] = observations["dynamic_inferred_extrinsic_wrench"]
                for key in ("actions", "rewards", "timestamps", "dones", "contact_phase"):
                    group.create_dataset(key, data=data[key][row], compression=compression)
                group["episode_lengths"] = np.int32(args.steps)
                group["episode_rewards"] = np.float32(data["rewards"][row].sum())
                metric_group = group.create_group("metrics")
                for key, value in metrics[spec.episode_id].items():
                    metric_group.attrs[key] = value
                labels = group.create_group("labels")
                labels["physical_valid"] = np.bool_(metrics[spec.episode_id]["physical_valid"])
                labels["dense_fidelity_pass"] = np.bool_(metrics[spec.episode_id]["dense_pass"])
                labels["environment_fidelity_pass"] = np.bool_(metrics[spec.episode_id]["environment_pass"])
                retained = metrics[spec.episode_id]["grasp_retained"]
                labels["grasp_retained"] = np.bool_(retained)
                labels["imitation_eligible"] = np.bool_(
                    metrics[spec.episode_id]["physical_valid"] and retained
                )
                labels["task_success"] = np.bool_(
                    metrics[spec.episode_id]["physical_valid"]
                    and retained
                    and metrics[spec.episode_id]["environment_contact_frames"] > 0
                )
        group = stream.create_group("benchmark")
        for key, value in benchmark.items():
            if isinstance(value, (str, bool, int, float, np.number)):
                group.attrs[key] = value
    partial.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-file", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_ROOT / "manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--parallel-envs", type=int, default=4)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--worker-threads", type=int, default=0)
    parser.add_argument("--compression", choices=("lzf", "gzip", "none"), default="lzf")
    parser.add_argument("--youngs-modulus-pa", type=float, default=200_000.0)
    parser.add_argument("--poisson-ratio", type=float, default=0.49)
    parser.add_argument("--mass-damping", type=float, default=5.0)
    parser.add_argument("--stiffness-damping", type=float, default=0.003)
    parser.add_argument("--gel-friction", type=float, default=1.4)
    parser.add_argument(
        "--grip-force-n", type=float, default=35.0,
        help="Inward effort per Franka finger; 35 N gives the Franka Hand's 70 N total limit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.pop("SUPERDEX_PRECISION", None)
    os.environ["SUPERDEX_ASSETS_PATH"] = str(PROJECT_ROOT / "assets")
    specs = read_specs(args.spec_file)
    manifest = load_manifest(args.manifest)
    records = {record["name"]: record for record in manifest["tools"]}
    from soft_gripper import GelMaterial
    import superdex.physics as physics

    material = GelMaterial(
        youngs_modulus_pa=args.youngs_modulus_pa,
        poisson_ratio=args.poisson_ratio,
        mass_damping_s_inv=args.mass_damping,
        stiffness_damping_s=args.stiffness_damping,
        friction_coefficient=args.gel_friction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    physics.initialize(num_worker_threads=args.worker_threads)
    chunks, timings, metrics = [], [], {}
    try:
        for offset in range(0, len(specs), args.parallel_envs):
            chunk_specs = specs[offset:offset + args.parallel_envs]
            print(f"cohort {offset}:{offset + len(chunk_specs)}", flush=True)
            data, timing = collect_cohort(
                chunk_specs, records, args.steps, args.dt, material, args.grip_force_n
            )
            chunks.append((chunk_specs, data))
            timings.append(timing)
            for row, spec in enumerate(chunk_specs):
                metrics[spec.episode_id] = episode_metrics(data, row, spec, args.dt)
    finally:
        physics.shutdown()
    simulation_seconds = time.perf_counter() - start
    benchmark = {
        "episodes": len(specs), "steps": args.steps,
        "environment_frames": len(specs) * args.steps,
        "parallel_envs": args.parallel_envs, "worker_threads": args.worker_threads,
        "setup_seconds": sum(t["setup_seconds"] for t in timings),
        "rollout_seconds": sum(t["rollout_seconds"] for t in timings),
        "simulation_seconds": simulation_seconds,
        "rollout_environment_fps": len(specs) * args.steps / sum(t["rollout_seconds"] for t in timings),
        "hostname": platform.node(), "pid": os.getpid(),
    }
    serialization_start = time.perf_counter()
    write_shard(args.output, specs, chunks, args, metrics, benchmark)
    benchmark["serialization_seconds"] = time.perf_counter() - serialization_start
    benchmark["total_seconds"] = time.perf_counter() - start
    benchmark["end_to_end_environment_fps"] = len(specs) * args.steps / benchmark["total_seconds"]
    benchmark["hdf5_bytes"] = args.output.stat().st_size
    sidecar = args.output.with_suffix(".json")
    sidecar.write_text(json.dumps({"benchmark": benchmark, "metrics": metrics}, indent=2) + "\n")
    print(json.dumps(benchmark, indent=2), flush=True)


if __name__ == "__main__":
    main()
