#!/usr/bin/env python3
"""Collect parallel soft-GelSight plug/table episodes into VT-ACWM HDF5."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DEFAULT_OUTPUT = HERE / "output" / "parallel_128x200"
SCHEMA_VERSION = "vt_acwm_superdex_v1"
PHASE_NAMES = ("grasp", "settle", "approach", "press", "shear")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--parallel-envs",
        type=int,
        default=16,
        help="Episodes advanced in one physics scene (remaining episodes use more cohorts).",
    )
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--point-count", type=int, default=1024)
    parser.add_argument("--youngs-modulus-pa", type=float, default=200_000.0)
    parser.add_argument("--poisson-ratio", type=float, default=0.49)
    parser.add_argument("--mass-damping", type=float, default=5.0)
    parser.add_argument("--stiffness-damping", type=float, default=0.003)
    parser.add_argument(
        "--compression", choices=("lzf", "gzip", "none"), default="lzf"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use 2 episodes, 12 steps and 64 points while exercising the real simulator.",
    )
    args = parser.parse_args()
    if args.smoke_test:
        args.episodes, args.steps, args.parallel_envs, args.point_count = 2, 12, 2, 64
    if min(args.episodes, args.steps, args.parallel_envs, args.point_count) <= 0:
        parser.error("episode, step, parallel environment and point counts must be positive")
    if args.dt <= 0.0 or args.youngs_modulus_pa <= 0.0:
        parser.error("--dt and --youngs-modulus-pa must be positive")
    if not 0.0 < args.poisson_ratio < 0.5:
        parser.error("--poisson-ratio must be between 0 and 0.5")
    args.parallel_envs = min(args.parallel_envs, args.episodes)
    return args


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: int
    seed: int
    tool_id: int
    gripper_roll_deg: float
    gripper_pitch_deg: float
    plug_roll_deg: float
    plug_pitch_deg: float
    lateral_x_m: float
    lateral_y_m: float
    target_force_n: float

    @property
    def branch_group_id(self) -> str:
        return f"superdex-plug-table-{self.episode_id:06d}"

    @property
    def initial_state_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class Environment:
    spec: EpisodeSpec
    root_offset: Any
    gripper: Any
    osc: Any
    world_from_root: Any
    initial_ee_position: Any
    initial_ee_rotation: Any
    plug: Any
    table: Any
    plug_position: Any
    plug_rotation: Any
    table_center: Any
    table_top_z: float
    ee_z_command: float
    finger_dofs: tuple[int, int]
    all_dofs: Any
    released: bool = False
    filtered_table_force: float = 0.0
    previous_ee_pose: Any | None = None


def episode_specs(count: int, seed: int) -> list[EpisodeSpec]:
    import numpy as np

    rng = np.random.default_rng(seed)
    specs = []
    # Tool IDs 11--20 follow the AC-VTWM training split recorded in Notion.
    for episode_id in range(count):
        specs.append(
            EpisodeSpec(
                episode_id=episode_id,
                seed=int(rng.integers(0, 2**31 - 1)),
                tool_id=11 + episode_id % 10,
                gripper_roll_deg=float(rng.uniform(-15.0, 15.0)),
                gripper_pitch_deg=float(rng.uniform(-15.0, 15.0)),
                plug_roll_deg=float(rng.uniform(-20.0, 20.0)),
                plug_pitch_deg=float(rng.uniform(-20.0, 20.0)),
                # Keep the plug inside both 18 x 24 mm gel footprints. Roll and
                # pitch provide the intended contact diversity; large lateral
                # offsets can turn an episode into a one-finger miss.
                lateral_x_m=float(rng.uniform(-0.0015, 0.0015)),
                lateral_y_m=float(rng.uniform(-0.0005, 0.0005)),
                target_force_n=float(rng.uniform(2.0, 10.0)),
            )
        )
    return specs


def create_box_shape(physics: Any, np: Any, dimensions: Any) -> Any:
    x, y, z = np.asarray(dimensions) / 2.0
    vertices = np.array(
        [
            [-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
            [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z],
        ],
        dtype=np.float32,
    ).ravel()
    faces = np.array(
        [[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,4,7],[0,7,3],
         [1,2,6],[1,6,5],[0,1,5],[0,5,4],[3,7,6],[3,6,2]],
        dtype=np.int32,
    ).ravel()
    return physics.create_tri_mesh_shape(vertices, faces)


def sample_box_surface(dimensions: Any, count: int, seed: int) -> Any:
    """Deterministically sample a box surface, with samples proportional to face area."""
    import numpy as np

    rng = np.random.default_rng(seed)
    half = np.asarray(dimensions, dtype=np.float32) / 2.0
    face_areas = np.array(
        [half[1] * half[2], half[1] * half[2],
         half[0] * half[2], half[0] * half[2],
         half[0] * half[1], half[0] * half[1]], dtype=float,
    )
    face = rng.choice(6, size=count, p=face_areas / face_areas.sum())
    points = rng.uniform(-1.0, 1.0, size=(count, 3)).astype(np.float32) * half
    axes = face // 2
    signs = np.where(face % 2, 1.0, -1.0)
    points[np.arange(count), axes] = signs * half[axes]
    return points


def transform_points(transform: Any, points: Any) -> Any:
    import numpy as np
    from scipy.spatial.transform import Rotation

    rotation = Rotation.from_rotvec(
        np.asarray(transform.rotation.to_rotation_vector(), dtype=float)
    )
    return (
        rotation.apply(np.asarray(points, dtype=np.float32))
        + np.asarray(transform.translation, dtype=float)
    ).astype(np.float32)


def pose7(transform: Any, origin: Any | None = None) -> Any:
    import numpy as np
    from scipy.spatial.transform import Rotation

    translation = np.asarray(transform.translation, dtype=np.float32)
    if origin is not None:
        translation = translation - np.asarray(origin, dtype=np.float32)
    quaternion_xyzw = Rotation.from_rotvec(
        np.asarray(transform.rotation.to_rotation_vector(), dtype=float)
    ).as_quat().astype(np.float32)
    return np.concatenate((translation, quaternion_xyzw)).astype(np.float32)


def phase_for_step(step: int, total_steps: int) -> int:
    fraction = step / max(total_steps, 1)
    if fraction < 0.18:
        return 0
    if fraction < 0.28:
        return 1
    if fraction < 0.48:
        return 2
    if fraction < 0.75:
        return 3
    return 4


def allocate(count: int, steps: int, point_count: int) -> dict[str, Any]:
    import numpy as np

    shapes = {
        "objectpointcloud": (count, steps, point_count, 3),
        "env_point_cloud": (count, steps, point_count, 3),
        "ee_pose": (count, steps, 7),
        "ee_vel": (count, steps, 6),
        "tool_pose": (count, steps, 7),
        "tool_vel": (count, steps, 6),
        "tactile_force_field_left": (count, steps, 7, 9, 3),
        "tactile_force_field_right": (count, steps, 7, 9, 3),
        "tactile_coord_left": (count, steps, 7, 9, 3),
        "tactile_coord_right": (count, steps, 7, 9, 3),
        "extrinsic_contact_wrench": (count, steps, 6),
        "field_inferred_extrinsic_wrench": (count, steps, 6),
        "actions": (count, steps, 6),
        "rewards": (count, steps),
        "timestamps": (count, steps),
    }
    data = {name: np.zeros(shape, dtype=np.float32) for name, shape in shapes.items()}
    data["dones"] = np.zeros((count, steps), dtype=np.bool_)
    data["contact_phase"] = np.zeros((count, steps), dtype=np.int8)
    data["table_contact_count"] = np.zeros((count, steps), dtype=np.int16)
    data["left_contact_count"] = np.zeros((count, steps), dtype=np.int16)
    data["right_contact_count"] = np.zeros((count, steps), dtype=np.int16)
    return data


def make_environment(
    scene: Any,
    context: Any,
    spec: EpisodeSpec,
    local_index: int,
    cohort_size: int,
    material: Any,
    plug_shape: Any,
    table_shape: Any,
    plug_dimensions: Any,
    table_dimensions: Any,
    query_handles: list[Any],
    physics: Any,
    robotics: Any,
    np: Any,
    Rotation: Any,
) -> Environment:
    from soft_gripper import build_soft_gripper, create_osc

    columns = int(np.ceil(np.sqrt(cohort_size)))
    row, column = divmod(local_index, columns)
    root_offset = np.array([0.75 * column, 0.75 * row, 0.0], dtype=float)
    suffix = f"_{spec.episode_id:06d}"
    gripper = build_soft_gripper(
        scene,
        material,
        name_suffix=suffix,
        world_from_root=physics.TransformRT(translation=root_offset),
    )
    osc = create_osc(context, gripper, suffix)
    params = osc.get_params()
    params.kp_p, params.kd_p = 1100.0, 85.0
    params.kp_r, params.kd_r = 40.0, 4.0
    params.max_translation_error, params.max_rotation_error = 0.04, 0.3
    params.b_apply_max_osc_torque_normalization = True
    osc.set_params(params)
    observation = osc.get_current_observations_from_mochi()
    initial_ee = observation.world_from_ee_link
    initial_ee_position = np.asarray(initial_ee.translation, dtype=float)
    initial_ee_rotation = Rotation.from_rotvec(
        np.asarray(initial_ee.rotation.to_rotation_vector(), dtype=float)
    )

    gel_centers = {}
    for side in ("left", "right"):
        # Before the first step a tiled soft actor's AABB is still expressed in
        # articulation-root coordinates. Transform the complete rest mesh by
        # the root pose instead of reading that transient AABB.
        gel_world_rest = transform_points(
            observation.world_from_root, gripper.gel_rest_root[side]
        )
        gel_centers[side] = 0.5 * (
            gel_world_rest.min(axis=0) + gel_world_rest.max(axis=0)
        )
    opening_axis = gel_centers["left"] - gel_centers["right"]
    opening_axis /= np.linalg.norm(opening_axis)
    local_z = np.array([0.0, 0.0, 1.0])
    local_x = np.cross(opening_axis, local_z)
    local_x /= np.linalg.norm(local_x)
    base_plug_rotation = Rotation.from_matrix(
        np.column_stack((local_x, opening_axis, local_z))
    )
    plug_rotation = base_plug_rotation * Rotation.from_euler(
        "xy", [spec.plug_roll_deg, spec.plug_pitch_deg], degrees=True
    )
    plug_position = 0.5 * (gel_centers["left"] + gel_centers["right"])
    plug_position += np.array([spec.lateral_x_m, spec.lateral_y_m, -0.018])

    contact = physics.ContactParams()
    contact.coulomb_friction_coefficient = 1.2
    plug = scene.create_rigid_actor(
        name=f"plug{suffix}",
        shape=plug_shape,
        mass=0.08,
        contact=contact,
        world_from_local=physics.TransformRT(
            rotation=physics.Quaternion.from_rotation_vector(plug_rotation.as_rotvec()),
            translation=plug_position,
        ),
    )
    plug.add_boundary_condition_dofs_world(
        np.arange(6, dtype=np.int32),
        [*plug_position.tolist(), *plug_rotation.as_rotvec().tolist()],
    )
    vertical_half_extent = float(
        np.abs(plug_rotation.as_matrix()[2]) @ (plug_dimensions / 2.0)
    )
    table_top_z = float(plug_position[2] - vertical_half_extent - 0.004)
    table_center = np.array(
        [plug_position[0], plug_position[1], table_top_z - table_dimensions[2] / 2]
    )
    table = scene.create_rigid_actor(
        name=f"tabletop{suffix}",
        shape=table_shape,
        mass=100.0,
        has_gravity=False,
        contact=contact,
        world_from_local=physics.TransformRT(translation=table_center),
    )
    table.add_boundary_condition_dofs_world(
        np.arange(6, dtype=np.int32), [*table_center.tolist(), 0.0, 0.0, 0.0]
    )
    for housing in (
        gripper.links["franka_left_gelsight_housing"],
        gripper.links["franka_right_gelsight_housing"],
    ):
        scene.enable_actor_contact_symmetric(
            housing.get_handle(), plug.get_handle(), False,
            physics.IncludeNestedActors.NO,
        )
    for actor in (*gripper.gels.values(), plug, table):
        query_handles.append(actor.register_query(physics.QueryType.CONTACT_POINTS))
        if actor in gripper.gels.values():
            query_handles.append(actor.register_query(physics.QueryType.NODE_POSITIONS))
            query_handles.append(
                actor.register_query(physics.QueryType.NODE_CONTACT_FORCES)
            )
    return Environment(
        spec=spec,
        root_offset=root_offset,
        gripper=gripper,
        osc=osc,
        world_from_root=observation.world_from_root,
        initial_ee_position=initial_ee_position,
        initial_ee_rotation=initial_ee_rotation,
        plug=plug,
        table=table,
        plug_position=plug_position,
        plug_rotation=plug_rotation,
        table_center=table_center,
        table_top_z=table_top_z,
        ee_z_command=float(initial_ee_position[2]),
        finger_dofs=(
            gripper.finger_dofs["left"], gripper.finger_dofs["right"]
        ),
        all_dofs=np.arange(gripper.actor.get_num_dofs(), dtype=np.int32),
    )


def collect_cohort(
    specs: list[EpisodeSpec], args: argparse.Namespace, data: dict[str, Any]
) -> dict[str, float]:
    import numpy as np
    from scipy.spatial.transform import Rotation
    import superdex.physics as physics
    import superdex.robotics as robotics

    from soft_gripper import GelMaterial
    from wrench_math import aggregate_contact_points, integrate_surface_force_field

    gravity = np.array([0.0, 0.0, -9.81])
    plug_dimensions = np.array([0.025, 0.030, 0.060])
    table_dimensions = np.array([0.30, 0.30, 0.04])
    plug_local_cloud = sample_box_surface(plug_dimensions, args.point_count, args.seed + 1)
    table_local_cloud = sample_box_surface(table_dimensions, args.point_count, args.seed + 2)
    material = GelMaterial(
        youngs_modulus_pa=args.youngs_modulus_pa,
        poisson_ratio=args.poisson_ratio,
        mass_damping_s_inv=args.mass_damping,
        stiffness_damping_s=args.stiffness_damping,
    )
    setup_start = time.perf_counter()
    scene = physics.create_scene(f"parallel GelSight cohort {specs[0].episode_id}")
    scene.set_gravity(gravity)
    context = robotics.create_context()
    query_handles: list[Any] = []
    plug_shape = create_box_shape(physics, np, plug_dimensions)
    table_shape = create_box_shape(physics, np, table_dimensions)
    envs = [
        make_environment(
            scene, context, spec, local_index, len(specs), material,
            plug_shape, table_shape, plug_dimensions, table_dimensions,
            query_handles, physics, robotics, np, Rotation,
        )
        for local_index, spec in enumerate(specs)
    ]
    setup_seconds = time.perf_counter() - setup_start

    rollout_start = time.perf_counter()
    for step in range(args.steps):
        phase = phase_for_step(step, args.steps)
        for env in envs:
            if phase >= 1 and not env.released:
                env.plug.clear_boundary_conditions()
                env.released = True
            progress = step / max(args.steps - 1, 1)
            grasp_progress = min(1.0, progress / 0.18)
            smooth = grasp_progress * grasp_progress * (3.0 - 2.0 * grasp_progress)
            finger_force = -18.0 * smooth
            delta_z = 0.0
            rotation_delta = np.zeros(3)
            if phase == 2:
                delta_z = -0.00035
                env.ee_z_command += delta_z
            elif phase >= 3:
                error = env.filtered_table_force - env.spec.target_force_n
                delta_z = float(np.clip(0.00010 * error, -0.00025, 0.00025))
                env.ee_z_command += delta_z
                if phase == 4:
                    shear_phase = 2.0 * np.pi * (progress - 0.75) / 0.25
                    rotation_delta = np.array(
                        [0.0008 * np.cos(shear_phase), 0.0008 * np.sin(shear_phase), 0.0]
                    )
            target_rotation_vector = np.deg2rad(
                [env.spec.gripper_roll_deg, env.spec.gripper_pitch_deg, 0.0]
            )
            orientation_blend = min(1.0, max(0.0, (progress - 0.18) / 0.30))
            target_rotation = (
                Rotation.from_rotvec(target_rotation_vector * orientation_blend + rotation_delta)
                * env.initial_ee_rotation
            )
            world_target = physics.TransformRT(
                translation=[
                    env.initial_ee_position[0],
                    env.initial_ee_position[1],
                    env.ee_z_command,
                ],
                rotation=physics.Quaternion.from_rotation_vector(
                    target_rotation.as_rotvec()
                ),
            )
            root_target = env.world_from_root.inverse() * world_target
            effort = np.asarray(
                env.osc.compute_output(
                    env.osc.get_current_observations_from_mochi(),
                    robotics.ControllerBasicOscPdTarget(root_from_target_ee=root_target),
                ),
                dtype=np.float32,
            ).copy()
            effort[env.finger_dofs[0]] += finger_force
            effort[env.finger_dofs[1]] += finger_force
            env.gripper.actor.set_external_forces_on_dofs(env.all_dofs, effort)
            row = env.spec.episode_id
            data["actions"][row, step, :3] = [0.0, 0.0, delta_z]
            data["actions"][row, step, 3:] = rotation_delta
            data["contact_phase"][row, step] = phase
        scene.step(args.dt)

        now = float(scene.get_total_simulation_time())
        for env in envs:
            row = env.spec.episode_id
            observation = env.osc.get_current_observations_from_mochi()
            ee_transform = observation.world_from_ee_link
            current_ee_pose = pose7(ee_transform, env.root_offset)
            data["ee_pose"][row, step] = current_ee_pose
            if env.previous_ee_pose is not None:
                data["ee_vel"][row, step, :3] = (
                    current_ee_pose[:3] - env.previous_ee_pose[:3]
                ) / args.dt
                current_rotation = Rotation.from_quat(current_ee_pose[3:])
                previous_rotation = Rotation.from_quat(env.previous_ee_pose[3:])
                data["ee_vel"][row, step, 3:] = (
                    current_rotation * previous_rotation.inv()
                ).as_rotvec() / args.dt
            env.previous_ee_pose = current_ee_pose.copy()

            plug_transform = env.plug.get_center_of_mass_transform()
            plug_com = np.asarray(plug_transform.translation, dtype=float)
            data["tool_pose"][row, step] = pose7(plug_transform, env.root_offset)
            data["tool_vel"][row, step, :3] = env.plug.get_linear_velocity()
            data["tool_vel"][row, step, 3:] = env.plug.get_angular_velocity()
            data["objectpointcloud"][row, step] = (
                transform_points(plug_transform, plug_local_cloud) - env.root_offset
            )
            data["env_point_cloud"][row, step] = (
                transform_points(env.table.get_root_transform(), table_local_cloud)
                - env.root_offset
            )
            plug_contacts = list(env.plug.get_contact_points_world())
            table_wrench = aggregate_contact_points(
                plug_contacts, env.plug.get_handle(), env.table.get_handle(), plug_com
            )
            data["extrinsic_contact_wrench"][row, step] = np.concatenate(
                (table_wrench.force, table_wrench.torque)
            )
            data["table_contact_count"][row, step] = table_wrench.count
            table_normal = max(0.0, float(table_wrench.force[2]))
            env.filtered_table_force = 0.85 * env.filtered_table_force + 0.15 * table_normal

            field_force = np.zeros(3)
            field_torque = np.zeros(3)
            for side in ("left", "right"):
                gel = env.gripper.gels[side]
                housing = env.gripper.links[f"franka_{side}_gelsight_housing"]
                housing_transform = housing.get_root_transform()
                field = env.gripper.get_surface_force_field(side)
                data[f"tactile_force_field_{side}"][row, step] = field
                positions_sensor = env.gripper.gel_rest_sensor_surface[side]
                positions_world = transform_points(housing_transform, positions_sensor)
                data[f"tactile_coord_{side}"][row, step] = (
                    positions_world - env.root_offset
                )
                field_wrench = integrate_surface_force_field(
                    field,
                    positions_sensor,
                    housing_transform,
                    plug_com,
                    negate=True,
                )
                field_force += field_wrench.force
                field_torque += field_wrench.torque
                contacts = list(gel.get_contact_points_world())
                gel_wrench = aggregate_contact_points(
                    contacts, gel.get_handle(), env.plug.get_handle(),
                    np.asarray(housing_transform.translation, dtype=float),
                )
                data[f"{side}_contact_count"][row, step] = gel_wrench.count
            # Quasistatic table wrench inferred through both fingertip fields.
            inferred_force = -(field_force + 0.08 * gravity)
            data["field_inferred_extrinsic_wrench"][row, step] = np.concatenate(
                (inferred_force, -field_torque)
            )
            force_error = np.linalg.norm(
                data["extrinsic_contact_wrench"][row, step, :3]
                - data["field_inferred_extrinsic_wrench"][row, step, :3]
            )
            data["rewards"][row, step] = -abs(table_normal - env.spec.target_force_n) - 0.05 * force_error
            data["timestamps"][row, step] = now
            data["dones"][row, step] = step == args.steps - 1
    rollout_seconds = time.perf_counter() - rollout_start
    return {"setup_seconds": setup_seconds, "rollout_seconds": rollout_seconds}


def write_hdf5(
    path: Path,
    data: dict[str, Any],
    specs: list[EpisodeSpec],
    args: argparse.Namespace,
    benchmark: dict[str, Any],
) -> None:
    import h5py
    import numpy as np

    compression = None if args.compression == "none" else args.compression
    text_dtype = h5py.string_dtype("utf-8")
    with h5py.File(path, "w", libver="latest") as stream:
        stream.attrs.update(
            {
                "schema_version": SCHEMA_VERSION,
                "domain": "superdex",
                "simulator": "superdex_mochi",
                "interaction_family": "structured_overlap_plug_table_press",
                "split": "train",
                "episode_count": args.episodes,
                "steps_per_episode": args.steps,
                "sample_period_s": args.dt,
                "point_count": args.point_count,
                "tactile_grid_shape": np.asarray([7, 9, 3], dtype=np.int32),
                "force_units": "N",
                "torque_units": "N m",
                "position_units": "m",
                "quaternion_order": "xyzw",
            }
        )
        index = stream.create_group("index")
        index.create_dataset("episode_id", data=np.arange(args.episodes, dtype=np.int32))
        index.create_dataset("tool_id", data=np.asarray([s.tool_id for s in specs], dtype=np.int16))
        index.create_dataset(
            "branch_group_id", data=np.asarray([s.branch_group_id for s in specs], dtype=text_dtype)
        )
        index.create_dataset(
            "initial_state_hash", data=np.asarray([s.initial_state_hash for s in specs], dtype=text_dtype)
        )
        episodes = stream.create_group("episodes", track_order=True)
        observation_keys = {
            "objectpointcloud", "env_point_cloud", "ee_pose", "ee_vel",
            "tool_pose", "tool_vel", "tactile_force_field_left",
            "tactile_force_field_right", "tactile_coord_left",
            "tactile_coord_right", "extrinsic_contact_wrench",
            "field_inferred_extrinsic_wrench", "table_contact_count",
            "left_contact_count", "right_contact_count",
        }
        for spec in specs:
            group = episodes.create_group(f"episode_{spec.episode_id:06d}")
            group.attrs.update(
                {
                    "episode_id": spec.episode_id,
                    "domain": "superdex",
                    "simulator": "superdex_mochi",
                    "split": "train",
                    "training_role": "train",
                    "interaction_family": "structured_overlap_plug_table_press",
                    "tool_shape": "rectangular_plug",
                    "tool_id": spec.tool_id,
                    "branch_group_id": spec.branch_group_id,
                    "initial_state_hash": spec.initial_state_hash,
                    "sibling_complete": False,
                    "episode_seed": spec.seed,
                }
            )
            perturbations = group.create_group("perturbations")
            for key, value in asdict(spec).items():
                if key not in ("episode_id", "seed", "tool_id"):
                    perturbations.attrs[key] = value
            observations = group.create_group("observations")
            for key in sorted(observation_keys):
                observations.create_dataset(
                    key,
                    data=data[key][spec.episode_id],
                    compression=compression,
                    shuffle=compression == "gzip",
                )
            # Aliases used by the existing contact-field and real-data loaders.
            observations["point_cloud"] = observations["objectpointcloud"]
            observations["tactile_data_left"] = observations["tactile_force_field_left"]
            observations["tactile_data_right"] = observations["tactile_force_field_right"]
            for key in ("actions", "rewards", "dones", "contact_phase", "timestamps"):
                group.create_dataset(
                    key,
                    data=data[key][spec.episode_id],
                    compression=compression,
                    shuffle=compression == "gzip",
                )
            group["episode_lengths"] = np.int32(args.steps)
            group["episode_rewards"] = np.float32(data["rewards"][spec.episode_id].sum())
            labels = group.create_group("labels")
            finite = all(
                np.isfinite(data[key][spec.episode_id]).all()
                for key in ("ee_pose", "tool_pose", "tactile_force_field_left",
                            "tactile_force_field_right", "extrinsic_contact_wrench")
            )
            table_active = bool(np.any(data["table_contact_count"][spec.episode_id] > 0))
            tactile_active = bool(
                np.any(data["left_contact_count"][spec.episode_id] > 0)
                and np.any(data["right_contact_count"][spec.episode_id] > 0)
            )
            labels["physical_valid"] = np.bool_(finite and table_active)
            labels["tactile_valid"] = np.bool_(finite and tactile_active)
            labels["imitation_eligible"] = np.bool_(finite and table_active and tactile_active)
            labels["task_success"] = np.bool_(table_active)
        benchmark_group = stream.create_group("benchmark")
        for key, value in benchmark.items():
            if isinstance(value, (str, bool, int, float, np.number)):
                benchmark_group.attrs[key] = value


def validate_hdf5(path: Path, episodes: int, steps: int, point_count: int) -> dict[str, Any]:
    import h5py
    import numpy as np

    required = {
        "objectpointcloud": (steps, point_count, 3),
        "env_point_cloud": (steps, point_count, 3),
        "ee_pose": (steps, 7),
        "ee_vel": (steps, 6),
        "tactile_force_field_left": (steps, 7, 9, 3),
        "tactile_force_field_right": (steps, 7, 9, 3),
        "tactile_coord_left": (steps, 7, 9, 3),
        "tactile_coord_right": (steps, 7, 9, 3),
        "extrinsic_contact_wrench": (steps, 6),
    }
    valid = 0
    table_active = 0
    tactile_active = 0
    with h5py.File(path, "r") as stream:
        if stream.attrs["schema_version"] != SCHEMA_VERSION:
            raise ValueError("unexpected HDF5 schema version")
        if len(stream["episodes"]) != episodes:
            raise ValueError("episode count does not match")
        for group in stream["episodes"].values():
            observations = group["observations"]
            for key, shape in required.items():
                if observations[key].shape != shape:
                    raise ValueError(f"{group.name}/{key}: expected {shape}, got {observations[key].shape}")
                if not np.isfinite(observations[key][...]).all():
                    raise ValueError(f"{group.name}/{key} contains non-finite values")
            if group["actions"].shape != (steps, 6) or group["dones"].shape != (steps,):
                raise ValueError(f"{group.name}: invalid action/done shape")
            if not bool(group["dones"][-1]):
                raise ValueError(f"{group.name}: final done flag is false")
            valid += int(bool(group["labels/physical_valid"][()]))
            table_active += int(np.any(observations["table_contact_count"][...] > 0))
            tactile_active += int(bool(group["labels/tactile_valid"][()]))
    return {
        "schema_valid": True,
        "episodes_valid": valid,
        "episodes_with_table_contact": table_active,
        "episodes_with_bilateral_tactile": tactile_active,
    }


def estimates(benchmark: dict[str, Any], file_bytes: int, episodes: int) -> dict[str, Any]:
    targets = {
        "isaac_accepted_episodes": 23_678,
        "mujoco_accepted_episodes": 23_749,
        "combined_accepted_episodes": 47_427,
        "isaac_scheduled_episodes": 24_800,
        "mujoco_scheduled_episodes": 24_800,
        "combined_scheduled_episodes": 49_600,
    }
    seconds_per_episode = benchmark["total_seconds"] / episodes
    bytes_per_episode = file_bytes / episodes
    return {
        name: {
            "episodes": count,
            "estimated_seconds": seconds_per_episode * count,
            "estimated_hours": seconds_per_episode * count / 3600.0,
            "estimated_bytes": bytes_per_episode * count,
            "estimated_gib": bytes_per_episode * count / 2**30,
        }
        for name, count in targets.items()
    }


def gpu_snapshot() -> str:
    try:
        return subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unavailable"


def write_benchmark_report(path: Path, benchmark: dict[str, Any]) -> None:
    estimates_by_target = benchmark["dataset_size_estimates"]
    rows = []
    for name, estimate in estimates_by_target.items():
        rows.append(
            f"| {name.replace('_', ' ')} | {estimate['episodes']:,} | "
            f"{estimate['estimated_hours']:.2f} h | {estimate['estimated_gib']:.2f} GiB |"
        )
    validation = benchmark["validation"]
    path.write_text(
        "# Parallel collection benchmark\n\n"
        f"Collected `{benchmark['episodes']} × {benchmark['steps_per_episode']}` = "
        f"`{benchmark['environment_frames']:,}` environment-frames using "
        f"`{benchmark['parallel_envs']}` simultaneous environments per cohort.\n\n"
        f"- Rollout: `{benchmark['rollout_seconds']:.2f} s` "
        f"(`{benchmark['rollout_environment_frames_per_second']:.2f}` env-frames/s)\n"
        f"- End-to-end: `{benchmark['total_seconds']:.2f} s` "
        f"(`{benchmark['end_to_end_environment_frames_per_second']:.2f}` env-frames/s)\n"
        f"- HDF5: `{benchmark['hdf5_mib']:.2f} MiB`\n"
        f"- Valid/contact-complete: `{validation['episodes_valid']}/"
        f"{benchmark['episodes']}` physical, "
        f"`{validation['episodes_with_bilateral_tactile']}/{benchmark['episodes']}` "
        "bilateral tactile\n"
        f"- GPU snapshot before launch: `{benchmark['gpu_snapshot_before']}`\n\n"
        "## Linear projections\n\n"
        "These estimates scale the measured end-to-end time and compressed bytes per "
        "episode. They do not assume additional machines or overlap between collection "
        "jobs.\n\n"
        "| Notion dataset target | Episodes | Estimated time | Estimated HDF5 |\n"
        "|---|---:|---:|---:|\n"
        + "\n".join(rows)
        + "\n\nCounts come from the [AC-VTWM / Contact World Model Notion page]"
        "(https://www.notion.so/3a3838adf1828181896efdc39c5bccab).\n"
    )


def main() -> None:
    args = parse_args()
    os.environ.pop("SUPERDEX_PRECISION", None)  # selected FP32 production path
    os.environ["SUPERDEX_ASSETS_PATH"] = str(PROJECT_ROOT / "assets")
    import superdex.physics as physics

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = allocate(args.episodes, args.steps, args.point_count)
    specs = episode_specs(args.episodes, args.seed)
    wall_start = time.perf_counter()
    physics.initialize(num_worker_threads=0)
    cohort_metrics: list[dict[str, float]] = []
    try:
        for start in range(0, args.episodes, args.parallel_envs):
            stop = min(start + args.parallel_envs, args.episodes)
            print(f"collecting cohort {start}:{stop} ({stop-start} parallel environments)", flush=True)
            cohort_metrics.append(collect_cohort(specs[start:stop], args, data))
    finally:
        physics.shutdown()
    simulation_end = time.perf_counter()
    benchmark: dict[str, Any] = {
        "episodes": args.episodes,
        "steps_per_episode": args.steps,
        "environment_frames": args.episodes * args.steps,
        "parallel_envs": args.parallel_envs,
        "cohort_count": len(cohort_metrics),
        "setup_seconds": sum(item["setup_seconds"] for item in cohort_metrics),
        "rollout_seconds": sum(item["rollout_seconds"] for item in cohort_metrics),
        "simulation_total_seconds": simulation_end - wall_start,
        "precision": "fp32",
        "hostname": platform.node(),
        "cpu": platform.processor(),
        "python": platform.python_version(),
        "gpu_snapshot_before": gpu_snapshot(),
    }
    benchmark["rollout_environment_frames_per_second"] = (
        benchmark["environment_frames"] / benchmark["rollout_seconds"]
    )
    hdf5_path = output / f"superdex_{args.episodes}x{args.steps}.hdf5"
    serialization_start = time.perf_counter()
    write_hdf5(hdf5_path, data, specs, args, benchmark)
    benchmark["serialization_seconds"] = time.perf_counter() - serialization_start
    benchmark["total_seconds"] = time.perf_counter() - wall_start
    benchmark["end_to_end_environment_frames_per_second"] = (
        benchmark["environment_frames"] / benchmark["total_seconds"]
    )
    benchmark["hdf5_bytes"] = hdf5_path.stat().st_size
    benchmark["hdf5_mib"] = benchmark["hdf5_bytes"] / 2**20
    benchmark["validation"] = validate_hdf5(
        hdf5_path, args.episodes, args.steps, args.point_count
    )
    benchmark["dataset_size_estimates"] = estimates(
        benchmark, benchmark["hdf5_bytes"], args.episodes
    )
    # Add the final end-to-end figures after serialization has itself been
    # measured. The JSON remains the complete, nested benchmark record.
    import h5py
    with h5py.File(hdf5_path, "r+") as stream:
        for key in (
            "serialization_seconds", "total_seconds",
            "end_to_end_environment_frames_per_second", "hdf5_bytes", "hdf5_mib",
        ):
            stream["benchmark"].attrs[key] = benchmark[key]
    # Appending final benchmark attributes can grow the file by a small metadata
    # block. Record that final on-disk size in the JSON and HDF5 itself.
    benchmark["hdf5_bytes"] = hdf5_path.stat().st_size
    benchmark["hdf5_mib"] = benchmark["hdf5_bytes"] / 2**20
    benchmark["dataset_size_estimates"] = estimates(
        benchmark, benchmark["hdf5_bytes"], args.episodes
    )
    with h5py.File(hdf5_path, "r+") as stream:
        stream["benchmark"].attrs["hdf5_bytes"] = benchmark["hdf5_bytes"]
        stream["benchmark"].attrs["hdf5_mib"] = benchmark["hdf5_mib"]
    (output / "benchmark.json").write_text(json.dumps(benchmark, indent=2) + "\n")
    write_benchmark_report(output / "benchmark_report.md", benchmark)
    (output / "config.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "arguments": {key: str(value) if isinstance(value, Path) else value
                              for key, value in vars(args).items()},
                "phase_names": list(PHASE_NAMES),
                "notion_source": "AC-VTWM / Contact World Model",
                "notion_url": "https://www.notion.so/3a3838adf1828181896efdc39c5bccab",
            },
            indent=2,
        ) + "\n"
    )
    print(
        f"wrote {args.episodes}x{args.steps}={benchmark['environment_frames']:,} "
        f"frames to {hdf5_path}\n"
        f"rollout {benchmark['rollout_seconds']:.2f}s "
        f"({benchmark['rollout_environment_frames_per_second']:.1f} env-frames/s); "
        f"end-to-end {benchmark['total_seconds']:.2f}s "
        f"({benchmark['end_to_end_environment_frames_per_second']:.1f} env-frames/s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
