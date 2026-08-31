#!/usr/bin/env python3
"""Run the FR3 + 2F-85 contact-wrench validation experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
DEFAULT_OUTPUT = EXPERIMENT_DIR / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--precision", choices=("single", "double", "both"), default="both"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--case-name", default="baseline")
    parser.add_argument("--gripper-roll-deg", type=float, default=0.0)
    parser.add_argument("--gripper-pitch-deg", type=float, default=0.0)
    parser.add_argument("--plug-roll-deg", type=float, default=0.0)
    parser.add_argument("--plug-pitch-deg", type=float, default=0.0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.video_fps <= 0 or args.width <= 0 or args.height <= 0:
        parser.error("video FPS and dimensions must be positive")
    return args


def worker_command(args: argparse.Namespace, precision: str, output: Path) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--precision",
        precision,
        "--output",
        str(output),
        "--video-fps",
        str(args.video_fps),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--case-name",
        args.case_name,
        "--gripper-roll-deg",
        str(args.gripper_roll_deg),
        "--gripper-pitch-deg",
        str(args.gripper_pitch_deg),
        "--plug-roll-deg",
        str(args.plug_roll_deg),
        "--plug-pitch-deg",
        str(args.plug_pitch_deg),
    ]
    if args.no_video:
        command.append("--no-video")
    return command


def run_coordinator(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    precisions = (
        ("single", "fp32"),
        ("double", "fp64"),
    ) if args.precision == "both" else ((args.precision, "fp32" if args.precision == "single" else "fp64"),)

    completed: dict[str, Path] = {}
    for precision, label in precisions:
        precision_output = output / label
        precision_output.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["SUPERDEX_ASSETS_PATH"] = str(PROJECT_ROOT / "assets")
        if precision == "double":
            environment["SUPERDEX_PRECISION"] = "double"
        else:
            environment.pop("SUPERDEX_PRECISION", None)
        print(f"\n=== Running {label} contact-wrench trial ===", flush=True)
        subprocess.run(
            worker_command(args, precision, precision_output),
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
        )
        completed[label] = precision_output

    if len(completed) == 2:
        build_precision_comparison(completed, output / "comparison")

    summary = {
        "experiment": "FR3 v2 + Robotiq 2F-85 contact-wrench validation",
        "case_name": args.case_name,
        "orientation_degrees": {
            "gripper_roll": args.gripper_roll_deg,
            "gripper_pitch": args.gripper_pitch_deg,
            "plug_roll": args.plug_roll_deg,
            "plug_pitch": args.plug_pitch_deg,
        },
        "precisions": {},
    }
    for label, directory in completed.items():
        summary["precisions"][label] = json.loads(
            (directory / "metrics.json").read_text()
        )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nResults written to {output}")


def read_csv_columns(path: Path) -> dict[str, list[float]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    columns: dict[str, list[float]] = {}
    for name in reader.fieldnames or []:
        try:
            columns[name] = [float(row[name]) for row in rows]
        except ValueError:
            continue
    return columns


def build_precision_comparison(inputs: dict[str, Path], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output.mkdir(parents=True, exist_ok=True)
    data = {
        label: read_csv_columns(directory / "contact_wrenches.csv")
        for label, directory in inputs.items()
    }
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    colors = {"fp32": "tab:blue", "fp64": "tab:orange"}
    for label, columns in data.items():
        time = np.asarray(columns["time_since_incidence_s"])
        table_force = np.asarray(columns["table_measured_force_world_z"])
        force_residual = np.sqrt(
            sum(np.square(columns[f"dynamic_force_residual_world_{axis}"]) for axis in "xyz")
        )
        torque_residual = np.sqrt(
            sum(np.square(columns[f"dynamic_torque_residual_world_{axis}"]) for axis in "xyz")
        )
        axes[0].plot(time, table_force, label=label, color=colors[label])
        axes[1].plot(time, force_residual, label=label, color=colors[label])
        axes[2].plot(time, torque_residual, label=label, color=colors[label])
    axes[0].set_ylabel("Table normal force [N]")
    axes[1].set_ylabel("Dynamic force residual [N]")
    axes[2].set_ylabel("Dynamic torque residual [N m]")
    axes[2].set_xlabel("Time since incidence [s]")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.suptitle("FP32 vs FP64 contact-wrench comparison")
    fig.tight_layout()
    fig.savefig(output / "precision_comparison.png", dpi=180)
    plt.close(fig)

    fp32, fp64 = data["fp32"], data["fp64"]
    t32 = np.asarray(fp32["time_since_incidence_s"])
    t64 = np.asarray(fp64["time_since_incidence_s"])
    overlap = (t32 >= np.nanmin(t64)) & (t32 <= np.nanmax(t64))
    comparisons: dict[str, float] = {}
    for column in (
        "table_measured_force_world_z",
        "table_measured_torque_world_x",
        "table_measured_torque_world_y",
        "table_measured_torque_world_z",
    ):
        interpolated = np.interp(t32[overlap], t64, np.asarray(fp64[column]))
        delta = np.asarray(fp32[column])[overlap] - interpolated
        comparisons[f"{column}_rms_difference"] = float(np.sqrt(np.mean(delta**2)))
        comparisons[f"{column}_peak_difference"] = float(np.max(np.abs(delta)))
    (output / "precision_metrics.json").write_text(
        json.dumps(comparisons, indent=2) + "\n"
    )


def run_worker(args: argparse.Namespace) -> None:
    # Precision and display selection must happen before importing SuperDex/Polyscope.
    if args.precision == "double":
        os.environ["SUPERDEX_PRECISION"] = "double"
    else:
        os.environ.pop("SUPERDEX_PRECISION", None)
    os.environ["SUPERDEX_ASSETS_PATH"] = str(PROJECT_ROOT / "assets")
    if not args.no_video:
        os.environ.pop("DISPLAY", None)
    _run_simulation(args)


def _run_simulation(args: argparse.Namespace) -> None:
    import imageio.v2 as imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image, ImageDraw
    from scipy.spatial.transform import Rotation
    import superdex.physics as physics
    import superdex.robotics as robotics
    from superdex.physics.paths import resolve_asset

    from wrench_math import (
        Wrench,
        aggregate_contact_points,
        append_vector,
        inertia_matrix,
        opposite_wrench,
        rotate_wrench_to_local,
    )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    np_real = np.float64 if physics.uses_double_precision() else np.float32
    precision_label = "fp64" if physics.uses_double_precision() else "fp32"

    time_step = 1.0 / 200.0
    grasp_duration = 1.0
    airborne_hold_end = 2.0
    incidence_threshold = 0.2
    table_top_z = 0.25
    plug_dimensions = np.array([0.025, 0.030, 0.060], dtype=float)
    table_dimensions = np.array([0.25, 0.25, 0.04], dtype=float)
    plug_mass = 0.08
    gravity = np.array([0.0, 0.0, -9.81], dtype=float)

    def create_box_shape(dimensions: np.ndarray):
        x, y, z = dimensions / 2.0
        vertices = np.array(
            [
                [-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
                [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z],
            ],
            dtype=np_real,
        ).flatten()
        faces = np.array(
            [
                [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
                [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2],
            ],
            dtype=np.int32,
        ).flatten()
        return physics.create_tri_mesh_shape(vertices, faces)

    def rotation_matrix(transform: Any) -> np.ndarray:
        return Rotation.from_rotvec(
            np.asarray(transform.rotation.to_rotation_vector(), dtype=float)
        ).as_matrix()

    def resolve_link(scene: Any, articulated_actor: Any, name: str):
        info = articulated_actor.get_articulated_shape_info()
        handles = list(articulated_actor.get_nested_link_actors())
        index = list(info.link_names).index(name)
        return scene.get_actor(handles[index])

    def target_force_and_phase(elapsed: float) -> tuple[float, str]:
        if elapsed < 0.5:
            return 4.0 * elapsed, "ramp_2N"
        if elapsed < 2.0:
            return 2.0, "hold_2N"
        if elapsed < 2.5:
            return 2.0 + 6.0 * (elapsed - 2.0), "ramp_5N"
        if elapsed < 4.0:
            return 5.0, "hold_5N"
        if elapsed < 4.5:
            return 5.0 + 10.0 * (elapsed - 4.0), "ramp_10N"
        if elapsed < 6.0:
            return 10.0, "hold_10N"
        if elapsed < 7.0:
            return 0.0, "retract"
        return 0.0, "post_contact_hold"

    def add_interface(
        row: dict[str, Any],
        name: str,
        measured: Wrench,
        reference: Wrench,
        rotation_world_from_local: np.ndarray,
    ) -> None:
        error = Wrench(
            measured.force - reference.force,
            measured.torque - reference.torque,
        )
        measured_local = rotate_wrench_to_local(
            measured.force, measured.torque, rotation_world_from_local
        )
        reference_local = rotate_wrench_to_local(
            reference.force, reference.torque, rotation_world_from_local
        )
        error_local = Wrench(
            measured_local.force - reference_local.force,
            measured_local.torque - reference_local.torque,
        )
        for label, wrench in (
            ("measured", measured), ("reference", reference), ("error", error)
        ):
            append_vector(row, f"{name}_{label}_force_world", wrench.force)
            append_vector(row, f"{name}_{label}_torque_world", wrench.torque)
        for label, wrench in (
            ("measured", measured_local),
            ("reference", reference_local),
            ("error", error_local),
        ):
            append_vector(row, f"{name}_{label}_force_local", wrench.force)
            append_vector(row, f"{name}_{label}_torque_local", wrench.torque)
        row[f"{name}_contact_count"] = reference.count

    physics.initialize(num_worker_threads=0)
    viewer = None
    video_writer = None
    bot = None
    robotics_context = None
    rows: list[dict[str, Any]] = []
    query_handles: list[Any] = []
    try:
        scene = physics.create_scene("FR3 2F-85 Contact Wrench Validation")
        scene.set_gravity(gravity)

        bot_prefab = robotics.load_bot_prefab_from_file(
            str(
                resolve_asset(
                    "bots/arm_hand_combos/fr3_v2_2f_85/fr3_v2_2f_85.superdex_bot"
                )
            )
        )
        for link_index in range(len(bot_prefab.links)):
            bot_prefab.links[link_index].has_gravity = False
        robotics_context = robotics.create_context()
        bot = robotics.create_bot(scene, bot_prefab, robotics_context)
        bot_actor = bot.get_articulated_actor()
        bot_name = bot.get_name()
        left_tip = resolve_link(scene, bot_actor, "2f_85_left_finger_tip_link")
        right_tip = resolve_link(scene, bot_actor, "2f_85_right_finger_tip_link")

        left_initial = np.asarray(
            left_tip.get_center_of_mass_transform().translation, dtype=float
        )
        right_initial = np.asarray(
            right_tip.get_center_of_mass_transform().translation, dtype=float
        )
        plug_initial_position = 0.5 * (left_initial + right_initial)
        # Collision-mesh inner planes are mildly asymmetric about the link COMs.
        # Center the coupon on those planes, measured from the authored asset.
        plug_initial_position[1] = -0.0001159
        # Present the coupon below the pad tips so the plug, rather than the
        # gripper collision geometry, is the first body to reach the tabletop.
        plug_initial_position[2] -= 0.025
        plug_initial_scipy_rotation = Rotation.from_euler(
            "xy", [args.plug_roll_deg, args.plug_pitch_deg], degrees=True
        )
        plug_initial_rotation_vector = plug_initial_scipy_rotation.as_rotvec()
        plug_initial_rotation = physics.Quaternion.from_rotation_vector(
            plug_initial_rotation_vector
        )

        contact = physics.ContactParams()
        contact.coulomb_friction_coefficient = 1.2
        plug = scene.create_rigid_actor(
            name="plug",
            shape=create_box_shape(plug_dimensions),
            mass=plug_mass,
            contact=contact,
            world_from_local=physics.TransformRT(
                rotation=plug_initial_rotation,
                translation=plug_initial_position,
            ),
        )
        # A temporary presentation constraint keeps the plug in place while the
        # gripper closes. It is removed for the airborne hold and all measurements.
        plug.add_boundary_condition_dofs_world(
            np.arange(6, dtype=np.int32),
            [*plug_initial_position.tolist(), *plug_initial_rotation_vector.tolist()],
        )

        table_center = np.array(
            [plug_initial_position[0], plug_initial_position[1], table_top_z - table_dimensions[2] / 2]
        )
        table = scene.create_rigid_actor(
            name="tabletop",
            shape=create_box_shape(table_dimensions),
            mass=100.0,
            has_gravity=False,
            contact=contact,
            world_from_local=physics.TransformRT(translation=table_center),
        )
        table.add_boundary_condition_dofs_world(
            np.arange(6, dtype=np.int32),
            [*table_center.tolist(), 0.0, 0.0, 0.0],
        )

        for actor in (left_tip, right_tip, plug, table):
            query_handles.append(actor.register_query(physics.QueryType.TOTAL_CONTACT_FORCE))
            query_handles.append(actor.register_query(physics.QueryType.CONTACT_POINTS))

        num_dofs = bot_actor.get_num_dofs()
        all_dofs = np.arange(num_dofs, dtype=np.int32)

        osc = bot.create_controller("BASIC_OSC_PD")
        osc.initialize(f"{bot_name}/fr3_link0", f"{bot_name}/fr3_link8")
        osc_params = osc.get_params()
        osc_params.kp_p = 1100.0
        osc_params.kd_p = 85.0
        osc_params.kp_r = 40.0
        osc_params.kd_r = 4.0
        osc_params.max_translation_error = 0.04
        osc_params.max_rotation_error = 0.3
        osc_params.b_apply_max_osc_torque_normalization = True
        osc.set_params(osc_params)

        initial_osc_observation = osc.get_current_observations_from_mochi()
        world_from_root = initial_osc_observation.world_from_root
        initial_ee = initial_osc_observation.world_from_ee_link
        initial_ee_position = np.asarray(initial_ee.translation, dtype=float)
        initial_ee_rotation = initial_ee.rotation
        initial_ee_scipy_rotation = Rotation.from_rotvec(
            np.asarray(initial_ee_rotation.to_rotation_vector(), dtype=float)
        )
        gripper_orientation_delta = Rotation.from_euler(
            "xy", [args.gripper_roll_deg, args.gripper_pitch_deg], degrees=True
        ).as_rotvec()
        ee_z_command = float(initial_ee_position[2])

        if not args.no_video:
            from superdex.physics.utils.coordinate_systems import CoordinateSystem
            from superdex.physics.viewer import Viewer, ViewerCfg

            viewer = Viewer(
                ViewerCfg(
                    backend="openGL3_egl",
                    size=(args.width, args.height),
                    offscreen=True,
                    coordinate_system=CoordinateSystem("-Y", "+Z", "+X"),
                )
            )
            viewer.set_scene(scene)
            look_from = np.array([1.35, -1.2, 1.05])
            look_at = np.array([plug_initial_position[0], 0.0, 0.38])
            look_direction = look_at - look_from
            viewer.frame_scene(look_direction / np.linalg.norm(look_direction))
            video_writer = imageio.get_writer(
                output / "simulation.mp4",
                format="ffmpeg",
                mode="I",
                fps=args.video_fps,
                codec="libx264",
                quality=8,
                macro_block_size=16,
            )

        next_video_time = 0.0
        plug_released = False
        incidence_time: float | None = None
        incidence_ee_z: float | None = None
        filtered_table_force = 0.0
        retract_start_z: float | None = None
        latest_table_force = 0.0
        latest_phase = "grasp"
        latest_target_force = 0.0

        for _step in range(int(18.0 / time_step)):
            time = float(scene.get_total_simulation_time())
            if time >= grasp_duration and not plug_released:
                plug.clear_boundary_conditions()
                plug_released = True

            if time < grasp_duration:
                phase = "grasp"
                blend = min(1.0, time / 0.8)
                blend = blend * blend * (3.0 - 2.0 * blend)
                grip_torque = 0.5 * blend
                target_force = 0.0
            else:
                grip_torque = 0.5
                if time < airborne_hold_end:
                    phase = "airborne_hold"
                    target_force = 0.0
                elif incidence_time is None:
                    phase = "descent"
                    target_force = 0.0
                    ee_z_command -= 0.06 * time_step
                else:
                    elapsed = time - incidence_time
                    target_force, phase = target_force_and_phase(elapsed)
                    if elapsed < 6.0:
                        velocity_command = np.clip(
                            0.008 * (filtered_table_force - target_force), -0.02, 0.02
                        )
                        ee_z_command += float(velocity_command) * time_step
                        assert incidence_ee_z is not None
                        ee_z_command = float(
                            np.clip(
                                ee_z_command,
                                incidence_ee_z - 0.045,
                                incidence_ee_z + 0.02,
                            )
                        )
                    elif elapsed < 7.0:
                        if retract_start_z is None:
                            retract_start_z = ee_z_command
                        ee_z_command = retract_start_z + 0.08 * (elapsed - 6.0)
                    else:
                        assert retract_start_z is not None
                        ee_z_command = retract_start_z + 0.08

            if time <= grasp_duration:
                orientation_blend = 0.0
            elif time < airborne_hold_end:
                orientation_blend = (time - grasp_duration) / (
                    airborne_hold_end - grasp_duration
                )
                orientation_blend = orientation_blend * orientation_blend * (
                    3.0 - 2.0 * orientation_blend
                )
            else:
                orientation_blend = 1.0
            target_ee_scipy_rotation = (
                Rotation.from_rotvec(gripper_orientation_delta * orientation_blend)
                * initial_ee_scipy_rotation
            )
            target_ee_rotation = physics.Quaternion.from_rotation_vector(
                target_ee_scipy_rotation.as_rotvec()
            )

            world_target = physics.TransformRT(
                translation=[
                    float(initial_ee_position[0]),
                    float(initial_ee_position[1]),
                    ee_z_command,
                ],
                rotation=target_ee_rotation,
            )
            root_target = world_from_root.inverse() * world_target
            osc_observation = osc.get_current_observations_from_mochi()
            arm_torque = np.asarray(
                osc.compute_output(
                    osc_observation,
                    robotics.ControllerBasicOscPdTarget(root_from_target_ee=root_target),
                ),
                dtype=np_real,
            ).copy()
            # The 2F-85 is a closed-loop linkage. Drive its outer knuckles in
            # opposite directions and let the remaining four joints follow
            # passively through the authored cycle constraints.
            gripper_torque = np.zeros(num_dofs, dtype=np_real)
            gripper_torque[8] = grip_torque
            gripper_torque[11] = -grip_torque
            bot_actor.set_external_forces_on_dofs(
                all_dofs, arm_torque + gripper_torque
            )
            scene.step(time_step)

            time_after = float(scene.get_total_simulation_time())
            actual_ee_transform = (
                osc.get_current_observations_from_mochi().world_from_ee_link
            )
            plug_com_transform = plug.get_center_of_mass_transform()
            table_com_transform = table.get_center_of_mass_transform()
            left_com_transform = left_tip.get_center_of_mass_transform()
            right_com_transform = right_tip.get_center_of_mass_transform()
            plug_com = np.asarray(plug_com_transform.translation, dtype=float)
            table_com = np.asarray(table_com_transform.translation, dtype=float)
            left_com = np.asarray(left_com_transform.translation, dtype=float)
            right_com = np.asarray(right_com_transform.translation, dtype=float)

            plug_contacts = list(plug.get_contact_points_world())
            left_contacts = list(left_tip.get_contact_points_world())
            right_contacts = list(right_tip.get_contact_points_world())
            left_tip_integrated = aggregate_contact_points(
                left_contacts, left_tip.get_handle(), plug.get_handle(), left_com
            )
            right_tip_integrated = aggregate_contact_points(
                right_contacts, right_tip.get_handle(), plug.get_handle(), right_com
            )
            table_reference = aggregate_contact_points(
                plug_contacts, plug.get_handle(), table.get_handle(), plug_com
            )
            left_on_plug = aggregate_contact_points(
                plug_contacts, plug.get_handle(), left_tip.get_handle(), plug_com
            )
            right_on_plug = aggregate_contact_points(
                plug_contacts, plug.get_handle(), right_tip.get_handle(), plug_com
            )
            left_reference = opposite_wrench(
                left_on_plug.force, left_on_plug.torque, plug_com, left_com
            )
            left_reference.count = left_on_plug.count
            right_reference = opposite_wrench(
                right_on_plug.force, right_on_plug.torque, plug_com, right_com
            )
            right_reference.count = right_on_plug.count

            left_measured = Wrench(
                np.asarray(left_tip.get_contact_force_from_actor_world(plug), dtype=float),
                left_tip_integrated.torque,
                left_tip_integrated.count,
            )
            right_measured = Wrench(
                np.asarray(right_tip.get_contact_force_from_actor_world(plug), dtype=float),
                right_tip_integrated.torque,
                right_tip_integrated.count,
            )
            table_opposite = opposite_wrench(
                table.get_contact_force_world(),
                table.get_contact_torque_world(),
                table_com,
                plug_com,
            )
            table_measured = Wrench(
                np.asarray(plug.get_contact_force_from_actor_world(table), dtype=float),
                table_opposite.torque,
            )
            latest_table_force = max(0.0, float(table_measured.force[2]))
            filtered_table_force = 0.85 * filtered_table_force + 0.15 * latest_table_force

            if (
                incidence_time is None
                and time_after >= airborne_hold_end
                and latest_table_force >= incidence_threshold
            ):
                incidence_time = time_after
                incidence_ee_z = ee_z_command
                print(
                    f"{precision_label}: contact incidence at {incidence_time:.3f} s, "
                    f"EE z={incidence_ee_z:.4f} m"
                )

            contact_force_sum = (
                left_on_plug.force + right_on_plug.force + table_reference.force
            )
            contact_torque_sum = (
                left_on_plug.torque + right_on_plug.torque + table_reference.torque
            )
            row: dict[str, Any] = {
                "time_s": time_after,
                "phase": phase,
                "precision": precision_label,
                "case_name": args.case_name,
                "target_table_normal_force_n": target_force,
                "filtered_table_normal_force_n": filtered_table_force,
                "ee_z_command_m": ee_z_command,
                "gripper_orientation_blend": orientation_blend,
            }
            append_vector(row, "ee_position_world", actual_ee_transform.translation)
            append_vector(
                row,
                "ee_rotation_vector_world",
                actual_ee_transform.rotation.to_rotation_vector(),
            )
            add_interface(
                row, "left", left_measured, left_reference, rotation_matrix(left_com_transform)
            )
            add_interface(
                row,
                "right",
                right_measured,
                right_reference,
                rotation_matrix(right_com_transform),
            )
            add_interface(
                row,
                "table",
                table_measured,
                table_reference,
                rotation_matrix(plug_com_transform),
            )
            append_vector(row, "plug_com_world", plug_com)
            append_vector(row, "plug_rotation_vector_world", plug_com_transform.rotation.to_rotation_vector())
            append_vector(row, "plug_linear_velocity_world", plug.get_linear_velocity())
            append_vector(row, "plug_angular_velocity_world", plug.get_angular_velocity())
            append_vector(row, "plug_contact_force_world", contact_force_sum)
            append_vector(row, "plug_contact_torque_world", contact_torque_sum)
            append_vector(row, "plug_total_query_force_world", plug.get_contact_force_world())
            append_vector(row, "plug_total_query_torque_world", plug.get_contact_torque_world())
            rows.append(row)
            latest_phase = phase
            latest_target_force = target_force

            if video_writer is not None and time_after + 1e-9 >= next_video_time:
                frame = viewer.render()
                if frame is None:
                    raise RuntimeError("Offscreen renderer returned no frame")
                image = Image.fromarray(np.asarray(frame)[..., :3])
                draw = ImageDraw.Draw(image)
                draw.rectangle((10, 10, 430, 107), fill=(255, 255, 255, 220))
                draw.text((20, 18), f"phase: {latest_phase}", fill=(10, 10, 10))
                draw.text(
                    (20, 40),
                    f"target: {latest_target_force:5.2f} N   table: {latest_table_force:5.2f} N",
                    fill=(10, 10, 10),
                )
                draw.text((20, 62), f"precision: {precision_label}", fill=(10, 10, 10))
                draw.text(
                    (20, 84),
                    f"case: {args.case_name}  grip r/p: {args.gripper_roll_deg:g}/{args.gripper_pitch_deg:g} deg",
                    fill=(10, 10, 10),
                )
                video_writer.append_data(np.asarray(image))
                next_video_time += 1.0 / args.video_fps

            if incidence_time is None and time_after >= 9.0:
                actual_ee = np.asarray(
                    osc.get_current_observations_from_mochi().world_from_ee_link.translation,
                    dtype=float,
                )
                raise RuntimeError(
                    "The plug did not contact the tabletop before timeout: "
                    f"plug COM={plug_com.tolist()}, actual EE={actual_ee.tolist()}, "
                    f"commanded EE z={ee_z_command:.4f}"
                )
            if incidence_time is not None and time_after - incidence_time >= 7.5:
                break
        else:
            raise RuntimeError("Experiment exceeded its maximum simulation duration")

        if incidence_time is None:
            raise RuntimeError("No tabletop contact incidence was recorded")

        for row in rows:
            row["time_since_incidence_s"] = float(row["time_s"] - incidence_time)

        plug_inertia_local = inertia_matrix(plug.get_rigid_moment_of_inertia_local())
        velocities = np.array(
            [[row[f"plug_linear_velocity_world_{axis}"] for axis in "xyz"] for row in rows]
        )
        angular_velocities = np.array(
            [[row[f"plug_angular_velocity_world_{axis}"] for axis in "xyz"] for row in rows]
        )
        accelerations = np.gradient(velocities, time_step, axis=0, edge_order=2)
        angular_accelerations = np.gradient(
            angular_velocities, time_step, axis=0, edge_order=2
        )
        for index, row in enumerate(rows):
            rotation = Rotation.from_rotvec(
                [row[f"plug_rotation_vector_world_{axis}"] for axis in "xyz"]
            ).as_matrix()
            inertia_world = rotation @ plug_inertia_local @ rotation.T
            inertial_force = plug_mass * accelerations[index]
            omega = angular_velocities[index]
            inertial_torque = (
                inertia_world @ angular_accelerations[index]
                + np.cross(omega, inertia_world @ omega)
            )
            contact_force = np.array(
                [row[f"plug_contact_force_world_{axis}"] for axis in "xyz"]
            )
            contact_torque = np.array(
                [row[f"plug_contact_torque_world_{axis}"] for axis in "xyz"]
            )
            append_vector(
                row,
                "dynamic_force_residual_world",
                contact_force + plug_mass * gravity - inertial_force,
            )
            append_vector(
                row,
                "dynamic_torque_residual_world",
                contact_torque - inertial_torque,
            )
            append_vector(
                row,
                "quasistatic_force_residual_world",
                contact_force + plug_mass * gravity,
            )
            append_vector(row, "quasistatic_torque_residual_world", contact_torque)

        csv_path = output / "contact_wrenches.csv"
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        metrics = compute_metrics(rows, incidence_time, np)
        metrics.update(
            {
                "precision": precision_label,
                "uses_double_precision": bool(physics.uses_double_precision()),
                "sample_rate_hz": int(round(1.0 / time_step)),
                "num_samples": len(rows),
                "incidence_time_s": incidence_time,
                "plug_mass_kg": plug_mass,
            }
        )
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        config = {
            "case_name": args.case_name,
            "robot": "bots/arm_hand_combos/fr3_v2_2f_85/fr3_v2_2f_85.superdex_bot",
            "plug_dimensions_m": plug_dimensions.tolist(),
            "plug_mass_kg": plug_mass,
            "table_dimensions_m": table_dimensions.tolist(),
            "table_top_z_m": table_top_z,
            "time_step_s": time_step,
            "incidence_threshold_n": incidence_threshold,
            "force_plateaus_n": [2.0, 5.0, 10.0],
            "gripper_outer_knuckle_torque_nm": 0.5,
            "plug_vertical_presentation_offset_m": -0.025,
            "normal_force_controller_gain_m_per_ns": 0.008,
            "orientation_degrees": {
                "gripper_roll": args.gripper_roll_deg,
                "gripper_pitch": args.gripper_pitch_deg,
                "plug_initial_roll": args.plug_roll_deg,
                "plug_initial_pitch": args.plug_pitch_deg,
            },
            "precision": precision_label,
            "video_enabled": not args.no_video,
        }
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        make_plots(rows, output, np, plt)
        print(
            f"{precision_label}: wrote {len(rows)} samples and analysis to {output}",
            flush=True,
        )
    finally:
        if video_writer is not None:
            video_writer.close()
        if viewer is not None:
            viewer.close()
        if bot is not None:
            try:
                robotics.destroy_bot(scene, bot)
            except Exception:
                pass
        physics.shutdown()


def compute_metrics(rows: list[dict[str, Any]], incidence_time: float, np: Any) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "interfaces": {},
        "plateaus": {},
        "execution_checks": {},
    }
    for interface in ("left", "right", "table"):
        active = np.array([row[f"{interface}_contact_count"] > 0 for row in rows])
        interface_metrics: dict[str, Any] = {"active_samples": int(active.sum())}
        for quantity in ("force", "torque"):
            measured = np.array(
                [
                    [row[f"{interface}_measured_{quantity}_world_{axis}"] for axis in "xyz"]
                    for row in rows
                ]
            )[active]
            reference = np.array(
                [
                    [row[f"{interface}_reference_{quantity}_world_{axis}"] for axis in "xyz"]
                    for row in rows
                ]
            )[active]
            if len(measured) == 0:
                interface_metrics[quantity] = None
                continue
            error = measured - reference
            rmse = float(np.sqrt(np.mean(np.sum(error**2, axis=1))))
            reference_rms = float(np.sqrt(np.mean(np.sum(reference**2, axis=1))))
            interface_metrics[quantity] = {
                "vector_rmse": rmse,
                "relative_rmse": rmse / max(reference_rms, 1e-12),
                "peak_vector_error": float(np.max(np.linalg.norm(error, axis=1))),
                "component_bias": np.mean(error, axis=0).tolist(),
            }
        metrics["interfaces"][interface] = interface_metrics

    for phase, target in (("hold_2N", 2.0), ("hold_5N", 5.0), ("hold_10N", 10.0)):
        values = np.array(
            [
                row["table_measured_force_world_z"]
                for row in rows
                if row["phase"] == phase
            ]
        )
        if len(values):
            phase_rows = [row for row in rows if row["phase"] == phase]
            metrics["plateaus"][phase] = {
                "target_n": target,
                "mean_n": float(np.mean(values)),
                "std_n": float(np.std(values)),
                "median_n": float(np.median(values)),
                "mean_tracking_error_n": float(np.mean(values) - target),
                "num_samples": int(len(values)),
                "both_fingertips_contact_fraction": float(
                    np.mean(
                        [
                            row["left_contact_count"] > 0
                            and row["right_contact_count"] > 0
                            for row in phase_rows
                        ]
                    )
                ),
                "table_contact_fraction": float(
                    np.mean([row["table_contact_count"] > 0 for row in phase_rows])
                ),
            }
    airborne_rows = [row for row in rows if row["phase"] == "airborne_hold"]
    metrics["execution_checks"] = {
        "airborne_hold_both_fingertips_contact_fraction": float(
            np.mean(
                [
                    row["left_contact_count"] > 0
                    and row["right_contact_count"] > 0
                    for row in airborne_rows
                ]
            )
        ),
        "airborne_hold_table_contact_fraction": float(
            np.mean([row["table_contact_count"] > 0 for row in airborne_rows])
        ),
    }
    metrics["incidence_time_s"] = incidence_time
    return metrics


def make_plots(rows: list[dict[str, Any]], output: Path, np: Any, plt: Any) -> None:
    time = np.array([row["time_since_incidence_s"] for row in rows])
    target = np.array([row["target_table_normal_force_n"] for row in rows])

    def vector(prefix: str) -> Any:
        return np.array([[row[f"{prefix}_{axis}"] for axis in "xyz"] for row in rows])

    table_force = vector("table_measured_force_world")
    table_force_ref = vector("table_reference_force_world")
    table_torque = vector("table_measured_torque_world")
    table_torque_ref = vector("table_reference_torque_world")
    left_force = vector("left_measured_force_world")
    right_force = vector("right_measured_force_world")
    force_residual = np.linalg.norm(vector("dynamic_force_residual_world"), axis=1)
    torque_residual = np.linalg.norm(vector("dynamic_torque_residual_world"), axis=1)

    fig, axes = plt.subplots(4, 1, figsize=(13, 13), sharex=True)
    axes[0].plot(time, target, "k--", label="target")
    axes[0].plot(time, table_force[:, 2], label="table query/opposite")
    axes[0].plot(time, table_force_ref[:, 2], ":", label="contact integration")
    axes[0].set_ylabel("Table Fz [N]")
    axes[1].plot(time, left_force[:, 2], label="left fingertip Fz")
    axes[1].plot(time, right_force[:, 2], label="right fingertip Fz")
    axes[1].set_ylabel("Fingertip Fz [N]")
    for index, axis_name in enumerate("xyz"):
        axes[2].plot(time, table_torque[:, index], label=f"measured T{axis_name}")
        axes[2].plot(
            time, table_torque_ref[:, index], ":", label=f"reference T{axis_name}"
        )
    axes[2].set_ylabel("Table torque [N m]")
    axes[3].plot(time, force_residual, label="force residual [N]")
    axes[3].plot(time, torque_residual, label="torque residual [N m]")
    axes[3].set_ylabel("Dynamic residual")
    axes[3].set_xlabel("Time since incidence [s]")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(ncol=3, fontsize=8)
    fig.suptitle("Contact-wrench time series")
    fig.tight_layout()
    fig.savefig(output / "wrench_timeseries.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for column, interface in enumerate(("left", "right", "table")):
        active = np.array([row[f"{interface}_contact_count"] > 0 for row in rows])
        for row_index, quantity in enumerate(("force", "torque")):
            measured = vector(f"{interface}_measured_{quantity}_world")[active].ravel()
            reference = vector(f"{interface}_reference_{quantity}_world")[active].ravel()
            axis = axes[row_index, column]
            axis.scatter(reference, measured, s=5, alpha=0.35)
            if len(reference):
                low = min(float(reference.min()), float(measured.min()))
                high = max(float(reference.max()), float(measured.max()))
                axis.plot([low, high], [low, high], "k--", linewidth=1)
            axis.set_title(f"{interface}: {quantity}")
            axis.set_xlabel("contact integration")
            axis.set_ylabel("actor query / opposite side")
            axis.grid(True, alpha=0.3)
    fig.suptitle("Measured versus independently aggregated contact wrenches")
    fig.tight_layout()
    fig.savefig(output / "wrench_accuracy.png", dpi=180)
    plt.close(fig)

    incidence_mask = (time >= -0.35) & (time <= 0.75)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(time[incidence_mask], table_force[incidence_mask, 2], label="table Fz")
    axes[0].plot(time[incidence_mask], target[incidence_mask], "k--", label="target")
    axes[0].axvline(0.0, color="tab:red", linestyle=":", label="incidence")
    axes[0].set_ylabel("Force [N]")
    for interface in ("left", "right", "table"):
        counts = np.array([row[f"{interface}_contact_count"] for row in rows])
        axes[1].plot(time[incidence_mask], counts[incidence_mask], label=interface)
    axes[1].set_ylabel("Contact count")
    axes[1].set_xlabel("Time since incidence [s]")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.suptitle("Contact incidence transient")
    fig.tight_layout()
    fig.savefig(output / "contact_incidence.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.worker:
        if args.precision == "both":
            raise ValueError("Worker precision must be single or double")
        run_worker(args)
    else:
        run_coordinator(args)


if __name__ == "__main__":
    main()
