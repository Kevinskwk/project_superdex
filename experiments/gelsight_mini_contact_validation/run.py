#!/usr/bin/env python3
"""Run the FR3 + dual soft GelSight Mini plug/table validation experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import time as wallclock
from typing import Any


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DEFAULT_OUTPUT = HERE / "output" / "baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case-name", default="baseline")
    parser.add_argument("--gel-geometry", choices=['legacy_box','matched_box','source_surface'], default='source_surface')
    parser.add_argument("--youngs-modulus-pa", type=float, default=200_000.0)
    parser.add_argument("--poisson-ratio", type=float, default=0.49)
    parser.add_argument("--mass-damping", type=float, default=5.0)
    parser.add_argument("--stiffness-damping", type=float, default=0.003)
    parser.add_argument("--gripper-roll-deg", type=float, default=0.0)
    parser.add_argument("--gripper-pitch-deg", type=float, default=0.0)
    parser.add_argument("--plug-roll-deg", type=float, default=0.0)
    parser.add_argument("--plug-pitch-deg", type=float, default=0.0)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--max-time", type=float, default=18.0)
    args = parser.parse_args()
    if not 0.0 < args.poisson_ratio < 0.5:
        parser.error("--poisson-ratio must be between 0 and 0.5")
    if min(args.youngs_modulus_pa, args.video_fps, args.width, args.height, args.max_time) <= 0:
        parser.error("positive modulus, video dimensions/FPS, and max time are required")
    return args


def create_box_shape(physics: Any, np: Any, dimensions: Any, dtype: Any) -> Any:
    x, y, z = np.asarray(dimensions) / 2.0
    vertices = np.array(
        [
            [-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
            [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z],
        ], dtype=dtype,
    ).ravel()
    faces = np.array(
        [[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,4,7],[0,7,3],
         [1,2,6],[1,6,5],[0,1,5],[0,5,4],[3,7,6],[3,6,2]],
        dtype=np.int32,
    ).ravel()
    return physics.create_tri_mesh_shape(vertices, faces)


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


def rotation_matrix(transform: Any, Rotation: Any, np: Any) -> Any:
    return Rotation.from_rotvec(
        np.asarray(transform.rotation.to_rotation_vector(), dtype=float)
    ).as_matrix()


def add_wrench(row: dict[str, Any], prefix: str, wrench: Any, append_vector: Any) -> None:
    append_vector(row, f"{prefix}_force_world", wrench.force)
    append_vector(row, f"{prefix}_torque_world", wrench.torque)
    row[f"{prefix}_contact_count"] = wrench.count


def run(args: argparse.Namespace) -> None:
    wall_start = wallclock.perf_counter()
    # FP32 is the selected production precision for this experiment.
    os.environ.pop("SUPERDEX_PRECISION", None)
    os.environ["SUPERDEX_ASSETS_PATH"] = str(PROJECT_ROOT / "assets")
    if not args.no_video:
        os.environ.pop("DISPLAY", None)

    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image, ImageDraw
    from scipy.spatial.transform import Rotation
    import superdex.physics as physics
    import superdex.robotics as robotics

    from soft_gripper import GelMaterial, build_soft_gripper, create_osc
    from tactile_grid import (
        GRID_X,
        GRID_Y,
        render_shear_field,
        surface_displacement,
        world_to_local,
    )
    from wrench_math import (
        aggregate_contact_points,
        append_vector,
        integrate_surface_force_field,
        opposite_at,
    )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    time_step = 1.0 / 100.0
    gravity = np.array([0.0, 0.0, -9.81])
    plug_dimensions = np.array([0.025, 0.030, 0.060])
    plug_mass = 0.08
    table_dimensions = np.array([0.30, 0.30, 0.04])
    table_top_z = 0.25
    grasp_duration = 1.5
    airborne_hold_end = 2.5
    incidence_threshold = 0.2
    np_real = np.float32

    material = GelMaterial(
        geometry=getattr(args, 'gel_geometry', 'source_surface'),
        youngs_modulus_pa=args.youngs_modulus_pa,
        poisson_ratio=args.poisson_ratio,
        mass_damping_s_inv=args.mass_damping,
        stiffness_damping_s=args.stiffness_damping,
    )
    physics.initialize(num_worker_threads=0)
    viewer = None
    writer = None
    query_handles: list[Any] = []
    rows: list[dict[str, Any]] = []
    tactile_times: list[float] = []
    force_grids: dict[str, list[Any]] = {"left": [], "right": []}
    displacements: dict[str, list[Any]] = {"left": [], "right": []}
    try:
        scene = physics.create_scene("FR3 Dual Soft GelSight Mini Validation")
        scene.set_gravity(gravity)
        gripper = build_soft_gripper(scene, material)
        context = robotics.create_context()
        osc = create_osc(context, gripper)
        params = osc.get_params()
        params.kp_p, params.kd_p = 1100.0, 85.0
        params.kp_r, params.kd_r = 40.0, 4.0
        params.max_translation_error, params.max_rotation_error = 0.04, 0.3
        params.b_apply_max_osc_torque_normalization = True
        osc.set_params(params)
        observation = osc.get_current_observations_from_mochi()
        world_from_root = observation.world_from_root
        initial_ee = observation.world_from_ee_link
        initial_ee_position = np.asarray(initial_ee.translation, dtype=float)
        initial_ee_rotation = Rotation.from_rotvec(
            np.asarray(initial_ee.rotation.to_rotation_vector(), dtype=float)
        )

        gel_centers = {}
        for side, gel in gripper.gels.items():
            bounds = gel.get_aabb_world()
            gel_centers[side] = 0.5 * (
                np.asarray(bounds.min, dtype=float) + np.asarray(bounds.max, dtype=float)
            )
        opening_axis = gel_centers["left"] - gel_centers["right"]
        opening_axis /= np.linalg.norm(opening_axis)
        local_z = np.array([0.0, 0.0, 1.0])
        local_x = np.cross(opening_axis, local_z)
        local_x /= np.linalg.norm(local_x)
        base_plug_rotation = Rotation.from_matrix(
            np.column_stack((local_x, opening_axis, local_z))
        )
        plug_delta = Rotation.from_euler(
            "xy", [args.plug_roll_deg, args.plug_pitch_deg], degrees=True
        )
        plug_rotation = base_plug_rotation * plug_delta
        plug_position = 0.5 * (gel_centers["left"] + gel_centers["right"])
        plug_position[2] -= 0.018

        contact = physics.ContactParams()
        contact.coulomb_friction_coefficient = 1.2
        plug = scene.create_rigid_actor(
            name="plug",
            shape=create_box_shape(physics, np, plug_dimensions, np_real),
            mass=plug_mass,
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
        table_center = np.array(
            [plug_position[0], plug_position[1], table_top_z - table_dimensions[2] / 2]
        )
        table = scene.create_rigid_actor(
            name="tabletop",
            shape=create_box_shape(physics, np, table_dimensions, np_real),
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

        num_dofs = gripper.actor.get_num_dofs()
        all_dofs = np.arange(num_dofs, dtype=np.int32)
        gripper_delta = Rotation.from_euler(
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
            direction = plug_position - np.array([1.25, -1.15, 1.0])
            viewer.frame_scene(direction / np.linalg.norm(direction))
            writer = imageio.get_writer(
                output / "simulation.mp4", fps=args.video_fps, codec="libx264",
                quality=8, macro_block_size=16,
            )

        next_video_time = 0.0
        released = False
        incidence_time = None
        incidence_ee_z = None
        retract_start_z = None
        filtered_table_force = 0.0
        latest_grids = {
            "left": np.zeros((GRID_X, GRID_Y, 3), dtype=np.float32),
            "right": np.zeros((GRID_X, GRID_Y, 3), dtype=np.float32),
        }
        latest_phase, latest_target, latest_table = "grasp", 0.0, 0.0

        for _ in range(int(args.max_time / time_step)):
            time = float(scene.get_total_simulation_time())
            if time >= grasp_duration and not released:
                plug.clear_boundary_conditions()
                released = True
            if time < grasp_duration:
                phase = "grasp"
                blend = min(1.0, time / 1.0)
                finger_force = -18.0 * blend * blend * (3.0 - 2.0 * blend)
                target = 0.0
            else:
                finger_force = -18.0
                if time < airborne_hold_end:
                    phase, target = "airborne_hold", 0.0
                elif incidence_time is None:
                    phase, target = "descent", 0.0
                    ee_z_command -= 0.06 * time_step
                else:
                    elapsed = time - incidence_time
                    target, phase = target_force_and_phase(elapsed)
                    if elapsed < 6.0:
                        ee_z_command += float(
                            np.clip(0.012 * (filtered_table_force - target), -0.025, 0.025)
                        ) * time_step
                        ee_z_command = float(np.clip(
                            ee_z_command, incidence_ee_z - 0.045, incidence_ee_z + 0.02
                        ))
                    elif elapsed < 7.0:
                        if retract_start_z is None:
                            retract_start_z = ee_z_command
                        ee_z_command = retract_start_z + 0.08 * (elapsed - 6.0)
                    else:
                        ee_z_command = retract_start_z + 0.08

            if time <= grasp_duration:
                orientation_blend = 0.0
            elif time < airborne_hold_end:
                orientation_blend = (time - grasp_duration) / (airborne_hold_end - grasp_duration)
                orientation_blend = orientation_blend**2 * (3.0 - 2.0 * orientation_blend)
            else:
                orientation_blend = 1.0
            target_rotation = Rotation.from_rotvec(gripper_delta * orientation_blend) * initial_ee_rotation
            world_target = physics.TransformRT(
                translation=[initial_ee_position[0], initial_ee_position[1], ee_z_command],
                rotation=physics.Quaternion.from_rotation_vector(target_rotation.as_rotvec()),
            )
            root_target = world_from_root.inverse() * world_target
            arm_force = np.asarray(
                osc.compute_output(
                    osc.get_current_observations_from_mochi(),
                    robotics.ControllerBasicOscPdTarget(root_from_target_ee=root_target),
                ), dtype=np_real,
            ).copy()
            effort = arm_force
            effort[gripper.finger_dofs["left"]] += finger_force
            effort[gripper.finger_dofs["right"]] += finger_force
            gripper.actor.set_external_forces_on_dofs(all_dofs, effort)
            scene.step(time_step)

            now = float(scene.get_total_simulation_time())
            plug_transform = plug.get_center_of_mass_transform()
            plug_com = np.asarray(plug_transform.translation, dtype=float)
            plug_contacts = list(plug.get_contact_points_world())
            table_wrench = aggregate_contact_points(
                plug_contacts, plug.get_handle(), table.get_handle(), plug_com
            )
            latest_table = max(0.0, float(table_wrench.force[2]))
            filtered_table_force = 0.85 * filtered_table_force + 0.15 * latest_table
            if incidence_time is None and latest_table >= incidence_threshold:
                incidence_time = now
                incidence_ee_z = ee_z_command
                print(f"contact incidence at {now:.3f}s", flush=True)

            row: dict[str, Any] = {
                "time_s": now,
                "phase": phase,
                "case_name": args.case_name,
                "target_table_normal_force_n": target,
                "filtered_table_normal_force_n": filtered_table_force,
                "ee_z_command_m": ee_z_command,
            }
            add_wrench(row, "table_on_plug", table_wrench, append_vector)
            append_vector(row, "plug_com_world", plug_com)
            append_vector(row, "plug_rotation_vector_world", plug_transform.rotation.to_rotation_vector())
            append_vector(row, "plug_linear_velocity_world", plug.get_linear_velocity())
            append_vector(row, "plug_angular_velocity_world", plug.get_angular_velocity())

            gel_on_plug_force = np.zeros(3)
            gel_on_plug_torque = np.zeros(3)
            inferred_on_plug_force = np.zeros(3)
            inferred_on_plug_torque = np.zeros(3)
            field_on_plug_force = np.zeros(3)
            field_on_plug_torque = np.zeros(3)
            grid_force_error = 0.0
            max_indentation = 0.0
            current_displacements = {}
            current_grids = {}
            for side in ("left", "right"):
                gel = gripper.gels[side]
                housing = gripper.links[f"franka_{side}_gelsight_housing"]
                # Sensor coordinates are authored at the link/root frame. The
                # housing COM is intentionally offset along the finger and must
                # not be used for tactile coordinates.
                housing_transform = housing.get_root_transform()
                housing_origin = np.asarray(housing_transform.translation, dtype=float)
                contacts = list(gel.get_contact_points_world())
                measured = aggregate_contact_points(
                    contacts, gel.get_handle(), plug.get_handle(), housing_origin
                )
                on_plug = aggregate_contact_points(
                    plug_contacts, plug.get_handle(), gel.get_handle(), plug_com
                )
                reference = opposite_at(on_plug, plug_com, housing_origin)
                inferred = opposite_at(measured, housing_origin, plug_com)
                add_wrench(row, f"{side}_measured", measured, append_vector)
                add_wrench(row, f"{side}_reference", reference, append_vector)
                gel_on_plug_force += on_plug.force
                gel_on_plug_torque += on_plug.torque
                inferred_on_plug_force += inferred.force
                inferred_on_plug_torque += inferred.torque
                grid = gripper.get_surface_force_field(side)
                dense_positions, dense_forces = gripper.get_dense_contact_field(side)
                append_vector(row, f'{side}_dense_on_plug_force_world', -dense_forces.sum(axis=0))
                append_vector(row, f'{side}_dense_on_plug_torque_world', -np.cross(dense_positions-plug_com,dense_forces).sum(axis=0))
                current_grids[side] = grid
                measured_local = world_to_local(housing_transform, measured.force, vectors=True)
                grid_force_error = max(
                    grid_force_error, float(np.linalg.norm(grid.sum(axis=(0,1)) - measured_local))
                )
                displacement = surface_displacement(
                    gel.get_node_positions_local(),
                    gripper.gel_surface_grid_indices[side],
                    housing_transform, world_from_root,
                    gripper.gel_rest_sensor_surface[side],
                )
                current_displacements[side] = displacement
                field_on_plug = integrate_surface_force_field(
                    grid,
                    gripper.gel_rest_sensor_surface[side] + displacement,
                    housing_transform,
                    plug_com,
                    negate=True,
                )
                add_wrench(row, f"{side}_field_on_plug", field_on_plug, append_vector)
                field_on_plug_force += field_on_plug.force
                field_on_plug_torque += field_on_plug.torque
                indentation = float(np.max(displacement[:, 2])) if len(displacement) else 0.0
                max_indentation = max(max_indentation, indentation)
                row[f"{side}_max_indentation_m"] = indentation
            append_vector(row, "gels_on_plug_force_world", gel_on_plug_force)
            append_vector(row, "gels_on_plug_torque_world", gel_on_plug_torque)
            append_vector(row, "tactile_inferred_gels_on_plug_force_world", inferred_on_plug_force)
            append_vector(row, "tactile_inferred_gels_on_plug_torque_world", inferred_on_plug_torque)
            append_vector(row, "field_gels_on_plug_force_world", field_on_plug_force)
            append_vector(row, "field_gels_on_plug_torque_world", field_on_plug_torque)
            # Quasistatic equilibrium predicts the extrinsic table-on-tool wrench
            # from the dense fingertip field. Gravity acts at the plug COM, so it
            # contributes force but no moment about this comparison origin.
            append_vector(
                row,
                "field_inferred_table_on_plug_force_world",
                -(field_on_plug_force + plug_mass * gravity),
            )
            append_vector(
                row,
                "field_inferred_table_on_plug_torque_world",
                -field_on_plug_torque,
            )
            row["grid_force_conservation_error_n"] = grid_force_error
            row["max_gel_indentation_m"] = max_indentation
            rows.append(row)
            tactile_times.append(now)
            for side in ("left", "right"):
                force_grids[side].append(current_grids[side])
                displacements[side].append(current_displacements[side])
            latest_grids = current_grids
            latest_phase, latest_target = phase, target

            if writer is not None and now + 1e-9 >= next_video_time:
                frame = viewer.render()
                if frame is None:
                    raise RuntimeError("offscreen renderer returned no frame")
                image = Image.fromarray(np.asarray(frame)[..., :3])
                draw = ImageDraw.Draw(image)
                draw.rectangle((10, 10, 455, 108), fill=(255,255,255,230))
                draw.text((20,18), f"phase: {latest_phase}", fill=(10,10,10))
                draw.text((20,41), f"target/table: {latest_target:5.2f}/{latest_table:5.2f} N", fill=(10,10,10))
                draw.text((20,64), f"soft GelSight E={args.youngs_modulus_pa/1000:g} kPa", fill=(10,10,10))
                draw.text((20,87), f"case: {args.case_name}", fill=(10,10,10))
                for index, side in enumerate(("left", "right")):
                    field = latest_grids[side]
                    max_shear = float(np.linalg.norm(field[..., :2], axis=2).max())
                    max_normal = float(np.abs(field[..., 2]).max())
                    panel = Image.fromarray(
                        render_shear_field(
                            field,
                            shear_scale_n=max(0.02, max_shear),
                            normal_scale_n=max(0.02, max_normal),
                        )
                    )
                    x = args.width - 154
                    y = 10 + index * 218
                    image.paste(panel, (x, y))
                    draw.text(
                        (x, y + 194),
                        f"{side} shear {max_shear:.2f} / normal {max_normal:.2f} N",
                        fill=(255,255,255),
                    )
                writer.append_data(np.asarray(image))
                next_video_time += 1.0 / args.video_fps

            if incidence_time is None and now >= args.max_time - time_step:
                raise RuntimeError("plug did not contact the tabletop before timeout")
            if incidence_time is not None and now - incidence_time >= 7.5:
                break

        if incidence_time is None:
            raise RuntimeError("no plug/table contact incidence recorded")
        finalize_results(
            rows, tactile_times, force_grids, displacements,
            gripper.gel_rest_sensor_surface, output, args, material,
            incidence_time, plug_mass, gravity, time_step, np, plt,
        )
        (output / 'runtime.json').write_text(json.dumps({'wall_s_including_plots_and_rendering': wallclock.perf_counter()-wall_start, 'frames':len(rows)},indent=2)+'\n')
        print(f"wrote {len(rows)} FP32 samples to {output}", flush=True)
    finally:
        if writer is not None:
            writer.close()
        if viewer is not None:
            viewer.close()
        physics.shutdown()


def finalize_results(
    rows: list[dict[str, Any]], tactile_times: list[float], force_grids: dict[str,list[Any]],
    displacements: dict[str,list[Any]], surface_rest: dict[str, Any],
    output: Path, args: argparse.Namespace,
    material: Any, incidence_time: float, plug_mass: float, gravity: Any,
    time_step: float, np: Any, plt: Any,
) -> None:
    from wrench_math import append_vector
    for row in rows:
        row["time_since_incidence_s"] = row["time_s"] - incidence_time
        contact_force = np.array([row[f"gels_on_plug_force_world_{a}"] + row[f"table_on_plug_force_world_{a}"] for a in "xyz"])
        append_vector(row, "quasistatic_force_residual_world", contact_force + plug_mass * gravity)
        contact_torque = np.array([row[f"gels_on_plug_torque_world_{a}"] + row[f"table_on_plug_torque_world_{a}"] for a in "xyz"])
        append_vector(row, "quasistatic_torque_residual_world", contact_torque)
    with (output / "contact_wrenches.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    left_force_field = np.asarray(force_grids["left"], dtype=np.float32)
    right_force_field = np.asarray(force_grids["right"], dtype=np.float32)

    def wrench_array(prefix: str) -> Any:
        return np.asarray(
            [
                [row[f"{prefix}_force_world_{axis}"] for axis in "xyz"]
                + [row[f"{prefix}_torque_world_{axis}"] for axis in "xyz"]
                for row in rows
            ],
            dtype=np.float32,
        )

    np.savez_compressed(
        output / "tactile_fields.npz",
        time_s=np.asarray(tactile_times, dtype=np.float32),
        left_force_field=left_force_field,
        right_force_field=right_force_field,
        # Compatibility aliases for results produced by earlier versions.
        left_force_grid=left_force_field,
        right_force_grid=right_force_field,
        left_surface_displacement=np.asarray(displacements["left"], dtype=np.float32),
        right_surface_displacement=np.asarray(displacements["right"], dtype=np.float32),
        left_surface_rest_sensor_m=np.asarray(surface_rest["left"], dtype=np.float32),
        right_surface_rest_sensor_m=np.asarray(surface_rest["right"], dtype=np.float32),
        field_gels_on_plug_wrench_world=wrench_array("field_gels_on_plug"),
        field_inferred_table_on_plug_wrench_world=wrench_array(
            "field_inferred_table_on_plug"
        ),
        extrinsic_table_on_plug_wrench_world=wrench_array("table_on_plug"),
    )
    metrics = compute_metrics(rows, plug_mass, np)
    metrics.update({
        "precision": "fp32", "sample_rate_hz": int(round(1.0 / time_step)), "incidence_time_s": incidence_time,
        "material": material.__dict__, "case_name": args.case_name,
        "field_representation": "legacy_surface_nodes" if material.geometry == 'legacy_box' else "7x9_nearest_marker_force_bins; torque approximate, use dense nodal wrench",
    })
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    config = {
        "robot": "FR3 v2 + Franka hand + dual HydroShear GelSight Mini",
        "precision": "fp32", "time_step_s": time_step,
        "plug_mass_kg": plug_mass, "force_plateaus_n": [2,5,10],
        "material": material.__dict__,
        "orientation_degrees": {
            "gripper_roll": args.gripper_roll_deg, "gripper_pitch": args.gripper_pitch_deg,
            "plug_roll": args.plug_roll_deg, "plug_pitch": args.plug_pitch_deg,
        },
        "tactile_force_field": {
            "shape": [7, 9, 3],
            "axes": ["sensor_x", "sensor_y", "sensor_xyz"],
            "frame": "gelsight_housing_root",
            "units": "N",
        },
        "integrated_wrench": {
            "field_force_sign": "equal-and-opposite force applied to plug",
            "origin": "instantaneous plug center of mass",
            "frame": "world",
            "force_units": "N",
            "torque_units": "N m",
            "extrinsic_prediction": "quasistatic: -(field gel wrench + gravity)",
        },
        "video_enabled": not args.no_video,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    make_plots(rows, force_grids, output, np, plt)


def compute_metrics(rows: list[dict[str, Any]], plug_mass: float, np: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"interfaces": {}, "plateaus": {}, "execution_checks": {}}

    def vectors(selected: list[dict[str, Any]], prefix: str, quantity: str) -> Any:
        return np.asarray(
            [
                [row[f"{prefix}_{quantity}_world_{axis}"] for axis in "xyz"]
                for row in selected
            ]
        )

    def vector_error(reference: Any, estimate: Any) -> dict[str, float]:
        error = estimate - reference
        rmse = float(np.sqrt(np.mean(np.sum(error**2, axis=1))))
        reference_rms = float(np.sqrt(np.mean(np.sum(reference**2, axis=1))))
        return {"vector_rmse": rmse, "relative_rmse": rmse / max(reference_rms, 1e-12)}
    for side in ("left", "right"):
        active = np.array([r[f"{side}_measured_contact_count"] > 0 for r in rows])
        side_metrics = {"active_samples": int(active.sum())}
        for quantity in ("force", "torque"):
            measured = np.array([[r[f"{side}_measured_{quantity}_world_{a}"] for a in "xyz"] for r in rows])[active]
            reference = np.array([[r[f"{side}_reference_{quantity}_world_{a}"] for a in "xyz"] for r in rows])[active]
            if len(measured):
                error = measured - reference
                rmse = float(np.sqrt(np.mean(np.sum(error**2, axis=1))))
                rms = float(np.sqrt(np.mean(np.sum(reference**2, axis=1))))
                side_metrics[quantity] = {"vector_rmse": rmse, "relative_rmse": rmse/max(rms,1e-12)}
        result["interfaces"][side] = side_metrics
    holds = (("hold_2N",2.0),("hold_5N",5.0),("hold_10N",10.0))
    for phase, target in holds:
        selected = [r for r in rows if r["phase"] == phase]
        table = np.array([r["table_on_plug_force_world_z"] for r in selected])
        tactile = np.array([r["left_measured_force_world_z"] + r["right_measured_force_world_z"] + plug_mass*9.81 for r in selected])
        table_torque = np.array([[r[f"table_on_plug_torque_world_{a}"] for a in "xyz"] for r in selected])
        tactile_torque = -np.array([[r[f"tactile_inferred_gels_on_plug_torque_world_{a}"] for a in "xyz"] for r in selected])
        field_table_force = vectors(
            selected, "field_inferred_table_on_plug", "force"
        )
        field_table_torque = vectors(
            selected, "field_inferred_table_on_plug", "torque"
        )
        table_force = vectors(selected, "table_on_plug", "force")
        result["plateaus"][phase] = {
            "target_n": target, "mean_n": float(table.mean()), "std_n": float(table.std()),
            "mean_tracking_error_n": float(table.mean()-target),
            "indirect_rmse_n": float(np.sqrt(np.mean((table-tactile)**2))),
            "indirect_correlation": float(np.corrcoef(table,tactile)[0,1]) if len(table)>1 else None,
            "indirect_torque_rmse_nm": float(np.sqrt(np.mean(np.sum((table_torque-tactile_torque)**2,axis=1)))),
            "field_extrinsic_force_vector_rmse_n": vector_error(
                table_force, field_table_force
            )["vector_rmse"],
            "field_extrinsic_torque_vector_rmse_nm": vector_error(
                table_torque, field_table_torque
            )["vector_rmse"],
            "both_gels_contact_fraction": float(np.mean([r["left_measured_contact_count"]>0 and r["right_measured_contact_count"]>0 for r in selected])),
        }
    load_rows = [r for r in rows if r["phase"].startswith("ramp_") or r["phase"].startswith("hold_")]
    extrinsic_force = vectors(load_rows, "table_on_plug", "force")
    extrinsic_torque = vectors(load_rows, "table_on_plug", "torque")
    field_extrinsic_force = vectors(
        load_rows, "field_inferred_table_on_plug", "force"
    )
    field_extrinsic_torque = vectors(
        load_rows, "field_inferred_table_on_plug", "torque"
    )
    direct_gel_force = vectors(load_rows, "gels_on_plug", "force")
    direct_gel_torque = vectors(load_rows, "gels_on_plug", "torque")
    field_gel_force = vectors(load_rows, "field_gels_on_plug", "force")
    field_gel_torque = vectors(load_rows, "field_gels_on_plug", "torque")
    result["dense_field_wrench"] = {
        "field_vs_direct_gel_force": vector_error(direct_gel_force, field_gel_force),
        "field_vs_direct_gel_torque": vector_error(direct_gel_torque, field_gel_torque),
        "inferred_vs_extrinsic_force": vector_error(
            extrinsic_force, field_extrinsic_force
        ),
        "inferred_vs_extrinsic_torque": vector_error(
            extrinsic_torque, field_extrinsic_torque
        ),
        "extrinsic_normal_force_correlation": float(
            np.corrcoef(extrinsic_force[:, 2], field_extrinsic_force[:, 2])[0, 1]
        ),
    }
    table = np.array([r["table_on_plug_force_world_z"] for r in load_rows])
    tactile = np.array([r["left_measured_force_world_z"] + r["right_measured_force_world_z"] + plug_mass*9.81 for r in load_rows])
    first, second = np.diff(table), np.diff(tactile)
    lag_candidates = range(-5, 6)
    lag_correlations = []
    for lag in lag_candidates:
        if lag < 0:
            a, b = first[-lag:], second[:lag]
        elif lag > 0:
            a, b = first[:-lag], second[lag:]
        else:
            a, b = first, second
        lag_correlations.append(float(np.corrcoef(a,b)[0,1]))
    best_index = int(np.nanargmax(lag_correlations))
    best_lag = list(lag_candidates)[best_index]
    result["causality"] = {
        "force_correlation": float(np.corrcoef(table,tactile)[0,1]),
        "best_derivative_lag_samples": best_lag,
        "best_derivative_lag_s": best_lag * 0.01,
        "best_derivative_correlation": lag_correlations[best_index],
        "max_indentation_m": float(max(r["max_gel_indentation_m"] for r in rows)),
        "peak_grid_conservation_error_n": float(max(r["grid_force_conservation_error_n"] for r in rows)),
    }
    result["execution_checks"] = {
        "airborne_both_gels_contact_fraction": float(np.mean([r["left_measured_contact_count"]>0 and r["right_measured_contact_count"]>0 for r in rows if r["phase"]=="airborne_hold"])),
        "table_contact_during_airborne_fraction": float(np.mean([r["table_on_plug_contact_count"]>0 for r in rows if r["phase"]=="airborne_hold"])),
    }
    return result


def make_plots(rows: list[dict[str, Any]], grids: dict[str,list[Any]], output: Path, np: Any, plt: Any) -> None:
    time = np.array([r["time_since_incidence_s"] for r in rows])
    target = np.array([r["target_table_normal_force_n"] for r in rows])
    table = np.array([r["table_on_plug_force_world_z"] for r in rows])
    left = np.array([r["left_measured_force_world_z"] for r in rows])
    right = np.array([r["right_measured_force_world_z"] for r in rows])
    indentation = 1000*np.array([r["max_gel_indentation_m"] for r in rows])
    residual_f = np.array([[r[f"quasistatic_force_residual_world_{a}"] for a in "xyz"] for r in rows])
    residual_t = np.array([[r[f"quasistatic_torque_residual_world_{a}"] for a in "xyz"] for r in rows])
    fig, axes = plt.subplots(4,1,figsize=(13,13),sharex=True)
    axes[0].plot(time,target,"k--",label="target"); axes[0].plot(time,table,label="table Fz")
    axes[1].plot(time,left,label="left gel Fz"); axes[1].plot(time,right,label="right gel Fz")
    axes[2].plot(time,indentation,label="maximum gel indentation")
    axes[3].plot(time,np.linalg.norm(residual_f,axis=1),label="force residual [N]")
    axes[3].plot(time,np.linalg.norm(residual_t,axis=1),label="torque residual [N m]")
    for axis,label in zip(axes,("Force [N]","Force [N]","Indentation [mm]","Residual")):
        axis.set_ylabel(label); axis.grid(alpha=.3); axis.legend()
    axes[-1].set_xlabel("Time since table incidence [s]")
    fig.suptitle("Dual soft GelSight contact-wrench validation"); fig.tight_layout()
    fig.savefig(output/"wrench_timeseries.png",dpi=180); plt.close(fig)

    fig, axes = plt.subplots(2,2,figsize=(11,10))
    for col, side in enumerate(("left","right")):
        active = np.array([r[f"{side}_measured_contact_count"]>0 for r in rows])
        for row_index, quantity in enumerate(("force","torque")):
            m=np.array([[r[f"{side}_measured_{quantity}_world_{a}"] for a in "xyz"] for r in rows])[active].ravel()
            q=np.array([[r[f"{side}_reference_{quantity}_world_{a}"] for a in "xyz"] for r in rows])[active].ravel()
            axes[row_index,col].scatter(q,m,s=4,alpha=.3)
            if len(q):
                lo=min(q.min(),m.min()); hi=max(q.max(),m.max()); axes[row_index,col].plot([lo,hi],[lo,hi],"k--")
            axes[row_index,col].set_title(f"{side} {quantity}"); axes[row_index,col].grid(alpha=.3)
            axes[row_index,col].set_xlabel("plug-side reference"); axes[row_index,col].set_ylabel("gel-side tactile")
    fig.tight_layout(); fig.savefig(output/"wrench_accuracy.png",dpi=180); plt.close(fig)

    peak = int(np.argmax(table))
    fig, axes = plt.subplots(1,2,figsize=(10,5))
    for axis,side in zip(axes,("left","right")):
        image=np.linalg.norm(np.asarray(grids[side][peak]),axis=2)
        shown=axis.imshow(image,origin="lower",cmap="inferno",aspect="auto")
        axis.set_title(f"{side} gel force magnitude"); fig.colorbar(shown,ax=axis,label="N/cell")
    fig.tight_layout(); fig.savefig(output/"tactile_force_maps.png",dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    grid_x, grid_y = np.meshgrid(np.arange(7), np.arange(9), indexing="ij")
    for axis, side in zip(axes, ("left", "right")):
        field = np.asarray(grids[side][peak])
        max_shear = max(1e-9, float(np.linalg.norm(field[..., :2], axis=2).max()))
        arrows = axis.quiver(
            grid_x,
            grid_y,
            field[..., 0],
            field[..., 1],
            np.abs(field[..., 2]),
            cmap="winter_r",
            angles="xy",
            scale_units="xy",
            scale=max_shear / 0.7,
            pivot="mid",
        )
        axis.set_aspect("equal")
        axis.set_xlim(-0.6, 6.6)
        axis.set_ylim(-0.6, 8.6)
        axis.set_xlabel("sensor X node")
        axis.set_ylabel("sensor Y node")
        axis.set_title(f"{side}: arrows=(Fx,Fy), color=|Fz|")
        axis.grid(alpha=0.2)
        fig.colorbar(arrows, ax=axis, label="normal force magnitude [N/node]")
    fig.suptitle("HydroShear-style dense fingertip shear field at peak table load")
    fig.tight_layout()
    fig.savefig(output / "tactile_shear_fields.png", dpi=180)
    plt.close(fig)

    extrinsic_force = np.array(
        [[r[f"table_on_plug_force_world_{axis}"] for axis in "xyz"] for r in rows]
    )
    extrinsic_torque = np.array(
        [[r[f"table_on_plug_torque_world_{axis}"] for axis in "xyz"] for r in rows]
    )
    field_force = np.array(
        [
            [r[f"field_inferred_table_on_plug_force_world_{axis}"] for axis in "xyz"]
            for r in rows
        ]
    )
    field_torque = np.array(
        [
            [r[f"field_inferred_table_on_plug_torque_world_{axis}"] for axis in "xyz"]
            for r in rows
        ]
    )
    contact_window = time >= 0.0
    comparison_time = time[contact_window]
    extrinsic_force = extrinsic_force[contact_window]
    extrinsic_torque = extrinsic_torque[contact_window]
    field_force = field_force[contact_window]
    field_torque = field_torque[contact_window]
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    for component, axis_name in enumerate("xyz"):
        axes[component, 0].plot(comparison_time, extrinsic_force[:, component], label="extrinsic plug-table")
        axes[component, 0].plot(comparison_time, field_force[:, component], "--", label="from 7x9x3 field")
        axes[component, 0].set_ylabel(f"F{axis_name} [N]")
        axes[component, 1].plot(comparison_time, extrinsic_torque[:, component], label="extrinsic plug-table")
        axes[component, 1].plot(comparison_time, field_torque[:, component], "--", label="from 7x9x3 field")
        axes[component, 1].set_ylabel(f"T{axis_name} [N m]")
        for axis in axes[component]:
            axis.grid(alpha=0.3)
            axis.legend(loc="best")
    axes[0, 0].set_title("Force about plug COM")
    axes[0, 1].set_title("Torque about plug COM")
    axes[-1, 0].set_xlabel("Time since table incidence [s]")
    axes[-1, 1].set_xlabel("Time since table incidence [s]")
    fig.suptitle("Dense tactile-field wrench vs extrinsic plug-table wrench")
    fig.tight_layout()
    fig.savefig(output / "field_wrench_vs_extrinsic.png", dpi=180)
    plt.close(fig)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
