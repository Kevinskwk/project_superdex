"""Rail-free, printable loaded tray with an integral pulling loop.

The body has six free DOFs. Ballast is represented by effective rigid-body mass;
there are no wheels, hidden guide forces, welds, or object-pose tracking servos.
"""

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from decisions import box_union, interpolate
from pilot import smooth
from transfer_world import TransferWorld


def straight_hook_mesh(width=0.018, depth=0.024):
    return box_union(
        [
            ([width, depth, 0.050], [0, 0, 0]),
            ([0.010, 0.010, 0.055], [0, 0, -0.0475]),
            ([0.043, 0.010, 0.008], [0.0165, 0, -0.078]),
            ([0.008, 0.010, 0.024], [0.034, 0, -0.066]),
        ]
    )


def loaded_tray_mesh(loop_height):
    """Loop crossbar at origin; tray behind it, not underneath the hook mouth."""
    floor = -loop_height
    return box_union(
        [
            ([0.120, 0.100, 0.006], [0.090, 0, floor + 0.003]),
            ([0.006, 0.100, 0.020], [0.147, 0, floor + 0.012]),
            ([0.006, 0.100, 0.020], [0.033, 0, floor + 0.012]),
            ([0.120, 0.006, 0.020], [0.090, -0.047, floor + 0.012]),
            ([0.120, 0.006, 0.020], [0.090, 0.047, floor + 0.012]),
            ([0.008, 0.064, 0.008], [0, 0, 0]),
            ([0.008, 0.008, loop_height], [0, -0.028, -loop_height / 2]),
            ([0.008, 0.008, loop_height], [0, 0.028, -loop_height / 2]),
            ([0.044, 0.008, 0.006], [0.018, -0.028, floor + 0.003]),
            ([0.044, 0.008, 0.006], [0.018, 0.028, floor + 0.003]),
        ]
    )


def loaded_box_metrics(data, spec):
    """Actual body load and motion; force quality is diagnostic, not an RMSE gate."""
    active = ~data["initialization"]
    pull = (data["phase"] == "pull") & active
    force = data["box_contact_force_world_n"]
    horizontal = np.linalg.norm(force[:, :2], axis=1)
    contact = pull & (horizontal > 0.1)
    return dict(
        final_pull_mm=float(data["task_progress"][-1] * 1000),
        max_box_tilt_deg=float(np.rad2deg(data["object_tilt_rad"][active]).max()),
        max_cross_track_mm=float(data["guide_drift_m"][active].max() * 1000),
        horizontal_force_median_n=float(np.median(horizontal[contact]))
        if contact.any()
        else None,
        vertical_force_p95_n=float(np.quantile(np.abs(force[contact, 2]), 0.95))
        if contact.any()
        else None,
        vertical_force_peak_n=float(np.abs(force[active, 2]).max()),
        vertical_force_p95_weight_fraction=float(
            np.quantile(np.abs(force[contact, 2]), 0.95) / (spec.sled_mass_kg * 9.81)
        )
        if contact.any()
        else None,
        nominal_slide_resistance_n=spec.sled_mass_kg * 9.81 * spec.support_friction,
        note="Nominal mu*m*g is a reference, not an exact dynamic-force target. Vertical force is on the box, not the tool's gravity load.",
    )


class LoadedBoxWorld(TransferWorld):
    def make_tool_mesh(self):
        mesh = straight_hook_mesh(self.spec.handle_width_m, self.spec.handle_depth_m)
        authored_to_task = self.rot.inv() * self.grasp_rot
        mesh.vertices = authored_to_task.apply(mesh.vertices)
        self.loop_center_task = authored_to_task.apply([0.022, 0, -0.064])
        self.floor_z = min(
            float(mesh.bounds[0, 2]) - 0.020, float(self.loop_center_task[2]) - 0.026
        )
        if self.spec.box_loop_height_m is not None:
            self.floor_z = float(self.loop_center_task[2] - self.spec.box_loop_height_m)
        return self.contact_mesh(mesh)

    def build_environment(self):
        self.env, self.environment_meshes = [], []
        self.axis = self.rot.apply([-1, 0, 0])
        cross = self.loop_center_task.copy()
        mesh = loaded_tray_mesh(cross[2] - self.floor_z)
        self.slider = self.add_body(
            "loaded_box_with_loop",
            self.contact_mesh(mesh),
            cross,
            self.spec.sled_mass_kg,
            False,
        )
        # Front edge keeps a sideways lower finger outside the tabletop while
        # retaining support under the tray through the complete 54 mm pull.
        self.add_body(
            "level_table",
            box_union([([0.34, 0.40, 0.020], [0, 0, 0])]),
            [cross[0] + 0.13, 0, self.floor_z - 0.010],
        )
        self.slider0 = np.asarray(
            self.slider.get_center_of_mass_transform().translation
        ).copy()
        self.slider_com0 = self.slider0.copy()
        self.table_top = float((self.origin + self.rot.apply([0, 0, self.floor_z]))[2])
        self.support_queries = [
            a.register_query(self.p.QueryType.CONTACT_POINTS) for a in self.env
        ]
        self.queries_extra = self.support_queries

    def __init__(self, spec):
        super().__init__(spec)
        self.preflight_clearance()
        self.vertical_relief_m = 0.0
        self.last_vertical_force_n = 0.0
        self.branch_value = None
        self.pull_tip_start_task = None
        self.camera_target_world = self.origin + self.rot.apply(
            self.loop_center_task + [0.030, 0, -0.005]
        )

    def preflight_clearance(self):
        """Reject initial loop/tool or actual housing overlap before rollout."""
        from soft_gripper import GELSIGHT_ROOT

        objects = [(self.tool, self.mesh), *zip(self.env, self.environment_meshes)]
        housing = trimesh.load(
            GELSIGHT_ROOT / "generated/housing_visual.glb", force="mesh"
        )
        sources = [(a, m.vertices) for a, m in objects]
        sources += [
            (self.gripper.links[f"franka_{s}_gelsight_housing"], housing.vertices)
            for s in ("left", "right")
        ]
        peak = 0.0
        for source, vertices in sources:
            pose = source.get_root_transform()
            points = Rotation.from_rotvec(
                np.asarray(pose.rotation.to_rotation_vector())
            ).apply(vertices) + np.asarray(pose.translation)
            for target, mesh in objects:
                if target.get_handle() == source.get_handle():
                    continue
                pose = target.get_root_transform()
                local = (
                    Rotation.from_rotvec(np.asarray(pose.rotation.to_rotation_vector()))
                    .inv()
                    .apply(points - np.asarray(pose.translation))
                )
                candidate = local[
                    np.all(
                        (local >= mesh.bounds[0]) & (local <= mesh.bounds[1]), axis=1
                    )
                ]
                inside = (
                    candidate[mesh.contains(candidate)] if len(candidate) else candidate
                )
                if len(inside):
                    _, distance, _ = trimesh.proximity.closest_point(mesh, inside)
                    peak = max(peak, float(distance.max()))
        self.initial_sampled_overlap_m = peak
        if peak > 0.0001:
            raise ValueError(
                f"Prepared geometry overlap {peak * 1000:.3f} mm; do not run contact qualification"
            )

    def checkpoint(self):
        state, control = super().checkpoint()
        control["pull_tip_start_task"] = (
            None
            if self.pull_tip_start_task is None
            else self.pull_tip_start_task.copy()
        )
        return state, control

    def command_at(self, t, branch="nominal", anchor_command=None):
        tau = self.clock(t)
        pull = self.spec.hook_pull_m
        if branch not in ("nominal", "pull", "hold", "retract"):
            raise ValueError("Loaded box supports nominal/pull/hold/retract")
        # The hook starts inside the loop aperture, with millimetre clearance.
        # No lifting engagement stroke: only horizontal contact acquisition.
        value, phase = interpolate(
            [
                (0, [0, 0, 0], "grasp"),
                (4, [0, 0, 0], "engage"),
                (7, [-0.008, 0, 0], "pull"),
                (18, [-pull, 0, 0], "hold"),
                (22, [-pull, 0, 0], "done"),
            ],
            tau,
        )
        if branch in ("hold", "retract") and t >= self.spec.branch_time:
            if self.branch_value is None:
                self.branch_value = value.copy()
            value = self.branch_value.copy()
            if branch == "retract":
                value[0] += 0.006 * smooth((t - self.spec.branch_time) / 3)
            phase = "branch_" + branch
        # Explicit collector-only wrench feedback relieves vertical loading;
        # never use object pose to artificially keep the tray flat or on path.
        if tau >= 4:
            self.vertical_relief_m = float(
                np.clip(
                    self.vertical_relief_m
                    + self.spec.dt
                    * np.clip(0.002 * self.last_vertical_force_n, -0.0005, 0.0005),
                    -0.006,
                    0.006,
                )
            )
        value[2] += self.vertical_relief_m
        if t >= 1.5 and self.calibration is None:
            self.calibration = self.rot.inv().apply(
                np.asarray(self.tool.get_root_transform().translation) - self.origin
            )
        if self.calibration is not None:
            value -= np.clip(self.calibration, -0.006, 0.006) * smooth((t - 1.5) / 1.5)
        if self.spec.tool_pose_feedback and tau >= 4:
            pose = self.tool.get_root_transform()
            tip = Rotation.from_rotvec(
                np.asarray(pose.rotation.to_rotation_vector())
            ).apply(self.loop_center_task) + np.asarray(pose.translation)
            tip = self.rot.inv().apply(tip - self.origin)
            if self.pull_tip_start_task is None:
                self.pull_tip_start_task = tip.copy()
            # Compare working-end travel to the horizontal stroke, not wrist
            # translation. Privileged collector only; no lateral pose servo or
            # box-pose tracking. Small bounded correction preserves impedance.
            actual = tip[0] - self.pull_tip_start_task[0]
            target = value[0] + (
                0
                if self.calibration is None
                else np.clip(self.calibration[0], -0.006, 0.006)
            )
            self.tracking_correction[0] = np.clip(
                self.tracking_correction[0]
                + self.spec.dt * np.clip(4 * (target - actual), -0.001, 0.001),
                -0.012,
                0.012,
            )
            value[0] += self.tracking_correction[0]
        self.nominal_tool_target = value.copy()
        self.trajectory_time = tau
        return value, phase, np.zeros(3)

    def step(self, t, branch="nominal", anchor_command=None):
        row = super().step(t, branch, anchor_command)
        force = self.rot.inv().apply(row["extrinsic_contact_wrench"][:3])
        self.last_vertical_force_n = float(force[2])
        row["collector_vertical_relief_m"] = self.vertical_relief_m
        row["tool_contact_force_task_n"] = force
        row["collector_pull_tip_anchor_task_m"] = (
            np.zeros(3)
            if self.pull_tip_start_task is None
            else self.pull_tip_start_task.copy()
        )
        row["box_contact_force_world_n"] = -row["tool_environment_pair_wrenches"][0, :3]
        return row
