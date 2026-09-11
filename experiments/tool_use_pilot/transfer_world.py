"""Physical-contact sled/pushing and gravity-loaded lever transfer candidates.

No fabricated hardware is claimed. Free objects have six unconstrained DOFs.
"""

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from benchmark import BenchmarkMixin
from decisions import HookDecision, box_union, interpolate
from pilot import smooth
from key_stages import KeyStageWorld


def block(size, center):
    mesh = trimesh.creation.box(size)
    mesh.apply_translation(center)
    return mesh


def sled_geometry():
    # Crossbar at local origin; broad ballast-compatible base and two short posts.
    return box_union(
        [
            ([0.080, 0.060, 0.006], [0, 0, -0.053]),
            ([0.012, 0.008, 0.052], [0, -0.026, -0.025]),
            ([0.012, 0.008, 0.052], [0, 0.026, -0.025]),
            ([0.012, 0.060, 0.008], [0, 0, 0]),
        ]
    )


def t_block_geometry():
    return box_union(
        [
            ([0.030, 0.080, 0.015], [-0.025, 0, 0]),
            ([0.100, 0.025, 0.015], [0.010, 0, 0]),
        ]
    )


def spatula_material_factors(gel_tool, blade_object, support_object):
    """Factor requested PAIR coefficients for the engine's geometric-mean rule.

    Factors are not measured standalone-material coefficients. Preserve gel grip
    while distinguishing the smooth working blade from a high-friction handle.
    """
    return dict(
        tool=blade_object,
        pancake=blade_object,
        gel=gel_tool**2 / blade_object,
        surface=support_object**2 / blade_object,
    )


def sled_support_layout(cross, clearance):
    floor = cross[2] - 0.056
    layout = [("sled_base", [0.200, 0.090, 0.010], [-0.015, cross[1], floor - 0.005])]
    for sign in (-1, 1):
        layout += [
            (
                "clearance_rail",
                [0.200, 0.008, 0.012],
                [-0.015, cross[1] + sign * (0.034 + clearance), floor + 0.006],
            ),
            (
                "travel_stop",
                [0.008, 0.090, 0.012],
                [cross[0] + (0.046 if sign == 1 else -0.120), cross[1], floor + 0.006],
            ),
        ]
    return layout


def transfer_audit(data, spec, base):
    active = ~data["initialization"]
    n = max(1, round(0.3 / spec.dt))
    end = data["phase"][-1] == "done"
    tipped = (
        bool(np.max(data["object_tilt_rad"][active]) > np.deg2rad(10))
        if spec.family != "levering" or spec.variant == "spatula_lift"
        else False
    )
    escaped = bool(np.max(data["support_escape"][active]))
    # Legacy guide drift <1mm applies to an ideal rail, not this loose sled.
    finite = all(np.isfinite(v).all() for v in data.values() if v.dtype.kind in "fc")
    valid = bool(
        finite
        and base["grasp_retained"]
        and not base["sustained_penetration"]
        and base["min_gel_jacobian"] > 0
        and not tipped
        and not escaped
    )
    if spec.variant == "spatula_lift":
        goal = (data["task_progress"] >= 0.030) & (
            data["object_position_error_m"] < 0.035
        )
    elif spec.family == "levering":
        goal = data["flap_angle_rad"] >= np.deg2rad(20)
    elif spec.family == "pushing":
        goal = (data["object_position_error_m"] <= 0.010) & (
            np.abs(data["object_yaw_error_rad"]) <= np.deg2rad(5)
        )
    else:
        goal = data["task_progress"] >= 0.040
    held = len(goal) >= n and bool(np.all(goal[-n:]))
    return dict(
        task=spec.family,
        physical_valid=valid,
        task_success=end and held and valid,
        object_tipped=tipped,
        support_escape=escaped,
        goal_held=held,
        sequence_complete=end and held,
        final_position_error_m=float(data["object_position_error_m"][-1]),
        final_yaw_error_deg=float(np.rad2deg(data["object_yaw_error_rad"][-1])),
        final_flap_angle_deg=float(np.rad2deg(data["flap_angle_rad"][-1])),
        max_flap_angle_deg=float(np.rad2deg(data["flap_angle_rad"]).max()),
        final_object_lift_m=float(data["task_progress"][-1])
        if spec.variant == "spatula_lift"
        else None,
        mechanism="free_pancake_scoop_lift"
        if spec.variant == "spatula_lift"
        else "gravity_hinge"
        if spec.family == "levering"
        else "free_body_contact_friction",
    )


class TransferWorld(BenchmarkMixin, HookDecision):
    # Use the validated small contact transition, not the legacy 10mm default.
    contact_params = KeyStageWorld.contact_params

    def actor(self, name, mesh, pos, rot, mass, friction, static=False, gravity=True):
        if not (
            static
            and len(mesh.vertices) == 8
            and len(mesh.faces) == 12
            and self.spec.transfer_support_collider == "analytic_box"
        ):
            return KeyStageWorld.actor(
                self, name, mesh, pos, rot, mass, friction, static, gravity
            )
        p = self.p
        shape = p.create_tri_mesh_shape(
            np.asarray(mesh.vertices, np.float32).ravel(),
            np.asarray(mesh.faces, np.int32).ravel(),
        )
        actor = self.scene.create_rigid_actor(
            name=name,
            shape=shape,
            mass=mass,
            contact=self.contact_params(friction),
            collider_type=p.ColliderType.BOX,
            has_gravity=False,
            world_from_local=p.TransformRT(
                translation=pos,
                rotation=p.Quaternion.from_rotation_vector(rot.as_rotvec()),
            ),
        )
        com = actor.get_center_of_mass_transform()
        actor.add_boundary_condition_dofs_world(
            np.arange(6, dtype=np.int32),
            np.r_[np.asarray(com.translation), rot.as_rotvec()],
        )
        return actor

    def prepared_frame(self, centers, default_rot, default_origin):
        rot, origin = super().prepared_frame(centers, default_rot, default_origin)
        if self.spec.family == "pushing":
            rot = rot * Rotation.from_euler(
                "z", self.spec.push_heading_deg, degrees=True
            )
        elif self.spec.family == "hook":
            rot = rot * Rotation.from_euler(
                "z", self.spec.hook_heading_deg, degrees=True
            )
        return rot, origin

    def __init__(self, spec):
        super().__init__(spec)
        if spec.variant == "spatula_lift" and spec.spatula_contact_friction is not None:
            factors = spatula_material_factors(
                spec.gel_friction, spec.spatula_contact_friction, spec.support_friction
            )
            actors = [
                (self.tool, "tool"),
                (self.slider, "pancake"),
                (self.env[0], "surface"),
            ]
            actors += [(self.gripper.gels[side], "gel") for side in ("left", "right")]
            for actor, name in actors:
                contact = actor.get_contact_params()
                contact.coulomb_friction_coefficient = factors[name]
                actor.set_contact_params(contact)
        self.calibration = None
        self.camera_target_world = self.origin + self.rot.apply([0.065, 0, -0.04])

    def checkpoint(self):
        state, control = super().checkpoint()
        control["calibration"] = (
            None if self.calibration is None else self.calibration.copy()
        )
        return state, control

    def make_tool_mesh(self):
        s = self.spec
        if s.family == "hook":
            mesh = super().make_tool_mesh()
            mesh.vertices[:, 1] *= s.handle_depth_m / 0.024
            stem = np.abs(mesh.vertices[:, 0]) <= 0.009 + 1e-9
            mesh.vertices[stem, 0] *= s.handle_width_m / 0.018
            working = mesh.vertices[:, 2] < -0.025
            mesh.vertices[working, 0] *= s.scale
            # Keep the rectangular grasp aligned to the jaws while rotating the
            # hook's working plane and task pull into the jaw-closing direction.
            if s.hook_heading_deg:
                if not np.isclose(abs(s.hook_heading_deg), 90):
                    raise ValueError(
                        "Only the verified orthogonal hook heading is supported before diversification"
                    )
                blend = np.clip((mesh.vertices[:, 2] + 0.025) / 0.010, 0, 1)
                mesh.vertices[:, 0] *= 1 + blend * (
                    s.handle_depth_m / s.handle_width_m - 1
                )
                mesh.vertices[:, 1] *= 1 + blend * (
                    s.handle_width_m / s.handle_depth_m - 1
                )
            mesh.vertices[:, 2] -= s.hook_working_drop_m * np.clip(
                (-mesh.vertices[:, 2] - 0.015) / 0.009, 0, 1
            )
            return mesh
        if s.variant == "spatula_lift":
            mesh = box_union(
                [
                    ([0.100, 0.024, s.spatula_handle_height_m], [0.025, 0, 0]),
                    ([0.070, 0.070, 0.003], [0.110, 0, -0.0045]),
                ]
            )
            # Taper the blade to a 1 mm leading edge; no initial overlap.
            front = mesh.vertices[:, 0] > 0.135
            top = front & (mesh.vertices[:, 2] > -0.005)
            mesh.vertices[top, 2] = -0.005
            mesh.apply_translation([-s.spatula_grasp_x_m, 0, 0])
            return self.contact_mesh(mesh)
        parts = [([s.handle_width_m, s.handle_depth_m, 0.050], [0, 0, 0])]
        if s.family == "levering":
            parts += [
                ([0.014, 0.018, 0.033], [0, 0, -0.0365]),
                (
                    [s.lever_tip_x_m + 0.010, 0.018, 0.006],
                    [(s.lever_tip_x_m - 0.010) / 2, 0, -0.050],
                ),
            ]
        else:
            bottom = s.pusher_bottom_m
            parts += [([0.018, 0.024 * s.working_width_scale, -0.020 - bottom], [0, 0, (-0.020 + bottom) / 2])]
        mesh = self.contact_mesh(box_union(parts))
        if s.family == "pushing":
            mesh.apply_transform(
                trimesh.transformations.rotation_matrix(
                    np.deg2rad(-s.push_heading_deg), [0, 0, 1]
                )
            )
        return mesh

    def add_body(self, name, mesh, local, mass=1.0, static=True, friction=None):
        local = np.asarray(local, float).copy()
        if (
            self.spec.family == "levering"
            and self.spec.case == "miss"
            and name in ("hinge_support", "flap_rest")
        ):
            local[1] += 0.10
        if friction is None:
            friction = self.spec.support_friction
        actor = self.actor(
            name,
            mesh,
            self.origin + self.rot.apply(local),
            self.rot,
            mass,
            friction,
            static=static,
        )
        self.env.append(actor)
        self.environment_meshes.append(mesh)
        return actor

    def build_environment(self):
        s, p = self.spec, self.p
        self.env, self.environment_meshes = [], []
        self.slider = None
        self.axis = self.rot.apply([-1, 0, 0] if s.family == "hook" else [1, 0, 0])
        if s.family == "hook":
            cross = np.array(
                [
                    0.024,
                    s.lateral_m + (0.006 if s.mirror_fixture else -0.006),
                    -0.016 - s.hook_working_drop_m,
                ]
            )
            self.slider = self.add_body(
                "weighted_sled",
                self.contact_mesh(sled_geometry()),
                cross,
                s.sled_mass_kg,
                False,
            )
            self.floor_z = cross[2] - 0.056
            for name, size, position in sled_support_layout(cross, s.rail_clearance_m):
                self.add_body(name, block(size, [0, 0, 0]), position)
        elif s.family == "pushing":
            self.floor_z = -0.061
            self.add_body(
                "push_table",
                block([0.400, 0.300, 0.020], [0, 0, 0]),
                [0.080, 0, self.floor_z - 0.010],
            )
            self.slider = self.add_body(
                "t_block",
                self.contact_mesh(t_block_geometry()),
                [0.056, s.lateral_m, self.floor_z + 0.0075],
                s.object_mass_kg,
                False,
            )
        elif s.variant == "spatula_lift":
            self.floor_z = 0.003 if s.spatula_edge_entry else -0.007
            self.add_body(
                "serving_surface",
                block([0.30, 0.26, 0.016], [0, 0, 0]),
                [
                    (0.330 if s.spatula_edge_entry else 0.255) - s.spatula_grasp_x_m,
                    0,
                    self.floor_z - 0.008,
                ],
            )
            pancake = trimesh.creation.cylinder(radius=0.035, height=0.008, sections=48)
            self.slider = self.add_body(
                "pancake",
                self.contact_mesh(pancake),
                [
                    (0.195 if s.spatula_edge_entry else 0.190) - s.spatula_grasp_x_m,
                    0.10 if s.case == "miss" else 0,
                    self.floor_z + 0.004,
                ],
                0.05,
                False,
            )
            self.flap = self.slider
        else:
            self.floor_z = -0.065
            self.add_body(
                "lever_base",
                block(
                    [
                        0.290 - s.lever_base_front_m,
                        0.240 if s.case == "miss" else 0.140,
                        0.012,
                    ],
                    [0, 0, 0],
                ),
                [
                    (0.290 + s.lever_base_front_m) / 2,
                    0.05 if s.case == "miss" else 0,
                    self.floor_z - 0.006,
                ],
            )
            fulcrum = trimesh.creation.cylinder(radius=0.006, height=0.060, sections=48)
            fulcrum.apply_transform(
                trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
            )
            self.fulcrum = self.add_body("fulcrum", fulcrum, [0.090, 0, -0.059])
            for sign in (-1, 1):
                self.add_body(
                    "hinge_support",
                    block([0.016, 0.016, 0.025], [0, 0, 0]),
                    [0.220, sign * 0.047, -0.0525],
                )
            self.flap_mesh = block([0.100, 0.070, 0.008], [-0.050, 0, 0])
            shape = p.create_tri_mesh_shape(
                np.asarray(self.flap_mesh.vertices, np.float32).ravel(),
                np.asarray(self.flap_mesh.faces, np.int32).ravel(),
            )
            params = p.ArticulatedActorParams(name="gravity_flap")
            params.world_from_root = p.TransformRT(
                translation=self.origin
                + self.rot.apply([0.220, 0.10 if s.case == "miss" else 0, -0.040]),
                rotation=p.Quaternion.from_rotation_vector(self.rot.as_rotvec()),
            )
            joint = p.ArticulatedJointParams(
                name="hinge", type=p.ArticulatedJointType.REVOLUTE, axis=[0, 1, 0]
            )
            resistance = p.ArticulatedJointFrictionParams()
            resistance.coulomb = s.flap_hinge_friction_nm
            resistance.viscous = 0.0001
            joint.friction = resistance
            params.joints = [
                p.ArticulatedJointParams(
                    name="fixed", type=p.ArticulatedJointType.HARD
                ),
                joint,
            ]
            contact = self.contact_params(s.support_friction)
            params.links = [
                p.ArticulatedLinkParams(name="hinge_anchor", parent_link=-1),
                p.ArticulatedLinkParams(
                    name="flap",
                    parent_link=0,
                    shape=shape,
                    collider_type=p.ColliderType.MESH,
                    density=s.flap_mass_kg / self.flap_mesh.volume,
                    contact=contact,
                ),
            ]
            self.flap_fixture = self.scene.create_articulated_actor(params)
            self.flap = self.scene.get_actor(
                self.flap_fixture.get_nested_link_actors()[1]
            )
            self.env.append(self.flap)
            self.environment_meshes.append(self.flap_mesh)
            # Two resting supports define the closed angle physically, away from blade access.
            for sign in (-1, 1):
                self.add_body(
                    "flap_rest",
                    block([0.020, 0.010, 0.021], [0, 0, 0]),
                    [0.135, sign * 0.030, -0.0545],
                )
        if self.slider is not None:
            self.slider0 = np.asarray(
                self.slider.get_center_of_mass_transform().translation
            ).copy()
            self.slider_com0 = self.slider0.copy()
            if s.family == "pushing":
                self.goal_object_com_pose_world = np.r_[
                    self.slider0 + self.rot.apply([s.push_distance_m, 0, 0]),
                    (
                        self.rot
                        * Rotation.from_euler(
                            "z",
                            s.push_yaw_deg if s.variant == "pose" else 0,
                            degrees=True,
                        )
                    ).as_quat(),
                ]
        self.table_top = float((self.origin + self.rot.apply([0, 0, self.floor_z]))[2])
        self.support_queries = [
            a.register_query(p.QueryType.CONTACT_POINTS) for a in self.env
        ]
        self.queries_extra = self.support_queries

    def apply_environment_force(self):
        # Intentionally no spring, rail force, anti-yaw force or planar constraint.
        pass

    def command_at(self, t, branch="nominal", anchor_command=None):
        tau = self.clock(t)
        s = self.spec
        value = np.zeros(6)
        phase = "grasp"
        if s.family == "hook":
            y = 0.0
            z = 0.0 if s.case == "miss" else 0.018
            knots = [
                (0, np.zeros(6), "grasp"),
                (2, np.zeros(6), "align"),
                (4, [0, y, z, 0, 0, 0], "engage"),
                (6, [-0.006, y, z, 0, 0, 0], "pull"),
                (18, [-s.hook_pull_m, y, z, 0, 0, 0], "hold"),
                (22, [-s.hook_pull_m, y, z, 0, 0, 0], "done"),
            ]
            if branch == "hold":
                knots = knots[:3] + [(22, [0, 0, z, 0, 0, 0], "done")]
            elif branch in ("left", "right"):
                y = (-1 if branch == "left" else 1) * 0.003
                knots = knots[:4] + [
                    (8, [0, 0, 0, 0, 0, 0], "realign"),
                    (10, [0, y, 0, 0, 0, 0], "reengage"),
                    (12, [0, y, z, 0, 0, 0], "pull"),
                    (24, [-s.hook_pull_m, y, z, 0, 0, 0], "hold"),
                    (28, [-s.hook_pull_m, y, z, 0, 0, 0], "done"),
                ]
            value, phase = interpolate(knots, tau)
        elif s.family == "pushing":
            y = s.push_contact_y_m if s.push_contact_y_m is not None else -0.022 if s.variant == "pose" else 0.0
            if s.case == "miss":
                y = 0.09
            goal = s.push_distance_m
            knots = [
                (0, np.zeros(6), "grasp"),
                (3, np.zeros(6), "approach"),
                (5, [0, y, 0, 0, 0, 0], "push"),
                (16, [goal + s.push_overtravel_m, y + s.push_end_lateral_m, 0, 0, 0, 0], "hold"),
                (22, [goal + s.push_overtravel_m, y + s.push_end_lateral_m, 0, 0, 0, 0], "done"),
            ]
            value, phase = interpolate(knots, tau)
        elif s.variant == "spatula_lift":
            value, phase = interpolate(
                [
                    (0, np.zeros(6), "grasp"),
                    (3, np.zeros(6), "approach"),
                    (12, [0.075, 0, 0, 0, 0, 0], "scoop"),
                    (17, [0.075, 0, 0.045, 0, 0, 0], "hold"),
                    (19, [0.075, 0, 0.045, 0, 0, 0], "done"),
                    (20, [0.075, 0, 0.045, 0, 0, 0], "done"),
                ],
                tau,
            )
            if s.spatula_edge_entry:
                value, phase = interpolate(
                    [
                        (0, np.zeros(6), "grasp"),
                        (3, np.zeros(6), "approach_under_edge"),
                        (7, [0.030, 0, 0, 0, 0, 0], "lift_edge"),
                        (
                            10,
                            [0.030, 0, s.spatula_entry_lift_m, 0, 0, 0],
                            "slide_under",
                        ),
                        (15, [0.065, 0, s.spatula_entry_lift_m, 0, 0, 0], "lift"),
                        (18, [0.065, 0, 0.045, 0, 0, 0], "hold"),
                        (19, [0.065, 0, 0.045, 0, 0, 0], "done"),
                        (20, [0.065, 0, 0.045, 0, 0, 0], "done"),
                    ],
                    tau,
                )
        else:
            pivot = np.array([0.090, 0, -0.053])
            angle = -np.deg2rad(s.lever_angle_deg) * smooth((tau - 5) / 10)
            rotation = Rotation.from_rotvec([0, angle, 0])
            value[:3] = pivot - rotation.apply(pivot)
            value[4] = angle
            phase = (
                "grasp"
                if tau < 2
                else "engage"
                if tau < 5
                else "lever"
                if tau < 15
                else "hold"
                if tau < 19
                else "done"
            )
        # Rotation acts about the tool root, not the wrist. Small precontact
        # translation calibration compensates the settled compliant grasp only.
        if t >= 1.5 and self.calibration is None:
            self.calibration = self.rot.inv().apply(
                np.asarray(self.tool.get_root_transform().translation) - self.origin
            )
        if self.calibration is not None:
            value[:3] -= np.clip(self.calibration, -0.006, 0.006) * smooth(
                (t - 1.5) / 1.5
            )
        offset = self.rot.inv().apply(self.origin - self.ee0)
        value[:3] += offset - Rotation.from_rotvec(value[3:]).apply(offset)
        self.nominal_tool_target = value[:3].copy()
        return value[:3], phase, value[3:]

    def step(self, t, branch="nominal", anchor_command=None):
        row = super().step(t, branch, anchor_command)
        s = self.spec
        angle = 0.0
        tilt = 0.0
        error = 0.0
        yaw_error = 0.0
        escape = False
        if s.variant == "spatula_lift":
            pos = self.rot.inv().apply(
                np.asarray(self.slider.get_center_of_mass_transform().translation)
                - self.slider0
            )
            row["task_progress"] = float(pos[2])
            error = float(np.linalg.norm(pos[:2]))
            escape = error > 0.10 or pos[2] < -0.01
            r = Rotation.from_rotvec(
                np.asarray(
                    self.slider.get_root_transform().rotation.to_rotation_vector()
                )
            )
            tilt = float(
                np.arccos(np.clip((self.rot.inv() * r).apply([0, 0, 1])[2], -1, 1))
            )
        elif s.family == "levering":
            r = Rotation.from_rotvec(
                np.asarray(self.flap.get_root_transform().rotation.to_rotation_vector())
            )
            angle = float((self.rot.inv() * r).as_rotvec()[1])
            row["task_progress"] = angle
        else:
            pos = np.asarray(self.slider.get_center_of_mass_transform().translation)
            displacement = self.rot.inv().apply(pos - self.slider0)
            r = Rotation.from_rotvec(
                np.asarray(
                    self.slider.get_root_transform().rotation.to_rotation_vector()
                )
            )
            relative = self.rot.inv() * r
            tilt = float(np.arccos(np.clip(relative.apply([0, 0, 1])[2], -1, 1)))
            yaw = float(relative.as_euler("xyz")[2])
            if s.family == "pushing":
                error = float(np.linalg.norm(displacement[:2] - [s.push_distance_m, 0]))
                yaw_error = float(
                    np.arctan2(
                        np.sin(
                            yaw
                            - np.deg2rad(s.push_yaw_deg if s.variant == "pose" else 0)
                        ),
                        np.cos(
                            yaw
                            - np.deg2rad(s.push_yaw_deg if s.variant == "pose" else 0)
                        ),
                    )
                )
                escape = abs(displacement[1]) > 0.12 or abs(displacement[0]) > 0.18
            else:
                error = abs(float(row["task_progress"]) - 0.04)
                escape = abs(displacement[1]) > (0.06 if s.variant == "loaded_box" else 0.010) or abs(displacement[2]) > 0.010
        row.update(
            collector_grasp_calibration_m=np.zeros(3)
            if self.calibration is None
            else self.calibration.copy(),
            flap_angle_rad=angle,
            object_tilt_rad=tilt,
            object_position_error_m=error,
            object_yaw_error_rad=yaw_error,
            support_escape=escape,
        )
        # Contact oracles: world-axis wrench ON TOOL about current tool COM,
        # separately for each environment actor. Support forces on the moved
        # object are stored separately, never added to the tool contact target.
        from wrench_math import aggregate_contact_points

        center = np.asarray(self.tool.get_center_of_mass_transform().translation)
        contacts = list(self.tool.get_contact_points_world())
        pair = []
        for actor in self.env:
            w = aggregate_contact_points(
                contacts, self.tool.get_handle(), actor.get_handle(), center
            )
            pair.append(np.r_[w.force, w.torque])
        row["tool_environment_pair_wrenches"] = np.asarray(pair)
        moved = self.flap if s.family == "levering" else self.slider
        origin = np.asarray(moved.get_center_of_mass_transform().translation)
        support = []
        contacts = list(moved.get_contact_points_world())
        row["support_penetration_m"] = max(
            [0.0]
            + [
                -float(c.distance)
                for c in contacts
                if self.tool.get_handle() not in (c.actor_a, c.actor_b)
            ]
        )
        row["rigid_penetration_m"] = max(
            row["rigid_penetration_m"], row["support_penetration_m"]
        )
        for actor in self.env:
            w = aggregate_contact_points(
                contacts, moved.get_handle(), actor.get_handle(), origin
            )
            support.append(np.r_[w.force, w.torque])
        row["object_support_pair_wrenches"] = np.asarray(support)
        row["rewards"] = row["task_progress"]
        return row
