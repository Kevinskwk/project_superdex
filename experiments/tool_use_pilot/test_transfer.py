import unittest
from dataclasses import replace, asdict
import numpy as np
from scipy.spatial.transform import Rotation

from benchmark import BenchmarkSpec, specifications, composite_gate
from benchmark_geometry import (
    peg_mesh,
    socket_mesh,
    profile,
    contains_profile,
    friction_provenance,
)
from transfer_world import (
    TransferWorld,
    spatula_material_factors,
    sled_geometry,
    sled_support_layout,
    t_block_geometry,
    transfer_audit,
)


class TransferTests(unittest.TestCase):
    def test_repaired_tool_meshes_and_heading(self):
        for s in specifications("transfer"):
            if s.family not in ("hook", "levering"):
                continue
            world = object.__new__(TransferWorld)
            world.spec = s
            mesh = world.make_tool_mesh()
            self.assertTrue(mesh.is_watertight)
            self.assertGreater(mesh.volume, 0)
            if s.family == "hook":
                rotation, _ = world.prepared_frame([], Rotation.identity(), np.zeros(3))
                self.assertAlmostEqual(abs(rotation.apply([1, 0, 0]) @ [0, 1, 0]), 1)
                self.assertAlmostEqual(mesh.bounds[0, 2], -0.076)
            else:
                self.assertLessEqual(mesh.extents[2], 0.0241)
                self.assertEqual(s.grasp_depth_m, 0.018)

    def test_spatula_lift_requires_hold_and_retention(self):
        s = next(s for s in specifications("transfer") if s.variant == "spatula_lift")
        n = 60
        data = dict(
            initialization=np.zeros(n, bool),
            phase=np.repeat("done", n),
            object_tilt_rad=np.zeros(n),
            support_escape=np.zeros(n, bool),
            object_position_error_m=np.zeros(n),
            object_yaw_error_rad=np.zeros(n),
            flap_angle_rad=np.zeros(n),
            task_progress=np.full(n, 0.04),
        )
        base = dict(
            grasp_retained=True, sustained_penetration=False, min_gel_jacobian=0.8
        )
        self.assertTrue(transfer_audit(data, s, base)["task_success"])
        data["task_progress"][-1] = 0.01
        self.assertFalse(transfer_audit(data, s, base)["task_success"])
        data["object_tilt_rad"][-1] = 0.3
        self.assertFalse(transfer_audit(data, s, base)["physical_valid"])

    def test_spatula_finishes_in_done_phase(self):
        s = next(s for s in specifications("transfer") if s.variant == "spatula_lift")
        world = object.__new__(TransferWorld)
        world.spec = s
        world.calibration = np.zeros(3)
        world.rot = Rotation.identity()
        world.origin = np.zeros(3)
        world.ee0 = np.zeros(3)
        command, phase, rotation = world.command_at(s.duration - s.dt)
        self.assertEqual(phase, "done")
        np.testing.assert_allclose(command, [0.065, 0, 0.045])
        np.testing.assert_allclose(rotation, 0)

    def test_legacy_geometry_defaults(self):
        old = BenchmarkSpec(revision=18)
        new = replace(
            old,
            revision=19,
            shaft_diameter_m=None,
            tip_length_m=None,
            tip_scale=None,
            follower_diameter_m=None,
        )
        self.assertEqual(old.shaft_diameter_m, 0.005)
        self.assertEqual(new.shaft_diameter_m, 0.010)
        self.assertEqual(new.tip_scale, 2.0)

    def test_thick_profile_geometry(self):
        for shape in ("round", "square", "hex", "d", "key"):
            mesh = peg_mesh(shape, 2.0, shaft_diameter=0.010, tip_length=0.022)
            self.assertTrue(mesh.is_watertight)
            self.assertGreater(mesh.volume, 0)
            self.assertLess(len(mesh.faces), 12000)
            neck = mesh.vertices[np.isclose(mesh.vertices[:, 2], -0.030, atol=0.002)]
            self.assertGreater(len(neck), 0)
            self.assertLessEqual(np.linalg.norm(neck[:, :2], axis=1).max(), 0.00501)
            self.assertTrue(
                contains_profile(profile(shape, 2.0), neck[:, :2], 0.000001).all()
            )
            socket = socket_mesh(shape, 2.0, shaft=True)
            self.assertTrue(socket.is_watertight)
            self.assertGreater(socket.volume, 0)
            self.assertAlmostEqual(mesh.bounds[0, 2], -0.069)

    def test_sled_tool_clearance(self):
        mesh = sled_geometry()
        self.assertTrue(mesh.is_watertight)
        self.assertAlmostEqual(mesh.bounds[0, 2], -0.056)
        # Sled deck below the unraised hook foot; 10mm nominal gap.
        self.assertGreater(-0.056 - (-0.016 - 0.050), 0.009)
        # 44mm aperture accepts 24mm-thick hook with +/-6mm offsets.
        self.assertGreater(0.022 - (0.012 + 0.006), 0.003)

    def test_t_block_closed(self):
        mesh = t_block_geometry()
        self.assertTrue(mesh.is_watertight)
        self.assertAlmostEqual(mesh.extents[0], 0.100)
        self.assertAlmostEqual(mesh.extents[2], 0.015)

    def test_mirrored_sled_rail_clearance(self):
        for offset in (-0.006, 0.006):
            cross = np.array([0.024, offset, -0.016])
            for name, size, center in sled_support_layout(cross, 0.002):
                if name == "clearance_rail":
                    gap = abs(center[1] - cross[1]) - size[1] / 2 - 0.030
                    self.assertAlmostEqual(gap, 0.002)

    def test_recipe_covers_additions(self):
        rows = specifications("transfer")
        self.assertEqual(len(rows), 17)
        self.assertTrue(all(s.revision == 20 for s in rows))
        self.assertEqual(len([s for s in rows if s.family == "hook"]), 1)
        self.assertEqual(
            next(s for s in rows if s.family == "hook").hook_heading_deg, 90
        )

    def test_transfer_friction_uses_support_material(self):
        for s in specifications("transfer"):
            if s.family not in ("hook", "levering", "pushing"):
                continue
            actual = friction_provenance(asdict(s), "")
            self.assertAlmostEqual(actual["effective_gel_tool_friction"], 1.4)
            self.assertAlmostEqual(
                actual["effective_environment_tool_friction"],
                s.spatula_contact_friction
                if s.variant == "spatula_lift"
                else np.sqrt(1.4 * 0.4),
            )

    def test_spatula_pair_friction_preserves_grip(self):
        f = spatula_material_factors(1.4, 0.15, 0.4)
        self.assertAlmostEqual(np.sqrt(f["gel"] * f["tool"]), 1.4)
        self.assertAlmostEqual(np.sqrt(f["tool"] * f["pancake"]), 0.15)
        self.assertAlmostEqual(np.sqrt(f["surface"] * f["pancake"]), 0.4)
        self.assertAlmostEqual(np.sqrt(f["surface"] * f["tool"]), 0.4)

    def test_turning_negative_does_not_initialize_inside_pocket_wall(self):
        s = next(
            s for s in specifications("transfer_negatives") if s.family == "turning"
        )
        self.assertEqual(s.case, "hold_control")
        self.assertEqual(s.lateral_m, 0)
        self.assertEqual(s.turn_command_deg, 100)
        self.assertEqual(s.rotor_stop_deg, 90)

    def test_lever_tip_keeps_overlap_at_goal(self):
        s = BenchmarkSpec(
            family="levering",
            variant="gravity_flap",
            lever_tip_x_m=0.16,
            lever_angle_deg=35,
            lever_base_front_m=0.084,
        )
        pivot = np.array([0.09, 0, -0.053])
        tip = pivot + Rotation.from_euler("y", -s.lever_angle_deg, degrees=True).apply(
            np.array([s.lever_tip_x_m, 0, -0.047]) - pivot
        )
        edge = np.array([0.22, 0, -0.04]) + Rotation.from_euler(
            "y", 20, degrees=True
        ).apply([-0.1, 0, -0.004])
        self.assertGreater(tip[0] - edge[0], 0.010)
        underside = (
            -0.04
            + (0.22 - tip[0]) * np.tan(np.deg2rad(20))
            - 0.004 / np.cos(np.deg2rad(20))
        )
        self.assertGreater(tip[2], underside)
        # The descending effort arm must not strike the platform's front edge.
        for angle in np.linspace(0, s.lever_angle_deg, 31):
            lower_face = -0.053 + (s.lever_base_front_m - 0.09) * np.tan(
                np.deg2rad(angle)
            )
            self.assertGreater(lower_face, -0.065 + 0.003)

    def test_transfer_goal_and_tipping(self):
        s = BenchmarkSpec(revision=19, family="pushing", variant="translation")
        n = 40
        data = dict(
            initialization=np.zeros(n, bool),
            phase=np.repeat("done", n),
            object_tilt_rad=np.zeros(n),
            support_escape=np.zeros(n, bool),
            object_position_error_m=np.full(n, 0.005),
            object_yaw_error_rad=np.zeros(n),
            flap_angle_rad=np.zeros(n),
        )
        base = dict(
            grasp_retained=True, sustained_penetration=False, min_gel_jacobian=0.8
        )
        self.assertTrue(transfer_audit(data, s, base)["task_success"])
        data["object_tilt_rad"][-1] = 0.3
        self.assertFalse(transfer_audit(data, s, base)["physical_valid"])

    def test_lever_requires_final_held_goal_not_peak_lift(self):
        s = BenchmarkSpec(family="levering", variant="gravity_flap")
        n = 60
        data = dict(
            initialization=np.zeros(n, bool),
            phase=np.repeat("done", n),
            object_tilt_rad=np.zeros(n),
            support_escape=np.zeros(n, bool),
            object_position_error_m=np.zeros(n),
            object_yaw_error_rad=np.zeros(n),
            flap_angle_rad=np.full(n, np.deg2rad(25)),
        )
        base = dict(
            grasp_retained=True, sustained_penetration=False, min_gel_jacobian=0.8
        )
        self.assertTrue(transfer_audit(data, s, base)["task_success"])
        data["flap_angle_rad"][-1] = 0
        m = transfer_audit(data, s, base)
        self.assertTrue(m["physical_valid"])
        self.assertFalse(m["task_success"])
        self.assertAlmostEqual(m["max_flap_angle_deg"], 25)

    def test_transfer_requires_completion_and_retained_support(self):
        s = next(s for s in specifications("transfer") if s.family == "hook")
        n = 40
        data = dict(
            initialization=np.zeros(n, bool),
            phase=np.repeat("hold", n),
            object_tilt_rad=np.zeros(n),
            support_escape=np.zeros(n, bool),
            object_position_error_m=np.zeros(n),
            object_yaw_error_rad=np.zeros(n),
            flap_angle_rad=np.zeros(n),
            task_progress=np.full(n, 0.045),
        )
        base = dict(
            grasp_retained=True, sustained_penetration=False, min_gel_jacobian=0.8
        )
        self.assertFalse(transfer_audit(data, s, base)["task_success"])
        data["phase"][:] = "done"
        self.assertTrue(transfer_audit(data, s, base)["task_success"])
        data["support_escape"][-1] = True
        self.assertFalse(transfer_audit(data, s, base)["physical_valid"])

    def test_composite_resistance_and_revision_matching(self):
        records = []
        for family in ("insertion", "turning"):
            for i in range(3):
                spec = BenchmarkSpec(
                    family=family,
                    case="nominal",
                    load_mode="friction" if family == "turning" else "spring",
                    episode_id=i,
                )
                records.append(
                    dict(
                        spec=asdict(spec),
                        metrics=dict(
                            physical_valid=True,
                            task_success=True,
                            full_rollout_complete=True,
                            sequence_complete=True,
                        ),
                    )
                )
        self.assertEqual(composite_gate(records, "friction", 19)["turning"], [0, 1, 2])
        with self.assertRaises(ValueError):
            composite_gate(records, "spring", 19)
        with self.assertRaises(ValueError):
            composite_gate(records, "friction", 18)


if __name__ == "__main__":
    unittest.main()
