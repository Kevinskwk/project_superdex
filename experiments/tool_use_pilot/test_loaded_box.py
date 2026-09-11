"""Geometry and control invariants for independently posed rail-free tasks."""

import unittest
from types import SimpleNamespace
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from benchmark import BenchmarkSpec, BenchmarkMixin, SurfaceWorld
from diversification_repairs import repair_specs, preferred_pose_specs
from loaded_box import LoadedBoxWorld, loaded_tray_mesh, straight_hook_mesh


class LoadedBoxTests(unittest.TestCase):
    def test_closed_printable_geometry(self):
        for mesh in (
            straight_hook_mesh(),
            loaded_tray_mesh(0.038),
            loaded_tray_mesh(0.058),
        ):
            self.assertTrue(mesh.is_watertight)
            self.assertTrue(mesh.is_winding_consistent)
            self.assertEqual(len(mesh.split()), 1)
            self.assertGreater(mesh.volume, 0)

    def test_initial_loop_clearance_top_and_side(self):
        for pitch in (0, -90):
            rot = Rotation.from_euler("y", pitch, degrees=True)
            tool = straight_hook_mesh()
            tool.vertices = rot.apply(tool.vertices)
            cross = rot.apply([0.022, 0, -0.064])
            floor = min(tool.bounds[0, 2] - 0.020, cross[2] - 0.026)
            box = loaded_tray_mesh(cross[2] - floor)
            box.apply_translation(cross)
            self.assertFalse(box.contains(tool.vertices).any())
            self.assertFalse(tool.contains(box.vertices).any())

    def test_level_frame_is_opt_in_and_scoped(self):
        self.assertFalse(BenchmarkSpec().level_task_frame)
        with self.assertRaises(ValueError):
            BenchmarkSpec(level_task_frame=True)
        with self.assertRaises(ValueError):
            BenchmarkSpec(surface_lean_deg=10)

    def test_bounded_manifest(self):
        rows = repair_specs(4)
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(s.level_task_frame and s.revision == 22 for s in rows))
        self.assertEqual(
            {s.robot_pitch_deg for s in rows if s.family == "hook"}, {0, -90}
        )
        final = repair_specs(5)
        self.assertEqual(len(final), 4)
        self.assertEqual({s.box_loop_height_m for s in final}, {0.055})
        preferred = preferred_pose_specs()
        self.assertEqual(len(preferred), 5)
        self.assertTrue(
            all(s.robot_pose_in_task_frame for s in preferred if s.robot_pitch_deg)
        )
        self.assertEqual(preferred[-1].surface_lean_deg, 20)

    def test_independent_frame_and_grasp_transform(self):
        world = object.__new__(BenchmarkMixin)
        world.spec = repair_specs(4)[0]
        frame = Rotation.from_euler("z", 40, degrees=True)
        origin = np.array([0.2, 0.1, 0.4])
        actual, shifted = world.prepared_frame([], frame, origin)
        np.testing.assert_allclose(actual.as_matrix(), frame.as_matrix())
        np.testing.assert_allclose(world.grasp_rot.as_matrix(), frame.as_matrix())
        np.testing.assert_allclose(
            shifted, origin + frame.apply([0, 0, world.spec.grasp_depth_m])
        )

    def test_pull_has_no_lifting_stroke(self):
        s = repair_specs(4)[0]
        world = object.__new__(LoadedBoxWorld)
        world.spec = s
        world.calibration = np.zeros(3)
        world.vertical_relief_m = 0.0
        world.last_vertical_force_n = 0.0
        for t in np.linspace(0, s.duration, 50):
            command, phase, rot = world.command_at(t)
            self.assertAlmostEqual(command[2], 0)
            np.testing.assert_allclose(rot, 0)
        self.assertEqual(phase, "done")
        self.assertAlmostEqual(command[0], -s.hook_pull_m)
        world.last_vertical_force_n = -1
        command, _, _ = world.command_at(10)
        self.assertLess(command[2], 0)  # Upward force on box -> lower tool.

    def test_hold_branch_preserves_prefix_then_stops(self):
        world = object.__new__(LoadedBoxWorld)
        world.spec = repair_specs(5)[0]
        world.calibration = np.zeros(3)
        world.vertical_relief_m = world.last_vertical_force_n = 0.0
        world.branch_value = None
        before, _, _ = world.command_at(4.9, "hold")
        nominal, _, _ = world.command_at(4.9)
        np.testing.assert_allclose(before, nominal)
        anchor, _, _ = world.command_at(world.spec.branch_time, "hold")
        later, phase, _ = world.command_at(15, "hold")
        np.testing.assert_allclose(anchor, later)
        self.assertEqual(phase, "branch_hold")

    def test_lean_preserves_full_working_rim(self):
        for s in repair_specs(4)[4:]:
            world = object.__new__(SurfaceWorld)
            world.spec = s
            world.rot = Rotation.from_euler("z", s.surface_heading_deg, degrees=True)
            tilt = Rotation.from_euler("x", s.surface_lean_deg, degrees=True)
            world.grasp_rot = world.rot * tilt
            world.make_tool_mesh()
            points = world.working_tip_points
            self.assertGreaterEqual(len(points), 12)
            self.assertGreater(np.ptp(points[:, 2]), 0.001)
            self.assertGreater(points.mean(0)[1], 0)
            # The working rim is an authored plane, not just one lowest node.
            np.testing.assert_allclose(
                points @ tilt.apply([0, 0, 1]),
                np.repeat((points @ tilt.apply([0, 0, 1])).mean(), len(points)),
                atol=1e-9,
            )

    def test_preflight_rejects_initial_overlap(self):
        def actor(handle, position):
            transform = SimpleNamespace(
                translation=np.asarray(position),
                rotation=SimpleNamespace(to_rotation_vector=lambda: np.zeros(3)),
            )
            return SimpleNamespace(
                get_handle=lambda: handle, get_root_transform=lambda: transform
            )

        world = object.__new__(LoadedBoxWorld)
        world.tool = actor(0, [0, 0, 0])
        world.mesh = trimesh.creation.box([0.01] * 3)
        world.env = [actor(1, [0, 0, 0])]
        world.environment_meshes = [trimesh.creation.box([0.02] * 3)]
        world.gripper = SimpleNamespace(
            links={
                "franka_left_gelsight_housing": actor(2, [10, 0, 0]),
                "franka_right_gelsight_housing": actor(3, [-10, 0, 0]),
            }
        )
        with self.assertRaisesRegex(ValueError, "Prepared geometry overlap"):
            world.preflight_clearance()
        world.env = [actor(1, [0.03, 0, 0])]
        world.preflight_clearance()
        self.assertEqual(world.initial_sampled_overlap_m, 0)

    def test_pull_tracking_is_tangential_and_rate_limited(self):
        world = object.__new__(LoadedBoxWorld)
        world.spec = repair_specs(7)[0]
        world.calibration = np.zeros(3)
        world.vertical_relief_m = world.last_vertical_force_n = 0.0
        world.pull_tip_start_task = None
        world.tracking_correction = np.zeros(3)
        world.loop_center_task = np.array([0.022, 0, -0.064])
        world.rot = Rotation.identity()
        world.origin = np.zeros(3)
        pose = SimpleNamespace(
            translation=np.zeros(3),
            rotation=SimpleNamespace(to_rotation_vector=lambda: np.zeros(3)),
        )
        world.tool = SimpleNamespace(get_root_transform=lambda: pose)
        world.command_at(4)
        world.command_at(10)
        self.assertLess(world.tracking_correction[0], 0)
        self.assertLessEqual(abs(world.tracking_correction[0]), world.spec.dt * 0.001)
        np.testing.assert_array_equal(world.tracking_correction[1:], [0, 0])


if __name__ == "__main__":
    unittest.main()
