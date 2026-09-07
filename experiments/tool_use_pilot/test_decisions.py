"""Geometry, causality and observation-contract regression tests."""

import unittest
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
from decisions import DecisionSpec, candidates, target, entrance_mesh, rim_mesh
from decision_campaign import specs_for
from decision_vision import normalize_and_metric, transform, surface


class DecisionTests(unittest.TestCase):
    def test_hook_single_solid_and_filled_corner(self):
        from pilot import tool_mesh

        mesh = tool_mesh("hook")
        self.assertTrue(mesh.is_watertight)
        self.assertTrue(mesh.is_winding_consistent)
        self.assertEqual(len(mesh.split()), 1)
        # Exact J profile area, including the repaired 18 x 8 mm corner.
        self.assertAlmostEqual(mesh.volume, (18 * 81 + 33 * 8 + 8 * 24) * 24 * 1e-9)
        from trimesh.ray.ray_pyembree import RayMeshIntersector

        rays = RayMeshIntersector(mesh)
        self.assertTrue(rays.intersects_any([[0, -0.05, -0.052]], [[0, 1, 0]])[0])
        self.assertFalse(rays.intersects_any([[0.022, -0.05, -0.035]], [[0, 1, 0]])[0])

    def test_hook_contact_resolution_preserves_solid(self):
        from decisions import HookDecision
        from pilot import tool_mesh

        world = HookDecision.__new__(HookDecision)
        world.spec = DecisionSpec(revision=3)
        mesh = world.contact_mesh(tool_mesh("hook"))
        self.assertTrue(mesh.is_watertight)
        self.assertLessEqual(
            mesh.edges_unique_length.max(), world.spec.contact_edge_m + 1e-6
        )
        self.assertAlmostEqual(mesh.volume, tool_mesh("hook").volume)

    def test_repaired_fixture_has_no_internal_seams(self):
        from decisions import HookDecision

        world = HookDecision.__new__(HookDecision)
        for mirror in (False, True):
            world.spec = DecisionSpec(revision=6, mirror_fixture=mirror)
            mesh = world.make_slider_mesh()
            self.assertTrue(mesh.is_watertight)
            self.assertTrue(mesh.is_winding_consistent)
            self.assertEqual(len(mesh.split()), 1)
            self.assertAlmostEqual(
                mesh.volume,
                0.010 * 0.110 * 0.008 + 0.010 * 0.005 * 0.044 + 0.020 * 0.110 * 0.006,
            )

    def test_repair_prefix_independent_of_branch(self):
        from decisions import HookDecision
        from types import SimpleNamespace

        for branch in candidates("hook"):
            world = HookDecision.__new__(HookDecision)
            world.spec = DecisionSpec(revision=3)
            world.rot = Rotation.identity()
            world.origin = np.zeros(3)
            world.tracking_correction = np.zeros(3)
            world.tool = SimpleNamespace(
                get_root_transform=lambda: SimpleNamespace(
                    translation=np.array([0, -0.006, 0])
                )
            )
            for t in np.arange(0, 5, 0.01):
                command, _, _ = world.command_at(t, branch)
                self.assertLessEqual(abs(world.tracking_correction).max(), 0.012)
            if branch == "pull":
                reference = command.copy()
            else:
                np.testing.assert_array_equal(command, reference)

    def test_recovery_waits_for_clearance_and_times_out(self):
        from decisions import HookDecision
        from types import SimpleNamespace

        world = HookDecision.__new__(HookDecision)
        world.spec = DecisionSpec(revision=4)
        world.rot = Rotation.identity()
        world.origin = np.zeros(3)
        world.tracking_correction = np.zeros(3)
        world.trajectory_time = 7.995
        world.recovery_gate = 0
        world.recovery_wait = 0.0
        world.controller_abort = ""
        pose = SimpleNamespace(translation=np.array([0, 0, 0.018]))
        world.tool = SimpleNamespace(get_root_transform=lambda: pose)
        _, phase, _ = world.command_at(8.0, "right")
        self.assertEqual(phase, "wait_recovery_clearance")
        self.assertEqual(world.trajectory_time, 8.0)
        self.assertEqual(world.nominal_tool_target[1], 0)
        pose.translation = np.zeros(3)
        world.command_at(8.01, "right")
        self.assertEqual(world.recovery_gate, 1)
        world.trajectory_time = 10.0
        for _ in range(202):
            world.command_at(12.0, "right")
        self.assertEqual(world.controller_abort, "recovery_alignment_timeout")

    def test_bounded_manifests(self):
        for kind in ("hook", "key"):
            self.assertEqual(len(specs_for("grip", kind)), 9)
            self.assertEqual(len(specs_for("evaluation", kind)), 24)
        self.assertEqual(
            len(candidates("hook")) * 24 + len(candidates("key")) * 24, 240
        )

    def test_material_validation(self):
        with self.assertRaises(ValueError):
            DecisionSpec(gel_modulus_pa=-1)
        with self.assertRaises(ValueError):
            DecisionSpec(gel_friction=-1)

    def test_candidate_independent_prefix(self):
        for kind in ("hook", "key"):
            spec = DecisionSpec(
                kind=kind, correction_m=0.0015 if kind == "key" else 0.012
            )
            for t in np.linspace(0, spec.branch_time, 60):
                first = target(spec, t, candidates(kind)[0])[0]
                for branch in candidates(kind):
                    np.testing.assert_allclose(first, target(spec, t, branch)[0])

    def test_smooth_velocity_limits(self):
        for kind in ("hook", "key"):
            spec = DecisionSpec(
                kind=kind, correction_m=0.0015 if kind == "key" else 0.012
            )
            t = np.arange(0, spec.duration, 0.002)
            for branch in candidates(kind):
                values = np.array([target(spec, x, branch)[0] for x in t])
                speed = np.diff(values, axis=0) / 0.002
                self.assertLess(np.linalg.norm(speed[:, :3], axis=1).max(), 0.0201)
                self.assertLess(
                    np.rad2deg(np.linalg.norm(speed[:, 3:], axis=1)).max(), 15.01
                )

    def test_socket_opening_not_convexified(self):
        gate = entrance_mesh()
        rotor = rim_mesh(0.0112, 0.0072, 0.010, outer=0.024, floor=True)
        for mesh in (gate, rotor):
            self.assertTrue(mesh.is_watertight)
            # Centerline remains clear down to the socket floor.
            from trimesh.ray.ray_pyembree import RayMeshIntersector

            hits, _, _ = RayMeshIntersector(mesh).intersects_location(
                [[0, 0, 0.010]], [[0, 0, -1]], multiple_hits=False
            )
            if mesh is gate:
                self.assertEqual(len(hits), 0)
            else:
                self.assertAlmostEqual(hits[0, 2], -0.010, places=6)

    def test_metric_path_retains_translation_scale(self):
        rng = np.random.default_rng(0)
        cloud = rng.normal(size=(2, 1024, 3)) * 0.01
        ee = np.r_[np.zeros(3), [0, 0, 0, 1]]
        a, ma = normalize_and_metric(cloud, [1024, 1024], ee)
        b, mb = normalize_and_metric(cloud + 0.002, [1024, 1024], ee)
        np.testing.assert_allclose(a, b, atol=1e-6)
        self.assertFalse(np.allclose(ma, mb))
        empty, metric = normalize_and_metric(np.zeros_like(cloud), [0, 0], ee)
        self.assertTrue(np.isfinite(empty).all())
        self.assertFalse(metric.any())

    def test_pose_reconstruction(self):
        points = np.array([[1, 0, 0], [0, 1, 0]])
        pose = np.r_[
            [0.1, 0.2, 0.3], Rotation.from_euler("z", 90, degrees=True).as_quat()
        ]
        np.testing.assert_allclose(
            transform(points, pose), [[0.1, 1.2, 0.3], [-0.9, 0.2, 0.3]], atol=1e-7
        )

    def test_surface_sampling_repeatable(self):
        m = trimesh.creation.box()
        np.testing.assert_array_equal(
            surface(m, 1024, np.random.default_rng(5)),
            surface(m, 1024, np.random.default_rng(5)),
        )

    def test_static_composite_fixture_keeps_mesh_root_pose(self):
        import superdex.physics as p
        from pilot import World

        p.initialize(num_worker_threads=0)
        world = World.__new__(World)
        world.p = p
        world.scene = p.create_scene("static-com-test")
        try:
            mesh = trimesh.creation.box([0.02, 0.03, 0.01])
            mesh.apply_translation([0, 0, -0.025])
            actor = world.actor(
                "offset-shape",
                mesh,
                np.array([0.1, 0.2, 0.3]),
                Rotation.identity(),
                1,
                0.5,
                static=True,
            )
            original = np.asarray(actor.get_root_transform().translation).copy()
            for _ in range(3):
                world.scene.step(0.01)
            np.testing.assert_allclose(
                actor.get_root_transform().translation, original, atol=1e-6
            )
        finally:
            world.close()
            p.shutdown()


if __name__ == "__main__":
    unittest.main()
