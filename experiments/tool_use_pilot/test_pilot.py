"""Fast contract tests; physical outcomes are checked by the actual simulator."""

import unittest
import numpy as np
from pilot import Spec, smooth, tool_mesh, trajectory, slider_mesh
from campaign import specifications
from probes import split_group, past_indices
from dataclasses import asdict


class PilotTests(unittest.TestCase):
    def test_bounded_campaign(self):
        specs = specifications()
        self.assertEqual(len(specs), 180)
        self.assertEqual(len([s for s in specs if s.task == "probe"]), 96)
        self.assertEqual(len({s.episode_id for s in specs}), 180)
        for group in {s.group for s in specs if s.task == "probe"}:
            self.assertEqual(
                {s.stiffness for s in specs if s.group == group}, {30.0, 60.0, 90.0}
            )

    def test_prepared_grasp_stays_still(self):
        for task in ("hook", "probe", "push", "calibration"):
            np.testing.assert_array_equal(
                trajectory(Spec(task=task), 0.5)[0], np.zeros(3)
            )

    def test_geometry_preserves_hook_opening(self):
        mesh = tool_mesh("hook")
        self.assertTrue(mesh.is_watertight)
        # Crossbar's initial x lies in the gap between stem and tooth.
        self.assertLess(mesh.volume, mesh.convex_hull.volume * 0.8)

    def test_trajectory_continuity(self):
        for task in ("hook", "probe", "push", "calibration"):
            for pause in (0.25, 0.75, 1.5):
                duration = 14 if task == "probe" else 18.5 if task == "hook" else 12
                p = np.array(
                    [
                        trajectory(Spec(task=task, pause=pause), t)[0]
                        for t in np.arange(0, duration, 0.001)
                    ]
                )
                self.assertLess(
                    np.linalg.norm(np.diff(p, axis=0), axis=1).max() / 0.001, 0.0201
                )
        self.assertEqual(smooth(0), 0)
        self.assertEqual(smooth(1), 1)

    def test_slider_has_open_aperture_and_clearance(self):
        mesh = slider_mesh()
        self.assertTrue(mesh.is_watertight)
        self.assertLess(mesh.volume, mesh.convex_hull.volume * 0.65)
        self.assertAlmostEqual(mesh.bounds[0, 2], -0.054)

    def test_split_groups_tie_replicates_and_hidden_stiffness(self):
        specs = specifications()
        a = asdict(specs[72])
        b = asdict(specs[73])
        duplicate = asdict(specs[72 + 81])
        self.assertEqual(split_group(a), split_group(b))
        self.assertEqual(split_group(a), split_group(duplicate))

    def test_probe_schedule_makes_room_for_both_pulls(self):
        s = Spec(task="probe", duration=14)
        np.testing.assert_allclose(trajectory(s, 5.1)[0], [-0.018, 0, 0.018])
        np.testing.assert_allclose(trajectory(s, 9.0)[0], [0, 0, 0])
        np.testing.assert_allclose(trajectory(s, 14)[0], [-0.018, 0, 0.018])

    def test_shuffle_control_never_reads_future(self):
        indices = past_indices(140, np.random.default_rng(1))
        self.assertTrue(np.all(indices <= np.arange(140)))
        self.assertTrue(np.all(indices >= 0))

    def test_hook_unloads_before_lowering(self):
        s = Spec(task="hook")
        self.assertEqual(s.duration, 18.5)
        np.testing.assert_allclose(trajectory(s, 16.5)[0], [0, 0, 0.018])
        np.testing.assert_allclose(trajectory(s, 18.3)[0], [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
