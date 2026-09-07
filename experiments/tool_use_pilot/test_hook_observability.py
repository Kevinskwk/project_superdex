"""Leakage, corruption and grouping checks for the local sensing diagnostic."""

import unittest
import numpy as np
from hook_observability import (
    Condition,
    augmentation_seed,
    corrupt,
    design,
    observed_relations,
    split_level,
)


class HookObservabilityTests(unittest.TestCase):
    def test_unseen_anchor_has_unseen_corruptions(self):
        seeds = [augmentation_seed(a, v) for a in range(12) for v in range(4)]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertEqual(augmentation_seed(1, 2), augmentation_seed(1, 2))

    def test_corruption_repeatable_and_temporal_noise_changes(self):
        rng = np.random.default_rng(42)
        points = rng.normal(0, 0.03, (2000, 3))
        args = (
            [points, points],
            np.array([0.0, -0.3, 0.2]),
            np.zeros(3),
            Condition("test", 0.001, 0.25),
            123,
        )
        a, n = corrupt(*args, 0)
        b, m = corrupt(*args, 0)
        c, _ = corrupt(*args, 1)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(n, m)
        self.assertTrue(np.all(n == 1500))
        self.assertFalse(np.array_equal(a, c))

    def test_empty_visible_stream_is_not_hallucinated(self):
        cloud, n = corrupt(
            [np.empty((0, 3)), np.empty((0, 3))],
            np.array([0.0, -0.3, 0.2]),
            np.zeros(3),
            Condition("test", 0.002, 0.5, True),
            12,
            0,
        )
        self.assertFalse(cloud.any())
        self.assertFalse(n.any())
        self.assertEqual(cloud.shape, (2, 1024, 3))

    def test_scaler_cannot_see_heldout_values(self):
        x = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [4.0, 5.0]])
        original = design([x], [0, 1, 2])
        x[3] = 1e6
        changed = design([x], [0, 1, 2])
        np.testing.assert_array_equal(original[:3], changed[:3])

    def test_offset_level_groups_both_labels_and_mirrors(self):
        for delta in (-0.0012, 0, 0.0012):
            for case, base in [("captured", 0.0375), ("near_miss", 0.0455)]:
                for sign in (-1, 1):
                    self.assertEqual(
                        split_level(dict(case=case, offset=sign * (base + delta))),
                        round(delta * 1e6),
                    )

    def test_relations_include_mirror_invariant_distance(self):
        m = np.zeros((2, 102))
        m[:, 17] = [-0.04, 0.04]
        r = observed_relations(m)
        np.testing.assert_allclose(r[0, 3:7], r[1, 3:7])
        self.assertAlmostEqual(r[0, 6], 0.04)


if __name__ == "__main__":
    unittest.main()
