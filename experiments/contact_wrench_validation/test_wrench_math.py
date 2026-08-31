"""Unit tests for contact-wrench signs, moments, and frame transforms."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "contact_wrench_math", Path(__file__).with_name("wrench_math.py")
)
assert _SPEC is not None and _SPEC.loader is not None
_WRENCH_MATH = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _WRENCH_MATH
_SPEC.loader.exec_module(_WRENCH_MATH)
aggregate_contact_points = _WRENCH_MATH.aggregate_contact_points
inertia_matrix = _WRENCH_MATH.inertia_matrix
opposite_wrench = _WRENCH_MATH.opposite_wrench
rotate_wrench_to_local = _WRENCH_MATH.rotate_wrench_to_local
shift_wrench = _WRENCH_MATH.shift_wrench


class WrenchMathTest(unittest.TestCase):
    def test_shift_wrench(self) -> None:
        shifted = shift_wrench([0, 1, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0])
        np.testing.assert_allclose(shifted.force, [0, 1, 0])
        np.testing.assert_allclose(shifted.torque, [0, 0, 1])

    def test_opposite_wrench(self) -> None:
        opposite = opposite_wrench([0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 0])
        np.testing.assert_allclose(opposite.force, [0, -1, 0])
        np.testing.assert_allclose(opposite.torque, [0, 0, -2])

    def test_rotate_wrench_to_local(self) -> None:
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        local = rotate_wrench_to_local([0, 1, 0], [1, 0, 0], rotation)
        np.testing.assert_allclose(local.force, [1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(local.torque, [0, -1, 0], atol=1e-12)

    def test_contact_actor_a_and_b_normalization(self) -> None:
        receiver, other = 11, 22
        contacts = [
            SimpleNamespace(
                actor_a=receiver,
                actor_b=other,
                force=[1, 0, 0],
                pos_a=[0, 1, 0],
                pos_b=[0, 1, 0],
            ),
            SimpleNamespace(
                actor_a=other,
                actor_b=receiver,
                force=[-1, 0, 0],
                pos_a=[0, -1, 0],
                pos_b=[0, -1, 0],
            ),
        ]
        wrench = aggregate_contact_points(contacts, receiver, other, [0, 0, 0])
        np.testing.assert_allclose(wrench.force, [2, 0, 0])
        np.testing.assert_allclose(wrench.torque, [0, 0, 0])
        self.assertEqual(wrench.count, 2)

    def test_inertia_unpack(self) -> None:
        matrix = inertia_matrix([1, 2, 3, 4, 5, 6])
        np.testing.assert_allclose(matrix, [[1, 2, 3], [2, 4, 5], [3, 5, 6]])


if __name__ == "__main__":
    unittest.main()
