"""Stage separation, real outcome criteria, and closed collision geometry."""

import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from key_stages import KeyStageSpec, KeyStageWorld, stage_target, stage_audit, tapered_gate, pocket_mesh, measured_seat


class KeyStageTests(unittest.TestCase):
    def test_insertion_never_commands_turn(self):
        spec = KeyStageSpec(stage="insertion")
        commands = np.array([stage_target(spec, t)[0] for t in np.linspace(0, 17, 171)])
        np.testing.assert_array_equal(commands[:, 3:], 0)
        self.assertAlmostEqual(commands[:, 2].min(), -.019)

    def test_turning_never_commands_insertion(self):
        spec = KeyStageSpec(stage="turning")
        commands = np.array([stage_target(spec, t)[0] for t in np.linspace(0, 26, 261)])
        np.testing.assert_array_equal(commands[:, :3], 0)
        self.assertAlmostEqual(np.rad2deg(commands[:, 5].max()), 100)
        with self.assertRaises(ValueError):
            KeyStageSpec(stage="turning", lateral_m=.001)

    def test_hold_is_no_turn_negative_control(self):
        spec = KeyStageSpec(stage="turning", case="hold_control")
        np.testing.assert_array_equal(stage_target(spec, 17)[0], 0)

    def test_blocked_withdrawal_has_no_deep_target_jump(self):
        world = KeyStageWorld.__new__(KeyStageWorld)
        world.spec = KeyStageSpec(stage="insertion")
        world.rot = Rotation.identity()
        world.origin = np.zeros(3)
        world.tool = SimpleNamespace(get_root_transform=lambda: SimpleNamespace(
            translation=np.array([0, 0, -.007]), rotation=SimpleNamespace(to_rotation_vector=lambda: np.zeros(3))))
        world.last_force = 4.0
        world.force_guard = False
        world.tracking_correction = np.zeros(3)
        world.rotation_correction = np.zeros(3)
        world.command_at(8)
        self.assertTrue(world.force_guard)
        a = world.command_at(11.99)[0]
        b = world.command_at(12.01)[0]
        self.assertLess(np.linalg.norm(a-b), .0001)
        self.assertGreater(b[2], -.008)

    def test_meshes_are_closed_positive_solids(self):
        world = KeyStageWorld.__new__(KeyStageWorld)
        world.spec = KeyStageSpec()
        for mesh in (tapered_gate(), pocket_mesh(), world.make_tool_mesh()):
            self.assertTrue(mesh.is_watertight)
            self.assertTrue(mesh.is_winding_consistent)
            self.assertGreater(mesh.volume, 0)
            self.assertEqual(len(mesh.split()), 1)
        self.assertFalse(tapered_gate().contains([[0, 0, -.001]])[0])
        self.assertFalse(pocket_mesh().contains([[0, 0, -.006]])[0])

    def test_turning_rotation_pivots_at_tool_not_wrist(self):
        world = KeyStageWorld.__new__(KeyStageWorld)
        world.spec = KeyStageSpec(stage="turning")
        world.rot = Rotation.identity()
        world.origin = np.zeros(3)
        world.ee0 = np.array([0, 0, .161])
        world.tool = SimpleNamespace(get_root_transform=lambda: SimpleNamespace(
            translation=np.zeros(3), rotation=SimpleNamespace(to_rotation_vector=lambda: np.array([.02, 0, 0]))))
        world.last_force = 0.0
        world.force_guard = False
        world.tracking_correction = np.zeros(3)
        world.rotation_correction = np.zeros(3)
        command, _, rotation = world.command_at(8)
        offset = world.origin - world.ee0
        np.testing.assert_allclose(command + Rotation.from_rotvec(rotation).apply(offset), offset, atol=1e-12)

    def test_seating_uses_actual_chamfered_blade(self):
        world = KeyStageWorld.__new__(KeyStageWorld)
        world.spec = KeyStageSpec()
        vertices = world.make_tool_mesh().vertices
        blade = vertices[vertices[:, 2] <= -.061 + 1e-8]
        rotation = Rotation.from_euler("z", 9, degrees=True)
        seat, depth = measured_seat(blade, np.zeros(3), rotation,
            np.array([0, 0, -.054]), Rotation.identity(),
            np.array([0, 0, -.058]), Rotation.identity())
        self.assertTrue(seat)
        self.assertAlmostEqual(depth, .015)
        self.assertGreater((abs(rotation.as_matrix()) @ [.005, .003, .004])[1], .0037)

    def test_measured_stage_outcomes_are_distinct(self):
        def evaluate(stage, rotor, seated=True):
            data = dict(initialization=np.zeros(100, bool), timestamps=np.linspace(4, 5, 100),
                        seat_valid=np.full(100, seated), insertion_depth_m=np.full(100, .015),
                        rotor_angle_rad=np.full(100, rotor), rigid_penetration_m=np.zeros(100),
                        prepared_seat_verified=np.ones(100, bool), extrinsic_contact_wrench=np.zeros((100, 6)),
                        force_guard=np.zeros(100, bool))
            with patch("key_stages.metrics", return_value=dict(physical_valid=True, tactile_valid=True)):
                return stage_audit(data, KeyStageSpec(stage=stage), None)
        self.assertTrue(evaluate("insertion", 0)["task_success"])
        self.assertFalse(evaluate("turning", 0)["task_success"])
        self.assertTrue(evaluate("turning", np.pi/2)["task_success"])
        self.assertFalse(evaluate("turning", np.pi/2, False)["task_success"])


if __name__ == "__main__":
    unittest.main()
