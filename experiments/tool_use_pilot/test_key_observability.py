"""Leakage, complete-group gating, and outcome-frame tests for key probes."""

import json
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from key_observability import target_record, qualify, paired_gain, conditions, WINDOWS, tare_wrench_features, axial_wrench_features


class FakeFile(dict):
    pass


class KeyObservabilityTests(unittest.TestCase):
    def test_complete_physical_groups_required(self):
        rows = [dict(group=g, label=i % 2, metrics={"physical_valid": True})
                for g in (-2, 0, 2) for i in range(4)]
        self.assertTrue(qualify(rows)["passed"])
        self.assertFalse(qualify(rows[:-1])["passed"])
        rows[0]["metrics"]["physical_valid"] = False
        self.assertFalse(qualify(rows)["passed"])

    def test_turn_target_uses_measured_axis_moment_not_authored_spring(self):
        n = 101
        poses = np.zeros((n, 3, 7)); poses[:, :, 6] = 1
        com = np.zeros((n, 7)); com[:, 1] = 1; com[:, 6] = 1
        wrench = np.zeros((n, 6)); wrench[:, 0] = 1; wrench[:, 5] = .96
        # r x F = -1 Nm: moment at rotor is -0.04 Nm, hence +0.04 resistance.
        f = FakeFile(observations=dict(timestamps=np.linspace(16,17,n), body_root_poses=poses,
                                      tool_pose=com, extrinsic_contact_wrench=wrench))
        f.attrs = {"config_json": json.dumps(dict(stage="turning",dt=.01,grip_force_n=35,torsion_nm_rad=.001)),
                   "metrics_json": json.dumps(dict(physical_valid=True,task_success=True))}
        r = target_record(f)
        self.assertAlmostEqual(r["future_resistance_nm"], .04)
        self.assertEqual(r["label"], 1)
        self.assertEqual(r["group"], 35)

    def test_insertion_label_is_outcome_not_case_name(self):
        f = FakeFile(observations={})
        f.attrs = {"config_json": json.dumps(dict(stage="insertion",case="aligned",yaw_deg=2)),
                   "metrics_json": json.dumps(dict(physical_valid=True,task_success=False))}
        self.assertEqual(target_record(f)["label"], 0)

    def test_window_and_mask_protocol(self):
        self.assertLess(WINDOWS["insertion"]["after"], 10)
        self.assertLess(WINDOWS["turning"]["after"], 16)
        a, b = conditions("insertion"), conditions("turning")
        self.assertEqual([x.name for x in a],[x.name for x in b])
        for x, y in zip(a,b):
            if x.contact_window:
                self.assertAlmostEqual(y.window_center_z_m-x.window_center_z_m, .019)

    def test_paired_gain_balances_unequal_outcomes(self):
        y = np.array([0]*4+[1]*8)
        v = dict(per_anchor_accuracy=[1]*4+[0]*8)
        vt = dict(per_anchor_accuracy=[1]*12)
        self.assertAlmostEqual(paired_gain(y,v,vt)["balanced_accuracy_gain"], .5)

    def test_wrench_tare_preserves_temporal_difference(self):
        raw = np.arange(18,dtype=float)
        baseline = np.arange(6,dtype=float)
        corrected = tare_wrench_features(raw,baseline)
        np.testing.assert_array_equal(corrected[:6],0)
        np.testing.assert_array_equal(corrected[6:12],6)
        np.testing.assert_array_equal(corrected[12:],raw[12:])

    def test_axis_projection_uses_robot_frame_calibration(self):
        poses = np.zeros((2,7));poses[:,3:] = Rotation.from_euler("y",90,degrees=True).as_quat()
        f = FakeFile({"observations/ee_pose":poses})
        def wrench(_,indices):
            w = np.array([0,0,0,-.02,0,0]) if indices[0] else np.zeros(6)
            return np.r_[w,w,np.zeros(6)]
        with patch("key_observability.wrench_features",side_effect=wrench):
            result = axial_wrench_features(f,[1],0)
        np.testing.assert_allclose(result,[0,.02,0,.02,0,0],atol=1e-8)


if __name__ == "__main__":
    unittest.main()
