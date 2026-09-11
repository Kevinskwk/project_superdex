import unittest
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from dataclasses import asdict, replace

import numpy as np

from diversification import baseline, manifest, stamp, STAGES, reviewed_variants
from benchmark import SurfaceWorld
from transfer_world import TransferWorld
from scipy.spatial.transform import Rotation
from diversification_observations import CloudAugmenter, CONDITIONS, CloudCondition
from diversification_repairs import repair_specs
from diversification_review import repair_margin


class DiversificationTests(unittest.TestCase):
    def test_stronger_margin_does_not_relabel_legacy_success(self):
        s = dict(family="surface", variant="curved_slot", surface_stroke_m=0.02)
        old = dict(
            imitation_eligible=True,
            task_success=True,
            follow_tip_travel_mm=25.65,
            guide_contact_continuity=0.841,
        )
        self.assertFalse(repair_margin(s, old)["passed"])
        self.assertTrue(old["task_success"])
        self.assertTrue(
            repair_margin(
                s, dict(old, follow_tip_travel_mm=20.3, guide_contact_continuity=1.0)
            )["passed"]
        )
        p = dict(family="pushing", variant="pose")
        self.assertFalse(
            repair_margin(
                p,
                dict(
                    imitation_eligible=True,
                    final_position_error_m=0.00979,
                    final_yaw_error_deg=2.74,
                ),
            )["passed"]
        )

    def test_guide_tracks_tip_not_root_and_preserves_force_axes(self):
        s = next(s for s in repair_specs() if s.variant == "curved_slot")
        s = replace(s, guide_acquisition_gate=False, calibrate_precontact_grasp=False)
        w = object.__new__(SurfaceWorld)
        w.spec = s
        w.rot = w.surface_rot = Rotation.identity()
        w.origin = w.ee0 = w.surface_origin = np.zeros(3)
        w.working_end = np.array([0.0, 0.0, -0.06])
        rotation = Rotation.from_euler("y", -5, degrees=True).as_rotvec()
        root = SimpleNamespace(
            translation=np.array([0.020, 0, 0]),
            rotation=SimpleNamespace(to_rotation_vector=lambda: rotation),
        )
        w.tool = SimpleNamespace(get_root_transform=lambda: root)
        w.mesh = SimpleNamespace(
            vertices=np.array([[0.0, 0.0, -0.06]]),
            bounds=np.array([[0.0, 0.0, -0.06], [0.0, 0.0, 0.0]]),
        )
        w.normal_offset = w.surface_wait_s = w.lateral_offset = 0.0
        w.last_normal_force = w.last_guide_force = 0.5
        w.tracking_correction = np.zeros(3)
        value, _, _ = w.command_at(18)
        self.assertLess(w.tracking_correction[0], 0)
        self.assertLessEqual(abs(w.tracking_correction[0]), s.dt * 0.002)
        np.testing.assert_array_equal(w.tracking_correction[1:], [0, 0])
        # Actual tip is >25 mm along the curve, despite a 20 mm root position.
        self.assertGreater(value[1], 0.006)

    def test_push_path_is_gentle_and_does_not_move_goal(self):
        for s in repair_specs()[4:]:
            w = object.__new__(TransferWorld)
            w.spec = s
            w.calibration = np.zeros(3)
            w.rot = Rotation.identity()
            w.origin = w.ee0 = np.zeros(3)
            times = np.arange(0, s.duration, s.dt)
            values = np.array([w.command_at(t)[0] for t in times])
            speed = np.linalg.norm(np.diff(values, axis=0) / s.dt, axis=1)
            # Pre-contact lateral alignment is <21 mm/s (the old path was
            # 20.6 mm/s); actual pushing is <8 mm/s.
            self.assertLess(speed.max(), 0.021)
            self.assertLess(speed[times[:-1] >= 8].max(), 0.008)
            np.testing.assert_allclose(
                values[-1],
                [
                    s.push_distance_m + s.push_overtravel_m,
                    s.push_contact_y_m + s.push_end_lateral_m,
                    0,
                ],
            )
            self.assertEqual(s.push_distance_m, 0.04)
            self.assertEqual(s.push_yaw_deg, 15)

    def test_repairs_are_bounded_and_not_randomized(self):
        rows = repair_specs()
        self.assertEqual(len(rows), 6)
        for s in rows:
            self.assertEqual(s.diversification_stage, "repair")
            self.assertEqual(s.robot_roll_deg, 0)
            self.assertEqual(s.grip_force_n, 35)
            self.assertEqual(s.gel_modulus_pa, 200000)
        self.assertTrue(all(s.tool_pose_feedback for s in rows[:4]))
        self.assertTrue(all(abs(s.push_end_lateral_m) <= 0.010 for s in rows[4:]))
        followup = repair_specs(3)
        self.assertEqual({s.variant for s in followup}, {"rounded_wall", "curved_slot"})
        self.assertTrue(all(s.guide_max_offset_m == 0.014 for s in followup))
        self.assertTrue(all(s.guide_max_speed_m_s == 0.001 for s in followup))
        self.assertTrue(all(s.surface_force_n == 0.5 for s in followup))

    def test_review_gate_requires_entire_exact_envelope(self):
        specs = [s for s in manifest("geometry") if s.family == "hook"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries, rows = [], {}
            for i, s in enumerate(specs):
                p = root / f"{i}.h5"
                p.write_bytes(b"mock episode")
                p.with_suffix(".json").write_text("{}")
                entries.append(
                    dict(
                        episode=p.name,
                        sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                        visual_review_passed=True,
                        housing_clearance_passed=True,
                    )
                )
                rows[p] = dict(spec=asdict(s), metrics=dict(imitation_eligible=True))
            review = root / "review.json"
            review.write_text(json.dumps(dict(episodes=entries)))
            with (
                patch("benchmark_report.summarize", side_effect=lambda p: rows[p]),
                patch(
                    "benchmark_report.geometry_audit", return_value=dict(passed=True)
                ),
            ):
                self.assertEqual(
                    reviewed_variants(review, "geometry"), {("hook", "friction_normal")}
                )
                entries[0]["visual_review_passed"] = False
                review.write_text(json.dumps(dict(episodes=entries)))
                self.assertEqual(reviewed_variants(review, "geometry"), set())
                entries[0]["visual_review_passed"] = True
                review.write_text(json.dumps(dict(episodes=entries)))
                rows[root / "0.h5"]["spec"]["grip_force_n"] = 30
                self.assertEqual(reviewed_variants(review, "geometry"), set())

    def test_observation_reproducibility_and_empty_stream(self):
        cloud = np.random.default_rng(5).uniform(-0.1, 0.1, (300, 3))
        a = CloudAugmenter([cloud, cloud], np.array([0.0, -1, 0.5]), np.zeros(3), 4)
        x, counts = a.apply([cloud, cloud], CONDITIONS[-1], 7)
        y, counts2 = a.apply([cloud, cloud], CONDITIONS[-1], 7)
        np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(counts, counts2)
        self.assertLess(counts[0], len(cloud))
        empty, counts = a.apply([cloud[:0], cloud], CONDITIONS[0], 7)
        self.assertEqual(counts[0], 0)
        self.assertFalse(empty[0].any())
        with self.assertRaises(ValueError):
            CloudCondition("bad", patch_fraction=1)

    def test_roster_and_counts(self):
        self.assertEqual([len(manifest(s)) for s in STAGES], [19, 6, 2, 48, 64, 12, 76])
        self.assertFalse(
            any(s.family == "levering" or "mirrored" in s.variant for s in baseline())
        )
        self.assertEqual(sum(len(manifest(s)) for s in STAGES), 227)

    def test_reproducible_and_safe(self):
        for stage in STAGES:
            a, b = manifest(stage), manifest(stage)
            self.assertEqual([asdict(s) for s in a], [asdict(s) for s in b])
            for s in a:
                self.assertEqual(s.revision, 21)
                self.assertLessEqual(s.grip_force_n, 35)
                self.assertFalse(s.return_after_turn)
                self.assertEqual(s.gel_geometry, "source_surface")
                self.assertTrue(s.geometry_id and s.physics_id and s.split_group)

    def test_ids_follow_factors_not_observation_seed(self):
        s = baseline()[0]
        a = stamp(s, "geometry", "a", {}, 0)
        b = stamp(s, "geometry", "b", dict(camera_seed=333), 1)
        self.assertEqual(a.split_group, b.split_group)
        c = stamp(s, "geometry", "c", dict(handle_width_m=0.019), 2)
        self.assertNotEqual(a.geometry_id, c.geometry_id)
        self.assertEqual(a.physics_id, c.physics_id)

    def test_working_geometry_changes_mesh(self):
        for family, variant, cls in (
            ("surface", "flat_scrape", SurfaceWorld),
            ("surface", "cylindrical_peel", SurfaceWorld),
            ("pushing", "pose", TransferWorld),
        ):
            s = next(
                s for s in baseline() if s.family == family and s.variant == variant
            )

            def mesh(spec):
                w = object.__new__(cls)
                w.spec = spec
                w.contact_mesh = lambda m: m
                return w.make_tool_mesh()

            a = mesh(s)
            b = mesh(replace(s, handle_width_m=s.handle_width_m * 1.1))
            self.assertTrue(a.is_watertight and b.is_watertight)
            self.assertGreater(abs(a.volume - b.volume), 1e-9)


if __name__ == "__main__":
    unittest.main()
