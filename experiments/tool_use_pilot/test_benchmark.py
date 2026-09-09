"""Fast benchmark geometry, configuration and serialization contracts."""

import unittest
from dataclasses import asdict
import numpy as np
from benchmark import BenchmarkSpec, BenchmarkMixin, specifications, composite_gate
from benchmark_geometry import (
    profile,
    offset_profile,
    peg_mesh,
    socket_mesh,
    height_surface,
    guide_wall,
    sequence_completion,
    friction_provenance,
)
from benchmark_observations import TactileCorruption, corrupt_field


class BenchmarkTests(unittest.TestCase):
    def test_representative_gallery_covers_active_variants(self):
        import json
        from pathlib import Path
        from benchmark_gallery import validate_coverage

        rows = json.loads(
            Path(__file__).with_name("representative_episodes.json").read_text()
        )
        validate_coverage(rows)
        for invalid in (rows[:-1], rows + [rows[0]]):
            with self.assertRaises(ValueError):
                validate_coverage(invalid)

    def test_file_completion_honors_turn_and_hold(self):
        import h5py
        import tempfile
        from pathlib import Path
        from benchmark import completion_from_file

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode.h5"
            with h5py.File(path, "w") as f:
                d = f.create_group("observations")
                d["phase"] = np.array(["done"] * 30, dtype=h5py.string_dtype())
                d["rotor_angle_rad"] = np.full(30, np.pi / 2)
                d["insertion_depth_m"] = np.full(30, 0.016)
                d["seat_valid"] = np.ones(30, bool)
            spec = dict(dt=0.01, family="turning", return_after_turn=False)
            self.assertTrue(completion_from_file(path, spec)["sequence_complete"])
            self.assertFalse(
                completion_from_file(path, dict(spec, return_after_turn=True))[
                    "sequence_complete"
                ]
            )

    def test_active_recipes_only_use_flat_scraping(self):
        from benchmark import SURFACES, RETIRED_SURFACES

        self.assertNotIn("convex_scrape", SURFACES)
        self.assertIn("convex_scrape", RETIRED_SURFACES)
        for recipe in (
            "screen",
            "qualification",
            "canonical",
            "diversity",
            "randomized",
            "composite",
            "sensing",
        ):
            specs = specifications(recipe)
            self.assertFalse(any(s.variant == "convex_scrape" for s in specs), recipe)
            self.assertTrue(
                all(s.variant == "flat_scrape" for s in specs if "scrape" in s.variant),
                recipe,
            )
        cells = {(s.family, s.variant) for s in specifications("canonical")}
        self.assertEqual(len(cells), 18)
        # Historical configurations/results must not become unreadable.
        self.assertEqual(
            BenchmarkSpec(
                revision=18, family="surface", variant="convex_scrape"
            ).variant,
            "convex_scrape",
        )

    def test_cylindrical_peeling_is_not_a_flat_proxy(self):
        from benchmark_geometry import peeling_cylinder, peeler_working_edge
        from types import SimpleNamespace

        mesh = peeling_cylinder(0.02)
        self.assertTrue(mesh.is_watertight)
        self.assertGreater(mesh.volume, 0)
        np.testing.assert_allclose(mesh.bounds[:, 1], [-0.02, 0.02], atol=1e-12)
        self.assertAlmostEqual(mesh.bounds[1, 2], 0)
        tool = SimpleNamespace(
            vertices=np.array(
                [[0, 0, -0.095], [0, 0.028, -0.098], [0, -0.028, -0.098], [0, 0, 0.04]]
            )
        )
        np.testing.assert_allclose(peeler_working_edge(tool), [0, 0, -0.095])
        spec = BenchmarkSpec(family="surface", variant="cylindrical_peel")
        self.assertEqual(spec.surface_contact_model, "sdf")
        with self.assertRaises(ValueError):
            BenchmarkSpec(
                family="surface",
                variant="cylindrical_peel",
                surface_contact_model="analytic_box",
            )

    def test_peeling_holder_contact_is_not_blade_success(self):
        from benchmark_geometry import peeling_contact_audit

        data = dict(
            initialization=np.zeros(100, bool),
            phase=np.array(["follow"] * 100),
            blade_contact_force_n=np.ones(100),
            holder_contact_force_n=np.zeros(100),
        )
        self.assertTrue(peeling_contact_audit(data)["blade_contact_pass"])
        data["holder_contact_force_n"][0] = 1
        self.assertFalse(peeling_contact_audit(data)["blade_contact_pass"])
        data["blade_contact_force_n"][:] = 0
        self.assertFalse(peeling_contact_audit(data)["blade_contact_pass"])

    def test_extrusion_sampling_preserves_curved_shape(self):
        for kind in ("convex", "concave"):
            a = height_surface(kind, 0.12, 0.07)
            b = height_surface(kind, 0.12, 0.07, 2)
            self.assertTrue(b.is_watertight)
            np.testing.assert_allclose(a.bounds, b.bounds)
            self.assertAlmostEqual(a.volume, b.volume, places=12)

    def test_surface_success_uses_tip_not_root_travel(self):
        from benchmark_geometry import surface_task_pass

        m = dict(
            contact_continuity=1.0,
            lateral_error_p95_m=0.001,
            follow_stroke_pass=True,
            max_progress_mm=7.0,
        )
        self.assertTrue(surface_task_pass(m, "concave_draw"))
        m["follow_stroke_pass"] = False
        m["max_progress_mm"] = 100.0
        self.assertFalse(surface_task_pass(m, "concave_draw"))

    def test_wrench_error_never_rejects_valid_episode(self):
        from benchmark_geometry import episode_eligibility, gel_benchmark_role

        metrics = dict(
            physical_valid=True,
            task_success=True,
            full_rollout_complete=True,
            tactile_valid=False,
            force_rmse=100.0,
            force_nrmse=100.0,
            torque_rmse=100.0,
        )
        result = episode_eligibility(metrics)
        self.assertTrue(result["episode_accepted"])
        self.assertTrue(result["imitation_eligible"])
        self.assertFalse(result["wrench_matching_is_gate"])
        metrics["task_success"] = False
        self.assertTrue(episode_eligibility(metrics)["world_model_eligible"])
        self.assertFalse(episode_eligibility(metrics)["imitation_eligible"])
        for key, value in [
            ("physical_valid", False),
            ("full_rollout_complete", False),
            ("contact_model_review_required", True),
            ("benchmark_variant_retired", True),
        ]:
            self.assertFalse(
                episode_eligibility(dict(metrics, **{key: value}))["episode_accepted"]
            )
        self.assertFalse(episode_eligibility(metrics, True)["episode_accepted"])
        self.assertEqual(BenchmarkSpec().gel_geometry, "source_surface")
        self.assertEqual(gel_benchmark_role("source_surface"), "primary")
        self.assertEqual(gel_benchmark_role("matched_box"), "flat_reference")
        self.assertEqual(
            BenchmarkSpec(gel_geometry="matched_box").gel_geometry, "matched_box"
        )

    def test_surface_release_motion_cannot_count_as_scraping(self):
        from benchmark_geometry import surface_stroke_audit

        phase = np.array(["follow"] * 100 + ["release"] * 100)
        tip = np.zeros((200, 3))
        tip[100:, 0] = 0.030
        self.assertFalse(
            surface_stroke_audit(phase, tip, 0.01, 0.020)["follow_stroke_pass"]
        )
        tip[:100, 0] = np.linspace(0, 0.030, 100)
        self.assertTrue(
            surface_stroke_audit(phase, tip, 0.01, 0.020)["follow_stroke_pass"]
        )

    def test_supported_stroke_preserves_authored_grasp(self):
        from benchmark import SurfaceWorld
        from dataclasses import replace
        from scipy.spatial.transform import Rotation
        from types import SimpleNamespace

        s = BenchmarkSpec(
            family="surface", variant="straight_wall", surface_heading_deg=0
        )
        m0 = SurfaceWorld.make_tool_mesh(SimpleNamespace(spec=s))
        m1 = SurfaceWorld.make_tool_mesh(
            SimpleNamespace(spec=replace(s, surface_heading_deg=90))
        )
        np.testing.assert_allclose(
            m0.vertices,
            Rotation.from_euler("z", 90, degrees=True).apply(m1.vertices),
            atol=1e-12,
        )

    def test_supported_scrape_and_explicit_impedance(self):
        s = BenchmarkSpec(family="surface", variant="flat_scrape")
        self.assertEqual(s.surface_heading_deg, 90)
        self.assertFalse(s.tool_pose_feedback)
        self.assertEqual(
            BenchmarkSpec(
                revision=12, family="surface", variant="flat_scrape"
            ).surface_heading_deg,
            0,
        )
        self.assertTrue(
            BenchmarkSpec(
                revision=12, family="surface", variant="flat_scrape"
            ).tool_pose_feedback
        )
        for kw in (dict(arm_stiffness_n_m=0), dict(arm_damping_ns_m=float("nan"))):
            with self.assertRaises(ValueError):
                BenchmarkSpec(**kw)

    def test_curved_default_and_legacy_replay(self):
        self.assertEqual(BenchmarkSpec().gel_geometry, "source_surface")
        self.assertEqual(BenchmarkSpec(revision=11).gel_geometry, "legacy_box")
        self.assertEqual(
            BenchmarkSpec(revision=11, gel_geometry="source_surface").gel_geometry,
            "source_surface",
        )

    def test_repaired_surface_and_joint_defaults_preserve_replay(self):
        for variant in (
            "straight_wall",
            "rounded_wall",
            "straight_slot",
            "curved_slot",
        ):
            self.assertEqual(
                BenchmarkSpec(family="surface", variant=variant).surface_heading_deg, 45
            )
            self.assertEqual(
                BenchmarkSpec(
                    revision=17, family="surface", variant=variant
                ).surface_heading_deg,
                90,
            )
        peel = BenchmarkSpec(family="surface", variant="cylindrical_peel")
        self.assertEqual(
            (peel.surface_force_n, peel.surface_contact_model), (1.0, "sdf")
        )
        draw = BenchmarkSpec(family="surface", variant="concave_draw")
        self.assertTrue(draw.surface_working_tip_tracking)
        self.assertEqual(draw.surface_lateral_samples, 2)
        self.assertFalse(
            BenchmarkSpec(
                revision=17, family="surface", variant="concave_draw"
            ).surface_working_tip_tracking
        )
        self.assertEqual(
            BenchmarkSpec(family="turning", load_mode="friction").rotor_friction_model,
            "joint",
        )
        self.assertEqual(
            BenchmarkSpec(
                revision=17, family="turning", load_mode="friction"
            ).rotor_friction_model,
            "external",
        )

    def test_mechanics_defaults_preserve_historical_configuration(self):
        modern = BenchmarkSpec(family="surface", variant="straight_wall")
        old = BenchmarkSpec(revision=14, family="surface", variant="straight_wall")
        self.assertEqual(modern.finger_coupling_stiffness_n_m, 1e6)
        self.assertTrue(modern.ideal_arm_friction_compensation)
        self.assertTrue(modern.guide_acquisition_gate)
        self.assertEqual(modern.guide_acquisition_force_n, 0.1)
        self.assertEqual(old.finger_coupling_stiffness_n_m, 0)
        self.assertFalse(old.ideal_arm_friction_compensation)
        self.assertFalse(old.guide_acquisition_gate)
        self.assertEqual(old.guide_acquisition_force_n, 0.25)
        self.assertEqual(
            BenchmarkSpec(
                family="turning", load_mode="friction"
            ).turning_return_overtravel_deg,
            12,
        )
        self.assertEqual(
            BenchmarkSpec(
                revision=14, family="turning", load_mode="friction"
            ).turning_return_overtravel_deg,
            0,
        )

    def test_closed_profiles_and_pockets(self):
        for shape in ("round", "square", "hex", "d", "key"):
            for mesh in (peg_mesh(shape), socket_mesh(shape)):
                self.assertTrue(mesh.is_watertight, shape)
                self.assertTrue(mesh.is_winding_consistent, shape)
                self.assertGreater(mesh.volume, 0, shape)
            np.testing.assert_array_equal(
                socket_mesh(shape).contains(
                    [[0, 0, -0.01], [0, 0, -0.018], [0.02, 0, -0.01]]
                ),
                [False, True, True],
            )

    def test_clearance_is_outward(self):
        for shape in ("round", "square", "hex", "d", "key"):
            p = profile(shape)
            q = offset_profile(p, 0.0006)
            self.assertTrue(
                np.all(np.linalg.norm(q, axis=1) > np.linalg.norm(p, axis=1))
            )

    def test_closed_surfaces(self):
        for mesh in [height_surface(s) for s in ("flat", "convex", "concave")] + [
            guide_wall(c, s) for c in (False, True) for s in (-1, 1)
        ]:
            self.assertTrue(mesh.is_watertight)
            self.assertTrue(mesh.is_winding_consistent)
            self.assertGreater(mesh.volume, 0)

    def test_budget_and_roundtrip(self):
        specs = sum(
            [specifications(p) for p in ("canonical", "diversity", "composite")], []
        )
        self.assertLessEqual(len(specs) + 32, 256)
        for s in specs:
            self.assertEqual(s, BenchmarkSpec(**asdict(s)))
            self.assertLessEqual(s.grip_force_n, 35)

    def test_configuration_rejects_bad_inputs(self):
        for kwargs in (
            dict(grip_force_n=40),
            dict(clearance_m=-1),
            dict(load_mode="magic"),
            dict(family="unknown"),
        ):
            with self.assertRaises(ValueError):
                BenchmarkSpec(**kwargs)

    def test_stage_and_randomized_coverage(self):
        expected = {
            "surface": "surface_following",
            "hook": "capture_pull",
            "insertion": "insertion",
            "turning": "turning",
            "composite": "insertion",
        }
        for family, stage in expected.items():
            s = BenchmarkSpec(
                family=family, variant="flat_scrape" if family == "surface" else "key"
            )
            self.assertEqual(s.stage, stage)
            self.assertEqual(s.branch_time, 5 if family == "hook" else 8)
        specs = specifications("randomized")
        self.assertEqual(len(specs), 16)
        for family in ("surface", "hook", "insertion", "turning"):
            self.assertEqual(sum(s.family == family for s in specs), 4)
        self.assertTrue(all(s.speed_scale <= 1 for s in specs if s.family == "turning"))

    def test_tactile_corruption_is_reproducible_and_nonmutating(self):
        field = np.ones((2, 7, 9, 3))
        cfg = TactileCorruption(noise_n=0.01, bias_n=0.02, saturation_n=0.5)
        a = corrupt_field(field, 20, 0.01, 42, cfg)
        np.testing.assert_array_equal(a, corrupt_field(field, 20, 0.01, 42, cfg))
        np.testing.assert_array_equal(field, np.ones_like(field))
        self.assertLessEqual(abs(a).max(), 0.5)

    def test_effective_environment_friction_is_independent_of_gel(self):
        from types import SimpleNamespace

        for gel in (1.0, 1.4, 1.8):
            obj = SimpleNamespace(
                spec=BenchmarkSpec(
                    gel_friction=gel, friction=0.5, effective_friction=True
                )
            )
            coefficient = BenchmarkMixin.environment_friction_coefficient(obj)
            self.assertAlmostEqual(np.sqrt(coefficient * gel), 0.5)

    def test_composite_requires_complete_nominal_mechanics(self):
        records = []
        for family in ("insertion", "turning"):
            for i in range(3):
                s = BenchmarkSpec(
                    episode_id=i + (3 if family == "turning" else 0),
                    family=family,
                    case="nominal",
                )
                records.append(
                    dict(
                        spec=asdict(s),
                        metrics=dict(
                            physical_valid=True,
                            task_success=True,
                            full_rollout_complete=True,
                            sequence_complete=True,
                        ),
                    )
                )
        self.assertEqual(composite_gate(records)["insertion"], [0, 1, 2])
        records[-1]["metrics"]["full_rollout_complete"] = False
        with self.assertRaises(ValueError):
            composite_gate(records)
        records[-1]["metrics"]["full_rollout_complete"] = True
        records[-1]["spec"]["case"] = "near_miss"
        with self.assertRaises(ValueError):
            composite_gate(records)

    def test_independent_overlap_catches_missed_contact_depth(self):
        import tempfile
        import h5py
        import trimesh
        from benchmark_report import surface_overlap

        with (
            tempfile.NamedTemporaryFile(suffix=".h5") as file,
            h5py.File(file.name, "w") as f,
        ):
            for key, mesh in (
                ("0", trimesh.creation.box([0.004, 0.004, 0.004])),
                ("1", height_surface("flat")),
            ):
                g = f.create_group("geometry/" + key)
                g["vertices"] = mesh.vertices
                g["faces"] = mesh.faces
            d = f.create_group("observations")
            poses = np.zeros((4, 2, 7))
            poses[:, :, 6] = 1
            d["body_root_poses"] = poses
            d["initialization"] = np.zeros(4, bool)
            result = surface_overlap(f)
            self.assertTrue(result["failed"])
            self.assertAlmostEqual(result["max_sampled_overlap_mm"], 2)
            poses[:, 0, 2] = 0.003
            d["body_root_poses"][:] = poses
            self.assertFalse(surface_overlap(f)["failed"])

    def test_composite_done_command_is_not_enough(self):
        data = dict(
            phase=np.array(["done"] * 30),
            rotor_angle_rad=np.zeros(30),
            insertion_depth_m=np.full(30, 0.015),
        )
        self.assertFalse(
            sequence_completion(data, "composite", 0.01)["sequence_complete"]
        )
        data["insertion_depth_m"][:] = -0.003
        self.assertTrue(
            sequence_completion(data, "composite", 0.01)["sequence_complete"]
        )
        data["rotor_angle_rad"][:] = np.pi / 2
        self.assertFalse(
            sequence_completion(data, "composite", 0.01)["sequence_complete"]
        )

    def test_turn_and_hold_without_return(self):
        data = dict(
            phase=np.array(["done"] * 30),
            rotor_angle_rad=np.full(30, np.pi / 2),
            insertion_depth_m=np.full(30, 0.015),
            seat_valid=np.ones(30, bool),
        )
        for family in ("turning", "composite"):
            self.assertTrue(
                sequence_completion(data, family, 0.01, False)["sequence_complete"]
            )
            self.assertFalse(
                sequence_completion(data, family, 0.01, True)["sequence_complete"]
            )
        data["seat_valid"][-1] = False
        self.assertFalse(
            sequence_completion(data, "composite", 0.01, False)["sequence_complete"]
        )
        data["seat_valid"][:] = True
        data["rotor_angle_rad"][-1] = np.deg2rad(96)
        self.assertFalse(
            sequence_completion(data, "turning", 0.01, False)["sequence_complete"]
        )
        self.assertFalse(BenchmarkSpec(family="turning").return_after_turn)
        self.assertTrue(BenchmarkSpec(revision=15, family="turning").return_after_turn)

    def test_key_command_stays_turned_and_seated(self):
        from types import SimpleNamespace
        from scipy.spatial.transform import Rotation
        from benchmark import PegWorld

        for family in ("turning", "composite"):
            for return_after in (False, True):
                spec = BenchmarkSpec(family=family, return_after_turn=return_after)
                world = SimpleNamespace(
                    spec=spec,
                    clock=lambda t: t,
                    tool=SimpleNamespace(
                        get_root_transform=lambda: SimpleNamespace(
                            translation=np.zeros(3)
                        )
                    ),
                    rot=Rotation.identity(),
                    origin=np.zeros(3),
                    ee0=np.zeros(3),
                    turn_start=10.0,
                    turn_anchor_yaw=0.0,
                    search_start=-1.0,
                    force_guard=False,
                    last_force=0.0,
                )
                value, phase, rotation = PegWorld.command_at(world, 50.0)
                self.assertEqual(phase, "done")
                self.assertAlmostEqual(
                    rotation[2],
                    0.0 if return_after else np.deg2rad(spec.turn_command_deg),
                )
                self.assertAlmostEqual(
                    value[2],
                    -0.019 if family == "composite" and not return_after else 0.0,
                )

    def test_peg_tool_friction_is_not_environment_friction(self):
        from types import SimpleNamespace
        from benchmark import PegWorld

        context = SimpleNamespace(p=SimpleNamespace(ContactParams=SimpleNamespace))
        self.assertEqual(
            PegWorld.contact_params(context, 1.4).coulomb_friction_coefficient, 1.4
        )
        spec = asdict(BenchmarkSpec(family="turning", friction=0.5, gel_friction=1.4))
        old = friction_provenance(
            spec,
            "class PegWorld:\n def contact_params(self, friction): pass\n def make_tool_mesh",
        )
        self.assertAlmostEqual(old["effective_gel_tool_friction"], np.sqrt(0.7))
        self.assertAlmostEqual(old["effective_environment_tool_friction"], 0.5)
        self.assertFalse(old["independent_friction_sweep_valid"])


if __name__ == "__main__":
    unittest.main()
