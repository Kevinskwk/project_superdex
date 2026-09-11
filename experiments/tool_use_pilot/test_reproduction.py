"""The committed replay inputs must remain independent of ignored outputs."""

from dataclasses import asdict
import json
from pathlib import Path
import unittest

from benchmark import BenchmarkSpec


ROOT = Path(__file__).parent / "reproduction"


class ReproductionTests(unittest.TestCase):
    def test_frozen_specs_preserve_recorded_parameters(self):
        evidence = json.loads((ROOT / "expected_results.json").read_text())
        by_id = {r["spec"]["episode_id"]: r for r in evidence["episodes"]}
        for filename, success in [
            ("nominal_specs.json", True),
            ("negative_specs.json", False),
        ]:
            rows = json.loads((ROOT / filename).read_text())
            self.assertEqual(len(rows), 21 if success else 1)
            for payload in rows:
                with self.subTest(episode=payload["episode_id"]):
                    actual = asdict(BenchmarkSpec(**payload))
                    for key, value in payload.items():
                        self.assertEqual(actual[key], value, key)
                    reference = by_id[payload["episode_id"]]
                    self.assertEqual(reference["spec"], payload)
                    metrics = reference["metrics"]
                    self.assertEqual(
                        bool(metrics["physical_valid"] and metrics["task_success"]),
                        success,
                    )
                    self.assertTrue(metrics["full_rollout_complete"])
                    self.assertFalse(metrics["wrench_matching_is_gate"])
                    self.assertEqual(payload["gel_geometry"], "source_surface")
                    self.assertFalse(payload["record_dense_field"])

    def test_evidence_is_portable_and_failed_roll_not_promoted(self):
        negative = json.loads((ROOT / "negative_specs.json").read_text())
        self.assertEqual(negative[0]["case"], "roll_10")
        evidence = json.loads((ROOT / "expected_results.json").read_text())
        for row in evidence["episodes"]:
            self.assertFalse(Path(row["source_episode"]).is_absolute())
            self.assertEqual(len(row["source_episode_sha256"]), 64)
            self.assertEqual(len(row["source_video_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
