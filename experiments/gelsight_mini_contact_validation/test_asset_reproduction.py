"""Offline regression tests for immutable, relocatable asset provisioning."""

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import trimesh

import scfields_assets as assets


class AssetReproductionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="superdex-asset-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.raw = (
            trimesh.creation.box([0.01, 0.02, 0.08]).export(file_type="obj").encode()
        )
        self.digest = hashlib.sha256(self.raw).hexdigest()

    def fake_lock(self):
        source = self.root / "fixture.obj"
        source.write_bytes(self.raw)
        record = assets._mesh_record(
            "fixture",
            "scraper",
            source,
            self.root / "expected/fixture.obj",
            {"tool_type": "scraper", "grasp_offset": 0.025},
            "assets/fixture.obj",
        )
        entry = {
            k: record[k]
            for k in (
                "name",
                "family",
                "source",
                "source_sha256",
                "canonical_sha256",
                "surface_sha256",
                "density_kg_m3",
                "friction",
                "grasp_offset_m",
                "size_quantile",
            )
        }
        return dict(
            tools=[entry],
            revision=assets.HF_REVISION,
            source_dataset="https://example.invalid/dataset",
            license="fixture",
            benchmark_tools={"scraper": "fixture"},
        )

    def test_download_is_pinned_verified_and_cached(self):
        dest = self.root / "download.obj"
        with patch.object(
            assets.urllib.request, "urlopen", return_value=io.BytesIO(self.raw)
        ) as fetch:
            assets.download("assets/fixture.obj", dest, self.digest)
            self.assertIn(assets.HF_REVISION, fetch.call_args.args[0].full_url)
            assets.download("assets/fixture.obj", dest, self.digest)
            self.assertEqual(fetch.call_count, 1)
        self.assertEqual(dest.read_bytes(), self.raw)

    def test_bad_download_is_not_published(self):
        dest = self.root / "download.obj"
        with patch.object(
            assets.urllib.request, "urlopen", return_value=io.BytesIO(b"corrupt")
        ):
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                assets.download("fixture.obj", dest, self.digest)
        self.assertFalse(dest.exists())
        self.assertEqual(list(self.root.glob(".download-*")), [])

    def test_corrupt_existing_source_is_not_overwritten(self):
        dest = self.root / "download.obj"
        dest.write_bytes(b"user data")
        with patch.object(assets.urllib.request, "urlopen") as fetch:
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                assets.download("fixture.obj", dest, self.digest)
            fetch.assert_not_called()
        self.assertEqual(dest.read_bytes(), b"user data")

    def test_prepare_relocation_and_idempotence_without_tacsl(self):
        lock = self.fake_lock()
        with patch.object(assets, "asset_lock", return_value=lock):
            with patch.object(
                assets.urllib.request, "urlopen", return_value=io.BytesIO(self.raw)
            ):
                path = assets.prepare(self.root / "first")
            original = path.read_bytes()
            manifest = json.loads(original)
            self.assertEqual(manifest["capsules"], [])
            for field in ("source_path", "canonical_path", "surface_path"):
                self.assertFalse(Path(manifest["tools"][0][field]).is_absolute())
            moved = self.root / "moved"
            path.parent.rename(moved)
            with patch.object(assets.urllib.request, "urlopen") as fetch:
                assets.prepare(moved)
                fetch.assert_not_called()
            loaded = assets.load_manifest(moved / "manifest.json")
            self.assertTrue(
                Path(loaded["tools"][0]["canonical_path"]).is_relative_to(moved)
            )
            self.assertEqual((moved / "manifest.json").read_bytes(), original)

    def test_generation_drift_is_rejected_before_manifest(self):
        lock = self.fake_lock()
        lock["tools"][0]["canonical_sha256"] = "0" * 64
        dest = self.root / "output"
        with patch.object(assets, "asset_lock", return_value=lock):
            with patch.object(
                assets.urllib.request, "urlopen", return_value=io.BytesIO(self.raw)
            ):
                with self.assertRaisesRegex(ValueError, "Generated canonical mismatch"):
                    assets.prepare(dest)
        self.assertFalse((dest / "manifest.json").exists())
        self.assertFalse((dest / "canonical/fixture.obj").exists())

    def test_paths_cannot_escape_cache(self):
        with self.assertRaisesRegex(ValueError, "escapes"):
            assets._local_path(self.root, {"path": "../outside.obj"}, "path")
        # Legacy absolute paths are rebased even if another checkout still exists.
        path = assets._local_path(
            self.root, {"path": "/old/assets/scfields/canonical/a.obj"}, "path"
        )
        self.assertEqual(path, self.root / "canonical/a.obj")

    def test_checked_in_lock_freezes_recorded_selection(self):
        lock = assets.asset_lock()
        self.assertEqual(len(lock["tools"]), 27)
        self.assertEqual(lock["benchmark_tools"]["scraper"], "tool_scraper_25")
        self.assertEqual(
            lock["benchmark_tools"]["peeler"],
            "peeler_1_head_rectangular_handle_h0_96_hnd0_99",
        )
        for row in lock["tools"]:
            for field in ("source_sha256", "surface_sha256", "canonical_sha256"):
                self.assertEqual(len(row[field]), 64)


if __name__ == "__main__":
    unittest.main()
