"""Fast notebook-helper tests; no released checkpoint or network required."""

import csv
import hashlib
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from zipfile import ZipFile, ZipInfo

import numpy as np
from PIL import Image
import requests
import torch

from animaloc.eval.stitchers import HerdNet_Detection_Branch_Stitcher
from animaloc.models import LossWrapper
from animaloc.utils import owl_demo as demo


class HalfHeatmap(torch.nn.Module):
    def forward(self, images):
        return torch.nn.functional.avg_pool2d(images[:, :1], 2)


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        torch.set_num_threads(2)

    def sample_archive(self):
        image_bytes = io.BytesIO()
        Image.new("RGB", (32, 32)).save(image_bytes, format="PNG")
        content = image_bytes.getvalue()
        archive = self.root / "sample.zip"
        with ZipFile(archive, "w") as stream:
            stream.writestr("OWL_DATA/", b"")
            stream.writestr("OWL_DATA/sample.png", content)
        manifest = {
            "archive_name": "sample.zip", "archive_size": archive.stat().st_size,
            "archive_sha256": demo.sha256_file(archive), "url": None, "root": "OWL_DATA",
            "publication_status": "Synthetic test only.",
            "attribution": {"sources": [
                {"id": "synthetic", "name": "Synthetic fixture", "url": "https://example.org",
                 "verification_status": "test_fixture"},
            ]},
            "files": [{
                "path": "OWL_DATA/sample.png", "size": len(content),
                "source_id": "synthetic",
                "sha256": hashlib.sha256(content).hexdigest(), "width": 32, "height": 32,
            }],
        }
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        return archive, manifest_path, manifest

    def test_prepare_and_reuse_verified_samples(self):
        archive, manifest_path, _ = self.sample_archive()
        output = self.root / "data"
        images = demo.prepare_samples(output, archive, manifest_path=manifest_path)
        initial = (images / "sample.png").stat().st_mtime_ns
        demo.prepare_samples(output, archive, manifest_path=manifest_path)
        self.assertEqual((images / "sample.png").stat().st_mtime_ns, initial)
        self.assertTrue((output / "contact_sheet.png").is_file())
        (images / "sample.png").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "Integrity"):
            demo.prepare_samples(output, archive, manifest_path=manifest_path)
        self.assertEqual((images / "sample.png").read_bytes(), b"corrupt")

    def test_missing_public_sample_url_is_explicit(self):
        _, manifest_path, _ = self.sample_archive()
        with patch.object(demo, "ROOT", self.root / "empty"):
            with self.assertRaisesRegex(FileNotFoundError, "public sample URL"):
                demo.prepare_samples(self.root / "output", manifest_path=manifest_path)

    def test_release_staging_preserves_archive_and_never_overwrites(self):
        archive, manifest_path, manifest = self.sample_archive()
        destination = self.root / "demo_data/release"
        with patch.object(demo, "ROOT", self.root):
            demo.stage_release_assets(destination, archive, manifest_path)
            self.assertEqual(demo.sha256_file(destination / "sample.zip"), manifest["archive_sha256"])
            self.assertEqual(archive.read_bytes(), (destination / "sample.zip").read_bytes())
            self.assertIn("STAGING ONLY", (destination / "RELEASE_NOTES.txt").read_text())
            self.assertEqual(
                json.loads((destination / "SAMPLE_ATTRIBUTION.json").read_text()), manifest,
            )
            with self.assertRaisesRegex(ValueError, "not be overwritten"):
                demo.stage_release_assets(destination, archive, manifest_path)
            with self.assertRaisesRegex(ValueError, "ignored demo_data"):
                demo.stage_release_assets(self.root / "outside", archive, manifest_path)

    def test_sample_source_mapping_is_complete(self):
        manifest = json.loads(demo.MANIFEST.read_text())
        sources = {source["id"]: source for source in manifest["attribution"]["sources"]}
        self.assertEqual(set(sources), {"sheepcounter", "herdnet-data", "caribou"})
        for source_id in sources:
            self.assertEqual(sum(entry["source_id"] == source_id for entry in manifest["files"]), 2)
        self.assertEqual(sources["caribou"]["license_identifier"], "CC-BY-NC-SA-4.0")
        self.assertEqual(sources["sheepcounter"]["license_identifier"], "Public Domain")
        self.assertIsNone(sources["sheepcounter"]["license_url"])
        self.assertEqual(sources["herdnet-data"]["license_identifier"], "CC-BY-NC-SA-4.0")
        self.assertEqual(sources["sheepcounter"]["verification_status"], "contributor_confirmed")
        self.assertEqual(sources["herdnet-data"]["verification_status"], "contributor_confirmed")

    def test_isolated_model_cache_and_unchanged_default(self):
        for model_name in ("owl-d", "owl-c"):
            for cache in (None, self.root / "separate-models"):
                with self.subTest(model=model_name, cache=cache), patch.object(demo, "download_file") as download:
                    demo.fetch_model_checkpoint(model_name, cache)
                    spec = demo.MODEL_SPECS[model_name]
                    expected_cache = cache if cache is not None else demo.ROOT / "demo_data/models"
                    download.assert_called_once_with(
                        f"https://zenodo.org/records/20802844/files/{spec['filename']}?download=1",
                        expected_cache / spec["filename"], spec["size"], spec["sha256"],
                    )
        with self.assertRaisesRegex(ValueError, "not both"):
            demo.load_demo_model("owl-c", "cpu", self.root / "checkpoint", self.root / "cache")

    def test_model_download_fallback_keeps_exact_integrity_pins(self):
        spec = demo.MODEL_SPECS["owl-c"]
        target = self.root / spec["filename"]
        with patch.object(demo, "download_file", side_effect=[demo.DownloadError("network"), target]) as download:
            self.assertEqual(demo.fetch_model_checkpoint("owl-c", self.root), target)
        self.assertEqual(
            [call.args for call in download.call_args_list],
            [(url, target, spec["size"], spec["sha256"]) for url in demo.model_download_urls("owl-c")],
        )
        with patch.object(demo, "download_file", side_effect=ValueError("Integrity check failed")) as download:
            with self.assertRaisesRegex(ValueError, "Integrity"):
                demo.fetch_model_checkpoint("owl-c", self.root)
            self.assertEqual(download.call_count, 1)
        with patch.object(demo, "download_file", side_effect=demo.DownloadError("network")) as download:
            with self.assertRaises(demo.DownloadError):
                demo.fetch_model_checkpoint("owl-c", self.root)
            self.assertEqual(download.call_count, 2)

    def test_unsafe_and_unexpected_archive_members(self):
        _, _, manifest = self.sample_archive()
        for name in ("../outside.png", "/outside.png", "OWL_DATA/../outside.png",
                     "OWL_DATA\\sample.png", "other/sample.png", "OWL_DATA/extra.png"):
            with self.subTest(name=name):
                path = self.root / "bad.zip"
                with ZipFile(path, "w") as archive:
                    archive.writestr(name, b"x")
                with ZipFile(path) as archive, self.assertRaises(ValueError):
                    demo.validate_archive_members(archive, manifest)
        path = self.root / "symlink.zip"
        link = ZipInfo("OWL_DATA/sample.png")
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        with ZipFile(path, "w") as archive:
            archive.writestr(link, b"target")
        with ZipFile(path) as archive, self.assertRaisesRegex(ValueError, "Unsafe"):
            demo.validate_archive_members(archive, manifest)

    def test_download_cache_and_corruption(self):
        path = self.root / "asset.bin"
        path.write_bytes(b"verified")
        digest = demo.sha256_file(path)
        with patch("requests.get") as request:
            demo.download_file("https://example.org/asset", path, 8, digest)
            request.assert_not_called()
        path.write_bytes(b"bad")
        with self.assertRaisesRegex(ValueError, "Integrity"):
            demo.download_file("https://example.org/asset", path, 8, digest)

    def test_failed_download_does_not_leave_partial_cache(self):
        destination = self.root / "asset.bin"
        with patch("requests.get", side_effect=requests.Timeout("test timeout")), patch("time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "Could not download"):
                demo.download_file("https://example.org/asset", destination, 8, "0" * 64)
        self.assertFalse(destination.exists())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_successful_download_verifies_before_publish(self):
        content = b"verified"
        digest = hashlib.sha256(content).hexdigest()
        response = MagicMock()
        response.__enter__.return_value = response
        response.iter_content.return_value = [b"veri", b"fied"]
        path = self.root / "asset.bin"
        with patch("requests.get", return_value=response):
            demo.download_file("https://example.org/asset", path, len(content), digest)
        self.assertEqual(path.read_bytes(), content)
        self.assertEqual(list(self.root.iterdir()), [path])

    def test_bad_download_digest_never_publishes(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.iter_content.return_value = [b"corrupt!"]
        path = self.root / "asset.bin"
        with patch("requests.get", return_value=response):
            with self.assertRaisesRegex(ValueError, "Integrity"):
                demo.download_file("https://example.org/asset", path, 8, "0" * 64)
        self.assertFalse(path.exists())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_device_selection_does_not_change_model(self):
        self.assertEqual(demo.resolve_device("owl-d", "auto", True), "cuda")
        self.assertEqual(demo.resolve_device("owl-c", "auto", False), "cpu")
        self.assertEqual(demo.resolve_device("owl-c", "cpu", True), "cpu")
        for requested, available in (("auto", False), ("cpu", True), ("cpu", False)):
            with self.subTest(requested=requested, available=available):
                with self.assertRaisesRegex(ValueError, "NOT been changed"):
                    demo.resolve_device("owl-d", requested, available)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            demo.resolve_device("owl-c", "cuda", False)

    def test_coordinate_scaling_and_thresholds(self):
        heatmap = torch.zeros((1, 1, 16, 16))
        heatmap[0, 0, 3, 7] = 0.9
        heatmap[0, 0, 10, 11] = 0.4
        rows = demo.heatmap_detections(heatmap, "sample.png", 32, 32, 0.5, 0.3, 0.1)
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["x"], rows[0]["y"]), (14, 6))
        self.assertEqual(rows[0]["images"], "sample.png")
        self.assertEqual(rows[0]["labels"], 1)

    def test_zero_predictions_and_nonfinite_heatmap(self):
        heatmap = torch.zeros((1, 1, 16, 16))
        rows = demo.heatmap_detections(heatmap, "empty.png", 32, 32, 0.2, 0.3, 0.1)
        self.assertEqual(rows, [])
        path = self.root / "detections.csv"
        demo.write_csv(path, demo.DETECTION_COLUMNS, rows)
        self.assertEqual(path.read_text().strip(), "images,x,y,dscores,labels")
        heatmap[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            demo.heatmap_detections(heatmap, "bad.png", 32, 32, 0.2, 0.3, 0.1)

    def test_stitching_small_odd_and_multitile_images(self):
        model = LossWrapper(HalfHeatmap(), []).eval()
        stitcher = HerdNet_Detection_Branch_Stitcher(
            model, size=(32, 32), overlap=8, down_ratio=2, up=False,
            reduction="mean", device_name="cpu",
        )
        for height, width in ((16, 16), (31, 29), (32, 32), (64, 72), (65, 73)):
            with self.subTest(height=height, width=width):
                image = torch.zeros((3, height, width))
                image[:, 4:6, 10:12] = 0.9
                with torch.inference_mode():
                    heatmap = stitcher(image)
                self.assertEqual(tuple(heatmap.shape), (1, 1, height // 2, width // 2))
                rows = demo.heatmap_detections(heatmap, "test.png", width, height, 0.2, 0.3, 0.1)
                self.assertEqual([(row["x"], row["y"]) for row in rows], [(10, 4)])

    def test_selection_and_invalid_images(self):
        Image.new("RGB", (512, 512)).save(self.root / "b.JPG")
        Image.new("RGB", (800, 600)).save(self.root / "a.png")
        self.assertEqual([p.name for p in demo.select_images(self.root, "patches")], ["b.JPG"])
        self.assertEqual([p.name for p in demo.select_images(self.root, "full")], ["a.png"])
        self.assertEqual([p.name for p in demo.select_images(self.root, "all")], ["a.png", "b.JPG"])
        (self.root / "broken.jpg").write_bytes(b"bad")
        with self.assertRaises(OSError):
            demo.select_images(self.root, "all")

    def test_stub_run_exports_consistent_counts_and_images(self):
        images = self.root / "images"
        images.mkdir()
        Image.new("RGB", (32, 32), color="white").save(images / "positive.png")
        Image.new("RGB", (32, 32), color="black").save(images / "negative.png")
        model = LossWrapper(HalfHeatmap(), []).eval()
        with patch.object(demo, "environment_info", return_value={"device": "cpu"}), \
                patch.object(demo, "load_demo_model", return_value=(model, {"test": True})):
            output = self.root / "run"
            metadata = demo.run_inference(
                images, output, "owl-c", "cpu", "all", tile_size=32, overlap=8,
            )
        with (output / "summary.csv").open() as stream:
            summary = list(csv.DictReader(stream))
        with (output / "detections.csv").open() as stream:
            detections = list(csv.DictReader(stream))
        self.assertEqual(len(summary), 2)
        self.assertEqual(int(summary[0]["count"]), 0)
        self.assertEqual(sum(int(row["count"]) for row in summary), len(detections))
        self.assertEqual(metadata["total_predicted_detections"], len(detections))
        self.assertTrue(all((output / filename).is_file() for filename in metadata["figures"]))
        for path in (output / "overlays").iterdir():
            with Image.open(path) as image:
                self.assertEqual(image.size, (32, 32))
        with self.assertRaisesRegex(ValueError, "new or empty"):
            demo.run_inference(images, output, "owl-c", "cpu")

    def test_overlay_matches_original_pixel_coordinates(self):
        output = self.root
        (output / "overlays").mkdir()
        (output / "figures").mkdir()
        rows = [{"images": "test.png", "x": 14, "y": 6, "dscores": 0.9, "labels": 1}]
        demo.render_prediction(Image.new("RGB", (32, 32)), np.zeros((16, 16)), rows, output, "test")
        with Image.open(output / "overlays/test.png") as image:
            self.assertEqual(image.getpixel((14, 6)), (255, 0, 0))
            self.assertEqual(image.getpixel((28, 12)), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
