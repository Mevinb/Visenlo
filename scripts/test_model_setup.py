"""Model setup regressions; no network access or model weights required."""
import contextlib
import io
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import download_models as setup


class ModelSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.models = patch.object(setup, "MODELS", self.root)
        self.models.start()
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()

    def tearDown(self):
        self.output.__exit__(None, None, None)
        self.models.stop()
        self.temp.cleanup()

    def required_files(self):
        for name in setup.BUFFALO_FILES:
            path = self.root / "insightface/models/buffalo_l" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"model")
        (self.root / "inswapper_128.onnx").write_bytes(b"model")

    def test_optional_models_do_not_block_basic_readiness(self):
        self.required_files()
        self.assertEqual(setup.check(), 0)
        self.assertEqual(setup.check(all_models=True), 1)
        (self.root / "inswapper_128.onnx").write_bytes(b"")
        self.assertEqual(setup.check(), 1)

    def test_check_never_downloads(self):
        with patch.object(setup, "download") as download, patch.object(setup, "download_buffalo") as buffalo:
            self.assertEqual(setup.main(["--check"]), 1)
        download.assert_not_called()
        buffalo.assert_not_called()

    def test_failed_download_preserves_previous_file_and_cleans_partial(self):
        path = self.root / "model.onnx"
        path.write_bytes(b"old")
        response = io.BytesIO(b"short")
        response.headers = {"Content-Length": "999"}
        with patch.object(setup.urllib.request, "urlopen", return_value=response):
            self.assertFalse(setup.download("https://example.invalid/model", path))
        self.assertEqual(path.read_bytes(), b"old")
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_html_error_page_is_not_installed_as_model(self):
        response = io.BytesIO(b"<html>Error</html>")
        response.headers = {"Content-Type": "text/html"}
        with patch.object(setup.urllib.request, "urlopen", return_value=response):
            self.assertFalse(setup.download("https://example.invalid/model", self.root / "model.onnx"))
        self.assertFalse((self.root / "model.onnx").exists())

    def test_download_failure_exits_unsuccessfully(self):
        self.required_files()
        with patch.object(setup, "download_buffalo", return_value=False):
            self.assertEqual(setup.main(["--download"]), 1)

    def test_buffalo_extracts_only_known_models_to_fixed_paths(self):
        archive = self.root / "fixture.zip"
        with zipfile.ZipFile(archive, "w") as output:
            for name in setup.BUFFALO_FILES:
                output.writestr("../../" + name, b"model")
            output.writestr("../../injected.py", b"unwanted")
        def copy_archive(url, destination):
            shutil.copyfile(archive, destination)
            return True
        with patch.object(setup, "download", side_effect=copy_archive):
            self.assertTrue(setup.download_buffalo())
        files = {str(path.relative_to(self.root)) for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(files, {"fixture.zip"} | {"insightface/models/buffalo_l/" + name for name in setup.BUFFALO_FILES})


if __name__ == "__main__":
    unittest.main()
