"""Regression checks for packaging privacy and isolated source installs."""
from pathlib import Path
import tempfile
import unittest
import zipfile

from make_dist import create_archive, iter_distribution_files


class DistributionTests(unittest.TestCase):
    def test_only_source_is_packaged(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            allowed = ["app.py", "app.css", "README.md", "visenlo/pipeline.py", "scripts/selfcheck.py", "docs/SETUP.md"]
            private = [".env", ".env.local", "secrets.json", "testdata/reference.jpg",
                       ".venvs/worker/lib/runtime.py", ".venv/lib/library.py", "outputs/photo.png",
                       "models/model.onnx", "visenlo/__pycache__/pipeline.pyc", "scripts/.env",
                       "docs/screenshots/private.png", "scripts/results/metrics.txt", "docs/private.jpg", "docs/secrets.txt", "scripts/credentials.yml"]
            for name in allowed + private:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture")
            (root / "visenlo" / "linked.py").symlink_to(root / ".env")
            (root / "scripts" / "linked").symlink_to(root / ".venv", target_is_directory=True)
            files = {str(p.relative_to(root)) for p in iter_distribution_files(root)}
            self.assertEqual(files, set(allowed))
            out = create_archive(root, root / "dist" / "Visenlo.zip")
            with zipfile.ZipFile(out) as archive:
                self.assertEqual(set(archive.namelist()), {"Visenlo/" + p for p in allowed})
                self.assertIsNone(archive.testzip())
            self.assertFalse(out.with_suffix(".zip.tmp").exists())


if __name__ == "__main__":
    unittest.main()
