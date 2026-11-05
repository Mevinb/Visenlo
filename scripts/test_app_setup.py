"""First-run setup UX regressions. Child installers are mocked; no downloads."""
from __future__ import annotations

import io
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import app


class FakeInstaller:
    def __init__(self, output="[ok] Required models are ready.\n", returncode=0):
        self.stdout = io.StringIO(output)
        self.returncode = None
        self.expected_code = returncode
        self.terminated = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = self.expected_code
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.expected_code = -15

    def kill(self):
        self.terminated = True
        self.expected_code = -9


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.old_lock = app._setup_lock
        app._setup_lock = threading.Lock()
        self.addCleanup(setattr, app, "_setup_lock", self.old_lock)

    def test_first_run_without_models_opens_setup_and_disables_run(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(app, "MODELS", Path(directory)):
            ui = app.build_ui()
        components = ui.config["components"]
        run = next(c for c in components if c.get("props", {}).get("elem_id") == "run-swap")
        install = next(c for c in components if c.get("props", {}).get("elem_id") == "install-models")
        report = next(c for c in components if c.get("props", {}).get("label") == "Processing report")
        setup = next(c for c in components if c.get("props", {}).get("label") == "Model setup · local assets and readiness")
        self.assertFalse(run["props"]["interactive"])
        self.assertFalse(install["props"]["interactive"])
        self.assertTrue(setup["props"]["open"])
        self.assertIn("install the required models", report["props"]["value"])

    def test_license_acknowledgement_required_before_starting_a_child(self):
        with patch.object(app.subprocess, "Popen") as child:
            with self.assertRaises(app.gr.Error) as error:
                list(app.install_models(False, False))
        self.assertIn("acknowledgement", str(error.exception))
        child.assert_not_called()
        self.assertFalse(app._setup_lock.locked())

    def test_core_install_uses_fixed_command_and_custom_model_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            models = Path(directory) / "local models"
            child = FakeInstaller()
            with patch.object(app, "MODELS", models), patch.object(app.subprocess, "Popen", return_value=child) as spawn, \
                 patch.object(app, "_check_models", side_effect=[("before", False), ("after", True)]):
                updates = list(app.install_models(True, False))
            args, kwargs = spawn.call_args
            self.assertEqual(args[0], [sys.executable, "-u", str(ROOT / "scripts/download_models.py"), "--download"])
            self.assertEqual(kwargs["env"]["VISENLO_MODELS"], str(models.resolve()))
            self.assertEqual(kwargs["cwd"], str(ROOT))
            self.assertNotIn("shell", kwargs)
            self.assertEqual(updates[-1][1], "after")
            self.assertTrue(updates[-1][3]["interactive"])
            self.assertTrue(updates[-1][4]["interactive"])
            self.assertGreaterEqual(len(updates), 3)
            self.assertIn("Required models are ready", updates[-1][0])
        self.assertFalse(app._setup_lock.locked())

    def test_optional_models_are_an_explicit_additional_argument(self):
        with patch.object(app.subprocess, "Popen", return_value=FakeInstaller()) as spawn, \
             patch.object(app, "_check_models", return_value=("ready", True)):
            list(app.install_models(True, True))
        self.assertEqual(spawn.call_args.args[0][-2:], ["--download", "--all"])

    def test_failed_transfer_reports_retry_and_allows_the_next_attempt(self):
        with patch.object(app.subprocess, "Popen", return_value=FakeInstaller("[fail] Connection interrupted\n", 1)), \
             patch.object(app, "_check_models", return_value=("missing", False)):
            updates = list(app.install_models(True, False))
        self.assertIn("retry", updates[-1][0])
        self.assertIn("Connection interrupted", updates[-1][0])
        self.assertFalse(updates[-1][3]["interactive"])
        self.assertFalse(app._setup_lock.locked())
        with patch.object(app.subprocess, "Popen", return_value=FakeInstaller()), \
             patch.object(app, "_check_models", return_value=("ready", True)):
            retried = list(app.install_models(True, False))
        self.assertTrue(retried[-1][3]["interactive"])

    def test_child_start_failure_releases_installation_lock(self):
        with patch.object(app.subprocess, "Popen", side_effect=OSError("Cannot start installer")), \
             patch.object(app, "_check_models", return_value=("missing", False)):
            updates = list(app.install_models(True, False))
        self.assertIn("Cannot start installer", updates[-1][0])
        self.assertIn("retry", updates[-1][0])
        self.assertFalse(app._setup_lock.locked())

    def test_concurrent_install_is_rejected_before_spawning(self):
        app._setup_lock.acquire()
        try:
            with patch.object(app.subprocess, "Popen") as child:
                with self.assertRaises(app.gr.Error):
                    list(app.install_models(True, False))
            child.assert_not_called()
        finally:
            app._setup_lock.release()

    def test_recheck_does_not_enable_swapping_during_setup(self):
        app._setup_lock.acquire()
        try:
            with patch.object(app, "_check_models", return_value=("installed", True)):
                _, status, update = app.check_models()
            self.assertFalse(update["interactive"])
            self.assertIn("Installing", status)
        finally:
            app._setup_lock.release()

    def test_report_size_is_bounded(self):
        output = "\n".join(f"line{i:03}:" + "x" * 2000 for i in range(150))
        with patch.object(app.subprocess, "Popen", return_value=FakeInstaller(output)), \
             patch.object(app, "_check_models", return_value=("ready", True)):
            updates = list(app.install_models(True, False))
        report = updates[-1][0]
        self.assertIn("line149:", report)
        self.assertNotIn("line000:", report)
        self.assertLess(len(report), 125000)

    def test_closing_setup_stream_releases_lock_and_terminates_child(self):
        child = FakeInstaller()
        with patch.object(app.subprocess, "Popen", return_value=child), \
             patch.object(app, "_check_models", return_value=("missing", False)):
            stream = app.install_models(True, False)
            next(stream)
            next(stream)
            stream.close()
        self.assertFalse(app._setup_lock.locked())
        self.assertTrue(child.terminated)


if __name__ == "__main__":
    unittest.main()
