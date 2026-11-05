"""Startup contract checks without installing packages or launching servers."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import launcher


class LauncherTests(unittest.TestCase):
    def invoke(self, arguments, healthy=True):
        with patch.object(sys, 'argv', ['launcher.py', *arguments]), \
             patch.object(launcher, 'check_python'), patch.object(launcher, 'ensure_venv'), \
             patch.object(launcher, 'dependencies_ready', return_value=healthy), \
             patch.object(launcher, 'pip_install') as install, \
             patch.object(launcher, 'system_check'), patch.object(launcher, 'model_check'), \
             patch.object(launcher, 'selfcheck') as checks, \
             patch.object(launcher, 'find_free_port', return_value=7881), \
             patch.object(launcher.os, 'execv') as execute:
            launcher.main()
            return install, checks, execute.call_args.args

    def test_healthy_launch_reuses_dependencies_and_avoids_extended_checks(self):
        install, checks, (_, command) = self.invoke([])
        install.assert_not_called()
        checks.assert_not_called()
        self.assertIn(str(launcher.APP), command)
        self.assertEqual(command[command.index('--port') + 1], '7881')
        self.assertNotIn('--no-browser', command)

    def test_headless_flag_survives_bootstrap_and_busy_port_selection(self):
        _, _, (_, command) = self.invoke(['--no-browser', '--port', '7860'])
        self.assertIn('--no-browser', command)
        self.assertEqual(command[command.index('--port') + 1], '7881')

    def test_extended_diagnostics_are_explicit(self):
        install, checks, _ = self.invoke(['--self-check'])
        install.assert_not_called()
        checks.assert_called_once()

    def test_reinstall_remains_explicit(self):
        install, _, _ = self.invoke(['--reinstall'])
        install.assert_called_once()

    def test_missing_dependencies_trigger_setup(self):
        install, _, _ = self.invoke([], healthy=False)
        install.assert_called_once()


if __name__ == '__main__':
    unittest.main()
