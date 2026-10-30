import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from web.launch import latest_version, main
from web.server import open_browser


class LaunchTests(unittest.TestCase):
    def web_root(self, directory):
        root = Path(directory)
        interface = root / 'python/etazero/evaluation/analysis.py'
        interface.parent.mkdir(parents=True)
        interface.touch()
        return root

    def test_latest_numeric_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('EtaZero_V1', 'EtaZero_V2.9', 'EtaZero_V2.10', 'EtaZero_Vinvalid'):
                (root / name).mkdir()
            self.assertEqual(latest_version(root).name, 'EtaZero_V2.10')

    def test_launch_delegates_incremental_build_to_cmake(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.web_root(directory)
            with patch('web.launch.latest_version', return_value=root), \
                 patch('web.launch.subprocess.run') as build, \
                 patch('web.server.main') as serve, \
                 patch('sys.argv', ['web.launch']):
                main()
        build.assert_called_once_with(['bash', str(root/'scripts/build.sh')], check=True)
        serve.assert_called_once_with([])

    def test_custom_binary_skips_default_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.web_root(directory)
            with patch('web.launch.latest_version', return_value=root), \
                 patch('web.launch.subprocess.run') as build, \
                 patch('web.server.main') as serve, \
                 patch('sys.argv', ['web.launch', '--binary', '/tmp/native']):
                main()
        build.assert_not_called()
        serve.assert_called_once_with(['--binary', '/tmp/native'])

    def test_explicit_version_is_used_without_latest_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            lab = Path(directory)
            root = self.web_root(lab / 'EtaZero_V2.10')
            with patch('web.launch.__file__', str(lab / 'web/launch.py')), \
                 patch('web.launch.latest_version') as latest, \
                 patch('web.launch.subprocess.run') as build, \
                 patch('web.server.main') as serve, \
                 patch('sys.argv', ['web.launch', '--version', root.name]):
                main()
            latest.assert_not_called()
            build.assert_called_once_with(['bash', str(root / 'scripts/build.sh')], check=True)
            serve.assert_called_once_with([])

    def test_unconnected_latest_version_fails_before_build(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('web.launch.latest_version', return_value=Path(directory)), \
             patch('web.launch.subprocess.run') as build, \
             patch('web.server.main') as serve, \
             patch('sys.stderr', new_callable=io.StringIO) as errors, \
             patch('sys.argv', ['web.launch']):
            with self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)
            self.assertIn('尚未接入 Web 协议', errors.getvalue())
        build.assert_not_called()
        serve.assert_not_called()

    def test_open_browser_and_fallback(self):
        with patch('web.server.webbrowser.open', return_value=True) as opened:
            open_browser('http://127.0.0.1:8766')
            opened.assert_called_once_with('http://127.0.0.1:8766', new=2)
        with patch('web.server.webbrowser.open', return_value=False), patch('builtins.print') as printed:
            open_browser('http://127.0.0.1:8766')
            self.assertIn('http://127.0.0.1:8766', printed.call_args.args[0])
