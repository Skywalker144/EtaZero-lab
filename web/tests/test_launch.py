import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from web.launch import latest_version, needs_build
from web.server import open_browser


class LaunchTests(unittest.TestCase):
    def test_latest_numeric_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('EtaZero_V1', 'EtaZero_V2.9', 'EtaZero_V2.10', 'EtaZero_Vinvalid'):
                (root / name).mkdir()
            self.assertEqual(latest_version(root).name, 'EtaZero_V2.10')

    def test_build_reuse_and_changed_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / 'build/etazero'
            binary.parent.mkdir()
            (root / 'cpp').mkdir()
            schema = root / 'python/etazero/schema.py'
            schema.parent.mkdir(parents=True)
            source = root / 'cpp/main.cpp'
            source.write_text('original')
            schema.write_text('schema')
            binary.write_bytes(b'executable')
            manifest = {
                'with_torch': True,
                'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
                'sources': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (source, schema)},
            }
            path = binary.parent / 'build_manifest.json'
            self.assertTrue(needs_build(root, binary))
            path.write_text(json.dumps(manifest))
            self.assertFalse(needs_build(root, binary))
            source.write_text('changed')
            self.assertTrue(needs_build(root, binary))
            source.write_text('original')
            extra = root / 'cpp/new.cpp'
            extra.touch()
            self.assertTrue(needs_build(root, binary))
            extra.unlink()
            source.unlink()
            self.assertTrue(needs_build(root, binary))
            source.write_text('original')
            binary.write_bytes(b'changed executable')
            self.assertTrue(needs_build(root, binary))

    def test_open_browser_and_fallback(self):
        with patch('web.server.webbrowser.open', return_value=True) as opened:
            open_browser('http://127.0.0.1:8766')
            opened.assert_called_once_with('http://127.0.0.1:8766', new=2)
        with patch('web.server.webbrowser.open', return_value=False), patch('builtins.print') as printed:
            open_browser('http://127.0.0.1:8766')
            self.assertIn('http://127.0.0.1:8766', printed.call_args.args[0])
