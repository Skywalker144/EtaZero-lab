"""Opening configuration routing and protection of generated human-play prefixes."""
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from etazero.config import ROOT
from etazero.engine_config import load_engine_config, load_match_opening_config
from web.app import human_turns
from web.server import discover_catalog, load_web_config


class OpeningConfigTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        shutil.copytree(ROOT / 'tests/fixtures/configs/baseline', self.root / 'configs/baseline')
        self.child = self.root / 'configs/child'
        self.child.mkdir()
        (self.child / 'analysis.cfg').write_text('@include ../baseline/analysis.cfg\n')
        (self.child / 'match.cfg').write_text('@include ../baseline/match.cfg\n[opening]\nbalance_exponent=7\n')
        (self.child / 'match.cfg.local').write_text('[opening]\navg_dist_factor=1.2\n')

    def test_inheritance_local_and_environment_match_full_profile(self):
        env = {'MATCH_OPENING_REJECTION_PROBABILITY': '.91', 'ANALYSIS_OPENING_BALANCE_EXPONENT': '99'}
        opening = load_match_opening_config(self.child, environ=env)
        self.assertEqual(opening, load_engine_config(self.child, match=True, environ=env)['opening'])
        self.assertEqual(opening['balance_exponent'], 7)
        self.assertEqual(opening['avg_dist_factor'], 1.2)
        self.assertEqual(opening['rejection_probability'], .91)

    def test_unrelated_search_fields_do_not_control_web_opening(self):
        (self.child / 'analysis.cfg').write_text('@include ../baseline/analysis.cfg\n[analysis]\nvisits=31\n')
        env = {'MATCH_GAMES': '3', 'MATCH_VISITS': 'invalid', 'ANALYSIS_VISITS': '43'}
        opening = load_match_opening_config(self.child, environ=env)
        self.assertEqual(opening['balance_exponent'], 7)
        self.assertEqual(load_engine_config(self.child, environ=env)['analysis']['visits'], 43)
        with self.assertRaises(ValueError):
            load_engine_config(self.child, match=True, environ=env)

    def test_invalid_opening_is_rejected(self):
        for field, value in (('probability', '0'), ('max_tries', '0'), ('rejection_probability_fallback', '1'),
                             ('balance_exponent', 'nan'), ('avg_dist_factor', '-1')):
            with self.subTest(field=field), self.assertRaises(ValueError):
                load_match_opening_config(self.child, environ={'MATCH_OPENING_' + field.upper(): value})

    def test_catalog_uses_the_selected_match_profile(self):
        root = self.root / 'data/child'
        (root / 'config').mkdir(parents=True)
        (root / 'config/effective.json').write_text('{}')
        with patch('web.server.ROOT', self.root), patch.dict('os.environ', {}, clear=True):
            _, runs = discover_catalog(self.root / 'data')
        self.assertEqual(runs[str(root)]['opening'], load_match_opening_config(self.child, environ={}))

    def test_go_and_xiangqi_do_not_require_balanced_openings_or_match_profile(self):
        root = self.root / 'data/child'
        (root / 'config').mkdir(parents=True)
        (root / 'config/effective.json').write_text('{}')
        for rule in ('go', 'xiangqi'):
            with self.subTest(rule=rule):
                settings = f'rule={rule}\nboard_size=10\nnn_symmetry=0\nroot_num_symmetries_to_sample=1\n'
                (self.child / 'analysis.cfg').write_text('@include ../baseline/analysis.cfg\n[analysis]\n' + settings)
                (self.child / 'match.cfg').write_text('@include ../baseline/match.cfg\n[match]\n' + settings +
                                                       '[opening]\nprobability=0\n')
                self.assertEqual(load_engine_config(self.child, match=True, environ={})['opening']['probability'], 0)
                for with_match in (True, False):
                    with self.subTest(with_match=with_match):
                        if not with_match:
                            (self.child / 'match.cfg').unlink()
                        with patch('web.server.ROOT', self.root), patch.dict('os.environ', {}, clear=True):
                            self.assertEqual(set(load_web_config(self.child)), {'analysis'})
                            _, runs = discover_catalog(self.root / 'data')
                        self.assertIsNone(runs[str(root)]['opening'])
                        self.assertIsNone(runs[str(root)]['hex_opening'])

    def test_web_still_rejects_invalid_placement_opening(self):
        with patch.dict('os.environ', {'MATCH_OPENING_PROBABILITY': '0'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'successful balanced openings'):
                load_web_config(self.child)

    def test_human_turns_exclude_prefix_for_both_colors(self):
        game = dict(moves=[4, 8, 12, 13, 14], opening=dict(moves=[4, 8, 12]))
        self.assertEqual(human_turns(game, 1), [4])
        self.assertEqual(human_turns(game, -1), [3])
        self.assertEqual(human_turns(dict(moves=[4, 8, 12], opening=game['opening']), 1), [])
