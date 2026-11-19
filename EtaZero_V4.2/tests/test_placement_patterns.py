"""Point-game pattern transforms and native SGF/automatic-pattern consumers."""
import subprocess

import pytest

from etazero import ROOT
from etazero.evaluation import load_games
from match_dispatch_worker import worker_command
from test_match import match_config


@pytest.fixture(scope='module')
def records(tmp_path_factory):
    path = tmp_path_factory.mktemp('placement-patterns')
    result = subprocess.run([str(ROOT / 'build/katago'), 'runtests', '-placement-patterns', str(path)],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'valid symmetries, files and source searches passed' in result.stdout
    return path


def test_native_patterns_and_actual_rule_filtering(records):
    log = (records / 'native.log').read_text()
    assert 'numPosesUsed = 5' in log and 'numPosesInvalid = 15' in log
    assert 'numPosLoadErrors = 0' in log
    assert len(list((records / 'sgfs').glob('*.sgf'))) == 4


@pytest.mark.parametrize('game', ['freestyle', 'standard', 'renju', 'hex'])
def test_native_match_reads_mixed_pattern_files_and_actual_games(tmp_path, records, game):
    cfg = match_config(tmp_path / 'match.cfg', [('/dev/null', False)]*2, game=game, games=2, moves=100)
    cfg.write_text(cfg.read_text().replace('bSizes = 3\n', 'bSizes = 7\n')
                   .replace('subtreeValueBiasFactor = 0\n', 'subtreeValueBiasFactor = 0.4\n'))
    with cfg.open('a') as out:
        out.write(f'debugSkipNeuralNet = true\n'
                  f'avoidSgfPatternDirs1 = {records / "sgfs"}\navoidSgfPatternUtility1 = 0.1\n'
                  'avoidRepeatedPatternUtility = 0.03\n')
    output = tmp_path / 'results'
    result = subprocess.run(worker_command([str(ROOT / 'build/katago'), 'match', '-config', str(cfg),
                             '-results-dir', str(output)]), capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'shapes to penalize repeats for bot 1' in result.stdout
    plan, games = load_games(output)
    assert plan['game'] == game and len(games) == 2
    assert {row['outcome'] for row in games} <= {'B', 'W', 'D'}
    assert all(f'RU[{game}]' in row['sgf'] for row in games)
