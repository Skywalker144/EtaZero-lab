"""Complete Xiangqi history through native SGF, hint files and source sampling."""
import json
import subprocess

import pytest

from etazero import ROOT


@pytest.fixture(scope='module')
def records(tmp_path_factory):
    output = tmp_path_factory.mktemp('xiangqi-sgf')
    result = subprocess.run([str(ROOT / 'build/katago'), 'runtests', '-xiangqi-sgf', str(output)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return output


def test_native_state_sgf_hint_and_initializer(records):
    sgf = (records / 'position.sgf').read_text()
    assert 'GM[7]SZ[9:10]' in sgf and 'XQHIST[e8f8][f9e9]' in sgf
    sample = json.loads((records / 'position.hintposes.txt').read_text())
    assert sample['game'] == 'xiangqi' and 'board' not in sample
    assert sample['xiangqiHistory'] == ['e8f8', 'f9e9']
    assert sample['moveLocs'] == ['f8e8', 'e9f9', 'e8f8', 'f9e9', 'f8e8']
    assert sample['hintLoc'] == 'e9f9' and sample['trainingWeight'] == .375


def test_source_samplesgfs_and_viewstartposes(records, tmp_path):
    output = tmp_path / 'samples'
    result = subprocess.run([str(ROOT / 'build/katago'), 'samplesgfs', '-sgfdir', str(records),
        '-outdir', str(output), '-sample-prob', '1', '-turn-weight-lambda', '0', '-max-komi', '0',
        '-flip-if-pass', '-training-weight', '.625', '-for-testing'], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    paths = list(output.glob('*.startposes.txt'))
    samples = [json.loads(line) for path in paths for line in path.read_text().splitlines()]
    assert len(samples) == 6
    assert {s['initialTurnNumber']+len(s['moveLocs']) for s in samples} == set(range(97, 103))
    assert all(s['game'] == 'xiangqi' and s['trainingWeight'] == .625 for s in samples)
    # The black-to-move positions retain their actual side despite -flip-if-pass.
    assert any(s['movePlas'] and s['movePlas'][-1] == 'B' for s in samples)
    for path in [*paths, records / 'position.hintposes.txt']:
        for options in (['-check-legality'], []):
            result = subprocess.run([str(ROOT / 'build/katago'), 'viewstartposes', '-start-poses-file', str(path), *options],
                                    capture_output=True, text=True, timeout=30)
            assert result.returncode == 0, result.stdout + result.stderr
            assert 'Illegal' not in result.stdout and 'Mismatching' not in result.stdout
            if not options:
                assert 'Xiangqi FEN:' in result.stdout and 'e8f8' in result.stdout
