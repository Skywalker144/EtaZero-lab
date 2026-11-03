"""Source root representatives constrain Gumbel search and retain native reports."""
import json
import os
import re
import subprocess

import pytest

from etazero import ROOT
from test_analysis import analysis_config, complete, request, run
from test_gtp import gtp_config, run_gtp
from test_muzero_analysis import bundle_for, config_for, cuda
from test_xiangqi_gtp import config_for_gtp


def check_protocols(directory, game, model='/dev/null', muzero=False, fp16=False):
    settings = dict(rootSymmetryPruning='true', gumbelMaxConsideredActions=4,
                    numSearchThreads=4, cudaUseFP16=str(fp16).lower())
    config = (config_for(directory / 'analysis.cfg', game, gumbel=True, **settings) if muzero else
              analysis_config(directory / 'analysis.cfg', game=game, gumbel=True, **settings))
    shape = (9, 10) if game == 'xiangqi' else (5, 5)
    rules = 'tromp-taylor' if game == 'go' else game
    def query(id, **changes):
        return request(id, rules=rules, boardXSize=shape[0], boardYSize=shape[1], maxVisits=64, **changes)
    pruned = complete(run(config, [query('pruned')], model))['pruned', 0]
    originals = {m['move']: m for m in pruned['moveInfos'] if 'isSymmetryOf' not in m}
    duplicates = [m for m in pruned['moveInfos'] if 'isSymmetryOf' in m]
    assert duplicates
    assert pruned['rootInfo']['visits'] == 64
    assert sum(m['edgeVisits'] for m in originals.values()) == 63
    for move in duplicates:
        source = originals[move['isSymmetryOf']]
        assert move['edgeVisits'] == source['edgeVisits']
        assert move['visits'] == source['visits']
        assert move['pvVisits'] == source['pvVisits']
        assert move['pv'][0] == move['move']
    target = duplicates[0]['move']
    reports = complete(run(config, [query('focus', focusMoves=[target], focusProb=.8),
        query('avoid', avoidMoves=[dict(player='B', moves=[target], untilDepth=1)]),
        query('disabled', overrideSettings=dict(rootSymmetryPruning=False))], model))
    for (name, _), report in reports.items():
        originals = {m['move']: m for m in report['moveInfos'] if 'isSymmetryOf' not in m}
        assert report['rootInfo']['visits'] == 64
        assert sum(m['edgeVisits'] for m in originals.values()) == 63
        if name == 'focus':
            move = next(m for m in report['moveInfos'] if m['move'] == target)
            assert move['visits'] > move['edgeVisits']
        elif name == 'avoid':
            assert target not in {m['move'] for m in report['moveInfos']}
        else:
            assert all('isSymmetryOf' not in m for m in report['moveInfos'])
    settings.update(muzero=muzero, gumbel=True, random=not muzero)
    config = (config_for_gtp(directory / 'gtp.cfg', **settings) if game == 'xiangqi' else
              gtp_config(directory / 'gtp.cfg', game, **settings))
    # Also exercise the native GTP default (true), rather than only an explicit
    # setting. Runtime switching must keep the source parameter API usable.
    config.write_text(''.join(line for line in config.read_text().splitlines(True)
                              if not line.startswith('rootSymmetryPruning =')))
    replies = run_gtp(config, ['kata-search_analyze B 0 rootInfo true',
        'kata-set-param rootSymmetryPruning false', 'kata-search B',
        'kata-set-param rootSymmetryPruning true', 'kata-search B', 'quit'], model)
    assert all(not error for error, _ in replies), replies
    assert 'isSymmetryOf ' in replies[0][1]
    assert re.search(r'rootInfo visits 32\b', replies[0][1])


@pytest.mark.parametrize('game', ['go', 'freestyle', 'standard', 'renju', 'hex', 'xiangqi'])
def test_native_protocol_symmetry(tmp_path, game):
    check_protocols(tmp_path, game)


@cuda
@pytest.mark.parametrize('game', ['go', 'renju', 'hex', 'xiangqi'])
@pytest.mark.parametrize('fp16', [False, True])
def test_native_latent_symmetry_and_protocols(tmp_path, game, fp16):
    model = bundle_for(tmp_path / 'model', game)
    (tmp_path / 'request.json').write_text(json.dumps(dict(fp16=fp16, symmetry=True)))
    result = subprocess.run([str(ROOT / 'build/katago'), 'runtests', '-muzero-search', str(model),
        str(tmp_path / 'request.json'), str(tmp_path / 'response.json')],
        env=dict(os.environ, NVIDIA_TF32_OVERRIDE='0'), capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((tmp_path / 'response.json').read_text()) is True
    check_protocols(tmp_path, game, model, True, fp16)
