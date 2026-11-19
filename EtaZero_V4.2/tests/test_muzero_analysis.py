"""MuZero bundles in the native JSON service, including async control and real roots."""
import json
import os
import re
import subprocess

import pytest
import torch

from export_muzero import export_components
from katago.train.muzero_model import MuZeroConfig, MuZeroModel
from test_muzero_model import component_config
from test_analysis import (analysis_config, command, request, run, complete,
                           full_history_requests, check_history, check_live_controls)

cuda = pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires native CUDA analysis')


def config_for(path, game='xiangqi', gumbel=False, **changes):
    values = dict(useMuZero='true', useGraphSearch='false', subtreeValueBiasFactor=0,
                  avoidRepeatedPatternUtility=0, maxBoardSizeForNNBuffer=10 if game == 'xiangqi' else 5)
    values.update(changes)
    return analysis_config(path, game=game, random=False, gumbel=gumbel, **values)


def bundle_for(path, game='xiangqi', kind='regular', metadata=False):
    torch.manual_seed(1024)
    representation = component_config(game, kind)
    if metadata:
        representation['metadata_encoder'] = dict(meta_encoder_version=1, internal_num_channels=16)
    model = MuZeroModel(MuZeroConfig(game, 6, representation, component_config(game, 'nbt'),
                                    component_config(game)), 10 if game == 'xiangqi' else 5)
    model.initialize()
    model.eval()
    export_components(model, path, 'analysis-test')
    return path


@pytest.mark.parametrize('changes, human, error', [
    (dict(useGraphSearch='true'), None, 'real-board graph'),
    (dict(subtreeValueBiasFactor=.2), None, 'SVB'),
    (dict(avoidRepeatedPatternUtility=.1), None, 'repeated-pattern'),
    (dict(humanSLPlaExploreProbWeightful=.2), '/dev/null', 'ordinary HumanSL is root-only'),
])
def test_unsupported_search_modes_rejected_before_loading(tmp_path, changes, human, error):
    config = config_for(tmp_path / 'reject.cfg', **changes)
    result = subprocess.run(command(config, human=human), input='', text=True, capture_output=True, timeout=15)
    assert result.returncode != 0
    assert error in result.stdout + result.stderr


@cuda
@pytest.mark.parametrize('gumbel,fp16,kind', [(False, False, 'regular'), (True, True, 'transformer')])
def test_xiangqi_full_history_and_async_control(tmp_path, gumbel, fp16, kind):
    model = bundle_for(tmp_path / 'model', kind=kind)
    config = config_for(tmp_path / 'history.cfg', gumbel=gumbel,
        cudaUseFP16=str(fp16).lower(), inputsUseNHWC=str(fp16).lower(), logSearchInfo='true',
        rootNumSymmetriesToSample=2)
    reports = run(config, full_history_requests(), model)
    # Only the real root knows the rules result. A latent child must not obtain
    # the true forced-check result by playing its action on the real board.
    check_history(reports, latent=True)
    info = next(r for r in reports if r['id'] == 'models')['models'][0]
    assert info == dict(name=str(model), internalName='analysis-test-prediction', maxBatchSize=4,
                       usesHumanSLProfile=False, version=19, usingFP16=str(fp16).lower())
    log = config.with_suffix('.log').read_text()
    assert 'Xiangqi FEN:' in log and 'All cleaned up, quitting' in log
    assert all(int(n) > 0 for n in re.findall(r'NN (?:rows|batches): (\d+)', log))
    config = config_for(tmp_path / 'live.cfg', gumbel=gumbel, numAnalysisThreads=1,
                        cudaUseFP16=str(fp16).lower())
    check_live_controls(config, tmp_path / 'live.log', gumbel, model)


@cuda
@pytest.mark.parametrize('game', ['go', 'freestyle', 'standard', 'renju', 'hex'])
def test_point_games_real_roots_latent_pvs_and_request_boundaries(tmp_path, game):
    model = bundle_for(tmp_path / 'model', game)
    config = config_for(tmp_path / 'points.cfg', game, cudaUseFP16='false', numSearchThreads=1)
    rules = 'tromp-taylor' if game == 'go' else game
    def point(id, **changes):
        return request(id, rules=rules, boardXSize=5, boardYSize=5, **changes)
    queries = [point('puct', initialStones=[['B', 'C3']], initialPlayer='W'),
        point('gumbel', initialStones=[['B', 'C3']], initialPlayer='W',
              overrideSettings=dict(rootSearchAlgorithm='gumbel', nonrootSearchAlgorithm='gumbel')),
        point('latent-chain', allowMoves=[dict(player=p, moves=['A1'], untilDepth=100) for p in ('B', 'W')]),
        request('other-game', rules='hex' if game != 'hex' else 'renju', boardXSize=5, boardYSize=5),
        request('too-large', rules=rules, boardXSize=6, boardYSize=5),
        point('graph', overrideSettings=dict(useGraphSearch=True)),
        point('svb', overrideSettings=dict(subtreeValueBiasFactor=.4)),
        point('patterns', overrideSettings=dict(avoidRepeatedPatternUtility=.03)),
        point('switch-model-type', overrideSettings=dict(useMuZero=False)),
        point('missing-metadata-model', overrideSettings=dict(humanSLProfile='rank_5d')),
        point('after-errors', overrideSettings=dict(useMuZero=True), maxVisits=8)]
    if game != 'go':
        stones = [['B', f'C{i}'] for i in range(1, 6)] if game == 'hex' else [['B', f'{c}3'] for c in 'ABCDE']
        queries.append(point('terminal', initialStones=stones, initialPlayer='W'))
    reports = run(config, queries, model)
    errors = [r for r in reports if 'error' in r]
    assert {r['id'] for r in errors} == {'other-game', 'too-large', 'graph', 'svb', 'patterns',
                                        'switch-model-type', 'missing-metadata-model'}
    good = complete([r for r in reports if 'error' not in r])
    for id in ('puct', 'gumbel', 'after-errors', 'latent-chain'):
        report = good[id, 0]
        assert len(report['policy']) == 26
        assert sum(p for p in report['policy'] if p >= 0) == pytest.approx(1, abs=2e-4)
        assert 1 <= report['rootInfo']['visits'] <= (8 if id == 'after-errors' else 40)
        assert report['policy'][-1] >= 0 if game == 'go' else report['policy'][-1] == -1
        if game != 'go':
            assert report['rootInfo']['noResultValue'] == 0
        if id in ('puct', 'gumbel'):
            assert report['policy'][12] == -1  # Occupied C3 excluded at the real root.
    chain = good['latent-chain', 0]
    assert len(chain['moveInfos']) == 1
    assert chain['moveInfos'][0]['pv'][:4] == ['A1'] * 4  # Illegal in every real successor.
    if game != 'go':
        terminal = good['terminal', 0]
        assert terminal['rootInfo']['lossValue'] == 1
        assert terminal['rootInfo']['winrate'] == 0 and terminal['moveInfos'] == []
        assert terminal['policy'] == [-1] * 26
    if game == 'hex':
        config = config_for(tmp_path / 'symmetry.cfg', game, nnForcedSymmetry=1)
        result = subprocess.run(command(config, model), input='', capture_output=True, text=True, timeout=20)
        assert result.returncode != 0 and 'nnForcedSymmetry is invalid' in result.stdout + result.stderr


@cuda
def test_metadata_exact_canvas_and_bundle_game(tmp_path):
    model = bundle_for(tmp_path / 'model', 'go', metadata=True)
    config = config_for(tmp_path / 'metadata.cfg', 'go', cudaUseFP16='true', requireMaxBoardSize='true')
    def point(id, **changes):
        return request(id, rules='tromp-taylor', boardXSize=5, boardYSize=5, **changes)
    reports = run(config, [point('missing'), point('profile', overrideSettings=dict(humanSLProfile='rank_5d')),
        point('other-profile', overrideSettings=dict(humanSLProfile='rank_1k')),
        request('small', rules='tromp-taylor', boardXSize=4, boardYSize=5,
                overrideSettings=dict(humanSLProfile='rank_5d')),
        dict(id='models', action='query_models')], model)
    assert {r['id'] for r in reports if 'error' in r} == {'missing', 'small'}
    good = complete([r for r in reports if 'error' not in r])
    assert len(good) == 2
    assert next(r for r in reports if r['id'] == 'models')['models'][0]['usesHumanSLProfile'] is True
    assert "Unused key '" not in config.with_suffix('.log').read_text()
    for settings, error in [(dict(game='renju'), 'model.json game'),
                            (dict(maxBoardSizeForNNBuffer=6), 'buffer dimensions')]:
        game = settings.pop('game', 'go')
        config = config_for(tmp_path / 'bad.cfg', game, **settings)
        result = subprocess.run(command(config, model), input='', capture_output=True, text=True, timeout=20)
        assert result.returncode != 0 and error in result.stdout + result.stderr


@cuda
def test_shutdown_cancels_queued_and_running_latent_requests(tmp_path):
    model = bundle_for(tmp_path / 'model')
    config = config_for(tmp_path / 'shutdown.cfg', numAnalysisThreads=1)
    queries = [request(str(i), maxVisits=10**9) for i in range(4)]
    result = subprocess.run(command(config, model) + ['-quit-without-waiting'],
        input=''.join(json.dumps(q) + '\n' for q in queries), capture_output=True, text=True, timeout=25)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'All cleaned up, quitting' in result.stderr
    assert 'Async bot thread failed' not in result.stderr
