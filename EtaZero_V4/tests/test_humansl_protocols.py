"""Native HumanSL protocols with ordinary and MuZero model services."""
import json
import re
import subprocess

import pytest
import torch

from export_model_pytorch import write_model_binary
from export_muzero import export_components
from katago.train.model_pytorch import Model
from katago.train.muzero_model import MuZeroConfig, MuZeroModel
from test_analysis import (analysis_config, command as analysis_command, complete, request, run,
                           check_live_controls, full_history_requests, check_history)
from test_gtp import command as gtp_command, gtp_config, run_gtp
from test_muzero_analysis import cuda
from test_muzero_model import component_config
from test_muzero_gtp import check_live_gtp
from test_xiangqi_gtp import MOVES, config_for_gtp


def model_for(path, game, latent, *, metadata=False, width=6, canvas=None):
    canvas = canvas or (10 if game == 'xiangqi' else 5)
    settings = component_config(game)
    if metadata:
        settings['metadata_encoder'] = dict(meta_encoder_version=1, internal_num_channels=16)
    if latent:
        model = MuZeroModel(MuZeroConfig(game, width, settings, component_config(game, 'nbt'),
                                        component_config(game)), canvas)
        model.initialize(); model.eval()
        export_components(model, path, path.name)
    else:
        path = path.with_suffix('.bin')
        model = Model(settings, canvas)
        model.initialize(); model.eval()
        with path.open('wb') as out:
            write_model_binary(model, out, path.stem)
    return path


def options_for(game, fp16, primary_mu):
    return dict(useMuZero=str(primary_mu).lower(), useGraphSearch='false', subtreeValueBiasFactor=0,
                avoidRepeatedPatternUtility=0, maxBoardSizeForNNBuffer=10 if game == 'xiangqi' else 5,
                cudaUseFP16=str(fp16).lower(), nnForcedSymmetry=0, numSearchThreads=2,
                chosenMoveTemperature=0, chosenMoveTemperatureEarly=0)


def position(game, id, **changes):
    return request(id, rules='tromp-taylor' if game == 'go' else game,
                   boardXSize=9 if game == 'xiangqi' else 5, boardYSize=10 if game == 'xiangqi' else 5,
                   maxVisits=32, **changes)


def raw_policy(reply, game):
    policy = reply.split('\npolicy\n')[1]
    if game == 'xiangqi':
        return [float(value) for _, value in (line.split() for line in policy.splitlines())]
    return [float(value) for value in policy.replace('policyPass', '').split()]


@cuda
@pytest.mark.parametrize('game', ['go', 'freestyle', 'standard', 'renju', 'hex', 'xiangqi'])
@pytest.mark.parametrize('fp16', [False, True])
@pytest.mark.parametrize('primary_mu,secondary_mu', [(True, True), (True, False), (False, True)])
def test_models_exploration_raw_policy_and_lifetime(tmp_path, game, fp16, primary_mu, secondary_mu):
    torch.manual_seed(6512)
    primary = model_for(tmp_path/'primary', game, primary_mu)
    human = model_for(tmp_path/'human', game, secondary_mu, metadata=game == 'go', width=8)
    options = options_for(game, fp16, primary_mu)
    if game == 'go': options['humanSLProfile'] = 'rank_5d'
    config = analysis_config(tmp_path/'analysis.cfg', game=game, random=False, gumbel=fp16,
                             humanSLChosenMoveProp=1, **options)
    tree_fields = ['humanSLPlaExploreProbWeightful', 'humanSLPlaExploreProbWeightless',
                   'humanSLOppExploreProbWeightful', 'humanSLOppExploreProbWeightless']
    queries = [dict(id='models', action='query_models'), position(game, 'base'),
        position(game, 'puct', overrideSettings=dict(rootSearchAlgorithm='puct', nonrootSearchAlgorithm='puct')),
        position(game, 'gumbel', overrideSettings=dict(rootSearchAlgorithm='gumbel', nonrootSearchAlgorithm='gumbel')),
        position(game, 'root-weightless', overrideSettings=dict(humanSLRootExploreProbWeightless=1)),
        position(game, 'root-conflict', overrideSettings=dict(rootSearchAlgorithm='gumbel', humanSLRootExploreProbWeightful=.1)),
        *[position(game, field, overrideSettings={field: .5}) for field in tree_fields],
        dict(id='cache', action='clear_cache'), position(game, 'after-errors')]
    if not primary_mu:
        queries.append(position(game, 'graph', overrideSettings=dict(useGraphSearch=True,
            rootSearchAlgorithm='puct', nonrootSearchAlgorithm='puct', humanSLPlaExploreProbWeightful=.5,
            humanSLOppExploreProbWeightful=.5, subtreeValueBiasFactor=.4)))
    reports = run(config, queries, primary, human)
    errors = {r['id']: r['error'] for r in reports if 'error' in r}
    assert set(errors) == {'root-conflict', *([] if secondary_mu else tree_fields)}, errors
    assert 'root weightful HumanSL' in errors['root-conflict']
    for field in tree_fields:
        if not secondary_mu: assert 'ordinary HumanSL is root-only' in errors[field]
    models = next(r['models'] for r in reports if r['id'] == 'models')
    assert len(models) == 2
    assert models[1]['name'] == str(human)
    assert models[1]['internalName'] == ('human-prediction' if secondary_mu else 'human')
    assert models[1]['usesHumanSLProfile'] == (game == 'go')
    results = complete([r for r in reports if 'error' not in r])
    assert results['root-weightless', 0]['rootInfo']['visits'] == 1
    for report in results.values():
        assert 'humanPolicy' in report and sum(max(0, p) for p in report['humanPolicy']) == pytest.approx(1, abs=2e-4)

    # The secondary alone must produce the same real-root policy. This also
    # detects accidentally routing raw-human-NN or query output to the primary.
    reference_options = options_for(game, fp16, secondary_mu)
    if game == 'go': reference_options['humanSLProfile'] = 'rank_5d'
    reference = analysis_config(tmp_path/'reference.cfg', game=game, random=False, **reference_options)
    expected = complete(run(reference, [position(game, 'reference')], human))['reference', 0]['policy']
    tolerance = 2e-3 if fp16 else 5e-5
    assert results['base', 0]['humanPolicy'] == pytest.approx(expected, abs=tolerance)

    configure = config_for_gtp if game == 'xiangqi' else lambda path, **kw: gtp_config(path, game, **kw)
    gtp = configure(tmp_path/'gtp.cfg', muzero=primary_mu, gumbel=fp16, random=False,
                    humanSLChosenMoveProp=1, **options)
    cmds = ['kata-get-models', 'version', 'kata-raw-human-nn 0', 'kata-search B', 'clear_cache',
            'kata-set-param humanSLPlaExploreProbWeightful .5', 'kata-search B',
            'kata-set-param humanSLPlaExploreProbWeightful 0', 'kata-set-param humanSLRootExploreProbWeightless 1',
            'kata-search_analyze B 0 rootInfo true', 'clear_board']
    if game != 'xiangqi': cmds += ['boardsize 4', 'kata-raw-human-nn 0', 'boardsize 5']
    cmds += ['kata-search B', 'quit']
    replies = run_gtp(gtp, cmds, primary, human)
    assert {i for i,(err,_) in enumerate(replies) if err} == (set() if secondary_mu else {5}), replies
    assert json.loads(replies[0][1])[1]['internalName'] == models[1]['internalName']
    actual = raw_policy(replies[2][1], game)
    for p, q in zip(actual, expected):
        assert p != p if q < 0 else p == pytest.approx(q, abs=tolerance)
    assert len(actual) == len(expected)
    move = replies[3][1]
    index = MOVES.index(move) if game == 'xiangqi' else 25 if move == 'pass' else (5-int(move[1:]))*5 + 'ABCDEFGHJKLMNOPQRSTUVWXYZ'.index(move[0])
    assert expected[index] == pytest.approx(max(expected), abs=tolerance)
    assert re.search(r'rootInfo visits 1\b', replies[9][1])
    log = gtp.with_suffix('.log').read_text()
    assert log.count('Loaded human SL net with nnXLen') == 1
    assert 'All cleaned up, quitting' in log


@cuda
@pytest.mark.parametrize('secondary_mu', [False, True])
def test_profile_activation_and_recovery(tmp_path, secondary_mu):
    torch.manual_seed(203)
    primary = model_for(tmp_path/'primary', 'go', True)
    human = model_for(tmp_path/'human', 'go', secondary_mu, metadata=True, width=8)
    options = options_for('go', False, True)
    config = analysis_config(tmp_path/'analysis.cfg', game='go', random=False, humanSLChosenMoveProp=1, **options)
    reports = complete(run(config, [position('go', 'inactive'),
        position('go', 'active', overrideSettings=dict(humanSLProfile='rank_5d')),
        position('go', 'inactive-again')], primary, human))
    assert 'humanPolicy' not in reports['inactive', 0] and 'humanPolicy' not in reports['inactive-again', 0]
    assert 'humanPolicy' in reports['active', 0]
    gtp = gtp_config(tmp_path/'gtp.cfg', muzero=True, random=False, humanSLChosenMoveProp=1, **options)
    replies = run_gtp(gtp, ['kata-search B', 'kata-raw-human-nn 0',
        'kata-set-param humanSLProfile rank_5d', 'kata-raw-human-nn 0', 'kata-search B', 'quit'], primary, human)
    assert [error for error,_ in replies] == [False, True, False, False, False, False]


@cuda
@pytest.mark.parametrize('wrong', ['game', 'canvas'])
def test_secondary_mismatch_rejected_before_worker_start(tmp_path, wrong):
    primary = model_for(tmp_path/'primary', 'go', True)
    human = model_for(tmp_path/'human', 'hex' if wrong == 'game' else 'go', True, canvas=6 if wrong == 'canvas' else 5)
    options = options_for('go', False, True)
    for protocol, configure, command in [('analysis', analysis_config, analysis_command), ('gtp', gtp_config, gtp_command)]:
        config = configure(tmp_path/f'{protocol}.cfg', game='go', random=False, **options)
        result = subprocess.run(command(config, primary, human), input='', text=True, capture_output=True, timeout=30)
        assert result.returncode != 0
        assert wrong in (result.stdout + result.stderr).lower()


@cuda
@pytest.mark.parametrize('gumbel', [False, True])
def test_secondary_live_focus_cancel_ponder_and_clock(tmp_path, gumbel):
    torch.manual_seed(61)
    primary = model_for(tmp_path/'primary-xq', 'xiangqi', True)
    human = model_for(tmp_path/'human-xq', 'xiangqi', True, width=8)
    options = options_for('xiangqi', gumbel, True)
    options.update(humanSLPlaExploreProbWeightful=.3, humanSLOppExploreProbWeightless=.2)
    config = analysis_config(tmp_path/'analysis.cfg', game='xiangqi', random=False, gumbel=gumbel,
                             numAnalysisThreads=1, **options)
    check_live_controls(config, tmp_path/'analysis-live.log', gumbel, primary, human)
    primary = model_for(tmp_path/'primary-go', 'go', True)
    human = model_for(tmp_path/'human-go', 'go', True, width=8)
    options = options_for('go', gumbel, True)
    config = gtp_config(tmp_path/'gtp.cfg', muzero=True, gumbel=gumbel, random=False,
        maxVisits=1000000000, maxTime=100, lagBuffer=0, ponderingEnabled='true',
        humanSLPlaExploreProbWeightful=.3, humanSLOppExploreProbWeightless=.2, **options)
    check_live_gtp(config, primary, tmp_path/'gtp-live.log', human)


@cuda
def test_ordinary_primary_cannot_change_secondary_game(tmp_path):
    primary = model_for(tmp_path/'primary', 'renju', False)
    human = model_for(tmp_path/'human', 'renju', True)
    options = options_for('renju', False, False)
    config = analysis_config(tmp_path/'analysis.cfg', game='renju', random=False, **options)
    reports = run(config, [position('hex', 'wrong'), position('renju', 'good')], primary, human)
    error = next(r for r in reports if r['id'] == 'wrong')
    assert error['field'] == 'rules' and 'secondary' in error['error']
    assert 'humanPolicy' in next(r for r in reports if r['id'] == 'good')
    sgf = tmp_path/'hex.sgf'; sgf.write_text('(;GM[11]FF[4]SZ[5]RU[hex];B[aa])')
    config = gtp_config(tmp_path/'gtp.cfg', 'renju', random=False, **options)
    replies = run_gtp(config, ['play B C3', 'printsgf', 'kata-set-rules hex', f'loadsgf {sgf}',
                              'printsgf', 'kata-search W', 'quit'], primary, human)
    assert [error for error,_ in replies] == [False, False, True, True, False, False, False]
    assert 'secondary' in replies[2][1] and 'secondary' in replies[3][1]
    assert replies[1][1] == replies[4][1]


@cuda
@pytest.mark.parametrize('primary_mu', [False, True])
def test_secondary_preserves_xiangqi_history_and_terminal_boundary(tmp_path, primary_mu):
    primary = model_for(tmp_path/'primary', 'xiangqi', primary_mu)
    human = model_for(tmp_path/'human', 'xiangqi', True, width=8)
    options = options_for('xiangqi', True, primary_mu)
    config = analysis_config(tmp_path/'analysis.cfg', game='xiangqi', random=False, **options)
    check_history(run(config, full_history_requests(), primary, human), human=True, latent=primary_mu)
    config = config_for_gtp(tmp_path/'gtp.cfg', muzero=primary_mu, random=False, **options)
    replies = run_gtp(config, ['set_position fen 5k3/9/9/9/9/9/9/9/9/4K4 w - - 0 1',
                              'kata-raw-human-nn 0', 'clear_board', 'kata-raw-human-nn 0', 'quit'], primary, human)
    assert [error for error,_ in replies] == [False, True, False, False, False]
    assert 'terminal' in replies[1][1]
