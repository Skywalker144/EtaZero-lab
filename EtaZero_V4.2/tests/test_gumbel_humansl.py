"""Source HumanSL exploration and move style around strict Gumbel root quotas."""
import re

import pytest
import torch

from export_model_pytorch import write_model_binary
from katago.train.model_pytorch import Model
from test_analysis import analysis_config, request, run, complete
from test_gtp import gtp_config, run_gtp
from test_muzero_analysis import cuda
from test_muzero_model import component_config
from test_xiangqi_gtp import config_for_gtp, MOVES


def human_config(path, game, *, random=True, interior='gumbel', **changes):
    values = dict(maxBoardSizeForNNBuffer=10 if game == 'xiangqi' else 5, numSearchThreads=1,
                  subtreeValueBiasFactor=0, avoidRepeatedPatternUtility=0,
                  nonrootSearchAlgorithm=interior, humanSLChosenMoveProp=1, humanSLChosenMovePiklLambda=1e9)
    values.update(changes)
    return analysis_config(path, game=game, gumbel=True, random=random, **values)


def check_modes(config, game, model='/dev/null', human='/dev/null'):
    shape = (9, 10) if game == 'xiangqi' else (5, 5)
    def position(id, **extra):
        return request(id, rules='tromp-taylor' if game == 'go' else game,
                       boardXSize=shape[0], boardYSize=shape[1], maxVisits=64, **extra)
    reports = run(config, [position('style'),
        position('root-weightless', overrideSettings=dict(humanSLRootExploreProbWeightless=1)),
        position('tree-weightless', overrideSettings=dict(humanSLPlaExploreProbWeightless=1,
                                                         humanSLOppExploreProbWeightless=1)),
        position('tree-weightful', overrideSettings=dict(humanSLPlaExploreProbWeightful=1,
                                                        humanSLOppExploreProbWeightful=1)),
        position('mixed', overrideSettings=dict(humanSLRootExploreProbWeightless=.3,
                  humanSLPlaExploreProbWeightless=.2, humanSLOppExploreProbWeightless=.2,
                  humanSLPlaExploreProbWeightful=.4, humanSLOppExploreProbWeightful=.4),
                  focusMoves=['b0c2' if game == 'xiangqi' else 'A1'], focusProb=.3),
        position('incompatible', overrideSettings=dict(humanSLRootExploreProbWeightful=.1)),
        position('sampled', overrideSettings=dict(gumbelSampleVisits=True,
                                                 humanSLRootExploreProbWeightless=1))], model, human)
    errors = [r for r in reports if 'error' in r]
    assert len(errors) == 1 and errors[0]['id'] == 'incompatible'
    assert 'root weightful HumanSL' in errors[0]['error']
    results = complete([r for r in reports if 'error' not in r])
    for (id, _), report in results.items():
        human_policy = report['humanPolicy']
        assert len(human_policy) == (2086 if game == 'xiangqi' else 26)
        assert sum(max(0, p) for p in human_policy) == pytest.approx(1, abs=2e-4)
        assert 1 <= report['rootInfo']['visits'] <= 64
        assert sum(m['edgeVisits'] for m in report['moveInfos']) == report['rootInfo']['visits']-1
        if id in ('root-weightless', 'sampled'):
            assert report['rootInfo']['visits'] == 1
            assert sum(m['visits'] for m in report['moveInfos']) == 63
        if id in ('style', 'tree-weightful'):
            assert report['rootInfo']['visits'] == 64
        # The native report restricts this view to allocated children. With
        # full HumanSL blending and large PIKL lambda, its normalized weights
        # must match the secondary priors, independent of the SH allocation.
        weights = sum(m['playSelectionValue'] for m in report['moveInfos'])
        priors = sum(m['humanPrior'] for m in report['moveInfos'])
        assert weights > 0 and priors > 0
        for move in report['moveInfos']:
            assert move['playSelectionValue']/weights == pytest.approx(move['humanPrior']/priors, abs=2e-6)
    return results['style', 0]['humanPolicy']


@pytest.mark.parametrize('game', ['go', 'freestyle', 'standard', 'renju', 'hex', 'xiangqi'])
@pytest.mark.parametrize('interior', ['puct', 'gumbel'])
def test_native_human_exploration_modes(tmp_path, game, interior):
    config = human_config(tmp_path / 'analysis.cfg', game, interior=interior)
    check_modes(config, game)


@cuda
@pytest.mark.parametrize('game', ['go', 'renju', 'hex', 'xiangqi'])
@pytest.mark.parametrize('fp16', [False, True])
def test_cuda_secondary_model_and_actual_gtp_move(tmp_path, game, fp16):
    torch.manual_seed(997)
    models = []
    for index in range(2):
        settings = component_config(game)
        if game == 'go' and index == 1:
            settings['metadata_encoder'] = dict(meta_encoder_version=1, internal_num_channels=16)
        network = Model(settings, 10 if game == 'xiangqi' else 5)
        network.initialize(); network.eval()
        model = tmp_path / f'model{index}.bin'
        with model.open('wb') as out:
            write_model_binary(network, out, f'hu-man-{index}')
        models.append(model)
    options = dict(cudaUseFP16=str(fp16).lower(), nnForcedSymmetry=0)
    if game == 'go':
        options['humanSLProfile'] = 'rank_5d'
    config = human_config(tmp_path / 'analysis.cfg', game, random=False,
                          interior='gumbel' if fp16 else 'puct', **options)
    policy = check_modes(config, game, *models)
    options.update(humanSLChosenMoveProp=1, humanSLChosenMovePiklLambda=1e9,
                   chosenMoveTemperature=0, chosenMoveTemperatureEarly=0, nnMaxBatchSize=4,
                   numSearchThreads=2, gumbel=True, random=False)
    config = (config_for_gtp(tmp_path / 'gtp.cfg', **options) if game == 'xiangqi' else
              gtp_config(tmp_path / 'gtp.cfg', game, **options))
    replies = run_gtp(config, ['kata-search B', 'kata-set-param humanSLRootExploreProbWeightful .1',
                               'kata-search B', 'kata-set-param humanSLRootExploreProbWeightless 1',
                               'kata-search_analyze B 0 rootInfo true', 'quit'], *models)
    assert [error for error, _ in replies] == [False, True, False, False, False, False], replies
    for _, move in (replies[0], replies[2]):
        if game == 'xiangqi':
            index = MOVES.index(move)
        elif move == 'pass':
            index = 25
        else:
            index = (5-int(move[1:]))*5 + 'ABCDEFGHJKLMNOPQRSTUVWXYZ'.index(move[0])
        assert policy[index] == pytest.approx(max(policy), abs=2e-6)
    assert re.search(r'rootInfo visits 1\b', replies[4][1])
