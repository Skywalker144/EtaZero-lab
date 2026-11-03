"""Native weightless focus is extra work, including in a Gumbel latent tree."""
import re

import pytest
import torch

from export_model_pytorch import write_model_binary
from katago.train.model_pytorch import Model
from test_analysis import analysis_config, request, run, complete
from test_gtp import gtp_config, run_gtp
from test_muzero_analysis import bundle_for, config_for, cuda
from test_muzero_model import component_config
from test_xiangqi_gtp import config_for_gtp


def check_focus(config, game, model='/dev/null'):
    target = 'b0c2' if game == 'xiangqi' else 'A1'
    rules = 'tromp-taylor' if game == 'go' else game
    shape = (9, 10) if game == 'xiangqi' else (5, 5)
    def position(id, **extra):
        return request(id, rules=rules, boardXSize=shape[0], boardYSize=shape[1], maxVisits=32, **extra)
    reports = complete(run(config, [position('focus', focusMoves=[target], focusProb=.8),
        position('avoided', focusMoves=[target], focusProb=.8,
                 avoidMoves=[dict(player='B', moves=[target], untilDepth=1)]),
        position('no-focus')], model))
    for (name, _), report in reports.items():
        assert report['rootInfo']['visits'] == 32
        assert sum(m['edgeVisits'] for m in report['moveInfos']) == 31
        if name == 'focus':
            move = next(m for m in report['moveInfos'] if m['move'] == target)
            assert move['visits'] > move['edgeVisits']
        if name == 'avoided':
            assert target not in {m['move'] for m in report['moveInfos']}


@pytest.mark.parametrize('game', ['go', 'freestyle', 'standard', 'renju', 'hex', 'xiangqi'])
@pytest.mark.parametrize('interior', ['puct', 'gumbel'])
def test_focus_protocol_and_strict_root_budget(tmp_path, game, interior):
    config = analysis_config(tmp_path / 'analysis.cfg', game=game, gumbel=True,
                             maxBoardSizeForNNBuffer=10 if game == 'xiangqi' else 5,
                             gumbelMaxConsideredActions=1, nonrootSearchAlgorithm=interior)
    check_focus(config, game)
    config = (config_for_gtp(tmp_path / 'gtp.cfg', gumbel=True, gumbelMaxConsideredActions=1,
                            nonrootSearchAlgorithm=interior) if game == 'xiangqi' else
              gtp_config(tmp_path / 'gtp.cfg', game, gumbel=True, gumbelMaxConsideredActions=1,
                         nonrootSearchAlgorithm=interior))
    target = 'b0c2' if game == 'xiangqi' else 'A1'
    replies = run_gtp(config, [f'kata-search_analyze B 0 focus {target} 0.8 rootInfo true',
                               'kata-search_analyze B 0 rootInfo true', 'quit'])
    assert all(not error for error, _ in replies), replies
    assert f'info move {target} ' in replies[0][1]
    assert all(re.search(r'rootInfo visits 32\b', body) for _, body in replies[:2])


@cuda
@pytest.mark.parametrize('game', ['go', 'renju', 'hex', 'xiangqi'])
@pytest.mark.parametrize('muzero,fp16', [(False, False), (True, True)])
def test_cuda_ordinary_and_latent_focus(tmp_path, game, muzero, fp16):
    if muzero:
        model = bundle_for(tmp_path / 'bundle', game)
    else:
        torch.manual_seed(372)
        network = Model(component_config(game), 10 if game == 'xiangqi' else 5)
        network.initialize()
        network.eval()
        model = tmp_path / 'model.bin'
        with model.open('wb') as stream:
            write_model_binary(network, stream, 'focus-test')
    values = dict(gumbelMaxConsideredActions=1, numSearchThreads=4, cudaUseFP16=str(fp16).lower())
    config = (config_for(tmp_path / 'analysis.cfg', game, gumbel=True, **values) if muzero else
              analysis_config(tmp_path / 'analysis.cfg', game=game, random=False, gumbel=True,
                              maxBoardSizeForNNBuffer=10 if game == 'xiangqi' else 5, **values))
    check_focus(config, game, model)
    values.update(muzero=muzero, random=False, gumbel=True, nnMaxBatchSize=4)
    config = (config_for_gtp(tmp_path / 'gtp.cfg', **values) if game == 'xiangqi' else
              gtp_config(tmp_path / 'gtp.cfg', game, **values))
    target = 'b0c2' if game == 'xiangqi' else 'A1'
    replies = run_gtp(config, [f'kata-search_analyze B 0 focus {target} 0.8 rootInfo true',
                               'kata-search B', 'quit'], model)
    assert all(not error for error, _ in replies), replies
    assert f'info move {target} ' in replies[0][1]
    assert re.search(r'rootInfo visits 32\b', replies[0][1])
