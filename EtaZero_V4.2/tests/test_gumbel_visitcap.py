"""Native visit-cap snapshots with Gumbel, including async root conflicts."""
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


def check_analysis(config, game, model='/dev/null'):
    shape = (9, 10) if game == 'xiangqi' else (5, 5)
    move = 'b0c2' if game == 'xiangqi' else 'A1'
    def position(id, **extra):
        return request(id, rules='tromp-taylor' if game == 'go' else game,
                       boardXSize=shape[0], boardYSize=shape[1], maxVisits=96, **extra)
    reports = run(config, [position('default'),
        position('explicit', overrideSettings=dict(visitCapContemptPla='W')),
        position('root-conflict', overrideSettings=dict(visitCapContemptPla='B')),
        position('partial-conflict', moves=[['B', move]], analyzeTurns=[0, 1],
                 overrideSettings=dict(visitCapContemptPla='W')),
        position('next-turn', moves=[['B', move]]),
        position('clear-cap', overrideSettings=dict(visitCapContempt=0)),
        position('focus', focusMoves=[move], focusProb=.3)], model)
    errors = [r for r in reports if 'error' in r]
    assert {r['id'] for r in errors} == {'root-conflict', 'partial-conflict'}
    assert all('Gumbel root visit-cap' in r['error'] for r in errors)
    results = complete([r for r in reports if 'error' not in r])
    # No turn from a rejected multi-turn request is enqueued.
    assert {id for id, _ in results} == {'default', 'explicit', 'next-turn', 'clear-cap', 'focus'}
    for report in results.values():
        assert report['rootInfo']['visits'] == 96
        assert sum(m['edgeVisits'] for m in report['moveInfos']) == 95
    return results


def check_gtp(config, model='/dev/null'):
    commands = ['kata-search_analyze B 0 rootInfo true',
        'kata-set-param visitCapContemptPla B',
        'kata-search B', 'kata-search_cancellable B',
        'kata-search_analyze B 0 rootInfo true', 'kata-search_analyze_cancellable B 0',
        'kata-analyze B 0',
        'kata-set-param visitCapContemptPla W',
        'kata-search_analyze B 0 rootInfo true',
        'kata-set-param ponderingEnabled true',
        'genmove B', 'stop',  # The new white root is explicitly capped.
        'kata-set-param visitCapContemptPla B',
        'kata-search_analyze W 0 rootInfo true', 'quit']
    replies = run_gtp(config, commands, model)
    assert [i for i, (error, _) in enumerate(replies) if error] == [2, 3, 4, 5, 6], replies
    assert all('Gumbel root visit-cap' in replies[i][1] for i in (2, 3, 4, 5, 6))
    for i in (0, 8, 13):
        assert re.search(r'rootInfo visits 96\b', replies[i][1]), replies[i]
    log = config.with_suffix('.log').read_text()
    assert 'Ponder search not started: Gumbel root visit-cap' in log
    assert 'All cleaned up' in log


@pytest.mark.parametrize('game', ['go', 'freestyle', 'standard', 'renju', 'hex', 'xiangqi'])
@pytest.mark.parametrize('interior', ['puct', 'gumbel'])
def test_source_visit_cap_protocol_and_recovery(tmp_path, game, interior):
    values = dict(visitCapContempt=4, gumbelMaxConsideredActions=2,
                  nonrootSearchAlgorithm=interior, numSearchThreads=2, maxVisits=96)
    config = analysis_config(tmp_path / 'analysis.cfg', game=game, gumbel=True,
                             maxBoardSizeForNNBuffer=10 if game == 'xiangqi' else 5, **values)
    check_analysis(config, game)
    config = (config_for_gtp(tmp_path / 'gtp.cfg', gumbel=True, **values) if game == 'xiangqi' else
              gtp_config(tmp_path / 'gtp.cfg', game, gumbel=True, **values))
    check_gtp(config)


@cuda
@pytest.mark.parametrize('game', ['go', 'renju', 'hex', 'xiangqi'])
@pytest.mark.parametrize('muzero,fp16', [(False, False), (True, True)])
def test_cuda_ordinary_and_latent_visit_caps(tmp_path, game, muzero, fp16):
    if muzero:
        model = bundle_for(tmp_path / 'bundle', game)
    else:
        torch.manual_seed(204)
        network = Model(component_config(game), 10 if game == 'xiangqi' else 5)
        network.initialize(); network.eval()
        model = tmp_path / 'model.bin'
        with model.open('wb') as stream:
            write_model_binary(network, stream, 'visit-cap-test')
    values = dict(visitCapContempt=4, gumbelMaxConsideredActions=2, maxVisits=96,
                  numSearchThreads=4, cudaUseFP16=str(fp16).lower())
    config = (config_for(tmp_path / 'analysis.cfg', game, gumbel=True, **values) if muzero else
              analysis_config(tmp_path / 'analysis.cfg', game=game, random=False, gumbel=True,
                              maxBoardSizeForNNBuffer=10 if game == 'xiangqi' else 5, **values))
    check_analysis(config, game, model)
    values.update(muzero=muzero, random=False, gumbel=True, nnMaxBatchSize=4)
    config = (config_for_gtp(tmp_path / 'gtp.cfg', **values) if game == 'xiangqi' else
              gtp_config(tmp_path / 'gtp.cfg', game, **values))
    check_gtp(config, model)
