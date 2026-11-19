"""Native JSON queue, full game history, joint actions and asynchronous controls."""
import json
import os
from queue import Queue
import subprocess
from threading import Thread

import pytest
import torch

from etazero import ROOT
from export_model_pytorch import write_model_binary
from katago.game import xiangqi as xq
from katago.train.model_pytorch import Model
from test_muzero_model import component_config


def analysis_config(path, *, game='xiangqi', random=True, gumbel=False, **changes):
    values = dict(game=game, numAnalysisThreads=2, numSearchThreads=2, maxVisits=40,
        maxBoardSizeForNNBuffer=10, nnMaxBatchSize=4, nnCacheSizePowerOfTwo=10,
        nnMutexPoolSizePowerOfTwo=6, nnRandomize='false', rootSymmetryPruning='false',
        useGraphSearch=str(not gumbel).lower(), wideRootNoise=0,
        ignorePreRootHistory='false', reportAnalysisWinratesAs='WHITE',
        subtreeValueBiasFactor=.4, avoidRepeatedPatternUtility=.03,
        rootSearchAlgorithm='gumbel' if gumbel else 'puct',
        nonrootSearchAlgorithm='gumbel' if gumbel else 'puct',
        debugSkipNeuralNet=str(random).lower(), logToStdout='false')
    values.update(changes)
    path.write_text(''.join(f'{key} = {value}\n' for key, value in values.items()))
    return path


def request(id='position', **changes):
    result = dict(id=id, rules='xiangqi', boardXSize=9, boardYSize=10,
                  moves=[], includePolicy=True, includePVVisits=True, includeNoResultValue=True)
    result.update(changes)
    return result


def command(config, model='/dev/null', human=None):
    return [str(ROOT / 'build/katago'), 'analysis', '-config', str(config), '-model', str(model),
            *(['-human-model', str(human)] if human else [])]


def run(config, requests, model='/dev/null', human=None):
    result = subprocess.run(command(config, model, human),
        input=''.join(json.dumps(q) + '\n' for q in requests), capture_output=True, text=True, timeout=90)
    config.with_suffix('.log').write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return [json.loads(line) for line in result.stdout.splitlines()]


def complete(reports):
    assert not [r for r in reports if 'error' in r or 'warning' in r], reports
    return {(r['id'], r['turnNumber']): r for r in reports if 'rootInfo' in r and not r['isDuringSearch']}


def check_joint_report(report, human=False):
    assert len(report['policy']) == 2086
    assert sum(p for p in report['policy'] if p >= 0) == pytest.approx(1, abs=2e-4)
    assert report['rootInfo']['noResultValue'] == 0
    strings = {f'{chr(97+x)}{y}{chr(97+u)}{v}': a for a, (x, y, u, v) in enumerate(xq.MOVES)}
    for move in report['moveInfos']:
        assert move['move'] in strings and report['policy'][strings[move['move']]] >= 0
        assert all(ply in strings for ply in move['pv'])
        assert len(move['pvVisits']) == len(move['pv'])
        assert move['noResultValue'] == 0
        if human:
            assert move['humanPrior'] == pytest.approx(report['humanPolicy'][strings[move['move']]], abs=1e-7)
    if human:
        assert len(report['humanPolicy']) == 2086
        assert sum(p for p in report['humanPolicy'] if p >= 0) == pytest.approx(1, abs=2e-4)


def full_history_requests():
    # Same occupancy after four plies, different history and repetition identity.
    cycle = [['B', 'b0c2'], ['W', 'h9g7'], ['B', 'c2b0'], ['W', 'g7h9']]
    # Black has only one escape at the seventh ply. The eighth adjudicates
    # red's long check; reconstructing from the last FEN loses this result.
    checks = ['e8f8', 'f9e9', 'f8e8', 'e9f9', 'e8f8', 'f9e9', 'f8e8', 'e9f9']
    return [request('cycle', moves=cycle, analyzeTurns=[0, 1, 4], priorities=[0, 2, 1]),
        request('checking', xiangqiFen='5k3/3RR4/9/9/4P4/9/9/9/9/4K4 w - - 0 1',
                moves=[['B' if i % 2 == 0 else 'W', m] for i, m in enumerate(checks)], analyzeTurns=[7, 8],
                allowMoves=[dict(player='W', moves=['e9f9'], untilDepth=1)]),
        request('draw', xiangqiFen='5k3/9/9/9/9/9/9/9/9/4K4 w - - 0 1'),
        dict(id='models', action='query_models')]


def check_history(reports, human=False, latent=False):
    results = complete(reports)
    assert len(results) == 6, reports
    for turn in (0, 1, 4):
        check_joint_report(results['cycle', turn], human)
    assert results['cycle', 0]['rootInfo']['thisHash'] != results['cycle', 4]['rootInfo']['thisHash']
    assert results['cycle', 1]['rootInfo']['currentPlayer'] == 'W'
    forced = results['checking', 7]
    check_joint_report(forced, human)
    assert {m['move'] for m in forced['moveInfos']} == {'e9f9'}
    if not latent:
        assert forced['rootInfo']['winrate'] > .95
    for key, winner in [(('checking', 8), 1), (('draw', 0), .5)]:
        terminal = results[key]
        assert terminal['rootInfo']['winrate'] == winner
        assert terminal['rootInfo']['drawValue'] == (winner == .5)
        assert terminal['rootInfo']['noResultValue'] == 0
        assert terminal['moveInfos'] == [] and terminal['policy'] == [-1] * 2086
        assert 'rawWinrate' not in terminal['rootInfo']
    models = next(r for r in reports if r['id'] == 'models')['models']
    assert len(models) == (2 if human else 1)


@pytest.mark.parametrize('gumbel', [False, True])
def test_xiangqi_analysis_full_history_and_terminal_results(tmp_path, gumbel):
    config = analysis_config(tmp_path / 'analysis.cfg', gumbel=gumbel)
    check_history(run(config, full_history_requests()))


def test_request_errors_do_not_break_the_analysis_worker(tmp_path):
    config = analysis_config(tmp_path / 'analysis.cfg')
    invalid = [dict(boardXSize=10), dict(boardYSize=9), dict(rules='chinese'), dict(xiangqiFen='bad'),
        dict(xiangqiFen=7), dict(initialPlayer='W'), dict(initialStones=[['B', 'A1']]),
        dict(moves=[['B', 'pass']]), dict(moves=[['B', 'J10']]), dict(moves=[['B', 'a0a9']]),
        dict(moves=[['W', 'h9g7']]), dict(komi=1), dict(whiteHandicapBonus='N'),
        dict(avoidMoves=[dict(player='B', moves=['pass'], untilDepth=1)]),
        dict(focusMoves=['bad']), dict(overrideSettings=dict(rootSearchAlgorithm='gumbel')),
        dict(xiangqiFen='5k3/9/9/9/9/9/9/9/9/4K4 w - - 0 1', moves=[['B', 'e0d0']])]
    reports = run(config, [request(str(i), **q) for i, q in enumerate(invalid)] + [request('ok')])
    assert {r['id'] for r in reports if 'error' in r} == set(map(str, range(len(invalid))))
    check_joint_report(next(r for r in reports if r['id'] == 'ok'))


def test_gumbel_root_weightful_human_override_is_rejected_before_async_search(tmp_path):
    config = analysis_config(tmp_path / 'analysis.cfg')
    reports = run(config, [request('bad', overrideSettings=dict(rootSearchAlgorithm='gumbel', useGraphSearch=False,
                                                              humanSLRootExploreProbWeightful=.5)),
                          request('ok', overrideSettings=dict(rootSearchAlgorithm='gumbel', useGraphSearch=False))],
                  human='/dev/null')
    error = next(r for r in reports if r['id'] == 'bad')
    assert error['field'] == 'overrideSettings' and 'root weightful HumanSL' in error['error']
    check_joint_report(next(r for r in reports if r['id'] == 'ok'), human=True)


def test_native_exact_canvas_rejects_small_request_before_async_search(tmp_path):
    config = analysis_config(tmp_path / 'exact.cfg', game='hex', maxBoardSizeForNNBuffer=5,
                             requireMaxBoardSize='true')
    reports = run(config, [request('small', rules='hex', boardXSize=4, boardYSize=5),
                           request('ok', rules='hex', boardXSize=5, boardYSize=5)])
    error = next(r for r in reports if r['id'] == 'small')
    assert error['field'] == 'boardXSize' and 'canvas' in error['error']
    assert len(next(r for r in reports if r['id'] == 'ok')['policy']) == 26


@pytest.mark.parametrize('game', ['go', 'freestyle', 'standard', 'renju', 'hex'])
def test_point_game_requests_keep_their_own_setup_rules(tmp_path, game):
    config = analysis_config(tmp_path / 'analysis.cfg', game=game)
    stones = [['W', 'D4'], ['B', 'C4'], ['B', 'E4'], ['B', 'D3'], ['B', 'D5']]
    rules = 'tromp-taylor' if game == 'go' else game
    reports = run(config, [request('setup', rules=rules, boardXSize=7, boardYSize=7,
        initialStones=stones, initialPlayer='B')])
    report = complete(reports)['setup', 0]
    assert len(report['policy']) == 50
    # Go removes the zero-liberty stone during setup. Placement games retain it.
    assert (report['policy'][3*7+3] >= 0) == (game == 'go')
    if game != 'go':
        stones = [['B', f'D{i}'] for i in range(1, 8)] if game == 'hex' else [['B', f'{c}4'] for c in 'ABCDE']
        terminal = complete(run(config, [request('terminal', rules=rules, boardXSize=7, boardYSize=7,
            initialStones=stones, initialPlayer='W')]))['terminal', 0]
        assert terminal['rootInfo']['lossValue'] == 1 and terminal['rootInfo']['winrate'] == 0
        assert terminal['policy'] == [-1] * 50 and terminal['moveInfos'] == []


def test_placement_report_identity_obeys_game_and_valid_symmetries(tmp_path):
    config = analysis_config(tmp_path / 'analysis.cfg', game='go')
    stones = [('B', 0, 1), ('W', 2, 3), ('B', 4, 1)]
    requests = []
    for game in ('freestyle', 'standard', 'renju', 'hex'):
        for symmetry in ('identity', 'rotate', 'transpose'):
            placements = []
            for pla, x, y in stones:
                if symmetry == 'rotate': x, y = 6-x, 6-y
                if symmetry == 'transpose': x, y = y, x
                placements.append([pla, f'{chr(65+x)}{y+1}'])
            requests.append(request(f'{game}-{symmetry}', rules=game, boardXSize=7, boardYSize=7,
                                    initialStones=placements, initialPlayer='B'))
    reports = complete(run(config, requests))
    assert len({reports[f'{g}-identity', 0]['rootInfo']['thisHash'] for g in ('freestyle', 'standard', 'renju', 'hex')}) == 4
    for game in ('freestyle', 'standard', 'renju', 'hex'):
        original = reports[f'{game}-identity', 0]['rootInfo']['symHash']
        assert original == reports[f'{game}-rotate', 0]['rootInfo']['symHash']
        assert (original == reports[f'{game}-transpose', 0]['rootInfo']['symHash']) == (game != 'hex')


@pytest.mark.parametrize('gumbel', [False, True])
def test_live_focus_cancellation_and_subsequent_requests(tmp_path, gumbel):
    config = analysis_config(tmp_path / 'analysis.cfg', gumbel=gumbel, numAnalysisThreads=1)
    check_live_controls(config, tmp_path / 'live.log', gumbel)


def check_live_controls(config, log, gumbel, model='/dev/null', human=None):
    messages = Queue()
    with log.open('w') as stderr:
        proc = subprocess.Popen(command(config, model, human), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=stderr, text=True, bufsize=1)
        def read():
            for line in proc.stdout:
                messages.put(json.loads(line))
            messages.put(None)
        reader = Thread(target=read, daemon=True)
        reader.start()
        def send(q):
            proc.stdin.write(json.dumps(q) + '\n'); proc.stdin.flush()
        def until(predicate):
            for _ in range(500):
                r = messages.get(timeout=15)
                assert r is not None, log.read_text()
                if predicate(r):
                    return r
            pytest.fail('No matching analysis message')
        try:
            send(request('running', maxVisits=10**9, reportDuringSearchEvery=.01,
                         firstReportDuringSearchAfter=.001))
            until(lambda r: r.get('id') == 'running' and r.get('isDuringSearch'))
            send(request('queued', maxVisits=10**9))
            send(dict(id='cancel-queued', action='terminate', terminateId='queued'))
            cancelled = until(lambda r: r.get('id') == 'queued')
            assert cancelled['noResults'] is True
            send(dict(id='focus', action='set_focus', targetId='running', focusMoves=['b0c2'], focusProb=.8))
            focus = until(lambda r: r.get('id') == 'focus')
            assert 'error' not in focus
            focused = until(lambda r: r.get('id') == 'running' and any(
                m['move'] == 'b0c2' and m['visits'] > m['edgeVisits'] for m in r.get('moveInfos', [])))
            assert focused['isDuringSearch']
            send(dict(id='clear-focus', action='set_focus', targetId='running', focusMoves=[]))
            assert 'error' not in until(lambda r: r.get('id') == 'clear-focus')
            send(dict(id='bad-focus', action='set_focus', targetId='running', focusMoves=['pass']))
            bad = until(lambda r: r.get('id') == 'bad-focus')
            assert 'error' in bad
            send(dict(id='stop', action='terminate', terminateId='running'))
            final = until(lambda r: r.get('id') == 'running' and r.get('isDuringSearch') is False)
            check_joint_report(final, human=human is not None)
            send(dict(id='clear', action='clear_cache'))
            until(lambda r: r.get('id') == 'clear')
            send(request('next', avoidMoves=[dict(player='B', moves=['b0c2'], untilDepth=1)]))
            result = until(lambda r: r.get('id') == 'next')
            check_joint_report(result, human=human is not None)
            assert 'b0c2' not in {m['move'] for m in result['moveInfos']}
            proc.stdin.close()
            assert proc.wait(timeout=15) == 0, log.read_text()
            reader.join(timeout=2)
        finally:
            if proc.poll() is None:
                proc.kill(); proc.wait(timeout=10)


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires native CUDA analysis')
@pytest.mark.parametrize('kind,gumbel,human', [('regular', False, False), ('regular', False, True), ('transformer', True, False)])
def test_xiangqi_cuda_protocol_and_secondary_model(tmp_path, kind, gumbel, human):
    torch.manual_seed(512)
    paths = []
    for i in range(2 if human else 1):
        net = Model(component_config('xiangqi', kind), 10)
        net.initialize(); net.eval()
        path = tmp_path / f'model{i}.bin'
        with path.open('wb') as stream:
            write_model_binary(net, stream, f'analysis-{i}')
        paths.append(path)
    changes = dict(cudaUseFP16=str(human or gumbel).lower(), inputsUseNHWC=str(kind == 'transformer').lower(),
                   rootNumSymmetriesToSample=2, logSearchInfo='true')
    if human:
        changes.update(humanSLChosenMoveProp=.5, humanSLRootExploreProbWeightful=.4,
                       humanSLPlaExploreProbWeightless=.3, humanSLOppExploreProbWeightful=.3)
    config = analysis_config(tmp_path / 'cuda.cfg', random=False, gumbel=gumbel, **changes)
    check_history(run(config, full_history_requests(), paths[0], paths[1] if human else None), human)
    assert 'Xiangqi FEN:' in config.with_suffix('.log').read_text()


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires native CUDA model-family boundaries')
def test_loaded_model_families_reject_incompatible_queries_without_losing_the_worker(tmp_path):
    paths = {}
    for game in ('go', 'renju'):
        net = Model(component_config(game), 10)
        net.initialize(); net.eval()
        path = tmp_path / f'{game}.bin'
        with path.open('wb') as stream:
            write_model_binary(net, stream, game)
        paths[game] = path
    for game, other in [('go', 'renju'), ('renju', 'go')]:
        config = analysis_config(tmp_path / f'{game}.cfg', game=game, random=False, maxVisits=8)
        rules = lambda g: 'tromp-taylor' if g == 'go' else g
        reports = run(config, [request('bad', rules=rules(other), boardXSize=7, boardYSize=7),
            request('good', rules=rules(game), boardXSize=7, boardYSize=7)], paths[game])
        assert next(r for r in reports if r['id'] == 'bad')['field'] == 'rules'
        assert 'error' in next(r for r in reports if r['id'] == 'bad')
        assert len(next(r for r in reports if r['id'] == 'good')['policy']) == 50
    config = analysis_config(tmp_path / 'secondary.cfg', game='renju', random=False)
    reports = run(config, [request('bad', rules='renju', boardXSize=7, boardYSize=7),
        dict(id='models', action='query_models')], paths['renju'], paths['go'])
    assert 'secondary' in next(r for r in reports if r['id'] == 'bad')['error']
    assert len(next(r for r in reports if r['id'] == 'models')['models']) == 2
