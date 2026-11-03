"""The workbench reads native state and the existing analysis serializer."""
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from etazero import ROOT
from etazero.gtp import GtpProcess, GtpError
from test_gtp import gtp_config
from test_xiangqi_gtp import config_for_gtp, CHECK_FEN, CHECKS
from test_benchmarknn import model_for
from test_analysis import analysis_config, request, run as run_analysis


def profile(path, game, **options):
    return config_for_gtp(path, **options) if game == 'xiangqi' else gtp_config(path, game, **options)


def report_command(player='B', continuous=False):
    verb = 'kata-analyze' if continuous else 'kata-search_analyze_cancellable'
    return f'{verb} {player} 1 json true pvVisits true noResultValue true'


@pytest.mark.parametrize('game', ['go', 'freestyle', 'standard', 'renju', 'hex', 'xiangqi'])
@pytest.mark.parametrize('gumbel', [False, True])
def test_native_state_search_reports_and_undo(tmp_path, game, gumbel):
    config = profile(tmp_path / 'gtp.cfg', game, gumbel=gumbel)
    with GtpProcess(config, '/dev/null') as engine:
        start = engine.state()
        assert start['game'] == game and start['currentPlayer'] == 'B'
        assert not start['isGameFinished'] and start['result'] is None
        assert len(start['board']) == start['boardXSize'] * start['boardYSize']
        if game == 'xiangqi':
            assert len(start['pieces']) == 90 and len(start['legalMoves']) == 44
            assert start['board'] == [(p > 0) - (p < 0) for p in start['pieces']]
        else:
            assert start['board'] == [0] * 25
            assert len(start['legalMoves']) == 25 + (game == 'go')
        with pytest.raises(GtpError, match='No search result'):
            engine.analysis()
        move = 'b0c2' if game == 'xiangqi' else 'C3'
        engine.command(f'play B {move}')
        current = engine.state()
        assert current['currentPlayer'] == 'W' and current['moves'] == [['B', move]]
        assert move not in current['legalMoves']
        assert current['turnNumber'] == start['turnNumber'] + 1
        reports = []
        reply = engine.command(report_command('W'), on_report=reports.append)
        assert reply.body.startswith('play ') and reply.body[5:] in current['legalMoves']
        assert reports and reply.analysis == reports[-1]
        assert engine.state() == current  # Searching never played the suggested move.
        report = engine.analysis()
        assert report == reports[-1]
        assert report['turnNumber'] == current['turnNumber']
        root = report['rootInfo']
        assert root['currentPlayer'] == 'W' and root['visits'] > 0
        assert sum(root[key] for key in ('winValue', 'drawValue', 'lossValue', 'noResultValue')) == pytest.approx(1)
        assert root['winrate'] == pytest.approx(root['winValue'] + .5 * (root['drawValue'] + root['noResultValue']))
        # Source point-game JSON retains the pass slot, masked for non-Go.
        assert len(report['policy']) == (2086 if game == 'xiangqi' else 26)
        if game not in ('go', 'xiangqi'):
            assert report['policy'][-1] < 0
        assert all(row['move'] in current['legalMoves'] for row in report['moveInfos'])
        assert sum(p for p in report['policy'] if p >= 0) == pytest.approx(1, abs=1e-5)
        assert all(len(row['pvVisits']) == len(row['pv']) for row in report['moveInfos'])
        engine.command('undo')
        assert engine.state() == start
    assert engine.process.poll() == 0


@pytest.mark.parametrize('exact', [False, True])
def test_board_size_obeys_configured_nn_buffer_without_disconnect(tmp_path, exact):
    config = profile(tmp_path / 'gtp.cfg', 'go', maxBoardSizeForNNBuffer=7,
                     requireMaxBoardSize=str(exact).lower(), defaultBoardSize=7)
    with GtpProcess(config, '/dev/null') as engine:
        engine.command('play B D4')
        before = engine.state()
        if exact:
            with pytest.raises(GtpError, match='requires an exact board size'):
                engine.command('boardsize 5')
            assert engine.state() == before
        else:
            engine.command('boardsize 5')
            assert engine.state()['boardXSize'] == 5
            assert engine.command(report_command(), on_report=lambda _: None).body.startswith('play ')
        before = engine.state()
        with pytest.raises(GtpError, match='NN buffer'):
            engine.command('boardsize 9')
        assert engine.state() == before
        sgf = tmp_path / 'oversize.sgf'
        sgf.write_text('(;GM[1]SZ[9]RU[Tromp-Taylor]KM[0])')
        with pytest.raises(GtpError, match='NN buffer'):
            engine.command(f'loadsgf {sgf}')
        assert engine.state() == before
        assert engine.command(report_command(before['currentPlayer']), on_report=lambda _: None).body.startswith('play ')


@pytest.mark.skipif(os.environ.get('ETAZERO_EIGEN_TESTS') != '1', reason='requires Eigen real-model build')
@pytest.mark.parametrize('game', ['go', 'renju', 'hex'])
@pytest.mark.parametrize('muzero', [False, True])
def test_real_model_search_on_smaller_board(tmp_path, game, muzero):
    model = model_for(tmp_path / 'model', game, muzero)
    config = profile(tmp_path / 'gtp.cfg', game, muzero=muzero, random=False,
                     inputsUseNHWC='true', maxBoardSizeForNNBuffer=5,
                     requireMaxBoardSize='false', maxVisits=8)
    with GtpProcess(config, model, binary=ROOT / 'build-eigen/katago') as engine:
        engine.command('boardsize 3')
        assert engine.state()['boardXSize'] == 3
        reply = engine.command(report_command(), on_report=lambda _: None)
        assert reply.body.startswith('play ')
        assert reply.analysis['rootInfo']['visits'] > 0
        with pytest.raises(GtpError, match='NN buffer|model canvas'):
            engine.command('boardsize 7')
        assert engine.state()['boardXSize'] == 3


@pytest.mark.parametrize('game,moves', [
    ('go', ['pass', 'pass']),
    ('freestyle', ['A1', 'B1', 'A2', 'B2', 'A3', 'B3', 'A4', 'B4', 'A5']),
    ('standard', ['A1', 'B1', 'A2', 'B2', 'A3', 'B3', 'A4', 'B4', 'A5']),
    ('renju', ['A1', 'B1', 'A2', 'B2', 'A3', 'B3', 'A4', 'B4', 'A5']),
    ('hex', ['A1', 'B1', 'A2', 'B2', 'A3', 'B3', 'A4', 'B4', 'A5']),
])
def test_terminal_state_comes_from_rules_and_undo_restores_it(tmp_path, game, moves):
    config = profile(tmp_path / 'gtp.cfg', game)
    with GtpProcess(config, '/dev/null') as engine:
        for i, move in enumerate(moves):
            assert move in engine.state()['legalMoves']
            engine.command(f'play {"B" if i % 2 == 0 else "W"} {move}')
        end = engine.state()
        assert end['isGameFinished'] and not end['isNoResult'] and end['legalMoves'] == []
        assert end['result'] == engine.command('final_score').body
        engine.command('undo')
        assert not engine.state()['isGameFinished'] and engine.state()['result'] is None


def test_go_captures_and_xiangqi_complete_history_are_native(tmp_path):
    config = profile(tmp_path / 'go.cfg', 'go')
    with GtpProcess(config, '/dev/null') as engine:
        engine.command('set_position W C3 B B3 B D3 B C2')
        engine.command('play B C4')
        captured = engine.state()
        assert captured['board'][12] == 0 and sum(p == 1 for p in captured['board']) == 4
        engine.command('undo')
        assert engine.state()['board'][12] == -1
    config = profile(tmp_path / 'xiangqi.cfg', 'xiangqi')
    with GtpProcess(config, '/dev/null') as engine:
        engine.command('set_position fen ' + CHECK_FEN)
        states = [engine.state()]
        for i, move in enumerate(CHECKS):
            assert move in states[-1]['legalMoves']
            engine.command(f'play {"B" if i % 2 == 0 else "W"} {move}')
            states.append(engine.state())
        assert states[-1]['isGameFinished'] and states[-1]['result'] == 'W+'
        engine.command('undo')
        assert engine.state() == states[-2]
        engine.command('play W ' + CHECKS[-1])
        assert engine.state() == states[-1]


@pytest.mark.parametrize('continuous', [False, True])
@pytest.mark.parametrize('gumbel', [False, True])
def test_cancellation_drains_native_framing_and_preserves_position(tmp_path, continuous, gumbel):
    config = profile(tmp_path / 'gtp.cfg', 'hex', gumbel=gumbel, maxVisits=1000000)
    with GtpProcess(config, '/dev/null') as engine, ThreadPoolExecutor(1) as executor:
        before = engine.state()
        reports = []
        ready = Event()
        def on_report(report):
            reports.append(report)
            ready.set()
        future = executor.submit(engine.command, report_command(continuous=continuous), on_report=on_report)
        assert ready.wait(15)
        assert not future.done()  # A live search, not a completed job or an idle lock.
        engine.stop()
        reply = future.result(timeout=15)
        assert reply.analysis is not None and reports
        assert engine.state() == before
        # The newline has no separate response to pollute the next request.
        engine.command('play B C3')
        assert engine.state()['moves'] == [['B', 'C3']]
        engine.command('undo')
        assert engine.state() == before


@pytest.mark.parametrize('callback_error', [ValueError, GtpError])
def test_bad_command_is_recoverable_but_transport_failures_close_process(tmp_path, callback_error):
    config = profile(tmp_path / 'gtp.cfg', 'renju')
    with GtpProcess(config, '/dev/null') as engine:
        before = engine.state()
        for command in ['kata-get-state extra', 'kata-get-analysis extra',
                        'kata-search_analyze B 1 json true maxmoves 2', 'lz-analyze B 1 json true']:
            with pytest.raises(GtpError):
                engine.command(command)
            assert engine.state() == before
        for command in ['quit\nplay B C3', '1 quit', 'play\tB C3', '']:
            with pytest.raises(ValueError):
                engine.command(command)
        def bad_callback(report):
            raise callback_error('client callback failed')
        with pytest.raises(callback_error, match='client callback failed'):
            engine.command(report_command(), on_report=bad_callback)
        assert engine.process.poll() is not None
    with pytest.raises(RuntimeError, match='response timed out'):
        with GtpProcess(config, '/dev/null') as engine:
            engine.timeout = .001
            # An explicit long report interval makes this a verified timeout
            # regardless of the small random model's search speed.
            engine.command('kata-analyze B 10000 json true', on_report=lambda report: None)
    assert engine.process.poll() is not None


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires actual CUDA GTP and JSON engines')
@pytest.mark.parametrize('game', ['go', 'renju', 'hex', 'xiangqi'])
@pytest.mark.parametrize('muzero,fp16,gumbel', [(False, False, False), (False, True, True),
                                             (True, False, True), (True, True, False)])
def test_cuda_gtp_json_uses_same_network_report_as_native_analysis(tmp_path, game, muzero, fp16, gumbel):
    model = model_for(tmp_path / 'model', game, muzero)
    config = profile(tmp_path / 'gtp.cfg', game, muzero=muzero, random=False, gumbel=gumbel,
                     cudaUseFP16=str(fp16).lower(), reportAnalysisWinratesAs='WHITE', maxVisits=8,
                     maxBoardSizeForNNBuffer=10 if game == 'xiangqi' else 5)
    move = 'b0c2' if game == 'xiangqi' else 'C3'
    with GtpProcess(config, model) as engine:
        engine.command(f'play B {move}')
        reports = []
        reply = engine.command(report_command('W'), on_report=reports.append)
        assert reply.analysis == engine.analysis() == reports[-1]
        actual = reply.analysis
        engine.command(f'play W {reply.body.removeprefix("play ")}')
        assert len(engine.state()['moves']) == 2
        engine.command('undo')
        assert engine.state()['moves'] == [['B', move]]
    reference_config = analysis_config(tmp_path / 'analysis.cfg', game=game, random=False, gumbel=gumbel,
        maxVisits=8, maxBoardSizeForNNBuffer=10 if game == 'xiangqi' else 5,
        useMuZero=str(muzero).lower(), cudaUseFP16=str(fp16).lower(),
        useGraphSearch=str(not (muzero or gumbel)).lower(),
        subtreeValueBiasFactor=0 if muzero else .4, avoidRepeatedPatternUtility=0 if muzero else .03)
    query = request(rules='tromp-taylor' if game == 'go' else game,
                    boardXSize=9 if game == 'xiangqi' else 5, boardYSize=10 if game == 'xiangqi' else 5,
                    moves=[['B', move]])
    reference = run_analysis(reference_config, [query], model)[0]
    assert 'error' not in reference, reference
    # Compare model evidence, not nondeterministic search allocation. Both use
    # source fixed-orientation native evaluation of the identical real state.
    for key in ['rawWinrate', 'rawDrawProb', 'rawNoResultProb', 'rawStWrError', 'rawVarTimeLeft']:
        assert actual['rootInfo'][key] == pytest.approx(reference['rootInfo'][key], abs=2e-4 if fp16 else 2e-5)
    assert actual['policy'] == pytest.approx(reference['policy'], abs=2e-4 if fp16 else 2e-5)
