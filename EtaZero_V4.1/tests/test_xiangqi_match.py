"""Full Xiangqi state through native tournaments, gatekeeper and outer acceptance."""
from collections import Counter
from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from etazero import ROOT
from etazero.evaluation import load_games, run_match
from etazero.run import checkpoint_progress
from export_model_pytorch import write_model_binary
from export_muzero import export_components
from match_dispatch_worker import worker_command
from test_muzero_model import make_model, limited_cpu_threads
from test_xiangqi_controller import controller_config, selfplay_config
from test_xiangqi_network import model

controller = importlib.import_module('etazero.run')
gpu = pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires native CUDA and learner')


def game_config(path, *, position='draw', random=False, gumbel=False, **changes):
    values = dict(game='xiangqi', bSizesXY='9-10', bSizeRelProbs=1, numGameThreads=2,
        numSearchThreads=2, maxVisits=16, maxMovesPerGame=12,
        logSearchInfo='true', logMoves='true', logToStdout='true', logGamesEvery=1,
        allowResignation='false', resignThreshold=-.95, resignConsecTurns=3,
        initGamesWithPolicy='false', nnMaxBatchSize=4, nnCacheSizePowerOfTwo=10,
        nnMutexPoolSizePowerOfTwo=8, subtreeValueBiasFactor=0, avoidRepeatedPatternUtility=0,
        rootSymmetryPruning='false', useGraphSearch=str(not gumbel).lower(),
        rootSearchAlgorithm='gumbel' if gumbel else 'puct',
        cudaUseFP16='true', debugSkipNeuralNet=str(random).lower())
    if position:
        poses = path.with_suffix('.poses')
        poses.mkdir()
        if position in ('draw', 'multi_turn'):
            # The first fixture draws after one ply. The second exercises
            # both bots before a rule draw or a rook mate, including tree reuse.
            fen = ('3aka3/9/9/9/4P4/9/9/9/9/R3K4 w - - 119 60' if position == 'draw' else
                   '3k5/9/9/9/9/9/9/9/9/R4K3 w - - 116 59')
            (poses / 'position.sgf').write_text(f'(;GM[7]SZ[9:10]XQFEN[{fen}]XQTURN[73])')
            values.update(startPosesFromSgfDir=str(poses), startPosesProb=1,
                          startPosesLoadProb=1, startPosesTurnWeightLambda=0)
        else:
            assert position == 'long_check'
            # The d8 rook blocks the alternative king escape on d9. Black must
            # play e9f9; that eighth ply adjudicates red's long check only if the
            # complete prehistory is retained.
            sample = dict(game='xiangqi', xSize=9, ySize=10,
                xiangqiFen='5k3/3RR4/9/9/4P4/9/9/9/9/4K4 w - - 0 1',
                xiangqiHistory=['e8f8', 'f9e9', 'f8e8', 'e9f9', 'e8f8', 'f9e9', 'f8e8'],
                nextPla='W', moveLocs=[], movePlas=[], initialTurnNumber=97,
                hintLoc='e9f9', weight=1, trainingWeight=1)
            (poses / 'position.hintposes.txt').write_text(json.dumps(sample) + '\n')
            # Source hint search requires clearing the tree before installing
            # its hint prior; ordinary match positions retain source tree reuse.
            values.update(hintPosesDir=str(poses), hintPosesProb=1, clearBotBeforeSearch='true')
    values.update(changes)
    path.write_text(''.join(f'{key} = {value}\n' for key, value in values.items()))
    return path


def match_config(path, models, **kwargs):
    game_config(path, **kwargs)
    with path.open('a') as stream:
        stream.write(f'numBots = {len(models)}\nnumGamesTotal = 6\n')
        for i, filename in enumerate(models):
            stream.write(f'botName{i} = bot{i}\nnnModelFile{i} = {filename}\n')
        # Different searches may share the same source model service.
        stream.write('rootSearchAlgorithm1 = gumbel\nuseGraphSearch1 = false\n')
    return path


def native_match(config, results=None, sgfs=None, **kwargs):
    command = [str(ROOT / 'build/katago'), 'match', '-config', str(config),
        *(['-results-dir', str(results)] if results else []),
        *(['-sgf-output-dir', str(sgfs)] if sgfs else [])]
    return subprocess.run(worker_command(command) if results else command,
                          capture_output=True, text=True, timeout=90, **kwargs)


@pytest.mark.parametrize('position', ['draw', 'multi_turn', 'long_check'])
def test_native_random_match_full_state_results_pairings_and_resume(tmp_path, position):
    config = match_config(tmp_path / 'match.cfg', ['/dev/null'] * 3, position=position, random=True)
    output = tmp_path / 'results'
    result = native_match(config, output)
    assert result.returncode == 0, result.stdout + result.stderr
    plan, games = load_games(output)
    assert plan['game'] == 'xiangqi' and len(games) == 6
    assert Counter(tuple(pair) for pair in plan['matchups']) == Counter(
        (black, white) for black in range(3) for white in range(3) if black != white)
    if position == 'multi_turn':
        assert {game['outcome'] for game in games} <= {'B', 'D'}
    else:
        assert {game['outcome'] for game in games} == ({'W'} if position == 'long_check' else {'D'})
    for game in games:
        assert 'GM[7]' in game['sgf'] and 'SZ[9:10]' in game['sgf'] and 'RE[' in game['sgf']
        if position == 'long_check':
            assert 'XQHIST[e8f8][f9e9][f8e8][e9f9][e8f8][f9e9][f8e8]' in game['sgf']
            assert 'W[e9f9]' in game['sgf']
        elif position == 'multi_turn':
            assert game['moves'] >= 75  # Initial turn 73, with both sides playing.
    preserved = {path: path.read_bytes() for path in output.glob('game-*.json')}
    # A missing game's publication is the native recovery boundary; the other
    # committed games and schedule remain immutable.
    removed = next(iter(preserved)); del preserved[removed]; removed.unlink()
    assert not (output / 'summary.json').exists()
    resumed = native_match(config, output)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert len(load_games(output)[1]) == 6
    assert all(path.read_bytes() == data for path, data in preserved.items())
    cached = native_match(config, output, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    assert cached.returncode == 0 and 'no inference needed' in cached.stdout
    plain = native_match(config, sgfs=tmp_path / 'sgfs')
    assert plain.returncode == 0, plain.stdout + plain.stderr
    assert sum(path.read_text().count('RE[') for path in (tmp_path / 'sgfs').glob('*.sgfs')) == 6


def test_native_match_does_not_rate_truncated_xiangqi(tmp_path):
    config = match_config(tmp_path / 'match.cfg', ['/dev/null'] * 2,
                          position=None, random=True, maxMovesPerGame=1)
    result = native_match(config, tmp_path / 'results')
    assert result.returncode != 0 and 'move limit without a rule result' in result.stdout + result.stderr
    assert not list((tmp_path / 'results').glob('game-*.json'))


def test_native_match_loads_indexed_external_patterns(tmp_path):
    pattern = tmp_path / 'opening.sgf'
    pattern.write_text('(;GM[7]SZ[9:10];B[b0c2];W[h9g7])')
    cfg = match_config(tmp_path / 'match.cfg', ['/dev/null'] * 2, random=True,
                       subtreeValueBiasFactor=.4, avoidRepeatedPatternUtility=.03,
                       avoidSgf2PatternUtility1=.1, avoidSgf2PatternDirs1=pattern)
    result = native_match(cfg)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'shapes to penalize repeats for bot 1' in result.stdout


def exported(path):
    path.mkdir(parents=True)
    with (path / 'model.bin').open('wb') as stream:
        write_model_binary(model(with_q=True).eval(), stream, path.name)
    return path / 'model.bin'


def gate_command(path, *, threshold=.5, muzero=False, model_game='xiangqi', **kwargs):
    baseline = path / 'accepted/baseline'
    if muzero:
        export_components(make_model(model_game), baseline, 'baseline')
        kwargs.update(useMuZero='true', useGraphSearch='false')
    else:
        exported(baseline)
    shutil.copytree(baseline, path / 'test/candidate')
    config = game_config(path / 'gate.cfg', numGamesPerGating=6, **kwargs)
    receipt = path / 'result.json'
    command = [str(ROOT / 'build/katago'), 'gatekeeper', '-config', str(config),
        '-test-models-dir', str(path / 'test'), '-accepted-models-dir', str(path / 'accepted'),
        '-rejected-models-dir', str(path / 'rejected'), '-sgf-output-dir', str(path / 'matches'),
        '-required-candidate-win-prop', str(threshold), '-no-autoreject-old-models',
        '-result-file', str(receipt)]
    return command, receipt


@pytest.mark.parametrize('random,muzero', [(True, False), pytest.param(False, False, marks=gpu),
                                         pytest.param(False, True, marks=gpu)])
@pytest.mark.parametrize('threshold,gumbel', [(.5, False), (.75, True)])
def test_xiangqi_gatekeeper_accepts_or_rejects_real_draws(tmp_path, random, muzero, threshold, gumbel):
    command, receipt = gate_command(tmp_path, threshold=threshold, random=random, gumbel=gumbel, muzero=muzero)
    original = {file.name: file.read_bytes() for file in (tmp_path / 'test/candidate').iterdir()}
    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(receipt.read_text())
    assert report['planned_games'] == 6 and 1 <= report['games'] <= 6
    assert report['outcomes'] == dict(win=0, loss=0, draw=report['games'], no_result=0)
    assert report['candidate_points'] == report['baseline_points'] == report['games'] * .5
    assert report['accepted'] == (threshold == .5)
    destination = tmp_path / ('accepted' if report['accepted'] else 'rejected') / 'candidate'
    assert {file.name: file.read_bytes() for file in destination.iterdir()} == original
    assert not (tmp_path / 'test/candidate').exists()
    records = ''.join(file.read_text() for file in (tmp_path / 'matches/candidate').glob('*.sgfs'))
    assert records.count('RE[0]') == report['games'] and 'GM[7]' in records


def test_xiangqi_gatekeeper_refuses_truncated_result(tmp_path):
    command, receipt = gate_command(tmp_path, random=True, position=None, maxMovesPerGame=1)
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0 and 'move limit without a rule result' in result.stdout + result.stderr
    assert not receipt.exists() and (tmp_path / 'test/candidate/model.bin').is_file()


@pytest.mark.parametrize('command_name', ['match', 'gatekeeper'])
def test_xiangqi_muzero_rejects_other_game_bundles_before_inference(tmp_path, command_name):
    if command_name == 'match':
        bundle = tmp_path / 'placement'
        export_components(make_model('renju'), bundle, 'placement')
        cfg = match_config(tmp_path / 'match.cfg', [bundle] * 2, useMuZero='true')
        command = [str(ROOT / 'build/katago'), 'match', '-config', str(cfg)]
    else:
        command, _ = gate_command(tmp_path, muzero=True, model_game='renju')
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and f'MuZero model game differs from the configured {command_name} game' in result.stdout + result.stderr


@gpu
@pytest.mark.parametrize('muzero,position', [(False, 'multi_turn'), (True, 'multi_turn'), (True, 'long_check')])
def test_real_xiangqi_models_match_and_source_elo(tmp_path, muzero, position):
    model1 = exported(tmp_path / 'first')
    model2 = tmp_path / 'second'
    if muzero:
        export_components(make_model('xiangqi', pos_len=12), model2, 'second')
    else:
        model2 = exported(model2)
    cfg = match_config(tmp_path / 'match.cfg', [model1, model2, model2], position=position)
    with cfg.open('a') as stream:
        stream.write('nnRandSeed0 = ordinary-seed\nnnRandSeed1 = shared-model-seed\n')
        if muzero:
            stream.write('useMuZero1 = true\nuseMuZero2 = true\nuseGraphSearch2 = false\n')
    output = tmp_path / 'evaluation'
    result = run_match(output, cfg, anchor='bot0')
    assert result['games'] == 6 and result['outcomes']['NR'] == 0
    if position == 'long_check':
        assert result['outcomes']['W'] == 6
        assert all('XQHIST[e8f8][f9e9][f8e8][e9f9][e8f8][f9e9][f8e8]' in game['sgf']
                   for game in load_games(output / 'results')[1])
    else:
        assert result['outcomes']['W'] == 0
    assert all(game['moves'] >= 75 for game in load_games(output / 'results')[1])
    assert result['players'][0]['elo'] == 0
    assert (output / 'elo.png').is_file()
    native_log = next((output / 'invocations').glob('*.native.log')).read_text()
    if not muzero:
        assert native_log.count('Model version 19 useFP16 = true') == 2
    assert 'nnRandSeed0 = ordinary-seed' in native_log and 'nnRandSeed1 = shared-model-seed' in native_log
    assert 'nnRandSeed2 =' not in native_log  # Both searches share one model service.
    assert run_match(output, fit_only=True, binary='/nonexistent', anchor='bot0')['outcomes'] == result['outcomes']


@gpu
@pytest.mark.parametrize('algorithm,threshold', [('alphazero', .5), ('gumbel_alphazero', .75),
                                                ('muzero', .5), ('gumbel_muzero', .75)])
def test_xiangqi_outer_gate_decision_recovery_and_next_round(tmp_path, monkeypatch, algorithm, threshold):
    muzero = 'muzero' in algorithm
    gate = game_config(tmp_path / 'gate.cfg', numGamesPerGating=6, gumbel='gumbel' in algorithm,
                       **({'useGraphSearch': 'false'} if muzero else {}))
    network = dict(model_kind=None, muzero_network=asdict(make_model('xiangqi').network_config),
                   muzero_unroll_steps=3) if muzero else {}
    config = controller_config(tmp_path, algorithm=algorithm, gatekeeper_config=str(gate),
                               gatekeeper_required_win_prop=threshold, **network).validate()
    selfplay_config(tmp_path / 'selfplay.cfg', useGraphSearch=str(not (config.is_gumbel or muzero)).lower())
    calls = []
    actual = controller.execute
    interrupt = [True]
    def execute(command, log, **kwargs):
        name = Path(log).name.split('.')[0]
        calls.append(name)
        actual(command, log, **kwargs)
        if name == 'gatekeeper' and interrupt[0]:
            interrupt[0] = False
            raise RuntimeError('interrupted after native gate decision')
    monkeypatch.setattr(controller, 'execute', execute)
    directory = tmp_path / 'run'
    first = controller.run(directory, config, 1)
    assert 'gatekeeper' not in calls
    with pytest.raises(RuntimeError, match='interrupted after native gate decision'):
        controller.run(directory, config, 2)
    trained = checkpoint_progress(directory / 'train/checkpoint.ckpt')
    state = json.loads((directory / 'state.json').read_text())
    assert state['completed_rounds'] == 1 and state['accepted_model'] == first['accepted_model']
    gate_dir = next((directory / 'rounds/000002/gatekeeper').iterdir())
    candidate_name = gate_dir.name
    decision = json.loads((gate_dir / 'result.json').read_text())
    assert decision['accepted'] == (threshold == .5) and decision['outcomes']['draw'] == decision['games']
    moved = gate_dir / ('accepted' if decision['accepted'] else 'rejected') / candidate_name
    moved.rename(gate_dir / 'test' / candidate_name)
    previous_calls = list(calls)
    second = controller.run(directory, config, 2)
    assert calls == previous_calls and moved.exists()
    assert checkpoint_progress(directory / 'train/checkpoint.ckpt') == trained
    expected = f'models/{candidate_name}' if threshold == .5 else first['accepted_model']
    assert second['accepted_model'] == expected
    third = controller.run(directory, config, 3)
    report = json.loads((directory / 'rounds/000003/result.json').read_text())
    assert report['input_model'] == Path(expected).name and report['gatekeeper']['baseline'] == Path(expected).name
    assert third['total_samples'] > second['total_samples'] and checkpoint_progress(directory / 'train/checkpoint.ckpt')[1] > trained[1]
