"""Native SGF diagnostics: game history, action domains and real model inputs."""
import json
import os
import re
import subprocess

import numpy as np
import pytest
import torch

from etazero import ROOT
from export_model_pytorch import write_model_binary
from export_muzero import export_components
from katago.game import xiangqi as xq
from katago.train.model_pytorch import Model
from katago.train.muzero_model import MuZeroConfig, MuZeroModel
from test_gtp import gtp_config
from test_muzero_model import component_config
from test_xiangqi_gtp import CHECK_FEN, CHECKS


GAMES = ['go', 'freestyle', 'standard', 'renju', 'hex', 'xiangqi']
cuda = pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires real CUDA inference')


def record(game):
    if game == 'xiangqi':
        return '(;GM[7]SZ[9:10]RU[xiangqi];B[a3a4];W[a6a5])'
    gm = 1 if game == 'go' else 11 if game == 'hex' else 4
    rules = 'Tromp-Taylor' if game == 'go' else game
    return f'(;GM[{gm}]SZ[5]RU[{rules}]KM[0];B[aa];W[ee])'


def config_for(path, game, *, muzero=False, gumbel=False, random=True, **options):
    values = dict(numSearchThreads=1, maxVisits=24, nnMaxBatchSize=4,
        logToStdout='true', logToStderr='false', requireMaxBoardSize='false',
        maxBoardSizeForNNBuffer=10 if game == 'xiangqi' else 5,
        rootPolicyOptimism=.4, rootPolicyTemperature=1, rootPolicyTemperatureEarly=1,
        reportAnalysisWinratesAs='WHITE', cudaUseFP16='false')
    values.update(options)
    return gtp_config(path, game, muzero, gumbel, random, **values)


def run(config, sgf, *args, model='/dev/null', human=None, turn=2, binary=None):
    result = subprocess.run([str(binary or ROOT / 'build/katago'), 'evalsgf', '-config', str(config),
        '-model', str(model), str(sgf), '-m', str(turn),
        *(['-human-model', str(human)] if human else []), *map(str, args)],
        capture_output=True, text=True, timeout=120)
    index = len(list(config.parent.glob('evalsgf-*.log')))
    (config.parent / f'evalsgf-{index}.log').write_text(result.stdout + result.stderr)
    return result


def reports(result):
    assert result.returncode == 0, result.stdout + result.stderr
    return [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]


@pytest.mark.parametrize('game', GAMES)
@pytest.mark.parametrize('gumbel', [False, True])
def test_move_options_reports_and_game_inference(tmp_path, game, gumbel):
    config = config_for(tmp_path / 'eval.cfg', game, gumbel=gumbel)
    # As for benchmark, the SGF can supply the game when config omits it.
    config.write_text(re.sub(r'^game = .*\n', '', config.read_text(), flags=re.M))
    sgf = tmp_path / 'position.sgf'; sgf.write_text(record(game))
    extra, avoid, branch = ('b0c2', 'b9c7', 'h9g7') if game == 'xiangqi' else ('B3', 'D4', 'C3')
    result = run(config, sgf, '-extra-moves', extra, '-avoid-moves', avoid,
                 '-hint-loc', branch, '-print-branch', branch, '-print-json', '-print-policy',
                 '-print-log-policy', '-print-dirichlet-shape', '-print-root-nn-values', '-print-graph')
    report, = reports(result)
    assert report['rootInfo']['currentPlayer'] == 'W'
    assert report['rootInfo']['visits'] == 24
    assert avoid not in {m['move'] for m in report['moveInfos']}
    assert 'White draw:' in result.stdout and 'White noresult:' in result.stdout
    # Source JSON keeps the point-domain pass slot; non-Go masks it to -1.
    assert len(report['policy']) == (2086 if game == 'xiangqi' else 26)
    assert sum(p for p in report['policy'] if p >= 0) == pytest.approx(1, abs=2e-5)
    if game != 'go':
        assert report['rootInfo']['noResultValue'] == 0
        assert 'Pass ' not in result.stdout
        if game != 'xiangqi':
            assert report['policy'][-1] == -1
    if game == 'xiangqi':
        assert 'Xiangqi FEN:' in result.stdout and re.search(r'^h9g7\s', result.stdout, re.M)
    # The diagnostic uses the actual policy domain, including Go pass, and
    # leaves illegal entries masked instead of printing uninitialized alphas.
    alphas = result.stdout.split('Dirichlet alphas with 10.83 total concentration: ')[1].split('Time taken:')[0]
    if game == 'xiangqi':
        numbers = re.findall(r'^[a-i][0-9][a-i][0-9]\s+([\d.]+)$', alphas, re.M)
    else:
        numbers = re.findall(r'\b\d+\.\d+\b', alphas)
    assert sum(map(float, numbers)) == pytest.approx(10.83, abs=.01)


@pytest.mark.parametrize('game,args,change,error', [
    ('renju', ['-override-komi', '1'], {}, 'zero komi'),
    ('hex', ['-print-score-now'], {}, 'Go rules scoring'),
    ('xiangqi', ['-extra-moves', 'a3a4'], {}, 'Illegal extra move'),
    ('renju', ['-extra-moves', 'A5'], {}, 'Illegal extra move'),
    ('xiangqi', [], {'game': 'go'}, 'config game and SGF rules differ'),
    ('go', ['-move-num-end', '3'], {}, 'sgf has only 2'),
])
def test_invalid_tool_inputs(tmp_path, game, args, change, error):
    config = config_for(tmp_path / 'eval.cfg', game)
    text = config.read_text()
    for key, value in change.items():
        text = re.sub(rf'^{key} = .*$', f'{key} = {value}', text, flags=re.M)
    config.write_text(text)
    sgf = tmp_path / 'position.sgf'; sgf.write_text(record(game))
    result = run(config, sgf, *args)
    assert result.returncode != 0 and error in result.stdout + result.stderr


def make_model(path, game, muzero, *, metadata=False, seed=748):
    torch.manual_seed(seed)
    canvas = 10 if game == 'xiangqi' else 5
    config = component_config(game)
    if metadata:
        config['metadata_encoder'] = dict(meta_encoder_version=1, internal_num_channels=16)
    if muzero:
        net = MuZeroModel(MuZeroConfig(game, 6, config, component_config(game), component_config(game)), canvas)
        net.initialize(); net.eval()
        export_components(net, path, 'evalsgf-check')
    else:
        net = Model(config, canvas)
        net.initialize(); net.eval()
        path = path.with_suffix('.bin')
        with path.open('wb') as stream:
            write_model_binary(net, stream, 'evalsgf-check')
    return net, path


@cuda
def test_muzero_conflicts_fail_before_async_search(tmp_path):
    _, model = make_model(tmp_path / 'model', 'go', True)
    config = config_for(tmp_path / 'invalid.cfg', 'go', muzero=True, random=False)
    sgf = tmp_path / 'game.sgf'; sgf.write_text(record('go'))
    for setting in ['useGraphSearch=true', 'subtreeValueBiasFactor=0.2', 'avoidRepeatedPatternUtility=0.1']:
        result = subprocess.run([str(ROOT / 'build/katago'), 'evalsgf', '-config', str(config), '-model', str(model),
            str(sgf), '-m', '0', '-visits', '1', '-override-config', setting],
            capture_output=True, text=True, timeout=20)
        text = result.stdout + result.stderr
        assert result.returncode != 0 and 'MuZero latent search excludes' in text
        assert 'Async bot thread failed' not in text


@cuda
@pytest.mark.parametrize('game', GAMES)
@pytest.mark.parametrize('muzero,gumbel', [(False, False), (False, True), (True, False), (True, True)])
def test_exported_observations_match_real_inference(tmp_path, game, muzero, gumbel):
    check_exported_observations(tmp_path, game, muzero, gumbel)


def check_exported_observations(tmp_path, game, muzero, gumbel, *, binary=None):
    torch.set_num_threads(1)
    net, model = make_model(tmp_path / 'model', game, muzero)
    config = config_for(tmp_path / 'eval.cfg', game, muzero=muzero, gumbel=gumbel, random=False,
                        cudaUseFP16=str(gumbel).lower())
    sgf = tmp_path / 'position.sgf'; sgf.write_text(record(game))
    inputs_path = tmp_path / 'inputs.npz'
    extra = 'b0c2' if game == 'xiangqi' else 'B3'
    result = run(config, sgf, '-extra-moves', extra, '-print-json', '-print-policy',
                 '-print-root-nn-values', '-dump-npz-input-to', inputs_path, model=model, binary=binary)
    report, = reports(result)
    inputs = np.load(inputs_path)
    spatial, global_input = torch.from_numpy(inputs['binaryInputNCHW']), torch.from_numpy(inputs['globalInputNC'])
    assert spatial.shape == (1, 113 if game == 'xiangqi' else 22 if game == 'go' else 5, net.pos_len, net.pos_len)
    assert global_input.shape == (1, 19 if game == 'go' else 8)
    if game == 'xiangqi':
        assert spatial[0, 0, :10, :9].sum() == 90 and not spatial[..., 9:].any()
    else:
        assert spatial[0, 0].sum() == 25
        # After extra B3 it is White's turn: White E1 is own, Black A5/B3 are opponent.
        assert spatial[0, 1, 4, 4] == 1 and spatial[0, 2, 0, 0] == spatial[0, 2, 2, 1] == 1
    head = net.prediction if muzero else net
    with torch.no_grad():
        raw = net.initial(spatial, global_input)[1] if muzero else net(spatial, global_input)
        output = head.postprocess_output(raw)[0]
    logits = output[0][0, 0] + .4 * (output[0][0, 5] - output[0][0, 0])
    if game == 'xiangqi':
        logits = logits[torch.tensor(xq.action_permutation(black=True))]
    policy = np.asarray(report['policy'])
    if game not in {'go', 'xiangqi'}:
        assert policy[-1] == -1
        policy = policy[:-1]
    legal = torch.tensor(policy >= 0)
    expected_policy = logits.masked_fill(~legal, -torch.inf).softmax(-1).numpy()
    np.testing.assert_allclose(policy[legal], expected_policy[legal], atol=2e-4 if gumbel else 2e-5, rtol=3e-3)
    values = output[1][0].clone()
    values[3] = -torch.inf  # Tromp-Taylor and non-Go prohibit no-result.
    expected_value = values.softmax(-1).numpy()
    actual_value = [float(re.search(rf'^White {label}: (\S+)', result.stdout, re.M)[1])
                    for label in ('win', 'draw', 'loss', 'noresult')]
    np.testing.assert_allclose(actual_value, expected_value, atol=7e-4 if gumbel else 2e-5, rtol=3e-3)
    assert int(re.search(r'NN rows: (\d+)', result.stdout)[1]) > 0
    # Exercise the separate -raw-nn path; source debug output uses percent units.
    raw_result = run(config, sgf, '-extra-moves', extra, '-raw-nn', model=model, binary=binary)
    assert raw_result.returncode == 0, raw_result.stdout + raw_result.stderr
    raw_values = [float(re.search(rf'^{label} ([\d.]+)c$', raw_result.stdout, re.M)[1]) / 100
                  for label in ('Win', 'Draw', 'Loss', 'NoResult')]
    np.testing.assert_allclose(raw_values, expected_value, atol=8e-4 if gumbel else 6e-5, rtol=3e-3)


@pytest.mark.parametrize('gumbel', [False, True])
def test_xiangqi_full_history_and_terminal_result(tmp_path, gumbel):
    config = config_for(tmp_path / 'eval.cfg', 'xiangqi', gumbel=gumbel)
    sgf = tmp_path / 'checking.sgf'
    sgf.write_text(f'(;GM[7]SZ[9:10]XQFEN[{CHECK_FEN}]XQHIST[e8f8][f9e9]XQTURN[97]'
                   + ''.join(f';{"B" if i % 2 == 0 else "W"}[{m}]' for i, m in enumerate(CHECKS[2:])) + ')')
    result = run(config, sgf, '-move-num-end', '6', '-print-json', '-print-policy', turn=5)
    before, terminal = reports(result)
    assert before['rootInfo']['currentPlayer'] == 'W'
    assert {m['move'] for m in before['moveInfos']} == {'e9f9'}
    assert terminal['rootInfo']['winrate'] == 1 and terminal['rootInfo']['drawValue'] == 0
    assert terminal['moveInfos'] == [] and terminal['policy'] == [-1] * 2086
    invalid = run(config, sgf, '-raw-nn', turn=6)
    assert invalid.returncode != 0 and 'terminal non-Go' in invalid.stderr


def test_rectangular_go_input_axes_and_rules_score(tmp_path):
    config = config_for(tmp_path / 'eval.cfg', 'go', maxBoardXSizeForNNBuffer=3,
                        maxBoardYSizeForNNBuffer=5, requireMaxBoardSize='true')
    sgf = tmp_path / 'rectangle.sgf'
    sgf.write_text('(;GM[1]SZ[3:5]RU[Tromp-Taylor]KM[0];B[aa];W[ce])')
    inputs_path = tmp_path / 'rectangle.npz'
    result = run(config, sgf, '-dump-npz-input-to', inputs_path, '-print-json', '-print-score-now')
    report, = reports(result)
    assert report['rootInfo']['currentPlayer'] == 'B'
    spatial = np.load(inputs_path)['binaryInputNCHW']
    assert spatial.shape == (1, 22, 5, 3)
    assert spatial[0, 0].sum() == 15
    assert spatial[0, 1, 0, 0] == spatial[0, 2, 4, 2] == 1
    assert 'Score now (ROOT position):' in result.stdout and 'Final:' in result.stdout


@cuda
@pytest.mark.parametrize('game', ['go', 'xiangqi'])
@pytest.mark.parametrize('muzero,human_mu,gumbel', [
    (False, False, False), (False, True, True), (True, False, True), (True, True, False),
])
def test_primary_secondary_model_combinations(tmp_path, game, muzero, human_mu, gumbel):
    _, model = make_model(tmp_path / 'primary', game, muzero)
    _, human = make_model(tmp_path / 'secondary', game, human_mu, seed=749)
    exploration = .5 if not muzero or human_mu else 0
    config = config_for(tmp_path / 'eval.cfg', game, muzero=muzero, gumbel=gumbel, random=False,
                        humanSLRootExploreProbWeightless=exploration,
                        humanSLPlaExploreProbWeightless=exploration, humanSLOppExploreProbWeightless=exploration,
                        humanSLChosenMoveProp=1, numSearchThreads=2)
    sgf = tmp_path / 'position.sgf'; sgf.write_text(record(game))
    result = run(config, sgf, '-print-json', '-print-policy', '-print-log-policy', model=model, human=human)
    report, = reports(result)
    assert 'Root human policy:' in result.stdout
    assert len(report['humanPolicy']) == len(report['policy'])
    assert sum(p for p in report['humanPolicy'] if p >= 0) == pytest.approx(1, abs=2e-5)
    assert np.isfinite(report['rootInfo']['humanWinrate'])
    assert report['humanPolicy'] != report['policy']
    raw_result = run(config, sgf, '-raw-nn', model=model, human=human)
    assert raw_result.returncode == 0, raw_result.stdout + raw_result.stderr
    assert len(re.findall(r'^Win ', raw_result.stdout, re.M)) == 2
    # A latent diagnostic branch can leave the real board's legal continuation.
    branch = 'b0c2 b0c2' if game == 'xiangqi' else 'C3 C3'
    branched = run(config, sgf, '-print-branch', branch, '-print-json', model=model, human=human)
    if muzero:
        assert len(reports(branched)) == 1 and 'subsequent nodes are latent' in branched.stdout
    else:
        assert branched.returncode != 0 and 'Branch Illegal move' in branched.stderr


@cuda
@pytest.mark.parametrize('muzero', [False, True])
def test_go_metadata_input_and_unsuppressed_four_results(tmp_path, muzero):
    net, model = make_model(tmp_path / 'model', 'go', muzero, metadata=True)
    config = config_for(tmp_path / 'eval.cfg', 'go', muzero=muzero, random=False,
                        humanSLProfile='rank_5d', rootPolicyOptimism=0)
    sgf = tmp_path / 'japanese.sgf'; sgf.write_text(record('go').replace('Tromp-Taylor', 'Japanese'))
    inputs_path = tmp_path / 'metadata.npz'
    result = run(config, sgf, '-print-json', '-print-root-nn-values', '-dump-npz-input-to', inputs_path, model=model)
    report, = reports(result)
    inputs = np.load(inputs_path)
    spatial, globals_, metadata = (torch.from_numpy(inputs[k]) for k in ('binaryInputNCHW', 'globalInputNC', 'metadataInputNC'))
    assert metadata.shape == (1, 192) and torch.isfinite(metadata).all() and metadata.any()
    head = net.prediction if muzero else net
    with torch.no_grad():
        raw = net.initial(spatial, globals_, metadata)[1] if muzero else net(spatial, globals_, metadata)
        expected = head.postprocess_output(raw)[0][1][0].softmax(-1).numpy()[[2, 1, 0, 3]]
    actual = [float(re.search(rf'^White {label}: (\S+)', result.stdout, re.M)[1])
              for label in ('win', 'draw', 'loss', 'noresult')]
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=3e-3)
    assert actual[1] > 0 and actual[3] > 0 and report['rootInfo']['noResultValue'] > 0
    assert 'model(s) don\'t use it' not in result.stdout + result.stderr
