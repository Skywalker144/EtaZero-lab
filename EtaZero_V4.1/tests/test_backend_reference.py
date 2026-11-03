"""Native reference collection and explicit family-specific checks, not calibration."""
import json
import math
import os
import re
import subprocess

import pytest

from etazero import ROOT
from etazero.gtp import GtpProcess
from test_benchmarknn import cuda, model_for
from test_booktoposes import model_file
from test_gpu_error import config_for as gpu_config
from test_startpose_search import GAMES, position, terminal_sample
from test_web_native import profile
from katago.game.xiangqi import MOVES


def limits(path, **changes):
    # Deliberately synthetic thresholds for unit fixtures. They are not a V4
    # training-family calibration or a suggested production test profile.
    values = dict(mixtureSmoothing=1e-5, avgPolicyKL=.001, p95PolicyKL=.001,
                  consensusTopProb=.8, consensusMinProb=.1, maxConsensusViolations=0,
                  avgWinrateKL=.001, p95WinrateKL=.001,
                  avgStWinlossErrorAE=.001, p95StWinlossErrorAE=.001,
                  avgResultKL=.001, p95ResultKL=.001)
    values.update(changes)
    path.write_text(json.dumps(values))
    return path


def reference(path, game, *, samples=None):
    samples = samples or [position(game, moveLocs=position(game)['moveLocs'][:n],
                                  movePlas=position(game)['movePlas'][:n]) for n in [0, 1, 2, 0]]
    rows = [dict(sample=sample, rules='Tromp-Taylor' if game == 'go' else game,
                 komi=0, policyOptimism=i / max(1, len(samples) - 1), pda=0, maxHistory=i)
            for i, sample in enumerate(samples)]
    write_rows(path, rows)
    return rows


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def configure(path, game, muzero=False, fp16=False, **changes):
    values = dict(maxBoardSizeForNNBuffer=10 if game == 'xiangqi' else 7)
    values.update(changes)
    return gpu_config(path, game, muzero, fp16, **values)


def run(config, model='/dev/null', *args, binary=None):
    result = subprocess.run([str(binary or ROOT / 'build/katago'), 'testbackendreference',
        '-config', str(config), '-model', str(model), *map(str, args)],
        capture_output=True, text=True, timeout=120)
    n = len(list(config.parent.glob('reference-check-*.log')))
    (config.parent / f'reference-check-{n}.log').write_text(result.stdout + result.stderr)
    return result


def dump(config, model, ref, output, *, binary=None):
    result = run(config, model, '-reference-data-file', ref, '-dump-candidate', output, '-dump-only', binary=binary)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'reference checks were not run' in result.stdout
    assert 'PASSED' not in result.stdout
    return [json.loads(line) for line in output.read_text().splitlines()]


def use_outputs(rows, outputs):
    for row, output in zip(rows, outputs):
        row.update(policyMixture=output['policy'], winrateAvg=output['winrate'],
                   shorttermWinlossErrorAvg=output['shorttermWinlossError'],
                   resultAvg=[output[k] for k in ['whiteWinProb', 'whiteDrawProb', 'whiteLossProb', 'whiteNoResultProb']])


@cuda
@pytest.mark.parametrize('game', GAMES)
@pytest.mark.parametrize('muzero', [False, True])
@pytest.mark.parametrize('fp16', [False, True])
def test_all_games_export_and_explicit_reference_checks(tmp_path, game, muzero, fp16):
    _, model = model_file(tmp_path / 'model', game, muzero)
    cfg = configure(tmp_path / 'nn.cfg', game, muzero, fp16)
    ref = tmp_path / 'positions.jsonl'; rows = reference(ref, game)
    outputs = dump(cfg, model, ref, tmp_path / 'outputs.jsonl')
    assert len(outputs) == len(rows) == 4
    for i, (row, out) in enumerate(zip(rows, outputs)):
        assert out['posIdx'] == i and out['policyOptimism'] == row['policyOptimism']
        assert out['maxHistory'] == row['maxHistory']
        assert out['symmetryUsed'] in ({0, 2} if game == 'xiangqi' else {0, 3} if game == 'hex' else set(range(8)))
        assert out['sample']['initialTurnNumber'] == 73
        assert len(out['policy']) == (2086 if game == 'xiangqi' else 26)
        assert sum(out['policy']) == pytest.approx(1, abs=1e-5)
        expected = [.6, .2, .2, 0] if len(row['sample']['moveLocs']) == 1 else [.2, .2, .6, 0]
        actual = [out[k] for k in ['whiteWinProb', 'whiteDrawProb', 'whiteLossProb', 'whiteNoResultProb']]
        assert actual == pytest.approx(expected, abs=2e-4)
        positive = [p for p in out['policy'] if p > 0]
        assert max(positive) == pytest.approx(min(positive), abs=1e-6)
        if game not in ('go', 'xiangqi'):
            assert out['policy'][-1] == 0
        if game != 'xiangqi':
            count = len(row['sample']['moveLocs'])
            assert len(positive) == 25 - count + (game == 'go')
            if count >= 1: assert out['policy'][20] == 0  # A1 in physical row-major coordinates.
            if count >= 2: assert out['policy'][4] == 0   # E5, independent of the NN canvas.
    use_outputs(rows, outputs); write_rows(ref, rows)
    result = run(cfg, model, '-reference-data-file', ref, '-reference-limits-file', limits(tmp_path / 'limits.json'))
    assert result.returncode == 0, result.stdout + result.stderr
    assert '4 four-result' in result.stdout and 'size lenience 1.000' in result.stdout
    assert 'PASSED' in result.stdout and 'ignored' not in result.stdout


@cuda
@pytest.mark.parametrize('game', ['go', 'hex', 'xiangqi'])
@pytest.mark.parametrize('muzero', [False, True])
def test_four_outcomes_detect_shift_with_same_expected_score_and_no_small_model_exemption(tmp_path, game, muzero):
    _, model = model_file(tmp_path / 'model', game, muzero)
    cfg = configure(tmp_path / 'nn.cfg', game, muzero)
    ref = tmp_path / 'reference.jsonl'; rows = reference(ref, game)
    outputs = dump(cfg, model, ref, tmp_path / 'outputs.jsonl')
    use_outputs(rows, outputs)
    for row in rows:
        row['resultAvg'][0] -= .1; row['resultAvg'][1] += .2; row['resultAvg'][2] -= .1
    write_rows(ref, rows)
    result = run(cfg, model, '-reference-data-file', ref, '-reference-limits-file', limits(tmp_path / 'limits.json'))
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'avg resultKL' in result.stdout and 'ignored' not in result.stdout
    assert 'failed check' in result.stdout and 'avg winrateKL' not in result.stdout
    expected = sum(r * math.log(r / c) for r, c in zip([.1, .4, .5], [.2, .2, .6]))
    match = re.search(r'backend reference resultKL \(nats\):\s+(\S+)', result.stdout)
    assert float(match[1]) == pytest.approx(expected, abs=2e-5)
    # Explicit lenience affects these declared limits, without implicit size scaling.
    result = run(cfg, model, '-reference-data-file', ref, '-reference-limits-file', tmp_path / 'limits.json',
                 '-lenience-factor', 1000)
    assert result.returncode == 0 and 'size lenience 1.000' in result.stdout


@cuda
def test_xiangqi_joint_policy_tail_and_consensus_are_checked(tmp_path):
    _, model = model_file(tmp_path / 'model', 'xiangqi', False)
    cfg = configure(tmp_path / 'nn.cfg', 'xiangqi')
    ref = tmp_path / 'reference.jsonl'; rows = reference(ref, 'xiangqi')
    outputs = dump(cfg, model, ref, tmp_path / 'outputs.jsonl'); use_outputs(rows, outputs)
    for row in rows:
        idx = next(i for i, p in enumerate(row['policyMixture']) if i > 361 and p > 0)
        row['policyMixture'] = [float(i == idx) for i in range(2086)]
    write_rows(ref, rows)
    lim = limits(tmp_path / 'limits.json', avgPolicyKL=100, p95PolicyKL=100, consensusMinProb=.9)
    result = run(cfg, model, '-reference-data-file', ref, '-reference-limits-file', lim)
    assert result.returncode == 1 and 'consensus move violations 4 > allowed 0' in result.stdout


@cuda
@pytest.mark.parametrize('muzero', [False, True])
def test_metadata_transformer_and_go_no_result(tmp_path, muzero):
    model = model_for(tmp_path / 'model', 'go', muzero, kind='transformer', metadata=True)
    cfg = configure(tmp_path / 'nn.cfg', 'go', muzero, maxBoardSizeForNNBuffer=5)
    ref = tmp_path / 'positions.jsonl'; rows = reference(ref, 'go')
    for row in rows:
        row['rules'] = 'Japanese'
    write_rows(ref, rows)
    outputs = dump(cfg, model, ref, tmp_path / 'outputs.jsonl')
    assert all(o['whiteNoResultProb'] > 0 and o['whiteDrawProb'] > 0 for o in outputs)
    use_outputs(rows, outputs); write_rows(ref, rows)
    result = run(cfg, model, '-reference-data-file', ref, '-reference-limits-file', limits(tmp_path / 'limits.json'))
    assert result.returncode == 0, result.stdout + result.stderr


@cuda
@pytest.mark.parametrize('muzero', [False, True])
def test_history_terminal_in_worker_fails_without_fake_policy(tmp_path, muzero):
    _, model = model_file(tmp_path / 'model', 'xiangqi', muzero)
    cfg = configure(tmp_path / 'nn.cfg', 'xiangqi', muzero)
    ref = tmp_path / 'positions.jsonl'
    terminal, _ = terminal_sample('xiangqi')
    reference(ref, 'xiangqi', samples=[position('xiangqi')] * 3 + [terminal])
    result = run(cfg, model, '-reference-data-file', ref, '-dump-only', '-dump-candidate', tmp_path / 'outputs.jsonl')
    assert result.returncode != 0 and 'ongoing position' in result.stderr
    assert 'PASSED' not in result.stdout


@cuda
@pytest.mark.parametrize('muzero', [False, True])
def test_xiangqi_complete_history_matches_native_raw_nn(tmp_path, muzero):
    _, model = model_file(tmp_path / 'model', 'xiangqi', muzero, uniform=False)
    cfg = configure(tmp_path / 'nn.cfg', 'xiangqi', muzero)
    prior = ['b0c2', 'b9c7', 'c2b0', 'c7b9']
    sample = position('xiangqi', xiangqiHistory=prior, initialTurnNumber=97)
    ref = tmp_path / 'positions.jsonl'; rows = reference(ref, 'xiangqi', samples=[sample])
    rows[0].update(maxHistory=1000, policyOptimism=.35); write_rows(ref, rows)
    output, = dump(cfg, model, ref, tmp_path / 'outputs.jsonl')
    sgf = tmp_path / 'position.sgf'
    sgf.write_text('(;GM[7]SZ[9:10]RU[xiangqi]XQFEN[' + sample['xiangqiFen'] +
                   ']XQHIST' + ''.join('[' + move + ']' for move in prior) + 'XQTURN[97];B[a3a4];W[a6a5])')
    gtp_cfg = profile(tmp_path / 'gtp.cfg', 'xiangqi', muzero=muzero, random=False,
                      maxBoardSizeForNNBuffer=10, cudaUseFP16='false',
                      cudaUseNHWC='true', inputsUseNHWC='true')
    with GtpProcess(gtp_cfg, model) as engine:
        engine.command('loadsgf ' + str(sgf))
        text = engine.command(f'kata-raw-nn {output["symmetryUsed"]} 0.35').body
        header, policy = text.split('\npolicy\n')
        values = dict(line.split() for line in header.splitlines())
        for raw, key in [('whiteWin', 'whiteWinProb'), ('draw', 'whiteDrawProb'),
                         ('whiteLoss', 'whiteLossProb'), ('noResult', 'whiteNoResultProb')]:
            assert output[key] == pytest.approx(float(values[raw]), abs=1e-6)
        policies = dict(line.split() for line in policy.splitlines())
        labels = [f'{chr(97+x)}{y}{chr(97+u)}{v}' for x, y, u, v in MOVES]
        expected = [0 if policies[move] == 'NAN' else float(policies[move]) for move in labels]
        assert output['policy'] == pytest.approx(expected, abs=1e-6)
        assert {move for move, p in zip(labels, output['policy']) if p > 0} == set(engine.state()['legalMoves'])


@cuda
@pytest.mark.parametrize('muzero', [False, True])
def test_zero_liberty_placement_and_rectangular_go_layout(tmp_path, muzero):
    for game, sample, length, occupied in [
        ('hex', position('hex', board='XO.../O..../...../...../...../', moveLocs=[], movePlas=[]), 26, [0, 1, 5]),
        ('go', position('go', xSize=4, board='..../' * 5, moveLocs=['A1', 'D5']), 21, [16, 3]),
    ]:
        _, model = model_file(tmp_path / game, game, muzero)
        cfg = configure(tmp_path / f'{game}.cfg', game, muzero)
        ref = tmp_path / f'{game}.jsonl'; reference(ref, game, samples=[sample])
        output, = dump(cfg, model, ref, tmp_path / f'{game}-dump.jsonl')
        assert len(output['policy']) == length
        assert all(output['policy'][i] == 0 for i in occupied)
        assert sum(p > 0 for p in output['policy']) == length - len(occupied) - (game != 'go')


@cuda
@pytest.mark.skipif(os.environ.get('ETAZERO_EIGEN_TESTS') != '1', reason='requires Eigen backend')
@pytest.mark.parametrize('game', ['go', 'hex', 'xiangqi'])
@pytest.mark.parametrize('muzero', [False, True])
def test_eigen_collection_checks_against_cuda(tmp_path, game, muzero):
    _, model = model_file(tmp_path / 'model', game, muzero, uniform=False)
    cfg = configure(tmp_path / 'nn.cfg', game, muzero)
    ref = tmp_path / 'positions.jsonl'; rows = reference(ref, game)
    outputs = dump(cfg, model, ref, tmp_path / 'eigen.jsonl', binary=ROOT / 'build-eigen/katago')
    use_outputs(rows, outputs); write_rows(ref, rows)
    result = run(cfg, model, '-reference-data-file', ref, '-reference-limits-file', limits(tmp_path / 'limits.json'))
    assert result.returncode == 0, result.stdout + result.stderr


@cuda
def test_source_go_size_lenience_and_exemption_are_retained(tmp_path):
    _, model = model_file(tmp_path / 'model', 'go', False)
    cfg = configure(tmp_path / 'nn.cfg', 'go')
    ref = tmp_path / 'reference.jsonl'; rows = reference(ref, 'go')
    for row in rows:
        row['shorttermWinlossErrorAvg'] = 1000000
    write_rows(ref, rows)
    result = run(cfg, model, '-reference-data-file', ref)
    assert result.returncode == 0 and 'failed checks ignored' in result.stdout
    assert 'checks not applicable at this model size' in result.stdout


@cuda
def test_new_family_requires_reference_metrics_and_explicit_limits(tmp_path):
    _, model = model_file(tmp_path / 'model', 'hex', False)
    cfg = configure(tmp_path / 'nn.cfg', 'hex')
    ref = tmp_path / 'reference.jsonl'; rows = reference(ref, 'hex')
    result = run(cfg, model, '-reference-data-file', ref)
    assert result.returncode != 0 and 'explicit reference-limits-file' in result.stderr
    lim = limits(tmp_path / 'limits.json')
    result = run(cfg, model, '-reference-data-file', ref, '-reference-limits-file', lim)
    assert result.returncode != 0 and 'No reference metrics' in result.stderr
    rows[0]['sample']['game'] = 'go'; write_rows(ref, rows)
    result = run(cfg, model, '-reference-data-file', ref, '-dump-only', '-dump-candidate', tmp_path / 'dump')
    assert result.returncode != 0 and 'sample and rules game differ' in result.stderr


@pytest.mark.parametrize('args,options,message', [
    (['-dump-only'], {}, 'requires dump-candidate'),
    (['-reference-limits-file', '/tmp/unused-reference-limits'], {}, 'require reference-data-file'),
    ([], {'useMuZero': 'true'}, 'explicit reference-data-file'),
    (['-reference-data-file', '/tmp/unused-reference-data'], {'useMuZero': 'true'}, 'explicit reference-limits-file'),
    ([], {'debugSkipNeuralNet': 'true'}, 'needs a real neural net model'),
])
def test_boundary_errors_are_not_passes(tmp_path, args, options, message):
    cfg = configure(tmp_path / 'nn.cfg', 'go', **options)
    result = run(cfg, '/dev/null', *args)
    assert result.returncode != 0 and message in result.stderr
    assert 'PASSED' not in result.stdout
