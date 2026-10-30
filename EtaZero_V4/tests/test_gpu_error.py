"""Source precision/batch comparisons on actual multigame and latent inference."""
import json
import os
import re
import subprocess

import pytest

from etazero import ROOT
from test_benchmarknn import cuda, model_for
from test_evalsgf import GAMES, record


def config_for(path, game, muzero=False, fp16=True, **changes):
    options = dict(game=game, useMuZero=str(muzero).lower(), nnMaxBatchSize=3,
                   numNNServerThreadsPerModel=2, nnCacheSizePowerOfTwo=8,
                   nnMutexPoolSizePowerOfTwo=4, cudaUseFP16=str(fp16).lower(),
                   cudaUseNHWC='true', inputsUseNHWC='true', policyOptimism=.35)
    options.update(changes)
    path.write_text(''.join(f'{k} = {v}\n' for k, v in options.items() if v is not None))
    return path


def run(config, model='/dev/null', *args, binary=None):
    result = subprocess.run([str(binary or ROOT / 'build/katago'), 'testgpuerror',
        '-config', str(config), '-model', str(model), *map(str, args)],
        capture_output=True, text=True, timeout=120)
    attempt = len(list(config.parent.glob('gpu-error-*.log')))
    (config.parent / f'gpu-error-{attempt}.log').write_text(result.stdout + result.stderr)
    return result


def check(result, *, muzero=False, fp16=True, count=None, recurrent_steps=3):
    assert result.returncode == 0, result.stdout + result.stderr
    text = result.stdout
    counts = list(map(int, re.findall(r'Computed stats on (\d+) positions', text)))
    assert len(counts) == (recurrent_steps + 1 if muzero else 1) and all(c > 0 for c in counts)
    if count is not None:
        assert counts == [count] * len(counts)
    assert 'batched fp32 error vs reference' in text
    assert ('batched current cfg error vs reference' in text) == fp16
    assert 'drawError:' in text and 'policyKLDiv:' in text
    depths = list(map(int, re.findall(r'Testing MuZero recurrent depth (\d+)', text)))
    assert depths == (list(range(1, recurrent_steps + 1)) if muzero else [])
    assert 'nan' not in text.lower() and ' inf' not in text.lower()


@cuda
@pytest.mark.parametrize('game', GAMES)
@pytest.mark.parametrize('muzero', [False, True])
@pytest.mark.parametrize('fp16', [False, True])
def test_actual_models_same_positions_in_each_precision_and_batch(tmp_path, game, muzero, fp16):
    model = model_for(tmp_path / 'model', game, muzero)
    cfg = config_for(tmp_path / 'test.cfg', game, muzero, fp16)
    sgf = tmp_path / 'game.sgf'; sgf.write_text(record(game))
    check(run(cfg, model, '-sgf', sgf), muzero=muzero, fp16=fp16, count=3)


@cuda
@pytest.mark.parametrize('game', GAMES)
def test_builtin_datasets_and_source_quick_filter(tmp_path, game):
    model = model_for(tmp_path / 'model', game, False)
    cfg = config_for(tmp_path / 'test.cfg', game)
    size = '9' if game == 'go' else '9x10' if game == 'xiangqi' else '5'
    full = run(cfg, model, '-boardsize', size)
    quick = run(cfg, model, '-boardsize', size, '-quick')
    check(full); check(quick)
    counts = [int(re.search(r'Computed stats on (\d+) positions', r.stdout)[1]) for r in [full, quick]]
    assert 0 < counts[1] < counts[0]
    again = run(cfg, model, '-boardsize', size, '-quick')
    check(again)
    assert int(re.search(r'Computed stats on (\d+) positions', again.stdout)[1]) == counts[1]


@cuda
@pytest.mark.skipif(os.environ.get('ETAZERO_EIGEN_TESTS') != '1', reason='requires Eigen reference exporter')
@pytest.mark.parametrize('game', GAMES)
@pytest.mark.parametrize('muzero', [False, True])
def test_eigen_reference_roundtrip_and_wrong_data_rejected(tmp_path, game, muzero):
    model = model_for(tmp_path / 'model', game, muzero)
    cfg = config_for(tmp_path / 'test.cfg', game, muzero, eigenUseFP16='false')
    sgf = tmp_path / 'game.sgf'; sgf.write_text(record(game))
    reference = tmp_path / 'reference.jsonl'
    check(run(cfg, model, '-sgf', sgf, '-reference-file', reference,
              binary=ROOT / 'build-eigen/katago'), muzero=muzero, fp16=False, count=3)
    references = [reference] + ([tmp_path / f'reference.jsonl.recurrent-{i}' for i in range(1, 4)] if muzero else [])
    hashes = []
    for path in references:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 3 and all('whiteDrawProb' in row for row in rows)
        assert all(len(row['policyProbs']) == (2086 if game == 'xiangqi' else 362) for row in rows)
        hashes.extend(row['nnHash'] for row in rows)
    assert len(set(hashes)) == len(hashes)  # Distinct roots and action prefixes.
    check(run(cfg, model, '-sgf', sgf, '-reference-file', reference), muzero=muzero, count=3)
    cfg32 = config_for(tmp_path / 'fp32.cfg', game, muzero, fp16=False)
    check(run(cfg32, model, '-sgf', sgf, '-reference-file', reference), muzero=muzero, fp16=False, count=3)
    # Mutate a recurrent reference when present, so a reader that only consumes
    # initial cannot pass the rejection checks.
    tested = references[-1]
    lines = tested.read_text().splitlines()
    values = [json.loads(line) for line in lines]
    # A different draw probability can leave W-L and NR unchanged. The new
    # channel must be checked, rather than disappearing in the old metric.
    for value in values:
        value['whiteDrawProb'] += .2
    tested.write_text('\n'.join(json.dumps(v) for v in values))
    bad = run(cfg, model, '-sgf', sgf, '-reference-file', reference)
    assert bad.returncode == 1 and 'drawError' in bad.stdout
    values = [json.loads(line) for line in lines]
    values[0]['nnHash'] = '0' * 32
    tested.write_text('\n'.join(json.dumps(v) for v in values))
    bad = run(cfg, model, '-sgf', sgf, '-reference-file', reference)
    assert bad.returncode != 0 and 'reference input hash differs' in bad.stderr
    if muzero:
        # Swapping two valid depth files retains the count and policy layout;
        # trajectory hashes must still reject them.
        tested.write_text(references[-2].read_text())
        bad = run(cfg, model, '-sgf', sgf, '-reference-file', reference)
        assert bad.returncode != 0 and 'reference input hash differs' in bad.stderr
        tested.unlink()
        bad = run(cfg, model, '-sgf', sgf, '-reference-file', reference)
        assert bad.returncode != 0 and str(tested) in bad.stderr


@cuda
@pytest.mark.parametrize('muzero', [False, True])
@pytest.mark.parametrize('reference', [False, True])
def test_go_metadata_transformer_and_rectangular_history(tmp_path, muzero, reference):
    if reference and os.environ.get('ETAZERO_EIGEN_TESTS') != '1':
        pytest.skip('requires Eigen reference exporter')
    model = model_for(tmp_path / 'model', 'go', muzero, kind='transformer', metadata=True)
    cfg = config_for(tmp_path / 'test.cfg', 'go', muzero, eigenUseFP16='false')
    sgf = tmp_path / 'game.sgf'
    sgf.write_text('(;GM[1]SZ[4:5]RU[Japanese]KM[0];B[aa];W[dc];B[];W[])')
    args = ['-reference-file', tmp_path / 'reference.jsonl'] if reference else []
    if reference:
        check(run(cfg, model, '-sgf', sgf, *args, binary=ROOT / 'build-eigen/katago'), muzero=muzero, fp16=False)
    check(run(cfg, model, '-sgf', sgf, *args), muzero=muzero)


@cuda
@pytest.mark.parametrize('muzero', [False, True])
@pytest.mark.parametrize('reference', [False, True])
def test_xiangqi_full_history_and_placement_zero_liberty(tmp_path, muzero, reference):
    if reference and os.environ.get('ETAZERO_EIGEN_TESTS') != '1':
        pytest.skip('requires Eigen reference exporter')
    for game in ['xiangqi', 'hex']:
        model = model_for(tmp_path / game, game, muzero)
        cfg = config_for(tmp_path / f'{game}.cfg', game, muzero, eigenUseFP16='false')
        sgf = tmp_path / f'{game}.sgf'
        sgf.write_text('(;GM[7]SZ[9:10]RU[xiangqi]XQHIST[a3a4][a6a5]XQTURN[97];B[b0c2];W[b9c7])'
                      if game == 'xiangqi' else '(;GM[11]SZ[5]RU[hex]AB[aa]AW[ab][ba];B[cc];W[dd])')
        args = ['-reference-file', tmp_path / f'{game}.jsonl'] if reference else []
        if reference:
            check(run(cfg, model, '-sgf', sgf, *args, binary=ROOT / 'build-eigen/katago'), muzero=muzero, fp16=False, count=3)
        check(run(cfg, model, '-sgf', sgf, *args), muzero=muzero, count=3)


@pytest.mark.skipif(os.environ.get('ETAZERO_EIGEN_TESTS') != '1', reason='requires Eigen reference exporter')
@pytest.mark.parametrize('game', ['go', 'xiangqi'])
def test_muzero_initial_only_and_repeated_cpu_references(tmp_path, game):
    model = model_for(tmp_path / 'model', game, True)
    cfg = config_for(tmp_path / 'test.cfg', game, True, eigenUseFP16='false')
    sgf = tmp_path / 'game.sgf'; sgf.write_text(record(game))
    initial = tmp_path / 'initial.jsonl'
    check(run(cfg, model, '-sgf', sgf, '-reference-file', initial, '-muzero-recurrent-steps', 0,
              binary=ROOT / 'build-eigen/katago'), muzero=True, fp16=False, count=3, recurrent_steps=0)
    assert list(tmp_path.glob('initial.jsonl*')) == [initial]
    runs = []
    for attempt in range(2):
        reference = tmp_path / f'reference-{attempt}.jsonl'
        check(run(cfg, model, '-sgf', sgf, '-reference-file', reference,
                  binary=ROOT / 'build-eigen/katago'), muzero=True, fp16=False, count=3)
        paths = [reference] + [tmp_path / f'{reference.name}.recurrent-{i}' for i in range(1, 4)]
        runs.append([[json.loads(line) for line in path.read_text().splitlines()] for path in paths])
    assert runs[0] == runs[1]
    assert runs[0][0] == [json.loads(line) for line in initial.read_text().splitlines()]


@pytest.mark.parametrize('options,args,message', [
    ({}, ['-boardsize', '5'], 'Unknown Go GPU-error dataset'),
    ({}, ['-muzero-recurrent-steps', '1'], 'require useMuZero'),
    ({}, ['-muzero-recurrent-steps', '-1'], 'nonnegative'),
    ({'game': 'hex', 'rules': 'renju'}, [], 'game and rules differ'),
    ({'game': 'hex'}, ['-boardsize', '5x6'], 'square board'),
    ({'game': 'xiangqi'}, ['-boardsize', '9'], 'requires a 9x10 board'),
])
def test_invalid_boundaries_fail_before_loading_model(tmp_path, options, args, message):
    cfg = config_for(tmp_path / 'test.cfg', **({'game': 'go'} | options))
    result = run(cfg, '/dev/null', *args)
    assert result.returncode != 0 and message in result.stderr


def test_sgf_game_mismatch_and_ambiguous_dataset(tmp_path):
    sgf = tmp_path / 'game.sgf'; sgf.write_text(record('hex'))
    cfg = config_for(tmp_path / 'test.cfg', 'go')
    for args, message in [(['-sgf', sgf], 'SGF and configured game differ'),
                          (['-sgf', sgf, '-boardsize', '9'], 'mutually exclusive')]:
        result = run(cfg, '/dev/null', *args)
        assert result.returncode != 0 and message in result.stderr


def test_random_evaluator_is_not_precision_evidence(tmp_path):
    cfg = config_for(tmp_path / 'test.cfg', 'go', debugSkipNeuralNet='true')
    result = run(cfg, '/dev/null', '-boardsize', '9')
    assert result.returncode != 0 and 'requires a real neural net model' in result.stderr
