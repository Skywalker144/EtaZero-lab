"""Native dataset traversal and policy entropy, including hand-computable models."""
import os
import re
import subprocess

import numpy as np
import pytest
import torch

from etazero import ROOT
from export_model_pytorch import write_model_binary
from export_muzero import export_components
from test_evalsgf import GAMES, config_for, make_model


cuda = pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires real CUDA search')


def configure(path, game, muzero=False, gumbel=False, random=True, **changes):
    settings = dict(maxVisits=9, minPlayoutsPerThread=0, rootPolicyOptimism=0, policyOptimism=0,
        rootNoiseEnabled='false', useNoisePruning='false', useUncertainty='false', useLcbForSelection='false',
        subtreeValueBiasFactor=0, avoidRepeatedPatternUtility=0, cpuctUtilityStdevScale=0,
        fpuReductionMax=0, rootFpuReductionMax=0, rootDesiredPerChildVisitsCoeff=0,
        chosenMoveSubtract=0, chosenMovePrune=0, gumbelMaxConsideredActions=4, logToStdout='true')
    settings.update(changes)
    return config_for(path, game, muzero=muzero, gumbel=gumbel, random=random, **settings)


def records(tmp_path, game):
    if game == 'xiangqi':
        texts = ['(;GM[7]SZ[9:10];B[a3a4];W[a6a5])'] * 2
        # The opening red pawn move does not change black's 44 legal moves.
        counts = [44] * 4
    else:
        gm = 1 if game == 'go' else 11 if game == 'hex' else 4
        rules = 'Tromp-Taylor' if game == 'go' else game
        header = f'(;GM[{gm}]SZ[5]RU[{rules}]KM[0]'
        texts = [header + ';B[aa];W[ee];B[ba];W[de])', header + ';B[aa];W[ee])']
        counts = np.array([25, 24, 23, 22, 25, 24]) + (game == 'go')
    paths = []
    for i, text in enumerate(texts):
        path = tmp_path / f'game-{i}.sgf'; path.write_text(text); paths.append(path)
    return paths, counts


def run(config, files=(), *, model='/dev/null', dataset=None, extra=(), binary=None):
    result = subprocess.run([str(binary or ROOT / 'build/katago'), 'searchentropyanalysis',
        '-config', str(config), '-model', str(model),
        *(['-boardsizedataset', dataset] if dataset is not None else []),
        *(arg for file in files for arg in ('-sgf', str(file))), *extra],
        capture_output=True, text=True, timeout=120)
    index = len(list(config.parent.glob('entropy-*.log')))
    (config.parent / f'entropy-{index}.log').write_text(result.stdout + result.stderr)
    return result


def outputs(result, count):
    assert result.returncode == 0, result.stdout + result.stderr
    counts = list(map(int, re.findall(r'^Searched numPositions: (\d+)$', result.stdout, re.M)))
    assert counts[-1] == count
    assert counts[:-1] == list(range(10, count + 1, 10))
    if count == 0:
        assert 'Mean search' not in result.stdout
        return None
    values = []
    for label in ['entropy', 'surprise']:
        matches = re.findall(rf'^Mean search {label}: (\S+) standard deviation: (\S+)$', result.stdout, re.M)
        assert len(matches) == len(counts)
        pair = list(map(float, matches[-1]))
        assert np.isfinite(pair).all() and min(pair) >= 0
        values.extend(pair)
    return values


@pytest.mark.parametrize('game', GAMES)
@pytest.mark.parametrize('gumbel', [False, True])
def test_game_inference_and_multiple_records(tmp_path, game, gumbel):
    files, counts = records(tmp_path, game)
    config = configure(tmp_path / 'entropy.cfg', game, gumbel=gumbel)
    config.write_text(re.sub(r'^game = .*\n', '', config.read_text(), flags=re.M))
    outputs(run(config, files), len(counts))


@pytest.mark.parametrize('dataset', ['9', '13', '19', '10x14', 'rectangle'])
def test_source_builtin_datasets_and_periodic_reports(tmp_path, dataset):
    config = configure(tmp_path / 'entropy.cfg', 'go', maxVisits=2, maxBoardSizeForNNBuffer=19)
    result = run(config, dataset=dataset)
    match = re.search(r'Total positions to search: (\d+)', result.stdout)
    assert match, result.stdout + result.stderr
    total = int(match[1])
    # Source traversal skips rules-terminal positions; enumerated SGF turns
    # and successfully searched positions are deliberately different counts.
    count = int(re.findall(r'^Searched numPositions: (\d+)$', result.stdout, re.M)[-1])
    assert 10 <= count <= total
    outputs(result, count)


def uniform_model(path, game, muzero):
    net, _ = make_model(path.with_name('initial'), game, muzero)
    head = net.prediction if muzero else net
    with torch.no_grad():
        head.policy_head.conv2p.weight.zero_()
        if head.policy_head.has_pass:
            head.policy_head.linear_pass2.weight.zero_()
        head.value_head.linear_valuehead.weight.zero_()
        head.value_head.linear_valuehead.bias.zero_()
    if muzero:
        export_components(net, path, 'uniform-policy-entropy')
    else:
        path = path.with_suffix('.bin')
        with path.open('wb') as stream:
            write_model_binary(net, stream, 'uniform-policy-entropy')
    return path


@cuda
@pytest.mark.parametrize('game', GAMES)
@pytest.mark.parametrize('muzero,gumbel', [(False, False), (False, True), (True, False), (True, True)])
def test_exact_uniform_entropy_and_population_variance(tmp_path, game, muzero, gumbel):
    check_exact_uniform_entropy(tmp_path, game, muzero, gumbel)


def check_exact_uniform_entropy(tmp_path, game, muzero, gumbel, *, binary=None, fp16=True):
    model = uniform_model(tmp_path / 'model', game, muzero)
    config = configure(tmp_path / 'entropy.cfg', game, muzero, gumbel, False, cudaUseFP16=str(fp16).lower())
    files, counts = records(tmp_path, game)
    actual = outputs(run(config, files, model=model, binary=binary), len(counts))
    # Nine visits give one root evaluation and eight distinct, equal children.
    # PUCT's target is uniform over those eight. Gumbel's completed-Q target
    # covers every legal action, even those not expanded by sequential halving.
    log_n = np.log(counts)
    expected = ([log_n.mean(), log_n.std(), 0, 0] if gumbel else
                [np.log(8), 0, log_n.mean() - np.log(8), log_n.std()])
    np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_empty_and_terminal_positions_do_not_fabricate_statistics(tmp_path):
    for game, record in [
        ('hex', '(;GM[11]SZ[5])'),
        ('renju', '(;GM[4]SZ[5]RU[renju]AB[aa][ba][ca][da][ea];W[ee])'),
        ('xiangqi', '(;GM[7]SZ[9:10]XQFEN[3aka3/9/9/9/4P4/9/9/9/9/R3K4 w - - 119 60]XQHIST[a0a1];W[d9e8])'),
    ]:
        path = tmp_path / f'{game}.sgf'; path.write_text(record)
        config = configure(tmp_path / f'{game}.cfg', game, gumbel=True)
        outputs(run(config, [path]), 0)


def test_input_errors_are_explicit(tmp_path):
    files, _ = records(tmp_path, 'renju')
    config = configure(tmp_path / 'entropy.cfg', 'renju')
    for args, dataset, message in [([], None, 'Specify either'), (files, '9', 'Specify either'),
                                  ([], 'invalid', 'Unknown search entropy dataset'),
                                  ([], '9', 'config and SGF game differ')]:
        result = run(config, args, dataset=dataset)
        assert result.returncode != 0 and message in result.stdout + result.stderr
    # The illegal second move is consumed when setting up the third position.
    files[0].write_text('(;GM[4]SZ[5]RU[renju];B[aa];W[aa];B[bb])')
    result = run(config, [files[0]])
    assert result.returncode != 0 and 'Illegal move' in result.stdout + result.stderr
    assert str(files[0]) in result.stdout + result.stderr
    files[0].write_text('(;GM[11]SZ[5];B[aa])')
    result = run(config, files)
    assert result.returncode != 0 and 'config and SGF game differ' in result.stdout + result.stderr


def test_muzero_game_and_canvas_checked_before_search(tmp_path):
    model = uniform_model(tmp_path / 'model', 'hex', True)
    config = configure(tmp_path / 'entropy.cfg', 'renju', muzero=True, random=False)
    files, _ = records(tmp_path, 'renju')
    result = run(config, files, model=model)
    assert result.returncode != 0 and 'game differs' in result.stdout + result.stderr
    files[0].write_text('(;GM[11]SZ[7];B[aa])')
    config = configure(tmp_path / 'hex.cfg', 'hex', muzero=True, random=False)
    result = run(config, files[:1], model=model)
    assert result.returncode != 0 and 'canvas must contain' in result.stdout + result.stderr
