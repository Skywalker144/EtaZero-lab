"""Real Eigen FP32 inference, using the same PyTorch oracles as the CUDA checks.

Build build-eigen/katago with USE_BACKEND=EIGEN and set ETAZERO_EIGEN_TESTS=1.
"""
import json
import os
import subprocess

import pytest
import torch

from etazero import ROOT
from test_evalsgf import check_exported_observations
from test_four_results import check_exported_four_logits, model
from test_generated_initializations import check_model_services
from test_hint_policy import check_uniform_hint_likelihood
from test_datamine import check_services as check_mining_services
from test_samplesgfs import check_exact_weights as check_fluctuation_weights
from test_booktoposes import check_source_weights as check_book_weights
from test_genbook import check_real_models as check_book_generation
from test_initializations import check_real_models_and_search
from test_placement_network import check_placement_export
from test_random_initializations import check_real_models as check_random_initializations
from test_search_entropy import check_exact_uniform_entropy
from test_startpose_search import check_real_model as check_startpose_search
from test_surprise_dump import check_four_outcome_kl_and_reanalysis
from test_xiangqi_evaluator import check_positions
from test_xiangqi_export import check_joint_export, export
from test_xiangqi_search import check_joint_search


pytestmark = pytest.mark.skipif(os.environ.get('ETAZERO_EIGEN_TESTS') != '1',
                                reason='requires the separate native Eigen build')


@pytest.fixture
def eigen_binary():
    binary = ROOT / 'build-eigen/katago'
    assert binary.is_file(), 'Build the Eigen backend before requesting its tests'
    return binary


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('with_q', [False, True])
def test_joint_logits_and_batch_reuse(tmp_path, eigen_binary, with_q):
    check_joint_export(tmp_path, eigen_binary, False, True, with_q, True)


@pytest.mark.parametrize('kind', ['regular', 'transformer'])
def test_positions_cache_and_parallel_queue(tmp_path, eigen_binary, kind):
    check_positions(tmp_path, eigen_binary, False, kind, inputs_nhwc=True)


@pytest.mark.parametrize('gumbel,threads,human_mode', [
    (False, 1, None), (True, 4, None), (False, 4, 'weightless'),
])
def test_joint_search(tmp_path, eigen_binary, gumbel, threads, human_mode):
    check_joint_search(tmp_path, eigen_binary, gumbel, threads, human_mode, False, inputs_nhwc=True)


@pytest.mark.parametrize('with_q', [False, True])
def test_placement_rules_symmetry_and_export(tmp_path, eigen_binary, with_q):
    check_placement_export(tmp_path, eigen_binary, False, with_q)


@pytest.mark.parametrize('with_q', [False, True])
def test_go_four_results_rule_mask_and_auxiliary(tmp_path, eigen_binary, model, with_q):
    check_exported_four_logits(tmp_path, eigen_binary, model, False, with_q)


@pytest.mark.parametrize('game', ['go', 'renju', 'xiangqi'])
def test_sgf_diagnostics_and_exported_inputs(tmp_path, eigen_binary, game):
    check_exported_observations(tmp_path, game, False, False, binary=eigen_binary)


@pytest.mark.parametrize('game,gumbel', [('go', False), ('xiangqi', True)])
def test_selfplay_surprise_and_reanalysis(tmp_path, eigen_binary, game, gumbel):
    check_four_outcome_kl_and_reanalysis(tmp_path, game, False, gumbel, binary=eigen_binary, fp16=False)


@pytest.mark.parametrize('game,gumbel', [('go', False), ('xiangqi', True)])
def test_initialization_sampling(tmp_path, eigen_binary, game, gumbel):
    check_real_models_and_search(tmp_path, game, False, gumbel, binary=eigen_binary, fp16=False)


@pytest.mark.parametrize('game,gumbel', [('go', False), ('xiangqi', True)])
def test_policy_entropy(tmp_path, eigen_binary, game, gumbel):
    check_exact_uniform_entropy(tmp_path, game, False, gumbel, binary=eigen_binary, fp16=False)


@pytest.mark.parametrize('game,gumbel', [('go', False), ('xiangqi', True)])
def test_random_policy_initializations(tmp_path, eigen_binary, game, gumbel):
    check_random_initializations(tmp_path, game, False, gumbel, binary=eigen_binary, fp16=False)


@pytest.mark.parametrize('game,gumbel', [('hex', False), ('xiangqi', True)])
def test_generated_initialization_files(tmp_path, eigen_binary, game, gumbel):
    check_model_services(tmp_path, game, False, gumbel, binary=eigen_binary, fp16=False)


@pytest.mark.parametrize('game,gumbel', [('go', False), ('xiangqi', True)])
def test_startpose_search(tmp_path, eigen_binary, game, gumbel):
    check_startpose_search(tmp_path, game, False, gumbel, binary=eigen_binary, fp16=False)


@pytest.mark.parametrize('game', ['go', 'hex', 'xiangqi'])
def test_sgf_hint_policy(tmp_path, eigen_binary, game):
    check_uniform_hint_likelihood(tmp_path, game, False, False, binary=eigen_binary, fp16=False)


@pytest.mark.parametrize('game,gumbel', [('go', False), ('xiangqi', True)])
def test_sgf_mining(tmp_path, eigen_binary, game, gumbel):
    check_mining_services(tmp_path, game, False, gumbel, binary=eigen_binary, fp16=False)


@pytest.mark.parametrize('game', ['go', 'hex', 'xiangqi'])
def test_sgf_value_fluctuation(tmp_path, eigen_binary, game):
    check_fluctuation_weights(tmp_path, game, False, False, binary=eigen_binary, fp16=False)


@pytest.mark.parametrize('game', ['go', 'hex', 'xiangqi'])
def test_book_sampling(tmp_path, eigen_binary, game):
    check_book_weights(tmp_path, game, False, False, binary=eigen_binary, fp16=False)


@pytest.mark.parametrize('game', ['go', 'hex', 'xiangqi'])
def test_book_generation(tmp_path, eigen_binary, game):
    check_book_generation(tmp_path, game, False, True, binary=eigen_binary, fp16=False)


@pytest.mark.parametrize('fp16,nhwc,message', [
    (True, True, 'FP16'), (False, False, 'inputsUseNHWC = false unsupported'),
])
def test_source_backend_layout_and_precision_limits(tmp_path, eigen_binary, fp16, nhwc, message):
    _, path = export(tmp_path, False)
    request = dict(width=10, height=10, fp16=fp16, inputs_nhwc=nhwc, batches=[1],
                   rows=[dict(spatial=[0.] * 11300, **{'global': [0.] * 8}, symmetry=0, optimism=0)])
    request_path = tmp_path / 'request.json'
    request_path.write_text(json.dumps(request))
    result = subprocess.run([str(eigen_binary), 'runtests', '-xiangqi-inference', str(path),
                             str(request_path), str(tmp_path / 'response.json')],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and message in result.stdout + result.stderr
