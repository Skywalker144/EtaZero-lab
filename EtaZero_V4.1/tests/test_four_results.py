import json
import math
import os
import re
import subprocess
import sys

import numpy as np
import pytest
import torch

from etazero import ROOT
from katago.train import modelconfigs
from katago.train.metrics_pytorch import Metrics
from katago.train.model_pytorch import Model


@pytest.fixture
def model():
    torch.manual_seed(4)
    model = Model(modelconfigs.config_of_name['b2c16-fson-mish'].copy(), pos_len=7)
    model.initialize()
    model.eval()
    with torch.no_grad():
        head = model.value_head
        head.linear_valuehead.weight.zero_()
        head.linear_valuehead.bias.copy_(torch.log(torch.tensor([1., 2., 3., 4.])))
        head.linear_miscvaluehead.weight.zero_()
        head.linear_miscvaluehead.bias[1:9].copy_(torch.arange(8.))
        head.linear_moremiscvaluehead.weight.zero_()
        head.linear_moremiscvaluehead.bias[1:5].copy_(torch.arange(8., 12.))
    return model


def batch_for(model):
    binary = torch.zeros(4, modelconfigs.get_num_bin_input_features(model.config), 7, 7)
    binary[:, 0] = 1
    global_input = torch.zeros(4, modelconfigs.get_num_global_input_features(model.config))
    targets = torch.zeros(4, 80)
    # Main and TD target groups deliberately use all four outcomes.
    for channels in ([0, 70, 1, 2], [4, 71, 5, 6], [8, 72, 9, 10], [12, 73, 13, 14]):
        targets[torch.arange(4), channels] = 1
    targets[:, 25] = torch.tensor([1., .25, .75, 2.])
    return dict(binaryInputNCHW=binary, globalInputNC=global_input,
                globalTargetsNC=targets, policyTargetsNCMove=torch.ones(4, 2, 50),
                valueTargetsNCHW=torch.zeros(4, 2, 7, 7))


def test_head_slices_targets_and_loss_gradients(model):
    batch = batch_for(model)
    outputs = model.postprocess_output(model(batch['binaryInputNCHW'], batch['globalInputNC']))
    value, td = outputs[0][1:3]
    assert value.shape == (4, 4) and td.shape == (4, 3, 4)
    torch.testing.assert_close(td[0], torch.arange(12.).reshape(3, 4))
    metrics = Metrics(1, model)
    losses = metrics.metrics_dict_batchwise(
        model, outputs, extra_outputs=None, batch=batch, is_training=False,
        soft_policy_weight_scale=1., disable_optimistic_policy=False, meta_kata_only_soft_policy=False,
        value_loss_scale=1., td_value_loss_scales=[1., 1., 1.], variance_time_loss_scale=.5, main_loss_scale=1., intermediate_loss_scale=None,
        include_model_norms=False,
    )
    assert torch.isfinite(losses['loss_sum'])
    weights = batch['globalTargetsNC'][:, 25]
    expected_main = 1.2 * (weights * -torch.log(torch.tensor([.1, .2, .3, .4]))).sum()
    torch.testing.assert_close(losses['vloss_sum'], expected_main)
    # Analytic CE gradient: 1.2 * row_weight * (probabilities - one_hot_target).
    grad = torch.autograd.grad(losses['loss_sum'], value, retain_graph=True)[0]
    torch.testing.assert_close(grad, 1.2 * weights[:, None] * (torch.tensor([.1, .2, .3, .4]) - torch.eye(4)))
    td_grad = torch.autograd.grad(losses['loss_sum'], td, retain_graph=True)[0]
    expected_td_grad = 1.2 * weights[:, None, None] * (torch.softmax(td.detach(), dim=2) - torch.eye(4)[:, None, :])
    torch.testing.assert_close(td_grad, expected_td_grad)
    losses['loss_sum'].backward()
    assert model.value_head.linear_valuehead.bias.grad.isfinite().all()


def test_draw_and_no_result_have_zero_base_value(model):
    metrics = Metrics(1, model)
    logits = torch.tensor([[-100., 0., -100., -100.], [-100., -100., -100., 0.]])
    assert metrics.square_value(logits, torch.ones(2)) == 0
    logits = torch.zeros(4, 4, requires_grad=True)
    loss = metrics.loss_value_samplewise(logits, torch.eye(4), torch.ones(4), torch.ones(4))
    torch.testing.assert_close(loss, torch.full((4,), 1.2 * math.log(4)))


@pytest.mark.parametrize('intermediate', [False, True])
def test_retained_future_position_loss_and_gradient(intermediate):
    config = modelconfigs.config_of_name['b2c16-fson-mish'].copy()
    config.update(has_intermediate_head=intermediate, intermediate_head_blocks=1)
    model = Model(config, pos_len=7)
    model.initialize()
    with torch.no_grad():
        model.value_head.conv_futurepos.weight.zero_()
        if intermediate:
            model.intermediate_value_head.conv_futurepos.weight.zero_()
    assert not any('conv_seki' in k or 'conv_scoring' in k or 'conv_ownership' in k for k in model.state_dict())
    batch = batch_for(model)
    batch['globalTargetsNC'][:, 33] = 1
    batch['valueTargetsNCHW'][:, 0, 0, 0] = 1
    batch['valueTargetsNCHW'][:, 1, 1, 1] = -1
    outputs = model.postprocess_output(model(batch['binaryInputNCHW'], batch['globalInputNC']))
    losses = Metrics(1, model).metrics_dict_batchwise(
        model, outputs, extra_outputs=None, batch=batch, is_training=True,
        soft_policy_weight_scale=1., disable_optimistic_policy=False, meta_kata_only_soft_policy=False,
        value_loss_scale=1., td_value_loss_scales=[1., 1., 1.],
        variance_time_loss_scale=.5, main_loss_scale=1., intermediate_loss_scale=.5 if intermediate else None,
        include_model_norms=False,
    )
    weights = batch['globalTargetsNC'][:, 25]
    expected_loss = weights.sum() * .25 * (1 + .25) / 7
    torch.testing.assert_close(losses['fploss_sum'], expected_loss)
    assert not {'oloss_sum', 'sloss_sum', 'skloss_sum', 'sekiweightscale_sum'} & losses.keys()
    for i, head_outputs in enumerate(outputs):
        future = head_outputs[3]
        grad = torch.autograd.grad(losses['loss_sum'], future, retain_graph=True)[0]
        expected_grad = torch.zeros_like(future)
        expected_grad[:, 0, 0, 0] = -.5 * weights / 7
        expected_grad[:, 1, 1, 1] = .125 * weights / 7
        if i == 1:
            expected_grad *= .5
            torch.testing.assert_close(losses['Ifploss_sum'], expected_loss)
        torch.testing.assert_close(grad, expected_grad)


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires CUDA and torch.compile')
def test_compiled_training_loss_matches_eager_after_head_removal(model):
    from katago.train.trainloop_helpers import make_training_metrics_fn

    model.cuda().train()
    batch = {key: value.cuda() for key, value in batch_for(model).items()}
    batch['globalTargetsNC'][:, 26:29] = 1
    batch['globalTargetsNC'][:, 33] = 1
    batch['valueTargetsNCHW'][:, 0, 0, 0] = 1
    metrics = Metrics(1, model)
    compiled = make_training_metrics_fn(metrics, no_compile=False, model_norms_only_at_print=True)

    def run(loss_fn):
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            outputs = model(batch['binaryInputNCHW'], batch['globalInputNC'])
        outputs = model.postprocess_output(model.float32ify_output(outputs))
        result = loss_fn(
            model, outputs, extra_outputs=None, batch=batch, is_training=True,
            soft_policy_weight_scale=1., disable_optimistic_policy=False, meta_kata_only_soft_policy=False,
            value_loss_scale=1., td_value_loss_scales=[1., 1., 1.], variance_time_loss_scale=.5,
            main_loss_scale=1., intermediate_loss_scale=None, include_model_norms=False,
        )
        result['loss_sum'].backward()
        return result['loss_sum'].detach(), {k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None}

    expected_loss, expected_gradients = run(metrics.metrics_dict_batchwise)
    for _ in range(2):
        loss, gradients = run(compiled)
        torch.testing.assert_close(loss, expected_loss, rtol=1e-4, atol=1e-4)
        assert gradients.keys() == expected_gradients.keys()
        for name, gradient in gradients.items():
            assert gradient.isfinite().all(), name
            torch.testing.assert_close(gradient, expected_gradients[name], rtol=1e-2, atol=2e-3)


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires native CUDA build and GPU')
@pytest.mark.parametrize('fp16', [False, True])
@pytest.mark.parametrize('with_q', [False, True])
def test_exported_cuda_four_logits_and_rule_mask(tmp_path, model, fp16, with_q):
    check_exported_four_logits(tmp_path, ROOT / 'build/katago', model, fp16, with_q)


def check_exported_four_logits(tmp_path, binary, model, fp16, with_q, *, backend_config=None):
    if with_q:
        q_model = Model(dict(model.config, version=17, predict_q_values=True), pos_len=7)
        q_model.initialize()
        q_model.eval()
        q_model.value_head.load_state_dict(model.value_head.state_dict())
        model = q_model
        assert model.policy_head.num_policy_outputs == 7
    checkpoint = tmp_path / 'model.ckpt'
    torch.save({'config': model.config, 'model': model.state_dict()}, checkpoint)
    subprocess.run([sys.executable, str(ROOT / 'python/export_model_pytorch.py'),
                    '-checkpoint', str(checkpoint), '-export-dir', str(tmp_path),
                    '-model-name', 'four_results', '-filename-prefix', 'model'], check=True, capture_output=True)
    config = tmp_path / 'gtp.cfg'
    config.write_text(f'rules = japanese\nnumSearchThreads = 1\nmaxVisits = 2\n'
                      f'cudaUseFP16 = {str(fp16).lower()}\nlogSearchInfo = false\nlogToStderr = false\n'
                      'logAllGTPCommunication = false\nponderingEnabled = false\n')
    if backend_config:
        with config.open('a') as stream:
            stream.write(''.join(f'{key} = {value}\n' for key, value in backend_config.items()))
    # The exported biases give mover W,D,L,NR = .1,.2,.3,.4, with no dependency on features.
    # Test both perspectives and source Go rules which prohibit NR.
    commands = ('boardsize 7\nkata-raw-nn w 0\nkata-raw-nn b 0\n'
                'kata-set-rules tromp-taylor\nkata-raw-nn w 0\nkata-raw-nn b 0\nquit\n')
    result = subprocess.run([str(binary), 'gtp', '-config', str(config),
                             '-model', str(tmp_path / 'model.bin')], input=commands, text=True,
                            capture_output=True, check=True, timeout=90)
    assert '?' not in result.stdout, result.stdout
    assert 'whiteOwnership' not in result.stdout
    assert 'Score' not in result.stdout and 'Lead' not in result.stdout
    # Auxiliary layers export variance time and short-term value error, not TD logits.
    batch = batch_for(model)
    with torch.no_grad():
        expected = model.postprocess_output(model(batch['binaryInputNCHW'], batch['globalInputNC']))[0]
    actual_time = np.array([float(v) for v in re.findall(r'varTimeLeft ([\d.]+)', result.stdout)])
    actual_error = np.array([float(v) for v in re.findall(r'shorttermWinlossError ([\d.]+)', result.stdout)])
    np.testing.assert_allclose(actual_time, expected[4][0].item(), atol=.002, rtol=.002)
    np.testing.assert_allclose(actual_error, expected[5][0].sqrt().item(), atol=.002, rtol=.002)
    actual = [tuple(map(float, m)) for m in re.findall(
        r'whiteWin ([\d.]+)\ndraw ([\d.]+)\nwhiteLoss ([\d.]+)\nnoResult ([\d.]+)', result.stdout)]
    np.testing.assert_allclose(actual, [[.1, .2, .3, .4], [.3, .2, .1, .4],
                                       [1/6, 2/6, 3/6, 0], [3/6, 2/6, 1/6, 0]], atol=3e-4, rtol=0)
    commands = ('boardsize 7\nkata-set-rules tromp-taylor\n'
                '90 final_score\n91 final_status_list alive\n92 kata-analyze ownership true\n'
                'play b pass\nplay w pass\n93 final_status_list alive\nquit\n')
    status = subprocess.run([str(binary), 'gtp', '-config', str(config),
                             '-model', str(tmp_path / 'model.bin')], input=commands, text=True,
                            capture_output=True, check=True, timeout=90)
    assert '?90 final_score requires a scored rules terminal' in status.stdout
    assert '?91 final_status_list requires a finished game' in status.stdout
    assert '?92 ' in status.stdout
    assert '=93' in status.stdout
    assert len(re.findall(r'^\?', status.stdout, re.M)) == 3, status.stdout

    analysis_config = tmp_path / 'analysis_ownership.cfg'
    analysis_config.write_text('numAnalysisThreads = 1\nnumSearchThreads = 1\nmaxVisits = 2\nnnMaxBatchSize = 4\n'
                               + ''.join(f'{k} = {v}\n' for k, v in (backend_config or {}).items()))
    fields = ['includeOwnership', 'includeOwnershipStdev', 'includeMovesOwnership', 'includeMovesOwnershipStdev']
    requests = [dict(id=field, boardXSize=7, boardYSize=7, rules='tromp-taylor',
                     komi=0, moves=[], **{field: False}) for field in fields]
    rejected_fields = subprocess.run([str(binary), 'analysis', '-config', str(analysis_config),
                                      '-model', str(tmp_path / 'model.bin')],
                                     input=''.join(json.dumps(r)+'\n' for r in requests),
                                     text=True, capture_output=True, check=True, timeout=90)
    reports = [json.loads(line) for line in rejected_fields.stdout.splitlines() if line.startswith('{')]
    assert len(reports) == len(fields)
    assert {r['id'] for r in reports} == set(fields)
    assert all('Ownership prediction is not supported' in r['error'] for r in reports), reports

    # A model with an obsolete trailing head must not be silently accepted.
    bad_model = tmp_path / 'obsolete.bin'
    bad_model.write_bytes((tmp_path / 'model.bin').read_bytes() + b'old_ownership_head\n')
    rejected = subprocess.run([str(binary), 'gtp', '-config', str(config),
                               '-model', str(bad_model)], input='quit\n', text=True,
                              capture_output=True, timeout=90)
    assert rejected.returncode != 0
    assert 'unexpected trailing model data' in rejected.stderr + rejected.stdout



@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires native CUDA build and GPU')
@pytest.mark.parametrize('graph', [False, True])
def test_training_only_heads_do_not_change_search_utility_weights_or_lcb(tmp_path, model, graph):
    # Change only training TD heads, which must not be exported for inference.
    # Both networks have identical policy and W/D/L/NR. Shared PUCT, weighting and LCB must agree.
    config = tmp_path / 'analysis.cfg'
    config.write_text('numAnalysisThreads = 1\nnumSearchThreads = 1\nmaxVisits = 80\n'
                      'nnMaxBatchSize = 4\n'
                      'cudaUseFP16 = false\nforDeterministicTesting = true\nnnRandomize = false\n'
                      'wideRootNoise = 0\nignorePreRootHistory = false\n'
                      'rootNoiseEnabled = false\nuseLcbForSelection = true\nuseUncertainty = true\n'
                      'subtreeValueBiasFactor = 0\nreportAnalysisWinratesAs = WHITE\n'
                      f'useGraphSearch = {str(graph).lower()}\n')
    request = dict(id='training_head_independence', boardXSize=7, boardYSize=7, rules='tromp-taylor',
                   komi=0, moves=[['B', 'D4'], ['W', 'E4']], includeNoResultValue=True)
    results = []
    for i, td_bias in enumerate([-10., 10.]):
        directory = tmp_path / str(i)
        directory.mkdir()
        with torch.no_grad():
            model.value_head.linear_miscvaluehead.bias[1:].fill_(td_bias)
            model.value_head.linear_moremiscvaluehead.bias[1:].fill_(td_bias)
        checkpoint = directory / 'model.ckpt'
        torch.save({'config': model.config, 'model': model.state_dict()}, checkpoint)
        subprocess.run([sys.executable, str(ROOT / 'python/export_model_pytorch.py'),
                        '-checkpoint', str(checkpoint), '-export-dir', str(directory),
                        '-model-name', 'training_head_independence', '-filename-prefix', 'model'], check=True, capture_output=True)
        result = subprocess.run([str(ROOT / 'build/katago'), 'analysis', '-config', str(config),
                                 '-model', str(directory / 'model.bin')], input=json.dumps(request)+'\n',
                                text=True, capture_output=True, timeout=90)
        assert result.returncode == 0, result.stderr
        reports = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
        assert len(reports) == 1 and 'error' not in reports[0], result.stdout
        results.append(reports[0])
    removed = {'scoreMean', 'scoreSelfplay', 'scoreLead', 'scoreStdev', 'rawLead', 'rawScoreSelfplay', 'rawStScoreError'}
    for result in results:
        assert not removed & result['rootInfo'].keys()
        for move in result['moveInfos']:
            assert not removed & move.keys()
    keys = ['visits', 'weight', 'winrate', 'drawValue', 'noResultValue', 'utility']
    assert {k: results[0]['rootInfo'][k] for k in keys} == {k: results[1]['rootInfo'][k] for k in keys}
    keys += ['move', 'prior', 'lcb', 'utilityLcb', 'edgeVisits', 'edgeWeight', 'playSelectionValue', 'pv']
    assert [{k: m[k] for k in keys} for m in results[0]['moveInfos']] == [
        {k: m[k] for k in keys} for m in results[1]['moveInfos']]


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires native CUDA build and GPU')
def test_pass_prior_is_uncapped_and_gtp_uses_the_search_move(tmp_path, model):
    with torch.no_grad():
        head = model.policy_head
        head.conv2p.weight.zero_()
        head.linear_pass.weight.zero_()
        head.linear_pass.bias.fill_(1.)
        head.linear_pass2.weight.fill_(20. / head.linear_pass2.in_features)
    checkpoint = tmp_path / 'model.ckpt'
    torch.save({'config': model.config, 'model': model.state_dict()}, checkpoint)
    subprocess.run([sys.executable, str(ROOT / 'python/export_model_pytorch.py'),
                    '-checkpoint', str(checkpoint), '-export-dir', str(tmp_path),
                    '-model-name', 'pass_rules', '-filename-prefix', 'model'], check=True, capture_output=True)
    config = tmp_path / 'gtp.cfg'
    config.write_text('rules = chinese\nnumSearchThreads = 1\nmaxVisits = 32\n'
                      'cudaUseFP16 = false\nallowResignation = false\nresignThreshold = -1\n'
                      'logSearchInfo = false\nlogToStderr = false\nlogAllGTPCommunication = false\n'
                      'chosenMoveTemperature = 0\nchosenMoveTemperatureEarly = 0\n')
    commands = ('boardsize 7\nkomi 100\nplay b pass\nkata-raw-nn w 0\n'
                '101 genmove w\n102 final_score\nquit\n')
    result = subprocess.run([str(ROOT / 'build/katago'), 'gtp', '-config', str(config),
                             '-model', str(tmp_path / 'model.bin')], input=commands, text=True,
                            capture_output=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert '?' not in result.stdout, result.stdout
    assert float(re.search(r'policyPass\s+([\d.]+)', result.stdout).group(1)) > .999
    assert '=101 pass\n' in result.stdout, result.stdout
    # Final score must be the rule result after two passes, without any predicted score.
    assert float(re.search(r'=102 W\+([\d.]+)', result.stdout).group(1)) == 100

    analysis_config = tmp_path / 'analysis.cfg'
    analysis_config.write_text('numAnalysisThreads = 1\nnumSearchThreads = 1\n'
                               'nnMaxBatchSize = 4\nmaxVisits = 32\ncudaUseFP16 = false\n')
    request = dict(id='pass_prior', boardXSize=7, boardYSize=7, rules='chinese',
                   komi=100, moves=[['B', 'pass']], includePolicy=True)
    result = subprocess.run([str(ROOT / 'build/katago'), 'analysis', '-config', str(analysis_config),
                             '-model', str(tmp_path / 'model.bin')], input=json.dumps(request)+'\n',
                            text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['policy'][-1] > .999


def test_optimistic_policy_retains_outcome_weights_and_detached_uncertainty(model):
    batch = batch_for(model)
    batch['globalTargetsNC'][:, 26] = torch.tensor([1., .5, .25, 1.])
    batch['globalTargetsNC'][:, 27] = torch.tensor([1., 1., 1., 0.])
    batch['policyTargetsNCMove'].zero_()
    batch['policyTargetsNCMove'][:, :, 0] = 1
    outputs = list(model.postprocess_output(model(batch['binaryInputNCHW'], batch['globalInputNC']))[0])
    outputs[0] = torch.zeros_like(outputs[0], requires_grad=True)
    outputs[2] = torch.zeros_like(outputs[2], requires_grad=True)
    # sqrt(error + 0.0001) = 1, so the short-term standardized excess is [1,0,-1,0].
    outputs[5] = torch.full_like(outputs[5], .9999, requires_grad=True)
    results = Metrics(1, model).metrics_dict_batchwise_single_heads_output(
        model, outputs, batch, is_training=True, soft_policy_weight_scale=1.,
        disable_optimistic_policy=False, meta_kata_only_soft_policy=False, value_loss_scale=1.,
        td_value_loss_scales=[1., 1., 1.], variance_time_loss_scale=.5,
        is_intermediate=False, include_model_norms=False,
    )
    # W has weight 1, D has weight 1/4, L has weight 0; NR row is masked by C27.
    long_weights = torch.tensor([1., .25 * .5 * .25, 0., 0.])
    short_weights = torch.tensor([
        1 / (1 + math.exp(1.5)),
        .25 * .5 / (1 + math.exp(4.5)),
        .75 * .25 / (1 + math.exp(7.5)),
        0.,
    ])
    torch.testing.assert_close(results['p0loptw_sum'], long_weights.sum())
    torch.testing.assert_close(results['p0soptw_sum'], short_weights.sum())
    torch.testing.assert_close(results['p0lopt_sum'], long_weights.sum() * math.log(50))
    torch.testing.assert_close(results['p0sopt_sum'], short_weights.sum() * math.log(50))
    grads = torch.autograd.grad(results['p0sopt_sum'], (outputs[0], outputs[2], outputs[5]), allow_unused=True)
    expected = torch.zeros_like(outputs[0])
    expected[:, 5, :] = short_weights[:, None] / 50
    expected[:, 5, 0] -= short_weights
    torch.testing.assert_close(grads[0], expected)
    assert grads[1] is None and grads[2] is None


def test_q_winloss_keeps_visit_weighting_and_zero_visit_gradient(model):
    metric = Metrics(1, model)
    logits = torch.zeros(2, 50, requires_grad=True)
    targets = torch.zeros(2, 50)
    targets[0, :2] = torch.tensor([1., -1.])
    visits = torch.zeros(2, 50)
    visits[0, :2] = torch.tensor([4., 9.])
    loss = metric.loss_qvalues_samplewise(logits, targets, visits, torch.tensor([2., 3.]))
    torch.testing.assert_close(loss, torch.tensor([2.5 * math.log(2), 0.]))
    gradient = torch.autograd.grad(loss.sum(), logits)[0]
    expected = torch.zeros_like(logits)
    expected[0, :2] = torch.tensor([-1., 1.5])
    torch.testing.assert_close(gradient, expected)
