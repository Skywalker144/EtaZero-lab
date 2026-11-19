"""Real positions through the source evaluator queue, cache, symmetries and postprocessing."""
import functools
import json
import os
import subprocess

import numpy as np
import pytest
import torch

from etazero import ROOT
from export_model_pytorch import write_model_binary
from katago.game import xiangqi as xq
from katago.train.model_pytorch import Model
from test_muzero_model import component_config


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires real native CUDA evaluator')
@pytest.mark.parametrize('fp16', [False, True])
@pytest.mark.parametrize('kind', ['regular', 'transformer'])
def test_positions_cache_symmetry_average_and_parallel_queue(tmp_path, fp16, kind):
    check_positions(tmp_path, ROOT / 'build/katago', fp16, kind, inputs_nhwc=kind == 'transformer')


def check_positions(tmp_path, binary, fp16, kind, *, inputs_nhwc, backend_config=None):
    torch.manual_seed(859)
    side = 12
    config = dict(component_config(kind=kind), version=19)
    net = Model(config, side)
    net.initialize()
    net.eval()
    model_path = tmp_path / 'model.bin'
    with model_path.open('wb') as out:
        write_model_binary(net, out, 'xiangqi-evaluator')
    cycle = ['b0c2', 'b9c7', 'c2b0', 'c7b9']
    cases = [dict(moves=[]),
             dict(moves=['b0c2'], symmetry=2, optimism=.65, temperature=.8, pda=-1.),
             dict(moves=cycle, max_history=0), dict(moves=[], max_history=0),
             dict(moves=cycle + ['b0c2'], max_history=1, optimism=.4, temperature=1.3, pda=1.5),
             dict(moves=cycle + ['b0c2'], max_history=7, optimism=.4, temperature=1.3, pda=1.5)]
    request = dict(evaluate=True, side=side, fp16=fp16, inputs_nhwc=inputs_nhwc, cases=cases,
                   backend_config=backend_config or {})
    request_path, response_path = tmp_path / 'request.json', tmp_path / 'response.json'
    request_path.write_text(json.dumps(request))
    result = subprocess.run([str(binary), 'runtests', '-xiangqi-inference', str(model_path),
                             str(request_path), str(response_path)], capture_output=True, text=True, timeout=240)
    (tmp_path / 'native.log').write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    actual = json.loads(response_path.read_text())
    assert len(actual['cases'][0]['legal']) == 44
    assert actual['cache_hits'] >= len(cases)
    assert actual['parallel_rows'] == 16 and 1 <= actual['parallel_batches'] <= 16
    assert {row['symmetry'] for row in actual['parallel']} == {0, 2}
    assert actual['cases'][2]['hash'] != actual['cases'][3]['hash']  # same board, different repetition history
    assert actual['cases'][4]['hash'] != actual['cases'][5]['hash']  # different visible feature history
    assert actual['cases'][4]['global'][5] == .25
    assert actual['cases'][5]['global'][5] == .75

    @functools.lru_cache(None)
    def expected(index, symmetry):
        row, params = actual['cases'][index], cases[index]
        if symmetry == -2:
            first, second = expected(index, 0), expected(index, 2)
            return {key: (first[key] + second[key]) / 2 for key in first}
        spatial = torch.tensor(row['spatial']).reshape(1, 113, side, side)
        if symmetry == 2:
            spatial[:, :, :10, :9] = spatial[:, :, :10, :9].flip(-1)
        globals_ = torch.tensor(row['global']).reshape(1, 8)
        with torch.no_grad():
            output = net.postprocess_output(net(spatial, globals_))[0]
        policy = output[0][0]
        logits = (policy[0] + (policy[5] - policy[0]) * params.get('optimism', 0)) / params.get('temperature', 1)
        permutation = torch.tensor(xq.action_permutation(black=row['player'] == 2, mirror=symmetry == 2))
        logits = logits[permutation]
        legal = torch.zeros(2086, dtype=torch.bool)
        legal[row['legal']] = True
        policy = logits.masked_fill(~legal, -torch.inf).softmax(-1).masked_fill(~legal, -1)
        value = output[1][0].softmax(-1)
        if row['player'] == 1:
            value = value[[2, 1, 0, 3]]
        return dict(policy=policy.numpy(), value=value.numpy(),
                    auxiliary=np.array([output[4][0].item(), output[5][0].sqrt().item()]))

    for row in actual['cases'] + actual['averages'] + actual['parallel'] + [actual['restarted']]:
        reference = expected(row['case'], row['symmetry'])
        for field in reference:
            tolerance = .15 if field == 'auxiliary' else 2e-3
            np.testing.assert_allclose(row[field], reference[field], atol=tolerance, rtol=3e-3, err_msg=field)
        assert row['value'][3] == 0 and abs(sum(row['value']) - 1) < 1e-6
        values = np.array(row['policy'])
        legal = np.zeros(2086, dtype=bool)
        legal[row['legal']] = True
        assert np.all(values[~legal] == -1) and np.all(values[legal] >= 0)
        assert abs(values[legal].sum() - 1) < 1e-6
