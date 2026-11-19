"""Joint actions through source PUCT/Gumbel, parallel CUDA evaluation and tree reuse."""
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


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires real native CUDA search')
@pytest.mark.parametrize('gumbel,threads,human_mode', [
    (False, 1, None), (False, 4, None), (True, 1, None), (True, 4, None),
    (False, 1, 'weightless'), (False, 4, 'weightful'),
])
@pytest.mark.parametrize('fp16', [False, True])
def test_native_joint_search(tmp_path, gumbel, threads, human_mode, fp16):
    check_joint_search(tmp_path, ROOT / 'build/katago', gumbel, threads, human_mode, fp16,
                       inputs_nhwc=threads == 4)


def check_joint_search(tmp_path, binary, gumbel, threads, human_mode, fp16, *, inputs_nhwc, backend_config=None):
    torch.manual_seed(912)
    kind = 'transformer' if threads == 4 else 'regular'
    net = Model(dict(component_config(kind=kind), version=19), 10)
    net.initialize()
    net.eval()
    model_path = tmp_path / 'model.bin'
    with model_path.open('wb') as out:
        write_model_binary(net, out, 'xiangqi-search')
    request = dict(evaluate=True, search=True, side=10, fp16=fp16,
                   inputs_nhwc=inputs_nhwc, gumbel=gumbel, threads=threads, backend_config=backend_config or {})
    if human_mode is not None:
        torch.manual_seed(913)
        human_net = Model(dict(component_config(kind=kind), version=19), 10)
        human_net.initialize()
        human_net.eval()
        human_path = tmp_path / 'human.bin'
        with human_path.open('wb') as out:
            write_model_binary(human_net, out, 'xiangqi-secondary')
        request.update(human_model=str(human_path), human_mode=human_mode)
    request_path, response_path = tmp_path / 'request.json', tmp_path / 'response.json'
    request_path.write_text(json.dumps(request))
    completed = subprocess.run([str(binary), 'runtests', '-xiangqi-inference',
                                str(model_path), str(request_path), str(response_path)],
                               capture_output=True, text=True, timeout=240)
    (tmp_path / 'native.log').write_text(completed.stdout + completed.stderr)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(response_path.read_text())
    assert 0 < result['batches'] <= result['rows']
    if human_mode is not None:
        assert 0 < result['human_batches'] <= result['human_rows']
    assert len(result['turns']) == 3
    move_strings = {f'{chr(97+x)}{y}{chr(97+u)}{v}': a for a, (x, y, u, v) in enumerate(xq.MOVES)}
    for turn in result['turns']:
        legal = set(turn['legal'])
        assert turn['chosen'] in legal and set(turn['actions']) <= legal
        assert np.isfinite(turn['target']).all() and min(turn['target']) >= 0 and sum(turn['target']) > 0
        if gumbel:
            assert set(turn['actions']) == legal
            assert sum(turn['target']) == pytest.approx(1)
        report = turn['analysis']
        assert len(report['policy']) == 2086
        assert sum(p for p in report['policy'] if p >= 0) == pytest.approx(1, abs=2e-4)
        assert {i for i, p in enumerate(report['policy']) if p >= 0} == legal
        if human_mode == 'weightless':
            assert report['rootInfo']['visits'] == 1
        else:
            assert 96 <= report['rootInfo']['visits'] <= 96 + threads
        assert report['rootInfo']['noResultValue'] == 0
        assert report['rootInfo']['drawValue'] >= 0
        if human_mode is not None:
            human = report['humanPolicy']
            assert len(human) == 2086
            assert {i for i, p in enumerate(human) if p >= 0} == legal
            assert sum(p for p in human if p >= 0) == pytest.approx(1, abs=2e-4)
            assert set(turn['play_actions']) == legal
            # Complete action selection uses the source secondary policy mix.
            probabilities = np.array(turn['play_values']) / sum(turn['play_values'])
            np.testing.assert_allclose(probabilities, [human[a] for a in turn['play_actions']], atol=2e-5)
            assert np.isfinite(report['rootInfo']['humanWinrate'])
        for entry in report['moveInfos']:
            assert move_strings[entry['move']] in legal
            assert entry['noResultValue'] == 0
            assert all(move in move_strings for move in entry['pv'])
            if human_mode is not None:
                assert entry['humanPrior'] == pytest.approx(human[move_strings[entry['move']]], abs=2e-5)
    assert result['turns'][0]['symmetries'] == ([0] if gumbel or human_mode else [0, 2])
