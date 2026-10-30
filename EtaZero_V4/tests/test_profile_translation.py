"""Role-based profiles feed the native engine and learner through one boundary."""
import json
import os
from pathlib import Path

import pytest

from etazero import ROOT
from etazero.config import load_config
from test_config_profiles import shell


def profile(path, family='alphazero', root='puct', game='renju'):
    path.mkdir(exist_ok=True)
    (path/'run.cfg').write_text('[run]\ngames_per_round = 4\niterations = 2\nauto_plot = false\n')
    (path/'env.cfg').write_text(f'[environment]\ngame = {game}\ncanvas = 5\nbSizes = 3\nbSizeRelProbs = 1\nallowRectangleProb = 0\nkomiMean = 0\n')
    (path/'net.cfg').write_text('[network]\narchitecture = b2c16-fson-mish\n[muzero]\nrepresentation = b2c16-fson-mish\ndynamics = b2c16-fson-mish\nprediction = b2c16-fson-mish\nlatent_channels = 6\nunroll_steps = 3\n')
    search_text = (f'[agent]\nalgorithm = {family}\nroot_search_algo = {root}\nnonroot_search_algo = {root}\n[search]\nmaxVisits = 8\nnumSearchThreads = 1\nuseGraphSearch = false\nsubtreeValueBiasFactor = 0\n[gumbel]\ngumbelMaxConsideredActions = 4\n')
    # Reuse the source engine settings, then pin this test's own execution budget.
    def native(p):
        values={}
        for raw in p.read_text().splitlines():
            line=raw.split('#',1)[0].strip()
            if line.startswith('@include '):values.update(native(p.parent/line.split(None,1)[1]))
            elif '=' in line:
                k,v=line.split('=',1);values[k.strip()]=v.strip()
        return values
    values=native(ROOT/'configs/checks/source_go.cfg')
    values.update(maxMovesPerGame='9',numGameThreads='2',inputsUseNHWC='true',eigenUseFP16='false',rootSymmetryPruning='false')
    for key in ('maxVisits', 'numSearchThreads', 'useGraphSearch', 'subtreeValueBiasFactor', 'gumbelMaxConsideredActions'):
        values.pop(key, None)
    (path/'selfplay.cfg').write_text(search_text + '\n[selfplay]\n'+''.join(f'{k} = {v}\n' for k,v in values.items()))
    (path/'train.cfg').write_text('[train]\nbatch-size = 4\nsamples-per-epoch = 9\nmax-epochs-this-instance = 1\nno-compile\nno-longterm-checkpoints\n')
    (path/'shuffle.cfg').write_text('[shuffle]\nnum-processes = 1\napprox-rows-per-out-file = 64\nkeep-target-rows = 512\nmin-rows = 1\n')
    return load_config(path)


@pytest.mark.parametrize('game,version', [('go',17),('renju',18),('hex',18),('xiangqi',19)])
def test_environment_controls_components_and_native_domain(tmp_path,game,version):
    if game=='xiangqi':
        # Construct first with a valid small canvas, then switch the complete environment.
        profile(tmp_path,'muzero',game='renju')
        (tmp_path/'env.cfg').write_text('[environment]\ngame = xiangqi\ncanvas = 10\nbSizesXY = 9-10\nbSizeRelProbs = 1\n')
        c=load_config(tmp_path)
        assert 'bSizes =' not in c.native_selfplay and 'allowRectangleProb' not in c.native_selfplay
    else:c=profile(tmp_path,'muzero',game=game)
    assert c.muzero_network['game']==game
    assert all(c.muzero_network[r]['version']==version for r in ('representation','dynamics','prediction'))
    assert f'game = {game}\n' in c.native_selfplay
    assert f'dataBoardLen = {c.canvas}\n' in c.native_selfplay
    assert c.model_kind is None and c.muzero_unroll_steps==3


def test_local_override_and_duplicate_ownership(tmp_path):
    profile(tmp_path)
    (tmp_path/'selfplay.cfg.local').write_text('[agent]\nroot_search_algo = gumbel\nnonroot_search_algo = puct\n')
    assert load_config(tmp_path).algorithm=='gumbel_alphazero'
    with (tmp_path/'train.cfg').open('a') as out:out.write('pos-len = 19\n')
    with pytest.raises(ValueError,match='belongs in env.cfg'):load_config(tmp_path)


@pytest.mark.skipif(os.environ.get('ETAZERO_EIGEN_TESTS')!='1',reason='requires Eigen')
@pytest.mark.parametrize('family,root',[('alphazero','puct'),('muzero','gumbel')])
def test_translated_profile_two_rounds_and_resume(tmp_path,family,root):
    directory=tmp_path/'profile';config=profile(directory,family,root)
    output=tmp_path/'run'
    args=('--config-dir',directory,'--run-dir',output,'--binary',ROOT/'build-eigen/katago')
    env=dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    result=shell('run.sh',*args,env=env)
    assert result.returncode==0,result.stdout[-4000:]+result.stderr[-4000:]
    assert (output/'selfplay.resolved.cfg').read_text()==config.native_selfplay
    assert json.loads((output/'state.json').read_text())['completed_rounds']==2
    checkpoint=(output/'train/checkpoint.ckpt').read_bytes()
    assert shell('run.sh',*args,env=env).returncode==0
    assert (output/'train/checkpoint.ckpt').read_bytes()==checkpoint


@pytest.mark.parametrize('blocks', [None, 0, 1, 3, 7])
def test_prediction_depth_override(tmp_path, blocks):
    config = profile(tmp_path, 'muzero')
    if blocks is not None:
        with (tmp_path/'net.cfg').open('a') as f:
            f.write(f'prediction_blocks = {blocks}\n')
        config = load_config(tmp_path)
    prediction = config.muzero_network['prediction']
    assert len(prediction['block_kind']) == (blocks or 0)
    if not blocks:
        assert prediction['trunk_num_channels'] == config.muzero_network['latent_channels']
    else:
        assert len({name for name, kind in prediction['block_kind']}) == blocks
