"""Native diagnostics and Web history operations use game-specific coordinates."""
import json
import subprocess

import pytest
import torch

from config_samples import CONFIGS
from etazero.config import write_native
from etazero.engine_config import load_engine_config
from etazero.games import XIANGQI_INDEX, spec_for_rule
from etazero.network import make_network, inference_network
from etazero.symmetry import augment_batch
from test_multigame import ROOT, BINARY, small_config


def cpu_model(tmp_path, rule, algorithm, size, canvas):
    if not BINARY.exists():pytest.skip('Build native engine first')
    torch.set_num_threads(1)
    torch.manual_seed(19)
    c=small_config('go' if rule=='go' else 'xiangqi',algorithm)
    c['environment'].update(rules=rule,sizes=str(size))
    c['network']['canvas']=canvas
    c['inference'].update(inference_precision='float32',server_threads=1)
    profile=load_engine_config(CONFIGS/'minimal_test',environ={})
    profile['analysis'].update(rule=rule,board_size=size,device='cpu',inference_precision='float32',
                               root_num_symmetries_to_sample=1,nn_randomize=False,nn_symmetry=0,
                               use_graph_search=False,subtree_value_bias_factor=0)
    c.update(profile)
    model=torch.jit.script(inference_network(make_network(c).eval()).eval())
    path=tmp_path/'model.pt';model.save(str(path))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(id='diagnostics',canvas=canvas,game=rule,
                                                        algorithm=algorithm,bytes=path.stat().st_size)))
    config=tmp_path/'effective.cfg';write_native(c,config)
    return model,path,config,profile


@pytest.mark.parametrize('rule,size,canvas,move',[
    ('hex',5,6,0),('go',5,6,0),('xiangqi',10,10,XIANGQI_INDEX[(1,0,2,2)])])
def test_muzero_recurrent_diagnostics_match_canonical_model(tmp_path,rule,size,canvas,move):
    model,path,config,_=cpu_model(tmp_path,rule,'muzero',size,canvas)
    replay=subprocess.run([str(ROOT/'build/rules_replay')],input=json.dumps(dict(
        size=size,canvas=canvas,rule=rule,actions=[move]))+'\n',text=True,capture_output=True,check=True)
    raw=json.loads(replay.stdout.splitlines()[-1])['observation']
    spec=spec_for_rule(rule);spatial=spec.planes*canvas*canvas
    batch=augment_batch(dict(obs=torch.tensor(raw[:spatial]).reshape(1,spec.planes,canvas,canvas),
                             globals=torch.tensor(raw[spatial:]).reshape(1,spec.globals),
                             actions=torch.arange(spec.actions(canvas))[None]),0,spec)
    mapping=batch['actions'][0]
    result=subprocess.run([str(BINARY),'infer','--config',str(config),'--model',str(path),
        '--model-id','diagnostics','--device','cpu','--seed','0','--size',str(size),'--rule',rule,
        '--moves',str(move)],text=True,capture_output=True,check=True,timeout=30)
    result=json.loads(result.stdout)
    with torch.inference_mode():
        output=model.initial(batch['obs'],batch['globals'])
        torch.testing.assert_close(torch.tensor(result['raw_logits']),output[1][0,mapping])
        assert len(result['recurrent'])==3
        for step in result['recurrent']:
            output=model.recurrent(output[0],mapping[step['action']].reshape(1))
            torch.testing.assert_close(torch.tensor(step['logits']),output[1][0,mapping])
            torch.testing.assert_close(torch.tensor(step['wdl']),output[2][0].softmax(0))
            torch.testing.assert_close(torch.tensor(step['optimistic_logits']),output[3][0,mapping])


def test_long_go_undo_and_web_branch_to_zero(tmp_path):
    from web.engine import Engine
    from web.app import App
    _,path,_,profile=cpu_model(tmp_path,'go','alphazero',19,19)
    # Every white pass is separated by a legal black placement: 650 plies,
    # with empty intersections remaining and no consecutive passes.
    moves=[action for point in range(325) for action in (point,361)]
    app=App(BINARY,{'test':path},profile,19)
    try:
        app.engine=Engine(BINARY,path,profile)
        initial=app.engine.command(dict(command='new',size=19,rule='go'))['state']
        game=app.engine.command(dict(command='new',size=19,rule='go',moves=moves))['state']
        assert game['turn']==650 and not game['finished']
        for count in (0,651):
            with pytest.raises(RuntimeError,match='integer range'):app.engine.command(f'undo {count}')
        restored=app.engine.command('undo 650')['state']
        for field in ('turn','moves','board','boards','players','legal_actions'):
            assert restored[field]==initial[field]
        game=app.engine.command(dict(command='new',size=19,rule='go',moves=moves))['state']
        app.publish(game=game,game_id='long',model='test',rule='go',mode='manual')
        app.submit('branch',dict(version=app.snapshot()['version'],turn=0))
        with app.condition:
            assert app.condition.wait_for(lambda:not app.state['busy'],timeout=15)
        result=app.snapshot()
        assert result['error'] is None and result['game']['turn']==0
        assert result['game']['board']==initial['board']
    finally:app.close()
