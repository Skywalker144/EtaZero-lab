"""Real CUDA coverage of standalone CLI profiles and the shared Web session."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch
from config_samples import CONFIGS
from etazero.analysis import analyze
from etazero.config import ROOT, load_config, write_native
from etazero.engine_config import load_engine_config
from etazero.export import export_model
from etazero.training import initialize

pytestmark=pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS')!='1',reason='Host CUDA required')


@pytest.fixture(params=['alphazero','muzero'])
def model_and_profiles(tmp_path,request):
    torch.set_num_threads(1)
    config=load_config(CONFIGS/'smoke_test')
    config['network'].update(canvas=11,channels=24,blocks=1)
    config['agent']['algorithm']=request.param
    if request.param=='muzero':
        config['muzero']=dict(latent_channels=16,dynamics_channels=24,dynamics_blocks=1,
                              prediction_channels=24,prediction_blocks=1)
        config['unroll']=dict(steps=3,hidden_gradient_scale=.5)
    run=tmp_path/request.param
    write_native(config,run/'config/effective.cfg')
    checkpoint=initialize(run,config)
    model=export_model(run,config,checkpoint)
    model_path=run/model['path']
    shutil.copytree(CONFIGS,tmp_path/'profiles')
    for path in (tmp_path/'profiles').rglob('run.cfg'):
        path.unlink()
    source=tmp_path/'profiles'/('muzero' if request.param=='muzero' else 'baseline')
    (source/'engine.cfg.local').write_text('[engine]\nvisits=16\nboard_size=5\nrule=renju\n'
                                           'nn_randomize=false\ncache_entries=0\n')
    return model_path,source


def command(*arguments,input=None):
    result=subprocess.run(['bash',str(ROOT/'scripts/run.sh'),*map(str,arguments)],input=input,
                          text=True,capture_output=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    return result.stdout


def test_single_analysis_and_stream_use_the_same_position_search(tmp_path,model_and_profiles):
    model,source=model_and_profiles
    output=tmp_path/'result.json'
    command('analysis','--config',source/'analysis.cfg','--model',model,'--run-dir',tmp_path,'--size','5','--rule','renju',
            '--moves','0,5','--output',output)
    saved=json.loads(output.read_text())
    assert saved['mode']=='analysis' and saved['config']['analysis']['board_size']==5
    assert saved['result']['root_visits']==16 and len(saved['result']['policy'])==25
    stdout=command('analysis','--config',source/'analysis.cfg','--model',model,'--stream',
                   input='new 5 renju\nplay 0\nplay 5\nanalyze 16\nstate\nquit\n')
    replies=[json.loads(line) for line in stdout.splitlines()]
    assert len(replies)==6 and all(reply['ok'] for reply in replies)
    result=replies[4]['analysis']
    assert replies[4]['state']['moves']==replies[5]['state']['moves']==[0,5]
    assert result['action']==saved['result']['action']
    assert result['root_value']==pytest.approx(saved['result']['value'],abs=1e-6)
    assert result['completed_visits']==16
    config=load_engine_config(source,environ={})
    _,api=analyze(config,ROOT/'build/etazero',tmp_path,model=model,size=5,rule='renju',moves='0,5')
    assert api['result']['visits']==saved['result']['visits']


def test_match_cli_reuses_complete_pairs_with_standalone_profiles(tmp_path,model_and_profiles):
    model,source=model_and_profiles
    output=tmp_path/'match'
    arguments=('match','--config',source/'match.cfg','--model',model,'--model-b',model,
               '--games','4','--output',output)
    command(*arguments)
    completed={path:path.read_bytes() for path in output.glob('pairs/*/games/*.json')}
    assert len(completed)==4
    assert all(json.loads(value)['same_bot'] for value in completed.values())
    command(*arguments)
    assert all(path.read_bytes()==value for path,value in completed.items())


def test_web_with_the_published_cuda_model(model_and_profiles):
    model,_=model_and_profiles
    result=subprocess.run([sys.executable,'-m','pytest',
                           str(ROOT.parent/'web/tests/test_web.py'),'-q','--tb=short'],
                          cwd=ROOT.parent,env={**os.environ,'ETAZERO_WEB_TEST_MODEL':str(model)},
                          text=True,capture_output=True,timeout=120)
    assert result.returncode==0,result.stdout+result.stderr
