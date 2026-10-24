"""Fixed-model position analysis and the persistent interactive analysis engine."""
import json
from pathlib import Path
import subprocess
import uuid
from etazero.config import fingerprint, write_native
from etazero.storage import load_json, save_json
from etazero.build import verify_build
from etazero.games import spec_for_rule


def model_info(path,canvas=None):
    path=Path(path).resolve()
    info=load_json(path.parent/"manifest.json")
    if (canvas is not None and info["canvas"]!=canvas):
        raise ValueError(f"Inference model canvas mismatch: {path}")
    if path.stat().st_size != info["bytes"]:
        raise ValueError(f"Inference model size mismatch: {path}")
    return path,info


def select_model(run_dir, model=None):
    root = Path(run_dir).resolve()
    return Path(model).resolve() if model is not None else root/load_json(root/'models/current.json')['model']['path']


def analyze(config,binary,run_dir,model=None,size=None,rule=None,moves="",output=None):
    verify_build(binary)
    root=Path(run_dir).resolve()
    model,a=model_info(select_model(root, model))
    from etazero.engine_config import validate_engine_config
    import copy
    config=copy.deepcopy(config)
    c=config['analysis']
    if size is not None: c['board_size']=size
    if rule is not None: c['rule']=rule
    validate_engine_config(config)
    size,rule=c['board_size'],c['rule']
    if spec_for_rule(a.get('game','gomoku')).layout != spec_for_rule(rule).layout:
        raise ValueError('Analysis rule and model game layout differ')
    if not 5<=size<=a['canvas']:
        raise ValueError('Analysis board size must be in [5, canvas]')
    # Public action indices use the actual board width; native uses its canvas.
    actions=[int(v) for v in moves.split(',')] if moves else []
    count=spec_for_rule(rule).actions(size)
    if any(v<0 or v>=count for v in actions):
        raise ValueError('Analysis moves must be board-row-major indices')
    encoded=','.join(str(v if rule=='xiangqi' else a['canvas']**2 if v==size*size else v//size*a['canvas']+v%size) for v in actions)
    directory=root/'analysis'/uuid.uuid4().hex;directory.mkdir(parents=True)
    write_native({**config,'network':{'canvas':a['canvas']}},directory/'effective.cfg')
    command=[str(binary),'analysis','--config',str(directory/'effective.cfg'),'--model',str(model),
             '--model-id',a['id'],'--device',c['device'],'--size',str(size),'--rule',rule,
             '--moves',encoded,'--seed',str(c['seed'])]
    result=json.loads(subprocess.run(command,text=True,capture_output=True,check=True).stdout)
    if 'action' in result:
        if result['action'] >= 0 and rule!='xiangqi':
            result['action']=size*size if result['action']==a['canvas']**2 else result['action']//a['canvas']*size+result['action']%a['canvas']
        for key in ('raw_logits','policy','visits','network_policy','search_policy'):
            if rule!='xiangqi': result[key]=[result[key][y*a['canvas']+x] for y in range(size) for x in range(size)]+([result[key][-1]] if rule=='go' else [])
    payload={'mode':'analysis','config_id':fingerprint(config),'config':config,'model_a':a,
             'board_size':size,'rule':rule,'opening_actions':actions,'seed':c['seed'],
             'command':command,'binary':str(Path(binary).resolve()),'result':result}
    path=Path(output).resolve() if output else directory/'result.json'
    save_json(path,payload,immutable=True)
    return path,payload


def stream_analysis(config, binary, run_dir, model=None, protocol='text'):
    """Run the same native session used by Web, retaining weights across queries."""
    import tempfile
    from etazero.engine_config import validate_engine_config
    verify_build(binary)
    validate_engine_config(config)
    model,info=model_info(select_model(run_dir,model))
    c=config['analysis']
    if spec_for_rule(info.get('game','gomoku')).layout != spec_for_rule(c['rule']).layout:
        raise ValueError('Analysis rule and model game layout differ')
    with tempfile.TemporaryDirectory(prefix='etazero-analysis-') as temporary:
        path=Path(temporary)/'effective.cfg'
        write_native({**config,'network':{'canvas':info['canvas']}},path)
        command=[str(binary),'gtp' if protocol=='gtp' else 'analysis',
                 *([] if protocol=='gtp' else ['--stream','true','--protocol',protocol]),'--config',str(path),'--model',str(model),
                 '--model-id',info['id'],'--device',c['device'],'--seed',str(c['seed'])]
        code=subprocess.call(command)
        if code:
            raise RuntimeError(f'Analysis engine exited {code}')
