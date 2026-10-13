"""Fixed-model position analysis and the persistent interactive analysis engine."""
import json
from pathlib import Path
import subprocess
import uuid
from .config import fingerprint, write_native
from .schema import CONTRACT_ID
from .storage import load_json, save_json, sha256


def model_info(path,canvas=None):
    path=Path(path).resolve()
    info=load_json(path.parent/"manifest.json")
    if info["contract"]!=CONTRACT_ID or (canvas is not None and info["canvas"]!=canvas) or sha256(path)!=info["sha256"]:
        raise ValueError(f"Inference model contract/checksum mismatch: {path}")
    return path,info


def select_model(run_dir, model=None):
    root = Path(run_dir).resolve()
    return Path(model).resolve() if model is not None else root/load_json(root/'models/current.json')['model']['path']


def analyze(config,binary,run_dir,model=None,size=None,rule=None,moves="",output=None):
    from .runtime import verify_build
    binary_hash=verify_build(binary)
    root=Path(run_dir).resolve()
    model,a=model_info(select_model(root, model))
    from .engine_config import validate_engine_config
    import copy
    config=copy.deepcopy(config)
    c=config['analysis']
    if size is not None: c['board_size']=size
    if rule is not None: c['rule']=rule
    validate_engine_config(config)
    size,rule=c['board_size'],c['rule']
    if not 5<=size<=a['canvas']:
        raise ValueError('Analysis board size must be in [5, canvas]')
    # Public action indices use the actual board width; native uses its canvas.
    actions=[int(v) for v in moves.split(',')] if moves else []
    if any(v<0 or v>=size*size for v in actions):
        raise ValueError('Analysis moves must be board-row-major indices')
    encoded=','.join(str(v//size*a['canvas']+v%size) for v in actions)
    directory=root/'analysis'/uuid.uuid4().hex;directory.mkdir(parents=True)
    write_native({**config,'network':{'canvas':a['canvas']}},directory/'effective.cfg')
    command=[str(binary),'analysis','--config',str(directory/'effective.cfg'),'--model',str(model),
             '--model-id',a['id'],'--device',c['device'],'--size',str(size),'--rule',rule,
             '--moves',encoded,'--seed',str(c['seed'])]
    result=json.loads(subprocess.run(command,text=True,capture_output=True,check=True).stdout)
    if 'action' in result:
        if result['action'] >= 0:
            result['action']=result['action']//a['canvas']*size+result['action']%a['canvas']
        for key in ('raw_logits','policy','visits','network_policy','search_policy'):
            result[key]=[result[key][y*a['canvas']+x] for y in range(size) for x in range(size)]
    payload={'mode':'analysis','config_id':fingerprint(config),'config':config,'model_a':a,
             'board_size':size,'rule':rule,'opening_actions':actions,'seed':c['seed'],
             'command':command,'binary_sha256':binary_hash,'result':result}
    path=Path(output).resolve() if output else directory/'result.json'
    save_json(path,payload,immutable=True)
    return path,payload


def stream_analysis(config, binary, run_dir, model=None):
    """Run the same native session used by Web, retaining weights across queries."""
    import tempfile
    from .runtime import verify_build
    from .engine_config import validate_engine_config
    verify_build(binary)
    validate_engine_config(config)
    model,info=model_info(select_model(run_dir,model))
    c=config['analysis']
    with tempfile.TemporaryDirectory(prefix='etazero-analysis-') as temporary:
        path=Path(temporary)/'effective.cfg'
        write_native({**config,'network':{'canvas':info['canvas']}},path)
        command=[str(binary),'analysis','--stream','true','--config',str(path),'--model',str(model),
                 '--model-id',info['id'],'--device',c['device'],'--seed',str(c['seed'])]
        code=subprocess.call(command)
        if code:
            raise RuntimeError(f'Analysis engine exited {code}')
