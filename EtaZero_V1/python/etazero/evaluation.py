"""Fixed-model evaluation and balanced-color matches with inspectable source identities."""
import json
from pathlib import Path
import subprocess
import uuid
from .config import csv, fingerprint, write_native
from .schema import CONTRACT_ID
from .storage import load_json, save_json, sha256


def model_info(path,canvas=None):
    path=Path(path).resolve()
    info=load_json(path.parent/"manifest.json")
    if info["contract"]!=CONTRACT_ID or (canvas is not None and info["canvas"]!=canvas) or sha256(path)!=info["sha256"]:
        raise ValueError(f"Evaluation model contract/checksum mismatch: {path}")
    return path,info


def evaluate(config,binary,run_dir,model=None,model_b=None,size=None,rule=None,moves="",output=None,games=None):
    from .runtime import verify_build
    binary_hash=verify_build(binary)
    root=Path(run_dir).resolve()
    if model is None:
        model=root/load_json(root/"models/current.json")["model"]["path"]
    if model_b:
        from .arena import single_match
        import copy
        config=copy.deepcopy(config)
        if size is not None: config['match']['board_size']=size
        if rule is not None: config['match']['rule']=rule
        if moves: raise ValueError('Paired matches generate balanced openings; --moves is only for evaluate')
        path, games_result=single_match(config,binary,model,model_b,output or root/'evaluations'/uuid.uuid4().hex,games)
        return path, {'result': {'complete': True, 'games': games_result}}
    model,a=model_info(model)
    from .eval_config import validate_evaluation
    validate_evaluation(config)
    c=config['evaluation']
    size=c['board_size'] if size is None else size
    rule=rule or c['rule']
    if not 5<=size<=a['canvas']:
        raise ValueError('Evaluation board size must be in [5, canvas]')
    directory=root/'evaluations'/uuid.uuid4().hex;directory.mkdir(parents=True)
    native={**config,'network':{'canvas':a['canvas']}}
    write_native(native,directory/'effective.cfg')
    # Public action indices use the actual board width; native uses its canvas.
    actions=[int(v) for v in moves.split(',')] if moves else []
    if any(v<0 or v>=size*size for v in actions):
        raise ValueError('Evaluation moves must be board-row-major indices')
    encoded=','.join(str(v//size*a['canvas']+v%size) for v in actions)
    command=[str(binary),'evaluate','--config',str(directory/'effective.cfg'),'--model',str(model),
             '--model-id',a['id'],'--device',c['device'],'--size',str(size),'--rule',rule,
             '--moves',encoded,'--seed',str(c['seed'])]
    result=json.loads(subprocess.run(command,text=True,capture_output=True,check=True).stdout)
    if 'action' in result:
        if result['action'] >= 0:
            result['action']=result['action']//a['canvas']*size+result['action']%a['canvas']
        for key in ('raw_logits','policy','visits','network_policy','search_policy'):
            result[key]=[result[key][y*a['canvas']+x] for y in range(size) for x in range(size)]
    payload={'mode':'evaluate','config_id':fingerprint(config),'config':config,'model_a':a,
             'board_size':size,'rule':rule,'opening_actions':actions,'seed':c['seed'],
             'command':command,'binary_sha256':binary_hash,'result':result}
    path=Path(output).resolve() if output else directory/'result.json'
    save_json(path,payload,immutable=True)
    return path,payload
