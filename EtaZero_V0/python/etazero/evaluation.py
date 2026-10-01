"""Fixed-model evaluation and balanced-color matches with inspectable source identities."""
import json
from pathlib import Path
import subprocess
import uuid
from .config import csv, fingerprint, write_native
from .schema import CONTRACT_ID
from .storage import load_json, save_json, sha256


def model_info(path,canvas):
    path=Path(path).resolve()
    info=load_json(path.parent/"manifest.json")
    if info["contract"]!=CONTRACT_ID or info["canvas"]!=canvas or sha256(path)!=info["sha256"]:
        raise ValueError(f"Evaluation model contract/checksum mismatch: {path}")
    return path,info


def evaluate(config,binary,run_dir,model=None,model_b=None,size=None,rule=None,moves="",output=None,games=None):
    from .runtime import verify_build
    binary_hash=verify_build(binary)
    root=Path(run_dir).resolve()
    if model is None:
        model=root/load_json(root/"current_model.json")["model"]["path"]
    canvas=config["network"]["canvas"]
    model,a=model_info(model,canvas)
    size=csv(config["environment"]["sizes"],int)[0] if size is None else size
    rule=rule or csv(config["environment"]["rules"])[0]
    if not 5<=size<=canvas:
        raise ValueError("Evaluation board size must be in [5, canvas]")
    count=config["match"]["games"] if games is None else games
    if model_b and (count<2 or count%2):
        raise ValueError("Matches require a positive even game count")
    directory=root/"evaluations"/uuid.uuid4().hex;directory.mkdir(parents=True)
    write_native(config,directory/"effective.cfg")
    mode="match" if model_b else "evaluate"
    command=[str(binary),mode,"--config",str(directory/"effective.cfg"),"--model",str(model),
             "--model-id",a["id"],"--device",csv(config["devices"]["selfplay"])[0],"--size",str(size),
             "--rule",rule,"--moves",moves,"--seed",str(config["run"]["seed"])]
    b=None
    if model_b:
        model_b,b=model_info(model_b,canvas)
        command += ["--model-b",str(model_b),"--model-b-id",b["id"],"--games",str(count)]
    result=subprocess.run(command,text=True,capture_output=True,check=True)
    payload={"mode":mode,"config_id":fingerprint(config),"config":config,"model_a":a,"model_b":b,
             "board_size":size,"rule":rule,"opening_actions":moves,"seed":config["run"]["seed"],
             "command":command,"binary_sha256":binary_hash,"result":json.loads(result.stdout)}
    path=Path(output).resolve() if output else directory/"result.json"
    save_json(path,payload,immutable=True)
    return path,payload
