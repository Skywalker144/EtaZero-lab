"""Round controller: plan -> selfplay -> snapshot -> fixed training -> verified publication."""
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import tarfile
import threading
import time
import uuid
from .config import ROOT, csv, fingerprint, write_native, native_text
from .data import Catalog
from .export import export_model
from .shuffle import build_snapshot
from .storage import atomic_write, load_json, save_json, sha256
from .training import initialize, train_cycle


class Journal:
    def __init__(self, root):
        self.path = Path(root)/"events.jsonl"
        self.previous = 0
        if self.path.exists():
            with self.path.open("rb+") as file:
                valid_end = 0
                while line := file.readline():
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        if file.read(1):
                            raise ValueError("Corrupt journal before its trailing partial record")
                        file.truncate(valid_end)
                        break
                    valid_end = file.tell()
                    self.previous = max(self.previous,event["active_seconds"])
        self.start = time.monotonic(); self.lock = threading.Lock(); self.closed = threading.Event();self.failure=None
        self.thread = threading.Thread(target=self._heartbeat,daemon=True);self.thread.start()

    def active(self):
        return self.previous + time.monotonic()-self.start

    def __call__(self,event,**fields):
        payload={"event":event,"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 "active_seconds":self.active(),**fields}
        with self.lock, self.path.open("a") as file:
            file.write(json.dumps(payload,sort_keys=True)+"\n");file.flush();os.fsync(file.fileno())

    def _heartbeat(self):
        try:
            while not self.closed.wait(1):
                self("heartbeat")
        except BaseException as error:
            self.failure=error;self.closed.set()

    def close(self):
        self.closed.set();self.thread.join();self("session_end")


def source_snapshot(root):
    import io
    paths=[]
    for path in ROOT.rglob("*"):
        if path.is_file() and not path.is_relative_to(Path(root)) and not set(path.relative_to(ROOT).parts)&{"build","runs","__pycache__",".pytest_cache"}:
            paths.append(path)
    # Include repository-level rules, but never unrelated reference working trees.
    paths += [ROOT.parent/"AGENTS.md",ROOT.parent/"RULES.md"]
    import hashlib
    contents={str(p.relative_to(ROOT.parent)):p.read_bytes() for p in sorted(paths)}
    manifest={name:hashlib.sha256(data).hexdigest() for name,data in contents.items()}
    identity=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    destination=Path(root)/"source"/(identity+".tar.gz")
    if not destination.exists():
        def write(path):
            with tarfile.open(path,"w:gz") as archive:
                for name,data in contents.items():
                    entry=tarfile.TarInfo(name);entry.size=len(data)
                    archive.addfile(entry,io.BytesIO(data))
        atomic_write(destination,write,immutable=True)
        save_json(destination.with_suffix(".json"),{"id":identity,"files":manifest},immutable=True)
    return identity


def verify_build(binary):
    binary=Path(binary)
    manifest=load_json(binary.parent/"build_manifest.json")
    if not manifest["with_torch"] or sha256(binary)!=manifest["binary_sha256"]:
        raise ValueError("Native executable differs from its build manifest; rebuild before running")
    for path,checksum in manifest["sources"].items():
        if sha256(ROOT/path)!=checksum:
            raise ValueError(f"Native build is stale for {path}; run scripts/build.sh")
    return manifest["binary_sha256"]


def provenance():
    import torch
    import numpy
    revision=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT.parent,text=True,capture_output=True)
    status=subprocess.run(["git","status","--porcelain"],cwd=ROOT.parent,text=True,capture_output=True,check=True)
    return {"git_head":revision.stdout.strip() if revision.returncode==0 else None,
            "git_status":status.stdout,"python":platform.python_version(),"torch":torch.__version__,
            "numpy":numpy.__version__,"cuda":torch.version.cuda,"platform":platform.platform(),
            "gpus":[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}


class Controller:
    def __init__(self,root,config,binary,max_cycles=None):
        self.root,self.config,self.binary=Path(root),config,Path(binary)
        self.config_id=fingerprint(config);self.children=[];self.stop=False
        self.max_cycles=config["run"]["max_cycles"] if max_cycles is None else max_cycles
        self.limit=config["run"]["max_seconds"]
        self.journal=Journal(root)
        self.previous_handlers={}
        for sig in (signal.SIGINT,signal.SIGTERM):
            self.previous_handlers[sig]=signal.signal(sig,self._signal)
        self.catalog=None

    def _signal(self,*_):
        self.stop=True
        for process in self.children:
            self.signal_child(process,signal.SIGINT)

    @staticmethod
    def signal_child(process,sig):
        if process.poll() is None:
            try:
                os.killpg(process.pid,sig)
            except ProcessLookupError:
                pass # The owned child completed between poll and signal.

    def stopping(self):
        if self.journal.failure:
            raise RuntimeError("Run journal writer failed") from self.journal.failure
        if self.limit and self.journal.active()>=self.limit and not self.stop:
            self._signal()
        return self.stop

    def close(self):
        if self.catalog:
            self.catalog.close()
        self.journal.close()
        for sig,handler in self.previous_handlers.items():
            signal.signal(sig,handler)

    def state(self,state):
        save_json(self.root/"state.json",state)

    def run(self,resume=False,weights=None):
        root,c=self.root,self.config
        import torch
        torch.set_num_threads(c["run"]["cpu_threads"])
        source=source_snapshot(root)
        self.source_id=source
        save_json(root/"session.json",{"source_id":source,"started_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()})
        self.journal("session_start",source_id=source,binary_sha256=sha256(self.binary),**provenance())
        if resume:
            info=load_json(root/"run.json")
            if info["config_id"]!=self.config_id or info["config"]!=c:
                raise ValueError("Resume configuration differs from saved effective configuration")
            state=load_json(root/"state.json") if (root/"state.json").exists() else {
                "run_id":info["id"],"cycle":1,"target_rows":0.0,"checkpoint":None,"model":None}
            if not (root/"effective.json").exists():
                save_json(root/"effective.json",c,immutable=True)
            if load_json(root/"effective.json")!=c:
                raise ValueError("Saved effective configuration is inconsistent")
            if not (root/"effective.cfg").exists():
                write_native(c,root/"effective.cfg")
            if (root/"effective.cfg").read_text()!=native_text(c):
                raise ValueError("Resolved native configuration differs from the saved effective configuration")
        else:
            if (root/"run.json").exists():
                raise ValueError("Run already exists; use --resume or a new directory")
            info={"id":uuid.uuid4().hex,"config_id":self.config_id,"config":c,"source_id":source,
                  "seed":c["run"]["seed"],"weights_initialization":{
                      "path":str(Path(weights).resolve()),"sha256":sha256(weights)} if weights else None}
            save_json(root/"run.json",info,immutable=True)
            save_json(root/"effective.json",c,immutable=True);write_native(c,root/"effective.cfg")
            state={"run_id":info["id"],"cycle":1,"target_rows":0.0,"checkpoint":None,"model":None}
            self.state(state)
        if state["run_id"]!=info["id"]:
            raise ValueError("Run state identity mismatch")
        self.run_id=info["id"]
        for name in ("data","cycles","logs","models","snapshots","checkpoints"):
            (root/name).mkdir(exist_ok=True)
        if state["checkpoint"] is None:
            origin=info["weights_initialization"]
            if origin and sha256(origin["path"])!=origin["sha256"]:
                raise ValueError("Weights initialization source changed")
            state["checkpoint"]=initialize(root,c,origin["path"] if origin else None);self.state(state)
        if state["model"] is None:
            state["model"]=export_model(root,c,state["checkpoint"],self.binary)
            self.publish(state["model"]);self.state(state)
        self.catalog=Catalog(root,info["id"],self.config_id)
        while not self.stopping() and (not self.max_cycles or state["cycle"]<=self.max_cycles):
            cycle=state["cycle"];directory=root/"cycles"/f"{cycle:06d}";directory.mkdir(exist_ok=True)
            plan_path,status_path=directory/"plan.json",directory/"status.json"
            if plan_path.exists():
                plan=load_json(plan_path)
                if plan["input_model"]!=state["model"] or plan["input_checkpoint"]!=state["checkpoint"]:
                    raise ValueError("Cycle plan differs from its parent committed state")
            else:
                training=c["training"]
                plan={"cycle":cycle,"train_steps":training["train_steps"],"batch_size":training["batch_size"],
                      "replay_ratio":c["replay"]["replay_ratio"],"target_rows":state["target_rows"]+
                      training["train_steps"]*training["batch_size"]/c["replay"]["replay_ratio"],
                      "input_model":state["model"],"input_checkpoint":state["checkpoint"]}
                save_json(plan_path,plan,immutable=True);self.journal("plan",**plan)
            status=load_json(status_path) if status_path.exists() else {"phase":"selfplay"}
            self.scan()
            if status["phase"]=="selfplay":
                self.phase("selfplay",cycle,lambda:self.produce(plan))
                if self.stopping():
                    break
                status={"phase":"shuffle"};save_json(status_path,status)
            if status["phase"]=="shuffle":
                snapshot=self.phase("shuffle",cycle,lambda:build_snapshot(root,cycle,self.catalog.entries(),c))
                status={"phase":"train","snapshot_id":snapshot};save_json(status_path,status)
            if self.stopping():
                break
            if status["phase"]=="train":
                train_plan={**plan,"snapshot_id":status["snapshot_id"]}
                checkpoint,done=self.phase("train",cycle,lambda:train_cycle(root,c,train_plan,plan["input_checkpoint"],self.journal,self.stopping))
                if not done:
                    break
                status={**status,"phase":"export","checkpoint":checkpoint};save_json(status_path,status)
            if self.stopping():
                break
            if status["phase"]=="export":
                model=self.phase("export",cycle,lambda:export_model(root,c,status["checkpoint"],self.binary))
                self.publish(model)
                status={**status,"phase":"completed","model":model};save_json(status_path,status)
            if status["phase"]!="completed":
                raise ValueError(f'Unknown persisted phase: {status["phase"]}')
            # Assign from the immutable plan; repeated recovery cannot add the target twice.
            state={"run_id":self.run_id,"cycle":cycle+1,"target_rows":plan["target_rows"],
                   "checkpoint":status["checkpoint"],"model":status["model"]}
            self.state(state)
            rows,games,_=self.catalog.counts(c["selfplay"]["recent_games"])
            self.journal("cycle_complete",cycle=cycle,unique_rows=rows,games=games,target_rows=plan["target_rows"],
                         total_samples=state["checkpoint"]["total_samples"],
                         replay_ratio=state["checkpoint"]["total_samples"]/rows,model_id=state["model"]["id"])
            print(f'cycle={cycle} complete rows={rows} trained_samples={state["checkpoint"]["total_samples"]}',flush=True)
        self.journal("run_stopped" if self.stop else "run_complete",completed_cycles=state["cycle"]-1)
        return state

    def phase(self,name,cycle,function):
        self.journal("phase_start",phase=name,cycle=cycle);start=time.monotonic()
        value=function()
        self.journal("phase_end",phase=name,cycle=cycle,seconds=time.monotonic()-start)
        return value

    def publish(self,model):
        path=self.root/"current_model.json"
        current=load_json(path) if path.exists() else None
        if current and current["model"]["id"]==model["id"]:
            if current["model"]!=model:
                raise ValueError("Published model identity collision")
            return
        publication={"model":model,"active_seconds":self.journal.active(),
                     "utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
        save_json(path,publication)
        self.journal("model_published",**publication)

    def scan(self):
        models={}
        for path in (self.root/"cycles").glob("*/plan.json"):
            plan=load_json(path);models[plan["cycle"]]=plan["input_model"]["id"]
        self.catalog.scan(models)

    def produce(self,plan):
        c=self.config
        rows,_,average=self.catalog.counts(c["selfplay"]["recent_games"])
        if average is None:
            self.launch(plan,c["selfplay"]["probe_games"]);self.scan()
            rows,_,average=self.catalog.counts(c["selfplay"]["recent_games"])
            if self.stopping():
                return
            if average is None:
                raise RuntimeError("Probe games produced no valid training rows")
        deficit=max(0,plan["target_rows"]-rows,c["replay"]["min_rows"]-rows)
        games=math.ceil(deficit/average)
        self.journal("selfplay_target",cycle=plan["cycle"],rows=rows,target=plan["target_rows"],deficit=deficit,
                     rows_per_game=average,planned_games=games)
        if games and not self.stopping():
            self.launch(plan,games);self.scan()
        while not self.stopping():
            new_rows,_,average=self.catalog.counts(c["selfplay"]["recent_games"])
            if new_rows>=c["replay"]["min_rows"]:
                break
            if new_rows<=rows:
                raise RuntimeError("Selfplay failed to produce enough valid data")
            rows=new_rows;self.launch(plan,math.ceil((c["replay"]["min_rows"]-rows)/average));self.scan()

    def launch(self,plan,games):
        c=self.config;cycle=plan["cycle"];attempt=uuid.uuid4().hex
        attempt_index=len(list((self.root/"cycles"/f"{cycle:06d}").glob("attempt_*.json")))
        devices=csv(c["devices"]["selfplay"]);workers=[]
        model=plan["input_model"]
        if sha256(self.root/model["path"])!=model["sha256"]:
            raise ValueError("Selfplay input model checksum mismatch")
        for worker,device in enumerate(devices):
            quota=games//len(devices)+(worker<games%len(devices))
            if quota:
                import hashlib
                namespace=f'{c["run"]["seed"]}:{cycle}:{attempt_index}:{worker}'.encode()
                seed=int.from_bytes(hashlib.sha256(namespace).digest()[:8],"little")
                workers.append({"id":worker,"device":device,"games":quota,"seed":seed})
        save_json(self.root/"cycles"/f"{cycle:06d}"/("attempt_"+attempt+".json"),
                  {"id":attempt,"index":attempt_index,"cycle":cycle,"workers":workers,
                   "model_id":plan["input_model"]["id"],"source_id":self.source_id},immutable=True)
        streams=[];self.children=[]
        try:
            for worker in workers:
                if self.stopping():
                    break
                output=self.root/"data"/f"cycle_{cycle:06d}"/f'{attempt}_worker_{worker["id"]}'
                stdout=self.root/"logs"/f'{attempt}_{worker["id"]}.jsonl'
                stderr=self.root/"logs"/f'{attempt}_{worker["id"]}.stderr'
                out,err=stdout.open("w"),stderr.open("w");streams.extend([out,err])
                command=[str(self.binary),"selfplay","--config",str(self.root/"effective.cfg"),
                         "--model",str(self.root/plan["input_model"]["path"]),"--model-id",plan["input_model"]["id"],
                         "--device",worker["device"],"--games",str(worker["games"]),"--output",str(output),
                         "--run-id",self.run_id,"--attempt-id",attempt,"--config-id",self.config_id,"--source-id",self.source_id,
                         "--cycle",str(cycle),"--worker",str(worker["id"]),"--seed",str(worker["seed"])]
                process=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True)
                self.children.append(process)
                if self.stop:
                    self.signal_child(process,signal.SIGINT)
                self.journal("worker_start",cycle=cycle,worker=worker,attempt=attempt,command=command,stdout=str(stdout),stderr=str(stderr))
            failed=False
            while any(p.poll() is None for p in self.children):
                self.stopping()
                if any(p.poll() not in (None,0,2) for p in self.children) and not failed:
                    failed=True
                    for p in self.children:
                        self.signal_child(p,signal.SIGTERM)
                time.sleep(0.05)
            codes=[p.returncode for p in self.children]
            for stream in streams:
                stream.flush()
            for worker in workers:
                stdout=self.root/"logs"/f'{attempt}_{worker["id"]}.jsonl'
                if not stdout.exists():
                    continue
                lines=stdout.read_text().splitlines()
                for i,line in enumerate(lines):
                    try:
                        record=json.loads(line)
                    except json.JSONDecodeError:
                        if i==len(lines)-1 and any(code!=0 for code in codes):
                            self.journal("partial_worker_log",file=str(stdout));continue
                        raise
                    if record["event"]=="inference":
                        self.journal("inference",cycle=cycle,worker_id=worker["id"],attempt=attempt,
                                     **{k:v for k,v in record.items() if k!="event"})
            if any(code!=0 for code in codes) and not self.stop:
                raise RuntimeError(f"Selfplay worker failed (codes={codes}); inspect {self.root/'logs'}")
        finally:
            for process in self.children:
                if process.poll() is None:
                    self.signal_child(process,signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self.signal_child(process,signal.SIGKILL);process.wait()
            self.children=[]
            for stream in streams:
                stream.close()


def run_training(root,config,binary,resume=False,weights=None,max_cycles=None):
    root=Path(root).resolve()
    root.mkdir(parents=True,exist_ok=True)
    with (root/"run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another controller already owns this run") from error
        if not resume and any(p.name!="run.lock" for p in root.iterdir()):
            raise ValueError("New runs require an empty output directory")
        if resume and weights:
            raise ValueError("Weights initialization and resume are separate operations")
        if not Path(binary).is_file():
            raise FileNotFoundError("Native executable is missing; run scripts/build.sh first")
        verify_build(binary)
        controller=Controller(root,config,binary,max_cycles)
        try:
            state=controller.run(resume,weights)
        except BaseException as error:
            controller.journal("failure",error=repr(error));raise
        finally:
            controller.close()
        return state
