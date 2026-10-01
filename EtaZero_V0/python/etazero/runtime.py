"""Round controller: plan -> selfplay -> snapshot -> fixed training -> verified publication."""
import datetime
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import math
import multiprocessing
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
from .native import NativeWorker
from .export import export_model
from .shuffle import build_snapshot, desired_window, prune_derived
from .storage import atomic_write, load_json, save_json, sha256, sync_directory
from .training import initialize, train_iteration


class Journal:
    def __init__(self, root):
        self.path = Path(root)/"logs/events.jsonl"
        self.path.parent.mkdir(parents=True,exist_ok=True)
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
        self.start = time.monotonic()
        self.changed = threading.Condition()
        self.queue = deque()
        self.issued = self.written = self.durable = self.barrier = 0
        self.closing = False;self.failure=None
        self.thread = threading.Thread(target=self._write_loop,daemon=True);self.thread.start()

    def active(self):
        return self.previous + time.monotonic()-self.start

    def __call__(self,event,**fields):
        payload={"event":event,"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 "active_seconds":self.active(),**fields}
        line = json.dumps(payload,sort_keys=True)+'\n'
        with self.changed:
            self.changed.wait_for(lambda:len(self.queue)<1024 or self.failure or self.closing)
            self._check()
            if self.closing:
                raise RuntimeError('Run journal is closing')
            self.issued += 1
            self.queue.append((self.issued,line))
            self.changed.notify_all()

    def _check(self):
        if self.failure:
            raise RuntimeError('Run journal writer failed') from self.failure

    def flush(self):
        """Durable barrier for every preceding event, especially checkpoint metrics."""
        with self.changed:
            self._check();target=self.issued;self.barrier=max(self.barrier,target)
            self.changed.notify_all()
            self.changed.wait_for(lambda:self.durable>=target or self.failure)
            self._check()

    def _write_loop(self):
        try:
            last_sync = time.monotonic()
            with self.path.open('a') as file:
                while True:
                    with self.changed:
                        self.changed.wait_for(lambda:self.queue or self.closing or self.barrier>self.durable,
                                              timeout=max(0,1-(time.monotonic()-last_sync)))
                        items=list(self.queue);self.queue.clear();self.changed.notify_all()
                        closing=self.closing
                    if items:
                        file.write(''.join(line for _,line in items));file.flush()
                        self.written=items[-1][0]
                    now=time.monotonic()
                    if now-last_sync>=1 or closing or self.barrier>self.durable:
                        if now-last_sync>=1 and not closing:
                            heartbeat={'event':'heartbeat','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                       'active_seconds':self.active()}
                            file.write(json.dumps(heartbeat,sort_keys=True)+'\n')
                        file.flush();os.fsync(file.fileno());last_sync=time.monotonic()
                        with self.changed:
                            self.durable=self.written;self.changed.notify_all()
                    if closing:
                        return
        except BaseException as error:
            with self.changed:
                self.failure=error;self.changed.notify_all()

    def close(self):
        if self.closing:
            self.thread.join();self._check();return
        self('session_end')
        with self.changed:
            self.closing=True;self.changed.notify_all()
        self.thread.join();self._check()


def source_snapshot(root):
    import io
    paths=[]
    for directory,names,files in os.walk(ROOT):
        parent=Path(directory)
        names[:]=[name for name in names
                  if name not in {"build","runs","__pycache__",".pytest_cache"}
                  and not (parent==ROOT and name=="data")
                  and not (parent/name).is_relative_to(Path(root))]
        paths.extend(parent/name for name in files if (parent/name).is_file())
    # Include repository-level rules, but never unrelated reference working trees.
    paths += [ROOT.parent/"AGENTS.md",ROOT.parent/"RULES.md"]
    import hashlib
    contents={str(p.relative_to(ROOT.parent)):p.read_bytes() for p in sorted(paths)}
    manifest={name:hashlib.sha256(data).hexdigest() for name,data in contents.items()}
    identity=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    destination=Path(root)/".internal/source"/(identity+".tar.gz")
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


def iteration_plan(state, config):
    """Cold-start surplus is excluded from the cumulative steady-state quota."""
    iteration = state['iteration']; training = config['training']
    model = state['model'] or {'id': f"random:{config['run']['seed']}", 'evaluator': 'random'}
    target = (0 if iteration == 0 else config['replay']['min_rows'] if iteration == 1 else
              state['target_rows'] + training['train_steps']*training['batch_size']/training['replay_ratio'])
    return {'iteration': iteration, 'train_steps': 0 if iteration == 0 else training['train_steps'],
            'batch_size': training['batch_size'], 'replay_ratio': training['replay_ratio'],
            'target_rows': target, 'input_model': model, 'input_checkpoint': state['checkpoint']}


class Controller:
    def __init__(self,root,config,binary,max_iteration=None,max_seconds=None):
        self.root,self.config,self.binary=Path(root),config,Path(binary)
        self.config_id=fingerprint(config);self.children=[];self.stop=False
        self.max_iteration=config["run"]["max_iteration"] if max_iteration is None else max_iteration
        self.limit=config["run"]["max_seconds"] if max_seconds is None else max_seconds
        self.journal=Journal(root)
        self.previous_handlers={}
        for sig in (signal.SIGINT,signal.SIGTERM):
            self.previous_handlers[sig]=signal.signal(sig,self._signal)
        self.catalog=None;self.shuffle_pool=None;self.models={};self.scanned=False
        self.hot_snapshots=deque()
        self.services={}

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
        return self.stop

    def close(self):
        errors=[]
        callbacks=[service.close for service in self.services.values()]
        if self.shuffle_pool:
            callbacks.append(lambda:self.shuffle_pool.shutdown(wait=True,cancel_futures=True))
        if self.catalog:
            callbacks.append(self.catalog.close)
        callbacks.append(self.journal.close)
        for callback in callbacks:
            try:
                callback()
            except BaseException as error:
                errors.append(error)
        for sig,handler in self.previous_handlers.items():
            signal.signal(sig,handler)
        if errors:
            raise errors[0]

    def state(self,state):
        self.journal.flush()
        save_json(self.root/".internal/state.json",state)

    def run(self,resume=False,weights=None):
        root,c=self.root,self.config
        import torch
        torch.set_num_threads(c["run"]["cpu_threads"])
        source=source_snapshot(root)
        self.source_id=source
        save_json(root/".internal/session.json",{"source_id":source,"started_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()})
        self.journal("session_start",source_id=source,binary_sha256=sha256(self.binary),
                     iteration_limit=self.max_iteration,active_seconds_limit=self.limit,**provenance())
        if resume:
            info=load_json(root/".internal/run.json")
            if info["config_id"]!=self.config_id or info["config"]!=c:
                raise ValueError("Resume configuration differs from saved effective configuration")
            state=load_json(root/".internal/state.json") if (root/".internal/state.json").exists() else {
                "run_id":info["id"],"iteration":0,"target_rows":0.0,"replay_origin_rows":None,"checkpoint":None,"model":None,"elapsed_seconds":0.0}
            if not (root/"config/effective.json").exists():
                save_json(root/"config/effective.json",c,immutable=True)
            if load_json(root/"config/effective.json")!=c:
                raise ValueError("Saved effective configuration is inconsistent")
            if not (root/"config/effective.cfg").exists():
                write_native(c,root/"config/effective.cfg")
            if (root/"config/effective.cfg").read_text()!=native_text(c):
                raise ValueError("Resolved native configuration differs from the saved effective configuration")
        else:
            if (root/".internal/run.json").exists():
                raise ValueError("Run already exists; use --resume or a new directory")
            info={"id":uuid.uuid4().hex,"config_id":self.config_id,"config":c,"source_id":source,
                  "seed":c["run"]["seed"],"weights_initialization":{
                      "path":str(Path(weights).resolve()),"sha256":sha256(weights)} if weights else None}
            save_json(root/".internal/run.json",info,immutable=True)
            save_json(root/"config/effective.json",c,immutable=True);write_native(c,root/"config/effective.cfg")
            state={"run_id":info["id"],"iteration":0,"target_rows":0.0,"replay_origin_rows":None,"checkpoint":None,"model":None,"elapsed_seconds":0.0}
            self.state(state)
        if state["run_id"]!=info["id"]:
            raise ValueError("Run state identity mismatch")
        if 'elapsed_seconds' not in state:
            raise ValueError('Run predates whole-iteration commits; use a new run directory')
        recover_iteration(root, state)
        self.run_id=info["id"]
        for name in ("selfplay",".internal/iterations","logs","models","snapshots","checkpoints"):
            (root/name).mkdir(parents=True,exist_ok=True)
        if state["checkpoint"] is None:
            origin=info["weights_initialization"]
            if origin and sha256(origin["path"])!=origin["sha256"]:
                raise ValueError("Weights initialization source changed")
            state["checkpoint"]=initialize(root,c,origin["path"] if origin else None);self.state(state)
        self.catalog=Catalog(root,info["id"],self.config_id)
        plotted=False
        while not self.stopping() and (not self.max_iteration or state["iteration"]<=self.max_iteration) and (not self.limit or state["elapsed_seconds"] < self.limit):
            plotted=False
            started=time.monotonic()
            iteration=state["iteration"];directory=root/".internal/iterations"/f"{iteration:06d}";directory.mkdir(exist_ok=True)
            plan_path,status_path=directory/"plan.json",directory/"status.json"
            if plan_path.exists():
                plan=load_json(plan_path)
                if plan != iteration_plan(state,c):
                    raise ValueError("Iteration plan differs from its parent committed state")
            else:
                plan=iteration_plan(state,c)
                save_json(plan_path,plan,immutable=True);self.journal("plan",**plan)
            status=load_json(status_path) if status_path.exists() else {"phase":"selfplay"}
            self.scan(plan)
            if status["phase"]=="selfplay":
                self.phase("selfplay",iteration,lambda:self.produce(plan))
                if self.stopping():
                    break
                if iteration == 0:
                    status={"phase":"completed","checkpoint":state["checkpoint"],"model":None}
                else:
                    status={"phase":"shuffle"}
                save_json(status_path,status)
            if status["phase"]=="shuffle":
                snapshot=self.phase('shuffle',iteration,self.snapshot)
                self.journal("shuffle_resources",iteration=iteration,**load_json(root/"snapshots"/snapshot/"manifest.json")["resource_plan"])
                status={"phase":"train","snapshot_id":snapshot};save_json(status_path,status)
            if self.stopping():
                break
            if status["phase"]=="train":
                train_plan={**plan,"snapshot_id":status["snapshot_id"]}
                checkpoint,done=self.phase("train",iteration,lambda:train_iteration(root,c,train_plan,plan["input_checkpoint"],self.journal,self.stopping))
                if not done:
                    break
                status={**status,"phase":"export","checkpoint":checkpoint};save_json(status_path,status)
            if self.stopping():
                break
            if status["phase"]=="export":
                model=self.phase("export",iteration,lambda:export_model(root,c,status["checkpoint"],self.binary))
                status={**status,"phase":"completed","model":model};save_json(status_path,status)
            if status["phase"]!="completed":
                raise ValueError(f'Unknown persisted phase: {status["phase"]}')
            # Assign from the immutable plan; repeated recovery cannot add the target twice.
            rows,games,_=self.catalog.counts(c["selfplay"]["recent_games"])
            origin = rows if iteration == 1 else state['replay_origin_rows']
            next_state={"run_id":self.run_id,"iteration":iteration+1,
                   "target_rows":rows if iteration == 1 else plan["target_rows"], "replay_origin_rows":origin,
                   "checkpoint":status["checkpoint"],"model":status["model"],"elapsed_seconds":state["elapsed_seconds"]}
            self.journal("selfplay_statistics",iteration=iteration,**self.catalog.statistics(iteration))
            self.journal("iteration_complete",iteration=iteration,unique_rows=rows,games=games,target_rows=next_state["target_rows"],
                         replay_origin_rows=origin,total_samples=next_state["checkpoint"]["total_samples"],
                         replay_ratio=next_state["checkpoint"]["total_samples"]/rows if rows else 0,
                         model_id=next_state["model"]["id"] if next_state["model"] else None)
            if iteration > 0:
                if status['snapshot_id'] not in self.hot_snapshots:
                    self.hot_snapshots.append(status['snapshot_id'])
                evicted=[]
                while len(self.hot_snapshots)>c['shuffle']['snapshot_keep']:
                    evicted.append(self.hot_snapshots.popleft())
                manifest=load_json(root/'snapshots'/status['snapshot_id']/'manifest.json')
                prune_derived(root,evicted,{e['sha256'] for e in manifest['sources']})
                if evicted:
                    self.journal('snapshots_evicted',snapshots=evicted)
            # Prepare figures before the atomic commit, as in MuZero V2. A
            # stopped/failed round has no committed metrics, time or model.
            self.plot(next_state)
            if self.stopping():
                break
            from .plotting import run_history
            metrics=run_history(root, next_state)[-1]
            next_state['elapsed_seconds']=state['elapsed_seconds']+time.monotonic()-started
            metrics.update(elapsed_seconds=next_state['elapsed_seconds'], model=next_state['model'])
            save_json(root/'logs/iterations'/f'{iteration:06d}.json',metrics,immutable=True)
            self.state(next_state)
            state=next_state
            if state['model']:
                self.publish(state['model'],state['elapsed_seconds'])
            plotted=True
            print(f'iteration={iteration} complete rows={rows} trained_samples={state["checkpoint"]["total_samples"]}',flush=True)
        if not plotted or self.stop:
            self.plot()
        self.journal("run_stopped" if self.stop else "run_complete",completed_iterations=state["iteration"]-1)
        return state

    def plot(self,state=None):
        from .plotting import plot_run
        self.journal.flush()
        plot_run(self.root,state)

    def phase(self,name,iteration,function):
        self.journal("phase_start",phase=name,iteration=iteration);start=time.monotonic()
        value=function()
        self.journal("phase_end",phase=name,iteration=iteration,seconds=time.monotonic()-start)
        return value

    def publish(self,model,elapsed_seconds):
        path=self.root/"models/current.json"
        current=load_json(path) if path.exists() else None
        if current and current["model"]["id"]==model["id"]:
            if current["model"]!=model:
                raise ValueError("Published model identity collision")
            return
        publication={"model":model,"elapsed_seconds":elapsed_seconds,
                     "utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
        save_json(path,publication)
        self.journal("model_published",**publication)

    def scan(self,plan=None):
        if plan:
            self.models[plan['iteration']]=plan['input_model']['id']
            self.current_iteration=plan['iteration']
        if not self.scanned:
            for path in sorted((self.root/'.internal/iterations').glob('*/plan.json')):
                saved=load_json(path);self.models[saved['iteration']]=saved['input_model']['id']
                status_path=path.parent/'status.json'
                if status_path.exists():
                    status=load_json(status_path)
                    if status['phase']=='completed' and 'snapshot_id' in status and (self.root/'snapshots'/status['snapshot_id']/'data').is_dir():
                        self.hot_snapshots.append(status['snapshot_id'])
            self.catalog.scan(self.models);self.scanned=True
        else:
            self.catalog.scan(self.models,[self.root/'selfplay'/f'iteration_{self.current_iteration:06d}'])

    def snapshot(self):
        if self.shuffle_pool is None:
            self.shuffle_pool=ProcessPoolExecutor(max_workers=self.config['shuffle']['workers'],
                                                 mp_context=multiprocessing.get_context('spawn'))
        total,_,_=self.catalog.counts(self.config['selfplay']['recent_games'])
        window=desired_window(total,self.config['replay'])
        return build_snapshot(self.root,self.current_iteration,self.catalog.entries(window),self.config,
                              pool=self.shuffle_pool,total_rows=total)

    def produce(self,plan):
        c=self.config;iteration=plan['iteration']
        if iteration == 0:
            _,completed=self.catalog.iteration_counts(0)
            remaining=c['selfplay']['bootstrap_games']-completed
            if remaining > 0 and not self.stopping():
                self.launch(plan,remaining);self.scan()
            rows,games,average=self.catalog.counts(c['selfplay']['recent_games'])
            if not self.stopping() and (games != c['selfplay']['bootstrap_games'] or not average):
                raise RuntimeError('Bootstrap did not produce the configured games and valid training rows')
        else:
            rows,_,average=self.catalog.counts(c['selfplay']['recent_games'])
            if not average:
                raise RuntimeError('Cold start produced no valid rows-per-game estimate')
            deficit=max(0,plan['target_rows']-rows)
            games=math.ceil(deficit/average)
            # A large cold-start surplus never pays for steady-state selfplay.
            # Normal whole-game overshoot is carried forward, with at least one
            # completed game in each subsequent iteration (as in MuZero V2).
            if iteration >= 2 and self.catalog.iteration_counts(iteration)[1] == 0:
                games=max(1,games)
            self.journal('selfplay_target',iteration=iteration,rows=rows,target=plan['target_rows'],deficit=deficit,
                         rows_per_game=average,planned_games=games)
            if games and not self.stopping():
                self.launch(plan,games);self.scan()
            while not self.stopping():
                new_rows,_,average=self.catalog.counts(c['selfplay']['recent_games'])
                if new_rows >= c['replay']['min_rows']:
                    break
                if new_rows <= rows or not average:
                    raise RuntimeError('Selfplay failed to produce enough valid training rows')
                rows=new_rows
                self.launch(plan,math.ceil((c['replay']['min_rows']-rows)/average));self.scan()
        if not self.stop:
            for service in self.services.values():
                service.release()

    def launch(self,plan,games):
        c=self.config;iteration=plan["iteration"];attempt=uuid.uuid4().hex
        attempt_index=len(list((self.root/".internal/iterations"/f"{iteration:06d}").glob("attempt_*.json")))
        devices=csv(c["devices"]["selfplay"]);workers=[]
        model=plan["input_model"]
        kind=model.get('evaluator','network')
        model_path=str(self.root/model['path']) if kind=='network' else ''
        if kind=='network' and sha256(model_path)!=model['sha256']:
            raise ValueError("Selfplay input model checksum mismatch")
        for worker,device in enumerate(devices):
            quota=games//len(devices)+(worker<games%len(devices))
            if quota:
                import hashlib
                namespace=f'{c["run"]["seed"]}:{iteration}:{attempt_index}:{worker}'.encode()
                seed=int.from_bytes(hashlib.sha256(namespace).digest()[:8],"little")
                workers.append({"id":worker,"device":device,"games":quota,"seed":seed})
        save_json(self.root/".internal/iterations"/f"{iteration:06d}"/("attempt_"+attempt+".json"),
                  {"id":attempt,"index":attempt_index,"iteration":iteration,"workers":workers,
                   "model_id":plan["input_model"]["id"],"source_id":self.source_id},immutable=True)
        pending={};codes=[];finished=False;self.children=[]
        try:
            for worker in workers:
                if self.stopping():
                    break
                output=self.root/"selfplay"/f"iteration_{iteration:06d}"/f'{attempt}_worker_{worker["id"]}'
                stdout=self.root/"logs"/f'{attempt}_{worker["id"]}.jsonl'
                command=[str(self.binary),"selfplay","--config",str(self.root/"config/effective.cfg"),
                         "--model",model_path,"--model-id",model["id"],"--evaluator",kind,
                         "--device",worker["device"],"--games",str(worker["games"]),"--output",str(output),
                         "--run-id",self.run_id,"--attempt-id",attempt,"--config-id",self.config_id,"--source-id",self.source_id,
                         "--iteration",str(iteration),"--worker",str(worker["id"]),"--seed",str(worker["seed"])]
                if worker['id'] not in self.services:
                    process_command=command.copy();process_command[1]='worker'
                    stderr=self.root/'logs'/f'service_{uuid.uuid4().hex}_{worker["id"]}.stderr'
                    self.services[worker['id']]=NativeWorker(process_command,stderr)
                service=self.services[worker['id']];process=service.process
                service.submit([model_path,model['id'],worker['games'],str(output),attempt,iteration,worker['seed'],kind],stdout)
                pending[worker['id']]=service
                self.children.append(process)
                if self.stop:
                    self.signal_child(process,signal.SIGINT)
                self.journal('worker_start',iteration=iteration,worker=worker,attempt=attempt,command=command,
                             process_command=service.command,pid=process.pid,stdout=str(stdout),stderr=str(service.stderr_path))
            while pending:
                self.stopping()
                for identity,service in list(pending.items()):
                    result=service.take()
                    if result is not None:
                        if result['event']!='worker_complete':
                            raise RuntimeError('Unexpected native selfplay response')
                        codes.append(result['code']);del pending[identity]
                if pending:
                    time.sleep(0.005)
            finished=True
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
                        self.journal("inference",iteration=iteration,worker_id=worker["id"],attempt=attempt,evaluator=kind,
                                     **{k:v for k,v in record.items() if k!="event"})
            if any(code!=0 for code in codes) and not self.stop:
                raise RuntimeError(f"Selfplay worker failed (codes={codes}); inspect {self.root/'logs'}")
        finally:
            if not finished:
                for process in self.children:
                    self.signal_child(process,signal.SIGTERM)
            self.children=[]


def recover_iteration(root, state):
    """Archive uncommitted products; the state pointer is the only commit authority.

    Renames are repeatable after a crash. Raw games/checkpoints are retained under
    discarded/, never silently reused or overwritten by the next attempt.
    """
    root=Path(root); iteration=state['iteration']
    archive=root/'.internal/discarded'/uuid.uuid4().hex
    paths=[]
    for folder, pattern in (('.internal/iterations', '*'), ('selfplay','iteration_*'),
                             ('snapshots','*iteration_*'), ('checkpoints','iteration_*'),
                             ('models','iteration_*'), ('logs/iterations','*.json')):
        for path in (root/folder).glob(pattern):
            import re
            match=re.search(r'iteration_(\d+)',path.name) if 'iteration_' in path.name else re.match(r'(\d+)',path.name)
            if not match or int(match[1]) < iteration:
                continue
            if state['checkpoint'] and path in (root/state['checkpoint']['path'], (root/state['checkpoint']['path']).with_suffix('.json')):
                continue
            paths.append(path)
    if paths:
        # Catalog is derived and may include games from the discarded round.
        paths=list((root/'.internal').glob('catalog.sqlite*'))+paths
        for path in paths:
            target=archive/path.relative_to(root)
            target.parent.mkdir(parents=True,exist_ok=True)
            path.rename(target)
            sync_directory(path.parent);sync_directory(target.parent)
        save_json(archive/'rollback.json',{'next_iteration':iteration,'elapsed_seconds':state['elapsed_seconds']})
    current=root/'models/current.json'
    if state['model']:
        save_json(current,{'model':state['model'],'elapsed_seconds':state['elapsed_seconds']})
    else:
        current.unlink(missing_ok=True)


def run_training(root,config,binary,resume=None,weights=None,max_iteration=None,max_seconds=None):
    root=Path(root).resolve()
    root.mkdir(parents=True,exist_ok=True)
    (root/'.internal').mkdir(exist_ok=True)
    with (root/".internal/run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another controller already owns this run") from error
        if resume is None:
            resume=(root/".internal/run.json").is_file()
        if resume and not (root/".internal/run.json").is_file():
            raise ValueError("Resume requires an existing run.json")
        if not resume and (any(p.name!='.internal' for p in root.iterdir()) or
                           any(p.name!='run.lock' for p in (root/'.internal').iterdir())):
            raise ValueError("New runs require an empty output directory")
        if resume and weights:
            raise ValueError("Weights initialization and resume are separate operations")
        if not Path(binary).is_file():
            raise FileNotFoundError("Native executable is missing; run scripts/build.sh first")
        verify_build(binary)
        controller=Controller(root,config,binary,max_iteration,max_seconds)
        try:
            state=controller.run(resume,weights)
        except BaseException as error:
            controller.journal("failure",error=repr(error));raise
        finally:
            controller.close()
        return state
