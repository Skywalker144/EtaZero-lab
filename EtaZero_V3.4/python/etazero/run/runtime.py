"""Round controller: plan -> selfplay -> snapshot -> credit-limited training -> publication."""
import datetime
from collections import deque
import fcntl
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import tarfile
import time
import uuid
import warnings
from etazero.config import ROOT, csv, fingerprint, resume_config, write_native
from etazero.train.compiler import WorkTimer, compilation_enabled, add_timing
from etazero.data.targets import Catalog
from etazero.run.native import NativeWorker
from etazero.models.export import export_model, verify_export
from etazero.data.shuffle import build_snapshot, desired_window, prune_derived, view_key, shuffle_pool
from etazero.storage import atomic_write, load_json, save_json, sync_directory
from etazero.train.learner import initialize, train_iteration, prune_checkpoints
from etazero.build import verify_build


def iteration_progress(iteration, elapsed_seconds, selfplay_rows, window_rows, trained_rows):
    days, minutes = divmod(int(elapsed_seconds)//60, 24*60)
    hours, minutes = divmod(minutes, 60)
    counts = []
    for rows in (selfplay_rows, window_rows, trained_rows):
        mantissa, exponent = f'{rows:.1e}'.split('e')
        counts.append(f'{mantissa}e{int(exponent)}')
    return (f'iter={iteration:<6} | elapsed={days:02d}d{hours:02d}h{minutes:02d}m'
            f' | selfplay={counts[0]:>6} | window={counts[1]:>6} | trained={counts[2]:>6}')


class Journal:
    """Append observational events; a logging failure does not abort training."""
    def __init__(self, root):
        self.path = Path(root)/"logs/events.jsonl"
        self.failure = None
        self.file = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Separate an interrupted trailing record without scanning old logs.
            if self.path.exists() and self.path.stat().st_size:
                with self.path.open('rb+') as file:
                    file.seek(-1, os.SEEK_END)
                    if file.read(1) != b'\n':
                        file.write(b'\n')
            self.file = self.path.open('a', buffering=1)
        except OSError as error:
            self._failed(error)

    def _failed(self, error):
        if self.failure is None:
            warnings.warn(f'Run journal unavailable; training continues: {error}', RuntimeWarning)
        self.failure = error

    def __call__(self, event, **fields):
        if self.failure is not None:
            return
        try:
            payload = {"event": event, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), **fields}
            self.file.write(json.dumps(payload, sort_keys=True)+'\n')
        except (OSError, ValueError, TypeError) as error:
            self._failed(error)

    def flush(self):
        if self.file is not None and self.failure is None:
            try:
                self.file.flush()
            except OSError as error:
                self._failed(error)

    def close(self):
        if self.file is not None and not self.file.closed:
            self('session_end')
            try:
                self.file.close()
            except OSError as error:
                self._failed(error)


def source_snapshot(root):
    paths=[]
    for directory,names,files in os.walk(ROOT):
        parent=Path(directory)
        names[:]=[name for name in names
                  if name not in {"build","runs","__pycache__",".pytest_cache"}
                  and not (parent==ROOT and name=="data")
                  and not (parent/name).is_relative_to(Path(root))]
        paths.extend(parent/name for name in files if (parent/name).is_file())
    paths += [ROOT.parent/"AGENTS.md",ROOT.parent/"RULES.md"]
    identity=uuid.uuid4().hex
    destination=Path(root)/".internal/source"/(identity+".tar.gz")
    def write(path):
        with tarfile.open(path,"w:gz") as archive:
            for source in sorted(paths):
                archive.add(source,arcname=str(source.relative_to(ROOT.parent)))
    atomic_write(destination,write,immutable=True)
    return identity


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
    """Plan new data from the baseline budget, independently of cold-start surplus."""
    iteration = state['iteration']; training = config['training']
    model = state['model'] or {'id': f"random:{config['run']['seed']}", 'evaluator': 'random'}
    target = (0 if iteration == 0 else config['replay']['min_rows'] if iteration == 1 else
              state['replay_rows'] + training['train_steps']*training['batch_size']/training['replay_ratio'])
    return {'iteration': iteration, 'train_steps': 0 if iteration == 0 else training['train_steps'],
            'batch_size': training['batch_size'], 'replay_ratio': training['replay_ratio'],
            'target_rows': target, 'start_rows': state['replay_rows'], 'train_credit': state['train_credit'],
            'input_model': model, 'input_checkpoint': state['checkpoint']}


def training_budget(plan, manifest, rows):
    """Bound consumption by the baseline, earned credit and snapshot batch count."""
    batches = manifest['resource_plan']['complete_batches_per_pass']
    if not batches:
        raise ValueError('Snapshot has no complete per-file training batch')
    new_rows = rows-plan['start_rows']
    if new_rows < 0:
        raise ValueError('Replay row count decreased during selfplay')
    credit = 0 if plan['iteration'] == 1 else plan['train_credit'] + new_rows*plan['replay_ratio']
    steps = min(plan['train_steps'], batches)
    if plan['iteration'] >= 2:
        steps = min(steps, math.floor(credit/plan['batch_size']))
    return dict(train_steps=steps, snapshot_batches=batches, available_credit=credit, new_rows=new_rows)


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
        save_json(self.root/".internal/state.json",state)

    def run(self,resume=False,weights=None):
        root,c=self.root,self.config
        import torch
        torch.set_num_threads(c["run"]["cpu_threads"])
        source=source_snapshot(root)
        self.source_id=source
        save_json(root/".internal/session.json",{"source_id":source,"started_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()})
        self.journal("session_start",source_id=source,binary=str(self.binary.resolve()),
                     iteration_limit=self.max_iteration,elapsed_seconds_limit=self.limit,
                     timing_basis='committed_wall_excluding_compile_and_plot',
                     compiler_cache={key:os.environ[key] for key in ('TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR')},
                     **provenance())
        if c['optimizer']['kind'] == 'muon':
            self.journal('optimizer_execution', kind='muon', environment={key: os.environ[key] for key in
                         ('KATAGO_MUON_BATCHED_NS', 'KATAGO_MUON_NS_BATCH_SIZE',
                          'KATAGO_MUON_NS_COMPILE_MODE', 'KATAGO_AUX_ADAM_FOREACH') if key in os.environ})
        if resume:
            info=load_json(root/".internal/run.json")
            if resume_config(info['config']) != resume_config(c):
                raise ValueError('Resume model/optimizer/data configuration differs from the saved run')
            state=load_json(root/".internal/state.json") if (root/".internal/state.json").exists() else {
                "run_id":info["id"],"iteration":0,"target_rows":0.0,"replay_origin_rows":None,"replay_rows":0,"train_credit":0.0,"checkpoint":None,"model":None,"elapsed_seconds":0.0}
        else:
            if (root/".internal/run.json").exists():
                raise ValueError("Run already exists; use --resume or a new directory")
            info={"id":uuid.uuid4().hex,"config_id":self.config_id,"config":c,"source_id":source,
                  "seed":c["run"]["seed"],"weights_initialization":{
                      "path":str(Path(weights).resolve())} if weights else None}
            save_json(root/".internal/run.json",info,immutable=True)

            state={"run_id":info["id"],"iteration":0,"target_rows":0.0,"replay_origin_rows":None,"replay_rows":0,"train_credit":0.0,"checkpoint":None,"model":None,"elapsed_seconds":0.0}
            self.state(state)
        if state["run_id"]!=info["id"]:
            raise ValueError("Run state identity mismatch")
        if 'elapsed_seconds' not in state:
            raise ValueError('Run predates whole-iteration commits; use a new run directory')
        if 'train_credit' not in state or 'replay_rows' not in state:
            raise ValueError('Run lacks replay-credit training budgets; use a new run directory')
        if state['checkpoint'] and not (root/state['checkpoint']['path']).is_file():
            raise FileNotFoundError('Committed checkpoint is missing')
        if state['model']:
            verify_export(root,state['model'],c['network']['canvas'])
            if state['model']['checkpoint']['total_samples'] > state['checkpoint']['total_samples']:
                raise ValueError('Published model is newer than committed learner state')
        # Keep the initial configuration immutable and record each effective
        # session before refreshing the native worker's current configuration.
        session_id = uuid.uuid4().hex
        save_json(root/'config/sessions'/(session_id+'.json'), c, immutable=True)
        save_json(root/'config/effective.json', c)
        write_native(c, root/'config/effective.cfg')
        self.journal('session_config', config_id=self.config_id, config_path=f'config/sessions/{session_id}.json')
        recover_iteration(root, state)
        self.run_id=info["id"]
        for name in ("selfplay",".internal/iterations","logs","models","snapshots","checkpoints"):
            (root/name).mkdir(parents=True,exist_ok=True)
        if state["checkpoint"] is None:
            origin=info["weights_initialization"]
            state["checkpoint"]=initialize(root,c,origin["path"] if origin else None);self.state(state)
        self.prune_checkpoints()
        self.catalog=Catalog(root,info["id"],self.config_id)
        plotted=False
        while not self.stopping() and (not self.max_iteration or state["iteration"]<=self.max_iteration) and (not self.limit or state["elapsed_seconds"] < self.limit):
            plotted=False
            iteration=state["iteration"];directory=root/".internal/iterations"/f"{iteration:06d}";directory.mkdir(exist_ok=True)
            timing_path = directory/'timing.json'
            previous_timing = load_json(timing_path) if timing_path.exists() else dict(seconds=0.,wall_seconds=0.,compile_seconds=0.)
            timer=WorkTimer(compilation_enabled(c))
            try:
                plan_path,status_path=directory/"plan.json",directory/"status.json"
                if plan_path.exists():
                    plan=load_json(plan_path)
                    if plan != iteration_plan(state,c):
                        raise ValueError("Iteration plan differs from its parent committed state")
                else:
                    plan=iteration_plan(state,c)
                    self.journal.flush()
                    save_json(directory/'journal.json', {'offset': self.journal.path.stat().st_size if self.journal.path.exists() else 0})
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
                    manifest=load_json(root/"snapshots"/snapshot/"manifest.json")
                    self.journal("shuffle_resources",iteration=iteration,**manifest["resource_plan"])
                    budget=training_budget(plan,manifest,self.catalog.counts()[0])
                    self.journal('training_budget',iteration=iteration,baseline_steps=plan['train_steps'],**budget)
                    status={"phase":"train","snapshot_id":snapshot,**budget};save_json(status_path,status)
                if self.stopping():
                    break
                if status["phase"]=="train":
                    if status['train_steps']:
                        train_plan={**plan,"snapshot_id":status["snapshot_id"],"train_steps":status['train_steps']}
                        checkpoint,done=self.phase("train",iteration,lambda:train_iteration(root,c,train_plan,plan["input_checkpoint"],self.journal,self.stopping))
                        if not done:
                            break
                        status={**status,"phase":"export","checkpoint":checkpoint}
                    else:
                        status={**status,"phase":"completed","checkpoint":state['checkpoint'],"model":state['model']}
                    save_json(status_path,status)
                if self.stopping():
                    break
                if status["phase"]=="export":
                    model=self.phase("export",iteration,lambda:export_model(root,c,status["checkpoint"]))
                    status={**status,"phase":"gatekeeper","candidate":model};save_json(status_path,status)
                if status['phase']=='gatekeeper':
                    from etazero.evaluation.gatekeeper import consider_candidate
                    try:
                        decision = self.phase('gatekeeper', iteration, lambda: consider_candidate(
                            root, c, self.binary, state['model'], status['candidate'], lambda: self.stop))
                    except InterruptedError:
                        if self.stopping():
                            break
                        raise
                    status={**status, 'phase':'completed', 'gatekeeper':decision,
                            'model':status['candidate'] if decision['accepted'] else state['model']}
                    save_json(status_path,status)
                if status["phase"]!="completed":
                    raise ValueError(f'Unknown persisted phase: {status["phase"]}')
                # Assign from the immutable plan; repeated recovery cannot add the target twice.
                rows,games=self.catalog.counts()
                origin = rows if iteration == 1 else state['replay_origin_rows']
                credit = (status['available_credit']-status['train_steps']*plan['batch_size']
                          if iteration >= 2 else 0.0)
                next_state={"run_id":self.run_id,"iteration":iteration+1,
                       "target_rows":rows if iteration == 1 else plan["target_rows"], "replay_origin_rows":origin,
                       "replay_rows":rows,"train_credit":credit,
                       "checkpoint":status["checkpoint"],"model":status["model"],"elapsed_seconds":state["elapsed_seconds"]}
                try:
                    self.journal("selfplay_statistics",iteration=iteration,**self.catalog.statistics(iteration))
                except (OSError, ValueError) as error:
                    warnings.warn(f'Selfplay statistics unavailable: {error}', RuntimeWarning)
                self.journal("iteration_complete",iteration=iteration,unique_rows=rows,games=games,target_rows=next_state["target_rows"],
                             replay_origin_rows=origin,total_samples=next_state["checkpoint"]["total_samples"],
                             train_steps=status['train_steps'] if iteration else 0,train_credit=credit,
                             replay_ratio=next_state["checkpoint"]["total_samples"]/rows if rows else 0,
                             model_id=next_state["model"]["id"] if next_state["model"] else None)
                window_rows = 0  # Bootstrap has no replay snapshot yet.
                if iteration > 0:
                    if status['snapshot_id'] not in self.hot_snapshots:
                        self.hot_snapshots.append(status['snapshot_id'])
                    evicted=[]
                    while len(self.hot_snapshots)>c['shuffle']['snapshot_keep']:
                        evicted.append(self.hot_snapshots.popleft())
                    manifest=load_json(root/'snapshots'/status['snapshot_id']/'manifest.json')
                    window_rows = manifest['window_rows']
                    prune_derived(root,evicted,{view_key(e,manifest['targets']) for e in manifest['sources']+manifest['validation_sources']})
                    if evicted:
                        self.journal('snapshots_evicted',snapshots=evicted)
                self.journal.flush()
                metrics_path = root/'logs/iterations'/f'{iteration:06d}.json'
                metrics = None
                if not metrics_path.exists():
                    from etazero.evaluation.plotting import round_history
                    try:
                        metrics = round_history(root, iteration,
                                                status['train_steps'] if iteration else 0,
                                                load_json(directory/'journal.json')['offset'])
                    except (OSError, ValueError, KeyError) as error:
                        warnings.warn(f'Round {iteration} metrics unavailable: {error}', RuntimeWarning)
                        metrics = dict(iteration=iteration, metrics_error=str(error))
                if 'commit' not in status:
                    timing = add_timing(previous_timing, timer.finish())
                    timing['timing_valid'] &= state.get('timing_valid', True)
                    status['commit'] = dict(iteration=iteration, model=next_state['model'],
                                            elapsed_seconds=state['elapsed_seconds']+timing['seconds'],
                                            total_rows=rows, total_samples=next_state['checkpoint']['total_samples'],
                                            steps=status['train_steps'] if iteration else 0, **timing,
                                            timing_basis='committed_wall_excluding_compile_and_plot')
                    # Persist the evaluation record before advancing the state pointer.
                    # A retry reuses this time even if the optional summary was lost.
                    save_json(status_path, status)
                if metrics is not None:
                    try:
                        save_json(metrics_path, metrics, immutable=True)
                    except OSError as error:
                        warnings.warn(f'Round {iteration} summary could not be saved: {error}', RuntimeWarning)
                next_state['elapsed_seconds'] = status['commit']['elapsed_seconds']
                next_state['timing_valid'] = status['commit']['timing_valid']
                self.state(next_state)
                state=next_state
                if state['model']:
                    self.publish(state['model'],state['elapsed_seconds'])
                self.prune_checkpoints()
                print(iteration_progress(iteration, state['elapsed_seconds'], rows, window_rows,
                                         state['checkpoint']['total_samples']), flush=True)
                self.plot()
                plotted=True
            finally:
                if state['iteration'] == iteration:
                    measured = timer.finish()
                    save_json(timing_path, add_timing(previous_timing, measured))
        if not plotted:
            self.plot()
        self.journal("run_stopped" if self.stop else "run_complete",completed_iterations=state["iteration"]-1)
        return state

    def prune_checkpoints(self):
        removed = prune_checkpoints(self.root, self.config['training']['checkpoint_keep'])
        if removed:
            self.journal('checkpoints_pruned', paths=removed,
                         keep=self.config['training']['checkpoint_keep'])

    def plot(self,state=None):
        from etazero.evaluation.plotting import plot_run
        self.journal.flush()
        try:
            plot_run(self.root,state)
        except Exception as error:
            self.journal('plot_failed', error=repr(error))
            warnings.warn(f'Training state is committed; plotting failed: {error}', RuntimeWarning)

    def phase(self,name,iteration,function):
        self.journal("phase_start",phase=name,iteration=iteration)
        timer=WorkTimer(compilation_enabled(self.config))
        succeeded=False
        try:
            value=function();succeeded=True
            return value
        finally:
            self.journal("phase_end",phase=name,iteration=iteration,succeeded=succeeded,**timer.finish())

    def publish(self,model,elapsed_seconds):
        verify_export(self.root,model,self.config['network']['canvas'])
        path=self.root/"models/current.json"
        current=load_json(path) if path.exists() else None
        if current and current["model"]["id"]==model["id"]:
            if current["model"]!=model:
                raise ValueError("Published model identity collision")
            # A zero-step round keeps the model but advances committed time.
            if current['elapsed_seconds'] != elapsed_seconds:
                save_json(path,{**current,'elapsed_seconds':elapsed_seconds})
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
        # Publication can finish immediately before a process dies at the phase
        # pointer. Reuse that complete dataset rather than reshuffling it.
        for path in sorted((self.root/'snapshots').glob(f'iteration_{self.current_iteration:06d}_*')):
            if (path/'manifest.json').is_file() and (path/'data').is_dir():
                manifest=load_json(path/'manifest.json')
                if (manifest['recipe']['replay'] == self.config['replay'] and
                        manifest['recipe']['shuffle'] == self.config['shuffle']):
                    return manifest['id']
        if self.shuffle_pool is None:
            self.shuffle_pool=shuffle_pool(self.config['shuffle']['workers'])
        from etazero.data.shuffle import desired_window
        counts=self.catalog.replay_counts(self.config['replay']['min_rows'])
        desired=desired_window(counts['usable_rows'],self.config['replay'])
        entries = self.catalog.entries(desired, include_game_rows=not self.config['training']['skip_validation'])
        return build_snapshot(self.root,self.current_iteration,entries,self.config,
                              pool=self.shuffle_pool,counts=counts)

    def produce(self,plan):
        c=self.config;iteration=plan['iteration']
        if iteration == 0:
            _,completed=self.catalog.iteration_counts(0)
            remaining=c['selfplay']['bootstrap_games']-completed
            if remaining > 0 and not self.stopping():
                self.launch(plan,remaining);self.scan()
            rows,games=self.catalog.counts()
            if not self.stopping() and (games != c['selfplay']['bootstrap_games'] or not rows):
                raise RuntimeError('Bootstrap did not produce the configured games and valid training rows')
        else:
            rows,completed=self.catalog.counts()
            average=self.catalog.previous_rows_per_game(iteration)
            if not average:
                raise RuntimeError('Cold start produced no valid rows-per-game estimate')
            deficit=max(0,plan['target_rows']-rows)
            if iteration >= 2:
                # The immutable starting row count and preceding-round mean
                # reproduce the same game quota after an interrupted attempt.
                planned=max(1,math.ceil((plan['target_rows']-plan['start_rows'])/average))
                games=max(0,planned-self.catalog.iteration_counts(iteration)[1])
            else:
                games=math.ceil(deficit/average)
            self.journal('selfplay_target',iteration=iteration,rows=rows,target=plan['target_rows'],deficit=deficit,
                         rows_per_game=average,planned_games=games)
            if games and not self.stopping():
                self.launch(plan,games);self.scan()
            if iteration >= 2 and games and not self.stopping():
                new_rows,new_completed=self.catalog.counts()
                if new_rows < rows or new_completed-completed != games:
                    raise RuntimeError('Selfplay failed to produce all planned complete games')
            # Only cold start fills a hard row threshold. Subsequent rounds
            # adjust training to the rows actually generated by their game quota.
            while iteration == 1 and not self.stopping():
                new_rows,new_completed=self.catalog.counts()
                if new_rows >= plan['target_rows']:
                    break
                # PCR and stochastic multiplicity can give completed games no
                # training rows. That is valid progress; keep filling the quota.
                # A worker returning neither rows nor games is a real failure.
                if new_rows < rows or (new_rows == rows and new_completed <= completed) or not average:
                    raise RuntimeError('Selfplay failed to produce enough valid training rows')
                rows,completed=new_rows,new_completed
                deficit=plan['target_rows']-rows
                planned=math.ceil(deficit/average)
                self.journal('selfplay_backfill',iteration=iteration,rows=rows,target=plan['target_rows'],
                             deficit=deficit,rows_per_game=average,planned_games=planned)
                self.launch(plan,planned);self.scan()
        if not self.stop:
            for identity,service in self.services.items():
                service.release()
                self.journal('worker_release',iteration=iteration,worker_id=identity,pid=service.process.pid,
                             model_id=plan['input_model']['id'])

    def launch(self,plan,games):
        c=self.config;iteration=plan["iteration"];attempt=uuid.uuid4().hex
        attempt_index=len(list((self.root/".internal/iterations"/f"{iteration:06d}").glob("attempt_*.json")))
        devices=csv(c["devices"]["selfplay"]);workers=[]
        model=plan["input_model"]
        kind=model.get('evaluator','network')
        model_path=str(self.root/model['path']) if kind=='network' else ''
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
                try:
                    lines=stdout.read_text().splitlines()
                    for i,line in enumerate(lines):
                        try:
                            record=json.loads(line)
                        except json.JSONDecodeError:
                            if i==len(lines)-1 and any(code!=0 for code in codes):
                                self.journal("partial_worker_log",file=str(stdout));continue
                            raise
                        if record['event']=='worker_prepared':
                            self.journal('inference_setup',iteration=iteration,worker_id=worker['id'],attempt=attempt,
                                         **{k:v for k,v in record.items() if k!='event'})
                        elif record["event"]=="inference":
                            self.journal("inference",iteration=iteration,worker_id=worker["id"],attempt=attempt,evaluator=kind,
                                         **{k:v for k,v in record.items() if k!="event"})
                except (OSError, ValueError, KeyError) as error:
                    warnings.warn(f'Native metrics unavailable from {stdout}: {error}', RuntimeWarning)
            if any(code!=0 for code in codes) and not self.stop:
                raise RuntimeError(f"Selfplay worker failed (codes={codes}); inspect {self.root/'logs'}")
        finally:
            if not finished:
                for process in self.children:
                    self.signal_child(process,signal.SIGTERM)
            self.children=[]


def recover_iteration(root, state):
    """Keep completed phases and learner progress; restore the publication pointer."""
    root=Path(root)
    # Only unpublished export scratch is abandoned. Complete raw shards,
    # selected snapshots and learner checkpoints remain available for resume.
    archive=root/'.internal/discarded'/uuid.uuid4().hex
    for path in (root/'models').glob('.tmp_*'):
        target=archive/path.relative_to(root)
        target.parent.mkdir(parents=True,exist_ok=True)
        path.rename(target)
        sync_directory(path.parent);sync_directory(target.parent)
    current=root/'models/current.json'
    if state['model']:
        publication={'model':state['model'],'elapsed_seconds':state['elapsed_seconds']}
        try:
            existing=load_json(current) if current.exists() else None
        except ValueError:
            existing=None
        if not isinstance(existing,dict) or any(existing.get(key)!=value for key,value in publication.items()):
            save_json(current,publication)
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
