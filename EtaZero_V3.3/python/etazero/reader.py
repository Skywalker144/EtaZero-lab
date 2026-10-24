"""Whole file batches and consumed cursors.

An immutable snapshot is one synchronous round dataset; switching datasets occurs
at the next round. Checkpoints additionally retain the current file batch cursor.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import random
import queue
import threading
import numpy as np
from .storage import load_json
from .schema import PLANES, GLOBALS, unpack_observations
from .games import spec_for_metadata
from .data import TRAIN_TARGETS, TRAIN_POLICY_TARGETS


def load_file(path, rows, manifest):
    with np.load(path, allow_pickle=False) as file:
        arrays = {key: file[key] for key in file.files}
    from .source_data import validate,decode_view
    validate(arrays,rows,manifest['canvas'],spec_for_metadata(manifest))
    arrays=decode_view(arrays,manifest.get('targets',TRAIN_TARGETS))
    if manifest.get('algorithm') == 'muzero':
        from .muzero.data import validate_view
        validate_view(arrays, rows, manifest['canvas'], manifest['unroll_steps'],
                      manifest.get('targets', TRAIN_TARGETS), manifest)
        return arrays
    if set(arrays) != {'obs', 'globals', *manifest.get('targets', TRAIN_TARGETS)} or len(arrays["value"]) != rows:
        raise ValueError(f"Invalid training view: {path}")
    canvas = manifest["canvas"]
    spec = spec_for_metadata(manifest)
    expected = {"obs": ((rows,spec.planes,(canvas*canvas+7)//8),np.uint8),
                "globals": ((rows,spec.globals),np.float32),
                "td_value": ((rows,3,4),np.float32),
                "full_game_weight": ((rows,),np.float32),
                'future_positions': ((rows,2,canvas,canvas),np.float32),
                'variance_time': ((rows,),np.float32),
                'training_weight': ((rows,),np.float32),
                "q_values": ((rows,spec.actions(canvas)),np.float32),
                "q_visits": ((rows,spec.actions(canvas)),np.float32),
                "policy": ((rows,spec.actions(canvas)),np.int16),
                "opponent_policy": ((rows,spec.actions(canvas)),np.int16),
                "opponent_policy_weight": ((rows,),np.float32),"value": ((rows,4),np.float32)}
    for key in arrays:
        shape,dtype = expected[key]
        if arrays[key].shape != shape or arrays[key].dtype != dtype:
            raise ValueError(f"Invalid training {key} layout: {path}")
    return arrays


class BatchReader:
    def __init__(self,snapshot,batch_size,prefetch_depth,seed,state=None,split="train",shuffle_files=True,max_batches=None):
        self.snapshot = Path(snapshot)
        if not (self.snapshot/'data').is_dir():
            from .shuffle import restore_snapshot
            restore_snapshot(self.snapshot)
        self.manifest = load_json(self.snapshot/"manifest.json")
        if split not in ('train','validation'):raise ValueError('Invalid reader split')
        self.split=split;self.directory=self.snapshot/'data'
        if split=='validation':self.directory=self.directory/'validation'
        self.files=self.manifest['files' if split=='train' else 'validation_files']
        total=self.manifest['rows' if split=='train' else 'validation_rows']
        if not self.files or sum(x['rows'] for x in self.files)!=total:
            raise ValueError("Invalid snapshot manifest")
        self.batch_size, self.depth = batch_size, prefetch_depth
        self.shuffle_files=shuffle_files
        self.random=random.Random(seed)
        self.index=0;self.offset=0
        self.order=list(range(len(self.files)))
        if shuffle_files:
            self.random.shuffle(self.order)
            # Source first-dir queue is uniformly interleaved with an empty old
            # queue, drawing once per file, then served in reverse order.
            for _ in self.order:self.random.random()
            self.order.reverse()
        self.usable_rows=sum(f['rows']//batch_size*batch_size for f in self.files)
        if not self.usable_rows:raise ValueError('Snapshot has no complete per-file batch')
        if state:
            if state['shuffle_files']!=shuffle_files or state['split']!=split or state['snapshot_id']!=self.manifest['id'] or state['batch_size']!=batch_size:
                raise ValueError('Checkpoint data snapshot/batch mode mismatch')
            self.random.setstate(state['rng'])
            self.index,self.offset,self.order=(state[k] for k in ('index','offset','order'))
            if sorted(self.order)!=list(range(len(self.files))) or not 0<=self.index<=len(self.order):
                raise ValueError('Invalid checkpoint data order')
            if self.offset<0 or self.offset%batch_size or (self.index==len(self.order) and self.offset):
                raise ValueError('Invalid checkpoint data cursor')
            if self.index<len(self.order) and self.offset>self.files[self.order[self.index]]['rows']//batch_size*batch_size:
                raise ValueError('Invalid checkpoint data cursor')
        # File order is fixed for the round. Share immutable snapshots across
        # queued cursors instead of copying the file list and RNG every batch.
        self._state_order=tuple(self.order)
        self._rng_state=self.random.getstate()
        self.remaining_batches=max_batches
        self.end_index=len(self.order)
        if max_batches is not None:
            if max_batches<0:raise ValueError('Batch budget must be nonnegative')
            self.end_index=self.index
            left=max_batches
            while left>0 and self.end_index<len(self.order):
                count=self.files[self.order[self.end_index]]['rows']//batch_size
                if self.end_index==self.index:count-=self.offset//batch_size
                left-=count;self.end_index+=1
            if left>0:raise ValueError('Single-pass snapshot has insufficient complete file batches')
        self.executor = ThreadPoolExecutor(max_workers=max(1,prefetch_depth))
        self.pending = {}

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)

    def state(self):
        return {"snapshot_id": self.manifest["id"], "split":self.split, "index": self.index,
                "offset":self.offset,"order":self._state_order,"rng":self._rng_state,
                "batch_size":self.batch_size,"shuffle_files":self.shuffle_files}

    def _load(self, index):
        info = self.files[self.order[index]]
        path = self.directory/info["path"]
        return load_file(path, info['rows'], self.manifest)

    def next(self):
        while True:
            if self.remaining_batches==0:
                raise StopIteration('Round batch budget exhausted')
            if self.index==len(self.order):
                raise StopIteration('Snapshot exhausted; a new snapshot is required')
            info=self.files[self.order[self.index]]
            usable=info['rows']//self.batch_size*self.batch_size
            if self.offset==usable:
                self.pending.pop(self.index,None)
                self.index+=1;self.offset=0;continue
            if not 0<=self.offset<usable:raise ValueError('Invalid data cursor offset')
            for i in range(self.index,min(self.end_index,self.index+self.depth+1)):
                if i not in self.pending:self.pending[i]=self.executor.submit(self._load,i)
            arrays=self.pending[self.index].result()
            begin=self.offset;self.offset+=self.batch_size
            batch={k:v[begin:self.offset] for k,v in arrays.items()}
            # Source drops each file's suffix. Never fill from another file or
            # wrap a partly filled batch across passes/snapshots.
            batch['obs']=unpack_observations(batch['obs'],self.manifest['canvas'])
            # Preserve integer weights through disk/cache/shuffle; expand only
            # the consumed batch before learner normalization and augmentation.
            for key in TRAIN_POLICY_TARGETS:
                if key in batch:batch[key]=batch[key].astype(np.float32)
            if self.manifest.get('algorithm') == 'muzero':
                from .muzero.data import prepare_batch
                batch = prepare_batch(batch, self.manifest['unroll_weight_mean'], self.random, spec_for_metadata(self.manifest))
                self._rng_state=self.random.getstate()
            if self.remaining_batches is not None:self.remaining_batches-=1
            return batch


class BatchPrefetcher:
    """One owner prepares complete CPU batches and their consumption cursors."""
    def __init__(self, reader, depth):
        self.reader=reader
        self.ready=queue.Queue(maxsize=max(1,depth))
        self.stopped=threading.Event()
        self.thread=threading.Thread(target=self._prepare,daemon=True);self.thread.start()

    def _put(self, item):
        while not self.stopped.is_set():
            try:
                self.ready.put(item,timeout=0.1);return
            except queue.Full:
                pass

    def _prepare(self):
        try:
            while not self.stopped.is_set():
                batch=self.reader.next()
                self._put((batch,self.reader.state()))
        except BaseException as error:
            self._put(error)

    def next(self):
        item=self.ready.get()
        if isinstance(item,BaseException):
            raise item
        return item

    def close(self):
        self.stopped.set();self.thread.join()


class CudaBatchPrefetcher:
    """Background uploads with two pinned slots and consumed-only cursors."""
    def __init__(self, reader, device):
        import torch
        self.reader, self.device = reader, device
        self.stream = torch.cuda.Stream(device=device)
        self.host = [{},{}]
        self.events = [torch.cuda.Event(),torch.cuda.Event()]
        self.used = [False,False]
        self.slot = 0
        self.pending = None
        self.executor = ThreadPoolExecutor(max_workers=1)

    def _enqueue(self):
        import torch
        # CUDA device/stream state is thread-local. Own all host slots and
        # uploads on this one worker; the learner waits only for its own batch.
        with torch.cuda.device(self.device):
            return self._upload()

    def _upload(self):
        import torch
        slot = self.slot
        self.slot = 1-slot
        if self.used[slot]:
            self.events[slot].synchronize() # CPU must not overwrite an in-flight DMA source.
        batch,cursor = self.reader.next()
        for key,array in batch.items():
            source = torch.from_numpy(array)
            if key not in self.host[slot] or self.host[slot][key].shape != source.shape:
                self.host[slot][key] = torch.empty(source.shape,dtype=torch.int64 if key in ('actions','recurrent_indices') else torch.float32,pin_memory=True)
            self.host[slot][key].copy_(source)
        with torch.cuda.stream(self.stream):
            tensors = {k:v.to(self.device,non_blocking=True) for k,v in self.host[slot].items()}
            self.events[slot].record(self.stream)
        self.used[slot] = True
        return tensors,cursor,slot

    def next(self, prefetch_next=True):
        import torch
        if self.pending is None:
            self.pending = self.executor.submit(self._enqueue)
        tensors,cursor,slot = self.pending.result()
        current = torch.cuda.current_stream(self.device)
        current.wait_event(self.events[slot])
        for tensor in tensors.values():
            tensor.record_stream(current)
        self.pending = None
        if prefetch_next:
            self.pending = self.executor.submit(self._enqueue)
        return tensors,cursor

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.stream.synchronize()
        self.pending = None
