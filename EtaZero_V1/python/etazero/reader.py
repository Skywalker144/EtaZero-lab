"""Whole file batches and consumed cursors.

Gap-delaying repeat order adapted from KataGo training_data_generator.py (MIT).
An immutable snapshot is one synchronous round dataset; switching datasets occurs
at the next round. Checkpoints additionally retain the current file batch cursor.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import copy
import random
import queue
import threading
import numpy as np
from .storage import load_json, sha256
from .schema import CONTRACT_ID, PLANES, GLOBALS, unpack_observations


class BatchReader:
    def __init__(self,snapshot,batch_size,prefetch_depth,seed,state=None,no_repeat_files=False,split="train",shuffle_files=True):
        self.snapshot = Path(snapshot)
        if not (self.snapshot/'data').is_dir():
            from .shuffle import restore_snapshot
            restore_snapshot(self.snapshot)
        self.manifest = load_json(self.snapshot/"manifest.json")
        if self.manifest["contract"] != CONTRACT_ID:
            raise ValueError("Snapshot contract mismatch")
        if split not in ('train','validation'):raise ValueError('Invalid reader split')
        self.split=split;self.directory=self.snapshot/'data'
        if split=='validation':self.directory=self.directory/'validation'
        self.files=self.manifest['files' if split=='train' else 'validation_files']
        total=self.manifest['rows' if split=='train' else 'validation_rows']
        if not self.files or sum(x['rows'] for x in self.files)!=total:
            raise ValueError("Invalid snapshot manifest")
        self.batch_size, self.depth = batch_size, prefetch_depth
        self.no_repeat_files=no_repeat_files;self.shuffle_files=shuffle_files
        self.random=random.Random(seed)
        self.epoch=0;self.index=0;self.offset=0;self.used=[]
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
            if state['shuffle_files']!=shuffle_files or state['split']!=split or state['snapshot_id']!=self.manifest['id'] or state['batch_size']!=batch_size or state['no_repeat_files']!=no_repeat_files:
                raise ValueError('Checkpoint data snapshot/batch/repeat mode mismatch')
            self.random.setstate(state['rng'])
            self.epoch,self.index,self.offset,self.order,self.used=(state[k] for k in ('epoch','index','offset','order','used'))
            if sorted(self.order)!=list(range(len(self.files))) or not 0<=self.index<=len(self.order) or self.used!=self.order[:self.index]:
                raise ValueError('Invalid checkpoint data order')
            if self.offset<0 or self.offset%batch_size or (self.index==len(self.order) and self.offset):
                raise ValueError('Invalid checkpoint data cursor')
            if self.index<len(self.order) and self.offset>self.files[self.order[self.index]]['rows']//batch_size*batch_size:
                raise ValueError('Invalid checkpoint data cursor')
        self.executor = ThreadPoolExecutor(max_workers=max(1,prefetch_depth))
        self.pending = {}

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)

    def state(self):
        return {"snapshot_id": self.manifest["id"], "split":self.split, "epoch": self.epoch, "index": self.index,
                "offset":self.offset,"order":self.order[:],"used":self.used[:],"rng":self.random.getstate(),
                "batch_size":self.batch_size,"no_repeat_files":self.no_repeat_files,"shuffle_files":self.shuffle_files}

    def _load(self, index):
        info = self.files[self.order[index]]
        path = self.directory/info["path"]
        if sha256(path) != info["sha256"]:
            raise ValueError(f"Snapshot file checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as file:
            arrays = {key: file[key] for key in file.files}
        if set(arrays) != {"obs", "globals", "policy", "opponent_policy", "opponent_policy_weight", "value", "td_value", "full_game_weight", "q_values", "q_visits"} or len(arrays["value"]) != info["rows"]:
            raise ValueError(f"Invalid training view: {path}")
        canvas = self.manifest["canvas"]
        expected = {"obs": ((info["rows"],len(PLANES),(canvas*canvas+7)//8),np.uint8),
                    "globals": ((info["rows"],len(GLOBALS)),np.float32),
                    "td_value": ((info["rows"],3,3),np.float32),
                    "full_game_weight": ((info["rows"],),np.float32),
                    "q_values": ((info['rows'],canvas*canvas),np.float32),
                    "q_visits": ((info['rows'],canvas*canvas),np.float32),
                    "policy": ((info["rows"],canvas*canvas),np.float32),
                    "opponent_policy": ((info["rows"],canvas*canvas),np.float32),
                    "opponent_policy_weight": ((info["rows"],),np.float32),"value": ((info["rows"],3),np.float32)}
        for key,(shape,dtype) in expected.items():
            if arrays[key].shape != shape or arrays[key].dtype != dtype:
                raise ValueError(f"Invalid training {key} layout: {path}")
        return arrays

    def _new_pass(self):
        if self.no_repeat_files:
            raise StopIteration('No-repeat snapshot exhausted; a new snapshot is required')
        previous=self.used;count=len(previous);k=(count*2+1)//3
        reservoir=previous[:k];order=[]
        while k<count:
            index=self.random.randrange(len(reservoir))
            reservoir[index],reservoir[-1]=reservoir[-1],reservoir[index]
            order.append(reservoir.pop());reservoir.append(previous[k]);k+=1
        self.random.shuffle(reservoir);order.extend(reservoir)
        self.order=order;self.used=[];self.index=self.offset=0;self.epoch+=1
        self.pending.clear()

    def next(self):
        while True:
            if self.index==len(self.order):self._new_pass()
            info=self.files[self.order[self.index]]
            usable=info['rows']//self.batch_size*self.batch_size
            if self.offset==usable:
                self.pending.pop(self.index,None);self.used.append(self.order[self.index])
                self.index+=1;self.offset=0;continue
            if not 0<=self.offset<usable:raise ValueError('Invalid data cursor offset')
            for i in range(self.index,min(len(self.order),self.index+self.depth+1)):
                if i not in self.pending:self.pending[i]=self.executor.submit(self._load,i)
            arrays=self.pending[self.index].result()
            begin=self.offset;self.offset+=self.batch_size
            batch={k:v[begin:self.offset] for k,v in arrays.items()}
            # Source drops each file's suffix. Never fill from another file or
            # wrap a partly filled batch across passes/snapshots.
            batch['obs']=unpack_observations(batch['obs'],self.manifest['canvas'])
            return batch


class BatchPrefetcher:
    """One owner prepares complete CPU batches and their consumption cursors."""
    def __init__(self, reader, depth):
        self.reader=reader;self.initial_state=reader.state()
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
    """Two pinned upload slots; the checkpoint cursor tracks consumed batches only."""
    def __init__(self, reader, device):
        import torch
        self.reader, self.device = reader, device
        self.stream = torch.cuda.Stream(device=device)
        self.host = [{},{}]
        self.events = [torch.cuda.Event(),torch.cuda.Event()]
        self.used = [False,False]
        self.slot = 0
        self.pending = None

    def _enqueue(self):
        import torch
        slot = self.slot
        self.slot = 1-slot
        if self.used[slot]:
            self.events[slot].synchronize() # CPU must not overwrite an in-flight DMA source.
        batch,cursor = self.reader.next()
        for key,array in batch.items():
            source = torch.from_numpy(array)
            if key not in self.host[slot]:
                self.host[slot][key] = torch.empty(source.shape,dtype=torch.float32,pin_memory=True)
            self.host[slot][key].copy_(source)
        with torch.cuda.stream(self.stream):
            tensors = {k:v.to(self.device,non_blocking=True) for k,v in self.host[slot].items()}
            self.events[slot].record(self.stream)
        self.used[slot] = True
        self.pending = tensors,cursor,slot

    def next(self, prefetch_next=True):
        import torch
        if self.pending is None:
            self._enqueue()
        tensors,cursor,slot = self.pending
        current = torch.cuda.current_stream(self.device)
        current.wait_event(self.events[slot])
        for tensor in tensors.values():
            tensor.record_stream(current)
        self.pending = None
        if prefetch_next:
            self._enqueue()
        return tensors,cursor

    def close(self):
        self.stream.synchronize()
        self.pending = None
