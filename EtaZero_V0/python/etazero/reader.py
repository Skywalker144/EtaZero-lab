"""Bounded file prefetch with a checkpointable cursor and no discarded batch tails."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import copy
import queue
import threading
import numpy as np
from .storage import load_json, sha256
from .schema import CONTRACT_ID, PLANES, GLOBALS, unpack_observations


class BatchReader:
    def __init__(self, snapshot, batch_size, prefetch_depth, seed, state=None):
        self.snapshot = Path(snapshot)
        if not (self.snapshot/'data').is_dir():
            from .shuffle import restore_snapshot
            restore_snapshot(self.snapshot)
        self.manifest = load_json(self.snapshot/"manifest.json")
        if self.manifest["contract"] != CONTRACT_ID:
            raise ValueError("Snapshot contract mismatch")
        self.files = self.manifest["files"]
        if not self.files or sum(x["rows"] for x in self.files) != self.manifest["rows"]:
            raise ValueError("Invalid snapshot manifest")
        self.batch_size, self.depth = batch_size, prefetch_depth
        self.random = np.random.default_rng(seed)
        self.epoch, self.index, self.offset = 0, 0, 0
        self.order = self.random.permutation(len(self.files)).tolist()
        if state:
            if state["snapshot_id"] != self.manifest["id"]:
                raise ValueError("Checkpoint data snapshot mismatch")
            self.random.bit_generator.state = state["rng"]
            self.epoch, self.index, self.offset, self.order = state["epoch"], state["index"], state["offset"], state["order"]
            if sorted(self.order) != list(range(len(self.files))) or not 0 <= self.index <= len(self.order):
                raise ValueError("Invalid checkpoint data order")
        self.executor = ThreadPoolExecutor(max_workers=max(1,prefetch_depth))
        self.pending = {}

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)

    def state(self):
        return {"snapshot_id": self.manifest["id"], "epoch": self.epoch, "index": self.index,
                "offset": self.offset, "order": self.order[:], "rng": copy.deepcopy(self.random.bit_generator.state)}

    def _load(self, index):
        info = self.files[self.order[index]]
        path = self.snapshot/"data"/info["path"]
        if sha256(path) != info["sha256"]:
            raise ValueError(f"Snapshot file checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as file:
            arrays = {key: file[key] for key in file.files}
        if set(arrays) != {"obs", "globals", "policy", "opponent_policy", "opponent_policy_weight", "value"} or len(arrays["value"]) != info["rows"]:
            raise ValueError(f"Invalid training view: {path}")
        canvas = self.manifest["canvas"]
        expected = {"obs": ((info["rows"],len(PLANES),(canvas*canvas+7)//8),np.uint8),
                    "globals": ((info["rows"],len(GLOBALS)),np.float32),
                    "policy": ((info["rows"],canvas*canvas),np.float32),
                    "opponent_policy": ((info["rows"],canvas*canvas),np.float32),
                    "opponent_policy_weight": ((info["rows"],),np.float32),"value": ((info["rows"],3),np.float32)}
        for key,(shape,dtype) in expected.items():
            if arrays[key].shape != shape or arrays[key].dtype != dtype:
                raise ValueError(f"Invalid training {key} layout: {path}")
        return arrays

    def _current(self):
        if self.index == len(self.order):
            self.epoch += 1; self.index = self.offset = 0
            self.order = self.random.permutation(len(self.files)).tolist()
            self.pending.clear()
        for i in range(self.index, min(len(self.order), self.index+self.depth+1)):
            if i not in self.pending:
                self.pending[i] = self.executor.submit(self._load, i)
        return self.pending[self.index].result()

    def next(self):
        parts, remaining = [], self.batch_size
        while remaining:
            arrays = self._current()
            if not 0 <= self.offset < len(arrays["value"]):
                raise ValueError("Invalid data cursor offset")
            n = min(remaining, len(arrays["value"])-self.offset)
            parts.append({k: a[self.offset:self.offset+n] for k,a in arrays.items()})
            self.offset += n; remaining -= n
            if self.offset == len(arrays["value"]):
                self.pending.pop(self.index); self.index += 1; self.offset = 0
        batch = dict(parts[0]) if len(parts)==1 else {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
        # Keep prefetch and cross-file assembly compact; expand only this batch.
        batch["obs"] = unpack_observations(batch["obs"],self.manifest["canvas"])
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
