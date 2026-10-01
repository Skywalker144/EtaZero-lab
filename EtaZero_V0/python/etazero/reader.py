"""Bounded file prefetch with a checkpointable cursor and no discarded batch tails."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import copy
import numpy as np
from .storage import load_json, sha256


class BatchReader:
    def __init__(self, snapshot, batch_size, prefetch_depth, seed, state=None):
        self.snapshot = Path(snapshot)
        self.manifest = load_json(self.snapshot/"manifest.json")
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
        if set(arrays) != {"obs", "policy", "value"} or len(arrays["value"]) != info["rows"]:
            raise ValueError(f"Invalid training view: {path}")
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
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
