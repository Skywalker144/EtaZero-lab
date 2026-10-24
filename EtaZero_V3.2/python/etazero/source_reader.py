"""KataGo file generator and whole-file subepochs, resumable between batches.

Outer replay credit may pause anywhere. Selected files and the generator survive
that pause; a new snapshot becomes eligible at the next subepoch selection.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import random
import numpy as np
from ._vendor.katago.training_data_generator import TrainingDataGenerator
from .reader import load_file
from .storage import load_json
from .schema import unpack_observations
from .games import spec_for_metadata
from .data import TRAIN_POLICY_TARGETS


class SourceBatchReader:
    def __init__(self, snapshot, batch_size, prefetch_depth, seed, epoch_steps, sub_epochs,
                 state=None, max_batches=None):
        self.snapshot = Path(snapshot).resolve()
        self.batch_size, self.depth = batch_size, prefetch_depth
        self.epoch_steps, self.sub_epochs = epoch_steps, sub_epochs
        self.remaining_batches = max_batches
        self.random = random.Random(seed)
        self.metadata = {}
        self.file_metadata = {}
        self.manifest = self._manifest(self.snapshot)
        if not any(f['rows'] >= batch_size for f in self.manifest['files']):
            raise ValueError('Snapshot has no complete per-file batch')
        self.generator_state = {} if state is None else {k:list(v) for k,v in state['generator'].items()}
        self.generator = TrainingDataGenerator(self.generator_state, False, self.random)
        self.selected = () if state is None else state['selected']
        self.index = 0 if state is None else state['index']
        self.offset = 0 if state is None else state['offset']
        self.subepoch_steps = 0 if state is None else state['subepoch_steps']
        self.epoch = 0 if state is None else state['epoch']
        self.subepoch = 0 if state is None else state['subepoch']
        self.subepoch_batches = 0 if state is None else state['subepoch_batches']
        if state is not None:
            if (state['batch_size'],state['epoch_steps'],state['sub_epochs']) != (batch_size,epoch_steps,sub_epochs):
                raise ValueError('Checkpoint source reader schedule mismatch')
            self.random.setstate(state['rng'])
        if state is None or state['snapshot_id'] != self.manifest['id']:
            self.generator.set_data_dir_if_has_remaining_files(str(self.snapshot/'data'))
        else:
            # Preserve the saved queue/RNG exactly when resuming the same data.
            self.generator._all_files = [str(self.snapshot/'data'/f['path']) for f in self.manifest['files']]
        self._save_queue()
        self.executor = ThreadPoolExecutor(max_workers=max(1,prefetch_depth))
        self.pending = {}

    def _manifest(self, snapshot):
        if snapshot not in self.metadata:
            if not (snapshot/'data').is_dir():
                from .shuffle import restore_snapshot
                restore_snapshot(snapshot)
            manifest = load_json(snapshot/'manifest.json')
            self.metadata[snapshot] = manifest
            self.file_metadata.update({snapshot/'data'/f['path']:(f,manifest) for f in manifest['files']})
        return self.metadata[snapshot]

    def _file_info(self, filename):
        path = Path(filename)
        self._manifest(path.parent.parent)
        return self.file_metadata[path]

    def _save_queue(self):
        # Immutable shared state: prefetch must not change a consumed cursor.
        self.saved_generator = {k:tuple(v) for k,v in self.generator_state.items()}

    def _select_subepoch(self):
        if self.selected:
            self.subepoch += 1
            if self.subepoch == self.sub_epochs:
                self.epoch += 1
                self.subepoch = 0
        selected = []
        batches = 0
        target = self.epoch_steps / self.sub_epochs
        while True:
            filename = self.generator.peek()
            info,_ = self._file_info(filename)
            count = info['rows'] // self.batch_size
            if count <= 0:
                self.generator.pop()
                continue
            if batches + count > target and batches > 0:
                if self.random.random() <= (batches + count - target) / count:
                    break
            self.generator.pop()
            selected.append(filename)
            batches += count
            if batches >= target or len(selected)>100000:
                break
        self.selected = tuple(selected)
        self.subepoch_steps = batches
        self.index = self.offset = self.subepoch_batches = 0
        self._save_queue()

    def state(self):
        return dict(snapshot_id=self.manifest['id'], batch_size=self.batch_size,
                    epoch_steps=self.epoch_steps, sub_epochs=self.sub_epochs,
                    generator=self.saved_generator, selected=self.selected,
                    index=self.index, offset=self.offset, subepoch_steps=self.subepoch_steps,
                    epoch=self.epoch, subepoch=self.subepoch, subepoch_batches=self.subepoch_batches,
                    epoch_end=(self.subepoch==self.sub_epochs-1 and self.subepoch_batches==self.subepoch_steps and bool(self.selected)),
                    rng=self.random.getstate())

    def _load(self, filename):
        info,manifest = self._file_info(filename)
        return load_file(filename, info['rows'], manifest)

    def next(self):
        if self.remaining_batches == 0:
            raise StopIteration('Outer replay credit exhausted')
        if not self.selected or self.subepoch_batches == self.subepoch_steps:
            self.pending.clear()
            self._select_subepoch()
        while True:
            filename = self.selected[self.index]
            info,manifest = self._file_info(filename)
            usable = info['rows']//self.batch_size*self.batch_size
            if self.offset < usable:
                break
            self.pending.pop(self.index, None)
            self.index += 1
            self.offset = 0
        for i in range(self.index,min(len(self.selected),self.index+self.depth+1)):
            if i not in self.pending:
                self.pending[i] = self.executor.submit(self._load, self.selected[i])
        arrays = self.pending[self.index].result()
        begin = self.offset
        self.offset += self.batch_size
        batch = {k:v[begin:self.offset] for k,v in arrays.items()}
        batch['obs'] = unpack_observations(batch['obs'],manifest['canvas'])
        for key in TRAIN_POLICY_TARGETS:
            if key in batch:
                batch[key] = batch[key].astype(np.float32)
        if manifest.get('algorithm') == 'muzero':
            from .muzero.data import prepare_batch
            batch = prepare_batch(batch,manifest['unroll_weight_mean'],self.random,spec_for_metadata(manifest))
        self.subepoch_batches += 1
        if self.remaining_batches is not None:
            self.remaining_batches -= 1
        return batch

    def close(self):
        self.executor.shutdown(wait=True,cancel_futures=True)
