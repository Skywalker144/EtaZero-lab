"""Checkpoint retention preserves committed metrics and unfinished-round recovery."""
from config_samples import CONFIGS
import json
from pathlib import Path
import pytest
from etazero.config import ROOT, load_config, validate
from etazero.plotting import run_history
from etazero.storage import load_json, save_json, sha256
from etazero.training import prune_checkpoints


@pytest.fixture
def checkpoint_run(tmp_path):
    parent = None
    references = {}
    for iteration, steps in ((0, (0,)), (1, (2, 4)), (2, (2, 4)), (3, (2, 4))):
        for step in steps:
            identity = f'iteration_{iteration:06d}_step_{step:08d}_test'
            path = tmp_path/'checkpoints'/(identity+'.pt')
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(identity.encode())
            reference = {'id': identity, 'path': str(path.relative_to(tmp_path)),
                         'sha256': sha256(path), 'iteration': iteration, 'step': step}
            references[iteration, step] = reference
            parent = reference
    save_json(tmp_path/'.internal/checkpoints.json',list(references.values()))
    for iteration in range(4):
        save_json(tmp_path/'logs/iterations'/f'{iteration:06d}.json',{'iteration':iteration,'steps':4 if iteration else 0})
    save_json(tmp_path/'.internal/state.json', {'iteration': 4, 'checkpoint': parent})
    # These files are outside the committed lineage and must never be pruned.
    untouched = [tmp_path/'checkpoints/unfinished.pt', tmp_path/'checkpoints/unfinished.json',
                 tmp_path/'.internal/discarded/old.pt', tmp_path/'models/old/model.pt',
                 tmp_path/'selfplay/game.npz']
    for path in untouched:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'preserve')
    return tmp_path, references, untouched


def test_retention_preserves_history_and_uncommitted_files(checkpoint_run):
    root, references, untouched = checkpoint_run
    json_hashes = {p: sha256(p) for p in (root/'checkpoints').glob('*.json')}
    other_hashes = {p: sha256(p) for p in untouched}
    history = run_history(root)
    assert [row['steps'] for row in history] == [0, 4, 4, 4]
    removed = prune_checkpoints(root, 2)
    expected = {r['path'] for (iteration, step), r in references.items()
                if (iteration, step) not in ((2, 4), (3, 4))}
    assert set(removed) == expected
    assert {r['path'] for r in references.values() if (root/r['path']).exists()} == {
        references[2, 4]['path'], references[3, 4]['path']}
    assert all(sha256(p) == digest for p, digest in {**json_hashes, **other_hashes}.items())
    assert run_history(root) == history
    assert prune_checkpoints(root, 2) == []


def test_missing_retained_payload_fails_before_deleting(checkpoint_run):
    root, references, _ = checkpoint_run
    latest = root/references[3, 4]['path']
    latest.unlink()
    before = {p: sha256(p) for p in (root/'checkpoints').glob('*.pt')}
    with pytest.raises((ValueError, FileNotFoundError)):
        prune_checkpoints(root, 2)
    assert before == {p: sha256(p) for p in (root/'checkpoints').glob('*.pt')}


def test_partial_deletion_can_be_retried(checkpoint_run, monkeypatch):
    root, references, _ = checkpoint_run
    unlink = Path.unlink
    deletions = []
    def interrupt(path, *args, **kwargs):
        if len(deletions) == 1:
            raise OSError('injected cleanup interruption')
        unlink(path, *args, **kwargs)
        deletions.append(path)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'unlink', interrupt)
        with pytest.raises(OSError, match='cleanup interruption'):
            prune_checkpoints(root, 2)
    assert len(deletions) == 1
    prune_checkpoints(root, 2)
    assert [row['steps'] for row in run_history(root)] == [0, 4, 4, 4]
    assert {key for key, r in references.items() if (root/r['path']).exists()} == {(2, 4), (3, 4)}


def test_checkpoint_keep_must_be_positive():
    config = load_config(CONFIGS / 'smoke_test')
    config['training']['checkpoint_keep'] = 0
    with pytest.raises(ValueError, match='checkpoint_keep'):
        validate(config)
