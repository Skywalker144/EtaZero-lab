"""Durable authority, interrupted rollback, artifact integrity and controller ownership."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from etazero.export import verify_export
from etazero.runtime import recover_iteration, run_training
from etazero.schema import CONTRACT_ID
from etazero.storage import save_json, sha256


def exported(root):
    path=root/'models/fixed/model.pt';path.parent.mkdir(parents=True);path.write_bytes(b'complete artifact')
    info={'id':'fixed','path':'models/fixed/model.pt','contract':CONTRACT_ID,'canvas':6,
          'checkpoint':{'id':'fixed'},'weights':'model','sha256':sha256(path),
          'verification':{'python_scripted':True,'native':True}}
    save_json(path.parent/'manifest.json',info)
    return path,info


def test_export_requires_complete_manifest_and_matching_payload(tmp_path):
    path,info=exported(tmp_path)
    assert verify_export(tmp_path,info,6)==info
    path.write_bytes(b'partial')
    with pytest.raises(ValueError,match='checksum'):verify_export(tmp_path,info,6)
    path.write_bytes(b'complete artifact')
    modified={**info,'weights':'swa'}
    with pytest.raises(ValueError,match='manifest'):verify_export(tmp_path,modified,6)
    with pytest.raises(ValueError,match='contract'):verify_export(tmp_path,info,7)
    modified={**info,'path':'models/.tmp_fixed/model.pt'}
    with pytest.raises(ValueError,match='identity'):verify_export(tmp_path,modified,6)


def test_rollback_interrupted_rename_retries_without_losing_artifacts(tmp_path,monkeypatch):
    state={'iteration':2,'elapsed_seconds':17,'checkpoint':None,'model':None}
    names=['models/.tmp_partial/model.pt','selfplay/iteration_000002/raw.npz',
           '.internal/iterations/000002/status.json','logs/iterations/000002.json']
    for name in names:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(name.encode())
    original=Path.rename;calls=0
    def interrupted(self,target):
        nonlocal calls
        calls+=1
        if calls==2:raise OSError('injected rename failure')
        return original(self,target)
    with monkeypatch.context() as patch:
        patch.setattr(Path,'rename',interrupted)
        with pytest.raises(OSError,match='injected'):recover_iteration(tmp_path,state)
    recover_iteration(tmp_path,state)
    for name in names:
        saved=list((tmp_path/'.internal/discarded').glob('*/'+name))
        assert len(saved)==1 and saved[0].read_bytes()==name.encode()
        assert not (tmp_path/name).exists()
    before=sorted(str(p) for p in (tmp_path/'.internal/discarded').rglob('*'))
    recover_iteration(tmp_path,state)
    assert before==sorted(str(p) for p in (tmp_path/'.internal/discarded').rglob('*'))


def test_second_controller_cannot_enter_owned_run(tmp_path):
    internal=tmp_path/'.internal';internal.mkdir()
    script="import fcntl,sys; f=open(sys.argv[1],'a+'); fcntl.flock(f,fcntl.LOCK_EX); print('ready',flush=True); sys.stdin.readline()"
    child=subprocess.Popen([sys.executable,'-c',script,str(internal/'run.lock')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    try:
        assert child.stdout.readline().strip()=='ready'
        with pytest.raises(RuntimeError,match='Another controller'):
            run_training(tmp_path,{},tmp_path/'missing_binary',resume=False)
    finally:
        child.communicate('\n',timeout=10)
    with pytest.raises(FileNotFoundError,match='executable'):
        run_training(tmp_path,{},tmp_path/'missing_binary',resume=False)


def test_atomic_immutable_competing_processes_publish_one_whole_payload(tmp_path):
    output=tmp_path/'payload';gate=tmp_path/'gate'
    code="""import sys,time
from pathlib import Path
from etazero.storage import atomic_write
p,gate=map(Path,sys.argv[1:3]);letter=sys.argv[3]
print('ready',flush=True)
while not gate.exists():time.sleep(.001)
try:
 atomic_write(p,lambda q:q.write_bytes(letter.encode()*1048576),immutable=True)
 print('won',flush=True)
except FileExistsError:print('lost',flush=True)
"""
    children=[subprocess.Popen([sys.executable,'-c',code,str(output),str(gate),letter],stdout=subprocess.PIPE,text=True) for letter in ('A','B')]
    try:
        assert all(p.stdout.readline().strip()=='ready' for p in children)
        gate.touch()
        outcomes=[p.communicate(timeout=10)[0].strip() for p in children]
        assert sorted(outcomes)==['lost','won'] and all(p.returncode==0 for p in children)
        assert output.read_bytes() in (b'A'*1048576,b'B'*1048576)
        assert not list(tmp_path.glob('payload.tmp.*'))
    finally:
        for p in children:
            if p.poll() is None:p.kill();p.wait()


def test_recovery_preserves_valid_publication_time_and_repairs_only_stale_pointer(tmp_path):
    state={'iteration':2,'elapsed_seconds':17,'checkpoint':None,'model':{'id':'committed'}}
    pointer=tmp_path/'models/current.json'
    save_json(pointer,{'model':state['model'],'elapsed_seconds':17,'utc':'original publication'})
    original=pointer.read_bytes()
    recover_iteration(tmp_path,state)
    assert pointer.read_bytes()==original
    for value in ({'model':{'id':'stale'},'elapsed_seconds':17},['malformed pointer'],None):
        if value is None:pointer.write_bytes(b'partial JSON')
        else:save_json(pointer,value)
        recover_iteration(tmp_path,state)
        assert json.loads(pointer.read_text())=={'model':state['model'],'elapsed_seconds':17}


def test_concurrent_reader_sees_only_complete_published_payloads(tmp_path):
    code="""import sys,time
from pathlib import Path
from etazero.storage import atomic_write,save_json,sha256
root=Path(sys.argv[1])
print('ready',flush=True)
while not (root/'gate').exists():time.sleep(.001)
for i in range(12):
 p=root/f'model_{i}'
 atomic_write(p,lambda q:q.write_bytes(bytes([i])*131072),immutable=True)
 save_json(root/'current.json',{'path':p.name,'sha256':sha256(p),'byte':i})
"""
    child=subprocess.Popen([sys.executable,'-c',code,str(tmp_path)],stdout=subprocess.PIPE,text=True)
    observed=set();pointer=tmp_path/'current.json'
    try:
        assert child.stdout.readline().strip()=='ready'
        (tmp_path/'gate').touch()
        while child.poll() is None:
            if pointer.exists():
                value=json.loads(pointer.read_text());payload=tmp_path/value['path']
                assert sha256(payload)==value['sha256'] and payload.read_bytes()==bytes([value['byte']])*131072
                observed.add(value['byte'])
        assert child.returncode==0 and observed
        value=json.loads(pointer.read_text());assert value['byte']==11 and sha256(tmp_path/value['path'])==value['sha256']
    finally:
        if child.poll() is None:child.kill();child.wait()
