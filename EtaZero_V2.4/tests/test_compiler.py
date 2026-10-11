"""Cache configuration and compiler-adjusted wall time."""
import os
import subprocess
import sys
import pytest
from etazero import compiler


def test_cache_defaults_apply_before_torch_and_preserve_overrides(tmp_path):
    env={k:v for k,v in os.environ.items() if k not in ('TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR')}
    env['XDG_CACHE_HOME']=str(tmp_path)
    code="import etazero,os,sys; assert 'torch' not in sys.modules; print(os.environ['TORCHINDUCTOR_CACHE_DIR']); print(os.environ['TRITON_CACHE_DIR'])"
    def paths():
        return subprocess.check_output([sys.executable,'-c',code],env=env,text=True).splitlines()
    assert paths()==[str(tmp_path/'etazero/compile-v1'/name) for name in ('inductor','triton')]
    env['TORCHINDUCTOR_CACHE_DIR']=str(tmp_path/'custom-inductor')
    assert paths()==[env['TORCHINDUCTOR_CACHE_DIR'],str(tmp_path/'etazero/compile-v1/triton')]
    env['TRITON_CACHE_DIR']=str(tmp_path/'custom-triton')
    assert paths()==[env['TORCHINDUCTOR_CACHE_DIR'],env['TRITON_CACHE_DIR']]
    assert not list(tmp_path.iterdir())  # Import alone does not create caches.


def test_timer_excludes_only_this_attempt_compilation(monkeypatch):
    wall=iter([100.,115.,200.,208.]);compile_time=iter([50.,56.,80.,82.])
    monkeypatch.setattr(compiler.time,'monotonic',lambda:next(wall))
    monkeypatch.setattr(compiler,'compilation_seconds',lambda:next(compile_time))
    assert compiler.WorkTimer(True).finish()==dict(seconds=9.,wall_seconds=15.,compile_seconds=6.)
    # Earlier failed attempts/other runs must not be subtracted again.
    assert compiler.WorkTimer(True).finish()==dict(seconds=6.,wall_seconds=8.,compile_seconds=2.)


def test_uncompiled_timer_and_invalid_counter_reset(monkeypatch):
    wall=iter([10.,14.,20.,25.])
    monkeypatch.setattr(compiler.time,'monotonic',lambda:next(wall))
    monkeypatch.setattr(compiler,'compilation_seconds',lambda:pytest.fail('Eager must not read compiler counters'))
    assert compiler.WorkTimer(False).finish()==dict(seconds=4.,wall_seconds=4.,compile_seconds=0.)
    compile_time=iter([20.,0.])
    monkeypatch.setattr(compiler,'compilation_seconds',lambda:next(compile_time))
    with pytest.raises(RuntimeError,match='Invalid compiler timing'):
        compiler.WorkTimer(True).finish()
