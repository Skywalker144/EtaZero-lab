"""Only callers with stopped writers may skip the source filesystem wait."""
import runpy
import sys
import time

import pytest

from etazero import ROOT


@pytest.mark.parametrize('stable', [False, True])
def test_shuffle_wait_is_opt_in(tmp_path, monkeypatch, stable):
    source = tmp_path / 'input'
    source.mkdir()
    sleeps = []
    monkeypatch.setattr(time, 'sleep', sleeps.append)
    args = [str(ROOT / 'python/shuffle.py'), str(source),
            '-out-dir', str(tmp_path / 'output'), '-out-tmp-dir', str(tmp_path / 'scratch'),
            '-num-processes', '1', '-keep-target-rows', 'all']
    if stable:
        args.append('-input-files-stable')
    monkeypatch.setattr(sys, 'argv', args)
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(args[0], run_name='__main__')
    assert exit_info.value.code == 0
    assert sleeps == ([] if stable else [3])
