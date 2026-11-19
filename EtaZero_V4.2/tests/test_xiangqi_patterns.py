"""Real joint-move patterns, source SVB lifecycle, SGF/auto-pattern filtering."""
import subprocess

from etazero import ROOT


def test_native_xiangqi_patterns_and_source_updates(tmp_path):
    result = subprocess.run([str(ROOT / 'build/katago'), 'runtests', '-xiangqi-patterns', str(tmp_path)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'source SVB updates and pattern files passed' in result.stdout
    assert len(list((tmp_path / 'poses').glob('*.txt'))) == 1
