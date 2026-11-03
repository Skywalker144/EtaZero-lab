"""Result counts describe completed games, independently of sampled training rows."""
import os
from pathlib import Path
import re
import subprocess

import pytest

from etazero import ROOT
from etazero.observations import OUTCOMES, selfplay_metrics


@pytest.mark.parametrize('game,size,limit,zero_rows', [
    ('freestyle', 3, 9, False), ('freestyle', 5, 25, False),
    ('freestyle', 5, 2, True), ('go', 5, 2, False),
])
def test_native_outcomes_match_sgf_and_keep_truncation_separate(tmp_path, game, size, limit, zero_rows):
    binary = Path(os.environ.get('ETAZERO_TEST_BINARY', ROOT / 'build/katago'))
    cfg = tmp_path / 'selfplay.cfg'
    cfg.write_text(f'@include {os.path.relpath(ROOT / "configs/checks/source_go.cfg", tmp_path)}\n'
                   f'game = {game}\nkomiMean = 0\nbSizes = {size}\ndataBoardLen = 5\n'
                   f'maxMovesPerGame = {limit}\nnumGameThreads = 2\nlogToStdout = false\n'
                   f'cheapSearchProb = {1 if zero_rows else 0}\ncheapSearchVisits = 4\ncheapSearchTargetWeight = 0\n')
    models, output = tmp_path / 'models', tmp_path / 'selfplay'
    models.mkdir(); output.mkdir()
    result = subprocess.run([str(binary), 'selfplay', '-config', str(cfg), '-models-dir', str(models),
                             '-output-dir', str(output), '-max-games-total', '12'],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    totals = selfplay_metrics(output, 1)
    assert totals['games'] == sum(totals[key] for key in OUTCOMES) == 12
    sgfs = [line for p in output.glob('*/sgfs/*.sgfs') for line in p.read_text().splitlines()]
    assert len(sgfs) == 12
    if limit < size * size:
        assert totals['truncated'] == 12
        assert all(totals[key] == 0 for key in OUTCOMES if key != 'truncated')
    else:
        results = [re.search(r'RE\[([^]]+)\]', line).group(1) for line in sgfs]
        assert totals['black_wins'] == sum(s.startswith('B+') for s in results)
        assert totals['white_wins'] == sum(s.startswith('W+') for s in results)
        assert totals['draws'] == sum(s in {'0', 'Draw'} for s in results)
        assert totals['no_results'] == totals['truncated'] == 0
    if zero_rows:
        assert totals['rows'] == 0

