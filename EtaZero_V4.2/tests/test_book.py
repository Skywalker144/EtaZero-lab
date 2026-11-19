"""Exercise native book storage and diagnostics without a neural evaluator."""
import json
import os
from pathlib import Path
import re
import subprocess

import pytest

from etazero import ROOT


GAMES = ('go', 'freestyle', 'standard', 'renju', 'hex', 'xiangqi')
BINARY = Path(os.environ.get('ETAZERO_BOOK_BINARY', ROOT / 'build/katago'))


def run(*args, cwd=None, success=True):
    result = subprocess.run([str(BINARY), *map(str, args)], cwd=cwd, capture_output=True, text=True, timeout=90)
    text = result.stdout + result.stderr
    assert (result.returncode == 0) == success, text
    return text


@pytest.fixture(scope='module')
def books(tmp_path_factory):
    directory = tmp_path_factory.mktemp('native-books')
    text = run('runtests', '-book-games', directory)
    assert 'Multi-game book tests passed' in text
    (directory / 'core.log').write_text(text)
    return directory


def test_source_go_book_regression(tmp_path):
    text = run('runtests', '-book', cwd=tmp_path)
    (tmp_path / 'source-book.log').write_text(text)
    assert 'Done' in text
    assert (tmp_path / 'tests/results/bookhtml/root/root.html').is_file()


@pytest.mark.parametrize('game', (*GAMES, 'xiangqi-history', 'xiangqi-check', 'xiangqi-chase', 'xiangqi-limit'))
def test_native_integrity_and_comparison(books, tmp_path, game):
    book = books / f'{game}.katabook'
    text = run('checkbook', '-book-file', book)
    assert 'Checking book' in text
    assert 'WARNING' not in text and 'failed integrity' not in text
    lines = book.read_text().splitlines()
    node_count = int(lines[1])
    same = run('comparebooks', '-book1', book, '-book2', book)
    assert f'Total common positions: {node_count}' in same
    assert 'Positions reported: 0' in same
    # Alter leaf values so the source comparison prints paths, including joint
    # Xiangqi moves, rather than only exercising the zero-difference branch.
    for i in range(node_count + 2, len(lines)):
        data = json.loads(lines[i])
        data['wl'] = -0.6
        lines[i] = json.dumps(data)
    changed = tmp_path / 'changed.katabook'
    changed.write_text('\n'.join(lines) + '\n')
    compared = run('comparebooks', '-book1', book, '-book2', changed)
    rows = [line.split(',') for line in compared.splitlines() if re.match(r'^[0-9A-Fa-f]{64},', line)]
    assert len(rows) == node_count
    for row in rows:
        assert float(row[1]) >= 0.6
        pattern = r'[a-i][0-9][a-i][0-9]' if game.startswith('xiangqi') else r'(?:[A-HJ-Z][0-9]+|pass)'
        assert all(re.fullmatch(pattern, move) for move in row[5].split())


@pytest.mark.parametrize('game', GAMES)
def test_native_bottleneck_paths(books, tmp_path, game):
    output = tmp_path / 'bottlenecks.jsonl'
    text = run('findbookbottlenecks', '-book-file', books / f'{game}.katabook', '-output-jsonl', output,
               '-max-depth', 10, '-min-visits', 0, '-add-wlchange-to-depth-factor', 1,
               '-winloss-delta', 0.1, '-discretization-increment', 0.05, '-num-threads', 2)
    assert 'WARNING' not in text
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows
    pattern = r'[a-i][0-9][a-i][0-9]' if game == 'xiangqi' else r'(?:[A-HJ-Z][0-9]+|pass)'
    for row in rows:
        for moves in [row['moveHistory'], *row['moveHistoryToChangeStrings']]:
            assert all(re.fullmatch(pattern, move) for move in moves)


def test_complete_xiangqi_initial_history(books, tmp_path):
    original = (books / 'xiangqi-history.katabook').read_text().splitlines()
    header = json.loads(original[0])
    assert header['initialTurnNumber'] == 97
    assert header['xiangqiHistory'] == ['b0c2', 'b9c7', 'c2b0', 'c7b9']
    assert header['xiangqiFen'].endswith('0 1')
    # Missing state and a mismatching player are rejected at the file boundary.
    for key, value in [('xiangqiFen', None), ('initialPla', 'W')]:
        changed = dict(header)
        if value is None:
            del changed[key]
        else:
            changed[key] = value
        path = tmp_path / f'{key}.katabook'
        path.write_text('\n'.join([json.dumps(changed), *original[1:]]) + '\n')
        text = run('checkbook', '-book-file', path, success=False)
        assert ('xiangqiFen' if value is None else 'complete initial state') in text

