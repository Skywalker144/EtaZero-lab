"""Joint action semantics and generated native/Python layout agreement."""
from pathlib import Path

import pytest

from katago.game import xiangqi as xq


def test_complete_vocabulary_and_native_generated_table():
    assert len(xq.MOVES) == len(set(xq.MOVES)) == 2086
    assert len(xq.DELTAS) == len(set(xq.DELTAS)) == 50
    root = Path(__file__).resolve().parents[1]
    assert (root / 'cpp/game/xiangqilayout.h').read_text() == xq.cpp_header()
    # All file/rank moves and horse jumps exist at every physical square.
    for x in range(9):
        for y in range(10):
            for u in range(9):
                if x != u:
                    assert (x, y, u, y) in xq.INDEX
            for v in range(10):
                if y != v:
                    assert (x, y, x, v) in xq.INDEX
    assert (3, 0, 4, 1) in xq.INDEX and (2, 0, 4, 2) in xq.INDEX
    assert (3, 2, 4, 3) not in xq.INDEX  # outside the advisor palace
    assert (2, 4, 4, 6) not in xq.INDEX  # elephant cannot cross the river


@pytest.mark.parametrize('black', [False, True])
@pytest.mark.parametrize('mirror', [False, True])
def test_orientation_is_an_involution_with_joint_policy_indices(black, mirror):
    permutation = xq.action_permutation(black=black, mirror=mirror)
    assert sorted(permutation) == list(range(2086))
    assert all(permutation[permutation[i]] == i for i in range(2086))
    indices = xq.policy_indices(10)
    assert len(set(indices)) == 2086
    for action, (x, y, u, v) in enumerate(xq.MOVES):
        cell, channel = indices[action] % 100, indices[action] // 100
        assert (cell % 10, cell // 10) == (x, y)
        dx, dy = xq.DELTAS[channel]
        assert (x + dx, y + dy) == (u, v)
    # b0c2 becomes h9g7 for black, then b9c7 if also mirrored.
    expected = (1, 9, 2, 7) if black and mirror else (7, 9, 6, 7) if black else (7, 0, 6, 2) if mirror else (1, 0, 2, 2)
    assert xq.MOVES[permutation[xq.INDEX[1, 0, 2, 2]]] == expected


def test_canvas_must_contain_the_physical_board():
    for size in (9, 0, True):
        with pytest.raises(ValueError):
            xq.policy_indices(size)
