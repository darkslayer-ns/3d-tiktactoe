import pytest

from backend.game.board import Board, P1, P2, build_lines, line_count


@pytest.mark.parametrize("n,expected", [(3, 49)])
def test_line_count(n, expected):
    assert line_count(n) == expected
    assert len(build_lines(n)) == expected


def test_line_lengths():
    for n in (3,):
        for line in build_lines(n):
            assert len(line) == n
            coords = set(line)
            assert len(coords) == n
            for c in coords:
                assert all(0 <= v < n for v in c)


def test_board_indexing():
    b = Board(3)
    assert b.idx(0, 0, 0) == 0
    assert b.idx(2, 2, 2) == 26
    assert b.idx(1, 2, 1) == 1 + 3 * 2 + 9 * 1 == 16
    b.set(1, 2, 1, P1)
    assert b.get(1, 2, 1) == P1
    assert b.coord(16) == (1, 2, 1)


def test_horizontal_win():
    b = Board(3)
    for x in range(3):
        b.set(x, 0, 0, P1)
    winner, line = b.winner()
    assert winner == P1
    assert len(line) == 3


def test_vertical_win():
    b = Board(3)
    for z in range(3):
        b.set(0, 0, z, P2)
    winner, _ = b.winner()
    assert winner == P2


def test_space_diagonal_win():
    b = Board(3)
    for i in range(3):
        b.set(i, i, i, P1)
    winner, line = b.winner()
    assert winner == P1
    assert set(line) == {(0, 0, 0), (1, 1, 1), (2, 2, 2)}


def test_no_win():
    b = Board(3)
    b.set(0, 0, 0, P1)
    b.set(1, 1, 1, P2)
    winner, line = b.winner()
    assert winner == 0
    assert line is None


def test_draw():
    b = Board(3)
    b.set(0, 0, 0, P1)
    b.set(1, 1, 1, P2)
    b.set(2, 2, 2, P1)
    assert b.is_full() is False
    winner, line, over = b.outcome()
    assert over is False
    assert winner == 0


def test_full_board_win_detected():
    """A full board where P1 owns an edge row is detected as a win."""
    cells = [
        P1, P1, P1,  # z=0, row y=0 (x 0..2) -> winning line
        P2, P2, P1,
        P2, P1, P2,
    ] + [P1, P2, P2, P2, P1, P1, P2, P1, P2] * 2
    b = Board(3, cells[:27])
    assert b.is_full()
    winner, line, over = b.outcome()
    assert over is True
    assert winner == P1
    assert line is not None