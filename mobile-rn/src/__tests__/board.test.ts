/**
 * Board core tests — mirror the semantics of backend/tests/test_board.py.
 */

import { Board, P1, P2, buildLines, lineCount } from '../game/board'
import { EMPTY } from '../game/types'
import type { Cell } from '../game/types'

describe('line counts', () => {
  it.each([
    [3, 49],
    [4, 76],
    [5, 109],
  ])('n=%i has %i lines', (n, expected) => {
    expect(lineCount(n)).toBe(expected)
    expect(buildLines(n)).toHaveLength(expected)
  })

  it('builds valid lines (length n, distinct in-range coords)', () => {
    for (const n of [3]) {
      for (const line of buildLines(n)) {
        expect(line).toHaveLength(n)
        const seen = new Set(line.map((c) => c.join(',')))
        expect(seen.size).toBe(n)
        for (const c of line) {
          for (const v of c) {
            expect(v).toBeGreaterThanOrEqual(0)
            expect(v).toBeLessThan(n)
          }
        }
      }
    }
  })
})

describe('indexing', () => {
  it('idx / coord round-trip', () => {
    const b = new Board(3)
    expect(b.idx(0, 0, 0)).toBe(0)
    expect(b.idx(2, 2, 2)).toBe(26)
    expect(b.idx(1, 2, 1)).toBe(16)
    expect(b.coord(16)).toEqual([1, 2, 1])
    for (let i = 0; i < 27; i++) {
      const [x, y, z] = b.coord(i)
      expect(b.idx(x, y, z)).toBe(i)
    }
  })

  it('set / get', () => {
    const b = new Board(3)
    b.set(1, 2, 1, P1)
    expect(b.get(1, 2, 1)).toBe(P1)
    expect(b.isEmpty(1, 2, 1)).toBe(false)
  })
})

describe('winner / outcome', () => {
  it('detects a horizontal win', () => {
    const b = new Board(3)
    for (let x = 0; x < 3; x++) b.set(x, 0, 0, P1)
    const { player, line } = b.winner()
    expect(player).toBe(P1)
    expect(line).toHaveLength(3)
  })

  it('detects a vertical win', () => {
    const b = new Board(3)
    for (let z = 0; z < 3; z++) b.set(0, 0, z, P2)
    const { player } = b.winner()
    expect(player).toBe(P2)
  })

  it('detects a space diagonal win', () => {
    const b = new Board(3)
    for (let i = 0; i < 3; i++) b.set(i, i, i, P1)
    const { player, line } = b.winner()
    expect(player).toBe(P1)
    expect(new Set(line!.map((c) => c.join(',')))).toEqual(
      new Set(['0,0,0', '1,1,1', '2,2,2']),
    )
  })

  it('reports no win', () => {
    const b = new Board(3)
    b.set(0, 0, 0, P1)
    b.set(1, 1, 1, P2)
    const { player, line } = b.winner()
    expect(player).toBe(EMPTY)
    expect(line).toBeNull()
  })

  it('draw: not full, not over', () => {
    const b = new Board(3)
    b.set(0, 0, 0, P1)
    b.set(1, 1, 1, P2)
    b.set(2, 2, 2, P1)
    expect(b.isFull()).toBe(false)
    const o = b.outcome()
    expect(o.over).toBe(false)
    expect(o.winner).toBe(EMPTY)
  })

  it('full board with a winning edge row is a win', () => {
    const cells: Cell[] = [
      P1, P1, P1, // z=0, row y=0 (x 0..2) -> winning line
      P2, P2, P1,
      P2, P1, P2,
      P1, P2, P2, P2, P1, P1, P2, P1, P2,
      P1, P2, P2, P2, P1, P1, P2, P1, P2,
    ]
    const b = new Board(3, cells)
    expect(b.isFull()).toBe(true)
    const o = b.outcome()
    expect(o.over).toBe(true)
    expect(o.winner).toBe(P1)
    expect(o.line).not.toBeNull()
  })
})

describe('state / moves', () => {
  it('moves() returns only empty cells', () => {
    const b = new Board(3)
    b.set(0, 0, 0, P1)
    b.set(1, 1, 1, P2)
    const ms = b.moves()
    expect(ms).toHaveLength(25)
    expect(ms).not.toContain(0)
    expect(ms).not.toContain(13)
    expect(ms).toContain(26)
    expect(b.emptyCount()).toBe(25)
    expect(b.moveCount()).toBe(25)
  })

  it('copy() is independent', () => {
    const b = new Board(3)
    b.set(0, 0, 0, P1)
    const c = b.copy()
    c.set(2, 2, 2, P2)
    expect(b.get(2, 2, 2)).toBe(EMPTY)
    expect(c.get(2, 2, 2)).toBe(P2)
    expect(b.get(0, 0, 0)).toBe(P1)
    expect(c.get(0, 0, 0)).toBe(P1)
  })

  it('apply rejects occupied cells', () => {
    const b = new Board(3)
    b.apply(0, P1)
    expect(() => b.apply(0, P2)).toThrow(/already occupied/)
  })

  it('rejects unsupported sizes', () => {
    expect(() => new Board(6)).toThrow(/unsupported board size/)
  })

  it('coordOfLine maps coords to indices', () => {
    const b = new Board(3)
    expect(b.coordOfLine([[0, 0, 0], [1, 0, 0], [2, 0, 0]])).toEqual([0, 1, 2])
    expect(b.coordOfLine([[2, 2, 2]])).toEqual([26])
  })
})