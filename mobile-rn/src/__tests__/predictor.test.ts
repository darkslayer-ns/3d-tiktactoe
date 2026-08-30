/**
 * OpponentPredictor tests — deterministic mock engine.
 */

import { Board, P1 } from '../game/board'
import { EMPTY } from '../game/types'
import type { Cell } from '../game/types'
import { OpponentPredictor } from '../ai/predictor'
import { createMockEngine } from '../ai/engine'

describe('OpponentPredictor', () => {
  it('returns a legal distribution summing to 1 (full topK)', () => {
    const b = new Board(3)
    const engine = createMockEngine(() => ({ value: 0.5, policy: new Array(27).fill(0) }))
    const pred = new OpponentPredictor(b, engine)
    const dist = pred.predictDistribution(P1, 1.0, 27)
    expect(dist).toHaveLength(27)
    let sum = 0
    for (const [idx, p] of dist) {
      expect(b.cells[idx]).toBe(EMPTY)
      expect(p).toBeGreaterThanOrEqual(0)
      expect(p).toBeLessThanOrEqual(1)
      sum += p
    }
    expect(sum).toBeCloseTo(1.0, 6)
  })

  it('likelyMoves truncates to topK and keeps the highest-probability cells', () => {
    const b = new Board(3)
    const engine = createMockEngine((cells, _player, n) => {
      const policy = new Array<number>(n ** 3).fill(-1)
      let heated = 0
      for (let i = 0; i < cells.length && heated < 2; i++) {
        if (cells[i] === EMPTY) {
          policy[i] = heated === 0 ? 5 : 4
          heated++
        }
      }
      return { value: 0, policy }
    })
    const pred = new OpponentPredictor(b, engine)
    const top = pred.likelyMoves(P1, 2)
    expect(top).toHaveLength(2)
    expect(top[0][0]).toBe(0)
    expect(top[1][0]).toBe(1)
  })

  it('returns [] for a full board', () => {
    const cells: Cell[] = []
    for (let i = 0; i < 27; i++) cells.push(((i % 2) + 1) as Cell)
    const b = new Board(3, cells)
    const engine = createMockEngine(() => ({ value: 0, policy: [] }))
    const pred = new OpponentPredictor(b, engine)
    expect(pred.predictDistribution(P1, 1.0, 8)).toEqual([])
  })

  it('ranks a recorded (affinity) cell first', () => {
    const b = new Board(3)
    const engine = createMockEngine(() => ({ value: 0.5, policy: new Array(27).fill(0) }))
    const pred = new OpponentPredictor(b, engine)
    pred.record(P1, 5)
    pred.record(P1, 5)
    const dist = pred.predictDistribution(P1, 1.0, 27)
    expect(dist[0][0]).toBe(5)
  })

  it('decays affinity across newGame()', () => {
    const b = new Board(3)
    const engine = createMockEngine(() => ({ value: 0.5, policy: new Array(27).fill(0) }))
    const pred = new OpponentPredictor(b, engine)
    pred.record(P1, 5)
    pred.newGame()
    const dist = pred.predictDistribution(P1, 1.0, 27)
    // affinity now 0.9: p(cell 5) = exp(0.9) / (exp(0.9) + 26*exp(0))
    expect(dist[0][0]).toBe(5)
    expect(dist[0][1]).toBeCloseTo(1 / (1 + 26 * Math.exp(-0.9)), 6)
  })

  it('predictNext returns the top cell', () => {
    const b = new Board(3)
    const engine = createMockEngine((cells, _player, n) => {
      const policy = new Array<number>(n ** 3).fill(-1)
      if (cells[0] === EMPTY) policy[0] = 5
      return { value: 0, policy }
    })
    const pred = new OpponentPredictor(b, engine)
    expect(pred.predictNext(P1)).toBe(0)
  })
})