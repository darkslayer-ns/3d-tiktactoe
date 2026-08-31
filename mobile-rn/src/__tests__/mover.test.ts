/**
 * LookaheadMover tests — deterministic mock engine + scripted RNG.
 */

import { Board, P1, P2 } from '../game/board'
import { EMPTY } from '../game/types'
import type { Cell } from '../game/types'
import type { EvalEngine } from '../ai/types'
import { LookaheadMover } from '../ai/mover'
import { OpponentPredictor } from '../ai/predictor'
import { createMockEngine } from '../ai/engine'
import { scriptedRng } from './rng-helper'

/**
 * A mock engine that always prefers the first EMPTY cell. Guarantees the
 * mover's greedy/argmax moves (and the predictor's likely_moves) always land
 * on legal cells, so the search never tries to apply onto an occupied cell.
 */
function legalEngine(value = 0.5): EvalEngine {
  return createMockEngine((cells, _player, n) => {
    const policy = new Array<number>(n ** 3).fill(-10)
    const firstEmpty = cells.indexOf(0)
    if (firstEmpty >= 0) policy[firstEmpty] = 1.0
    return { value, policy }
  })
}

describe('LookaheadMover', () => {
  it('takes an immediate win', async () => {
    const b = new Board(3)
    b.apply(b.idx(0, 0, 0), P1)
    b.apply(b.idx(1, 0, 0), P1)
    const engine = legalEngine()
    const mover = new LookaheadMover(engine, b, new OpponentPredictor(b, engine), 'hard')
    expect(await mover.chooseMove(P1)).toBe(b.idx(2, 0, 0))
  })

  it('blocks an immediate loss', async () => {
    const b = new Board(3)
    b.apply(b.idx(0, 0, 0), P2)
    b.apply(b.idx(1, 0, 0), P2)
    const engine = legalEngine()
    const mover = new LookaheadMover(engine, b, new OpponentPredictor(b, engine), 'hard')
    expect(await mover.chooseMove(P1)).toBe(b.idx(2, 0, 0))
  })

  it('returns a legal move on an empty board and records the search', async () => {
    const b = new Board(3)
    const engine = legalEngine()
    const mover = new LookaheadMover(engine, b, new OpponentPredictor(b, engine), 'hard')
    const move = await mover.chooseMove(P1)
    expect(b.cells[move]).toBe(EMPTY)
    expect(mover.lastDecision?.kind).toBe('search')
    expect(mover.lastDecision?.player).toBe(P1)
    expect(mover.lastDecision?.chosen).toBe(move)
    expect(mover.lastDecision?.coord).toEqual(b.coord(move))
    expect(mover.lastDecision?.scored.length).toBeGreaterThan(0)
  })

  it('easy makes blunders up to its budget (budget > 1)', async () => {
    const b = new Board(3)
    // No two cells of the same player are collinear -> no immediate win/block,
    // so the search runs and consumes the scripted draws exactly as expected.
    const placed: Array<[number, Cell]> = [
      [0, P1], [25, P1], [5, P1], [26, P2], [4, P2], [19, P2],
    ]
    for (const [i, pl] of placed) b.apply(i, pl)

    const engine = legalEngine()
    // call1: move_temp pick -> blunder check (0.1 < 0.45) -> choice
    // call2: same again — the budget (6) is not exhausted, so it blunders again.
    const { rng, remaining } = scriptedRng([
      { kind: 'random', value: 0.99 },
      { kind: 'random', value: 0.1 },
      { kind: 'choice', value: 8, len: 21 },
      { kind: 'random', value: 0.99 },
      { kind: 'random', value: 0.1 },
      { kind: 'choice', value: 10, len: 20 },
    ])
    const mover = new LookaheadMover(engine, b, new OpponentPredictor(b, engine), 'easy', rng)

    const m1 = await mover.chooseMove(P1)
    expect(b.cells[m1]).toBe(EMPTY)
    expect(mover.lastDecision?.kind).toBe('blunder')
    b.apply(m1, P1)

    const m2 = await mover.chooseMove(P1)
    expect(b.cells[m2]).toBe(EMPTY)
    expect(mover.lastDecision?.kind).toBe('blunder')
    expect(remaining()).toBe(0)
  })

  it('returns -1 when the board is full', async () => {
    const cells: Cell[] = []
    for (let i = 0; i < 27; i++) cells.push(((i % 2) + 1) as Cell)
    const b = new Board(3, cells)
    const engine = legalEngine()
    const mover = new LookaheadMover(engine, b, new OpponentPredictor(b, engine), 'hard')
    expect(await mover.chooseMove(P1)).toBe(-1)
  })
})