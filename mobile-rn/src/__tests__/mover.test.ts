/**
 * LookaheadMover tests — deterministic mock engine + scripted RNG.
 */

import { Board, P1, P2 } from '../game/board'
import { EMPTY } from '../game/types'
import type { Cell } from '../game/types'
import type { Difficulty, EvalEngine } from '../ai/types'
import { LookaheadMover, DIFFICULTY } from '../ai/mover'
import { OpponentPredictor } from '../ai/predictor'
import { createMockEngine } from '../ai/engine'
import { scriptedRng } from './rng-helper'
import type { Rng } from '../ai/math'

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

  it('getHint returns the immediate winning cell', async () => {
    const b = new Board(3)
    b.apply(b.idx(0, 0, 0), P1)
    b.apply(b.idx(1, 0, 0), P1)
    const engine = legalEngine()
    const mover = new LookaheadMover(engine, b, new OpponentPredictor(b, engine), 'hard')
    expect(await mover.getHint(P1)).toBe(b.idx(2, 0, 0))
  })

  it('getHint blocks an immediate threat for the human', async () => {
    const b = new Board(3)
    b.apply(b.idx(0, 0, 0), P2)
    b.apply(b.idx(1, 0, 0), P2)
    const engine = legalEngine()
    const mover = new LookaheadMover(engine, b, new OpponentPredictor(b, engine), 'hard')
    expect(await mover.getHint(P1)).toBe(b.idx(2, 0, 0))
  })

  it('getHint returns a legal empty move on a fresh board', async () => {
    const b = new Board(3)
    const engine = legalEngine()
    const mover = new LookaheadMover(engine, b, new OpponentPredictor(b, engine), 'hard')
    const hint = await mover.getHint(P1)
    expect(hint).toBeGreaterThanOrEqual(0)
    expect(b.cells[hint]).toBe(EMPTY)
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

  it('batched (evalPositions) picks the same moves as sequential evalPosition', async () => {
    // Deterministic LCG so both movers consume identical RNG draws.
    const lcg = (seed: number): Rng => {
      let s = seed >>> 0
      const next = () => {
        s = (s * 1664525 + 1013904223) >>> 0
        return s / 4294967296
      }
      return { random: next, choice: (arr) => arr[Math.floor(next() * arr.length)] }
    }
    // Raw net: value logit = sum of empties, policy favors low empty indices.
    const rawFn = (_norm: number[], mask: number[], n: number): { value: number; policy: number[] } => {
      const N = n ** 3
      const empties = mask.reduce((a, b) => a + (b ? 1 : 0), 0)
      const policy = new Array<number>(N).fill(-5)
      for (let i = 0; i < N; i++) {
        if (mask[i]) {
          policy[i] = i < empties ? 1.0 : 0.5
        }
      }
      return { value: empties * 0.1, policy }
    }

    const sequential: EvalEngine = {
      evalPosition(cells, player, n) {
        const size = n ?? Math.round(Math.cbrt(cells.length))
        const norm = new Array<number>(size ** 3)
        const mask = new Array<number>(size ** 3)
        for (let i = 0; i < cells.length; i++) {
          const c = cells[i]
          norm[i] = c === 0 ? 0 : c === player ? 1 : 2
          mask[i] = c === 0 ? 1 : 0
        }
        return rawFn(norm, mask, size)
      },
    }
    const batched: EvalEngine = {
      evalPosition: sequential.evalPosition,
      evalPositions(boards, masks, n) {
        const values: number[] = []
        const policies: number[] = []
        const N = n ** 3
        for (let i = 0; i < boards.length; i += N) {
          const r = rawFn(
            boards.slice(i, i + N) as number[],
            masks.slice(i, i + N) as number[],
            n,
          )
          values.push(r.value)
          policies.push(...r.policy)
        }
        return { values, policies }
      },
    }

    const positions: Array<[Cell[], Difficulty]> = [
      [new Array<Cell>(27).fill(0), 'easy'],
      [new Array<Cell>(27).fill(0), 'medium'],
      [new Array<Cell>(27).fill(0), 'hard'],
    ]
    for (const [cells, difficulty] of positions) {
      const b1 = new Board(3, cells)
      const b2 = new Board(3, cells)
      const m1 = new LookaheadMover(sequential, b1, new OpponentPredictor(b1, sequential), difficulty, lcg(1234))
      const m2 = new LookaheadMover(batched, b2, new OpponentPredictor(b2, batched), difficulty, lcg(1234))
      expect(await m2.chooseMove(P1)).toBe(await m1.chooseMove(P1))
    }
  })

  it("denies the player's overused cells (persisted heatmap)", async () => {
    const b = new Board(3)
    const engine = legalEngine()
    const heat = new Map<number, Map<number, number>>()
    heat.set(P2, new Map([[13, 10]])) // human (P2) heavily overuses the centre
    const pred = new OpponentPredictor(b, engine, 0.9, heat)
    const mover = new LookaheadMover(engine, b, pred, 'hard')
    const m = await mover.chooseMove(P1)
    const top = mover.lastDecision?.scored ?? []
    expect(top[0]?.index).toBe(13) // the overused centre ranks first (AI denies it)
    expect(m).toBeGreaterThanOrEqual(0)
  })

  it('difficulty blunder kinds are configured', () => {
    expect(DIFFICULTY.easy.blunder_kind).toBe('random')
    expect(DIFFICULTY.medium.blunder_kind).toBe('suboptimal')
    expect(DIFFICULTY.hard.blunder_kind).toBe('none')
  })

  it('medium blunders ONCE when about to win (not the winning move)', async () => {
    const b = new Board(3)
    b.apply(b.idx(0, 0, 0), P1)
    b.apply(b.idx(1, 0, 0), P1) // P1 (AI) can win at (2,0,0)
    const engine = legalEngine()
    // blunder check (0.1 < 0.7) → blunders; choice picks a NON-winning move (5th of the 24 others).
    const { rng, remaining } = scriptedRng([
      { kind: 'random', value: 0.1 },
      { kind: 'choice', value: 5, len: 24 },
    ])
    const mover = new LookaheadMover(engine, b, new OpponentPredictor(b, engine), 'medium', rng)
    expect(mover.blunderKind).toBe('suboptimal')
    const m = await mover.chooseMove(P1)
    expect(mover.lastDecision?.kind).toBe('blunder')
    expect(m).not.toBe(b.idx(2, 0, 0)) // throws away the win
    expect(b.cells[m]).toBe(EMPTY)
    expect(remaining()).toBe(0)

    // Budget is spent (wrong_move_budget = 1): next identical win is taken.
    const b2 = new Board(3)
    b2.apply(b2.idx(0, 0, 0), P1)
    b2.apply(b2.idx(1, 0, 0), P1)
    const rng2 = scriptedRng([{ kind: 'random', value: 0.9 }]) // blunder check fails → takes win
    const mover2 = new LookaheadMover(engine, b2, new OpponentPredictor(b2, engine), 'medium', rng2.rng)
    const m2 = await mover2.chooseMove(P1)
    expect(m2).toBe(b2.idx(2, 0, 0))
    expect(mover2.lastDecision?.kind).toBe('search')
    expect(rng2.remaining()).toBe(0)
  })
})