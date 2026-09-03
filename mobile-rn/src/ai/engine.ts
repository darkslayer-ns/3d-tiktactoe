/**
 * EvalEngine factories.
 *
 * - createNativeEngine(): the real on-device engine backed by the C++ JSI
 *   module (src/native/TfmEngine). Falls back to an error-throwing engine if
 *   the native module is missing (e.g. web/dev).
 * - createMockEngine(): injects a fixed eval function — used by tests and by
 *   the parity gate to replay recorded backend outputs.
 */

import type { Cell } from '../game/types'
import type { EvalEngine, EvalResult } from './types'
import {
  evalPosition,
  evalPositions,
  isAvailable,
  load,
  predictedLine,
  searchScored,
  searchScoredAsync,
} from '../native/TfmEngine'

/**
 * Bounded LRU over normalized board signatures. The search calls evalPosition
 * on the SAME position repeatedly (apply → eval → undo), so a tiny result
 * cache removes a chunk of duplicate native forwards. Pure function — safe.
 */
class BoardMemo {
  private max: number
  private map = new Map<string, EvalResult>()
  constructor(max = 4096) {
    this.max = max
  }
  private key(norm: readonly number[], mask: readonly number[], n: number): string {
    let s = n + ':'
    for (let i = 0; i < norm.length; i++) s += norm[i] + (mask[i] ? '1' : '0')
    return s
  }
  get(norm: readonly number[], mask: readonly number[], n: number): EvalResult | undefined {
    const k = this.key(norm, mask, n)
    const v = this.map.get(k)
    if (v) {
      // LRU touch: re-insert at the end
      this.map.delete(k)
      this.map.set(k, v)
    }
    return v
  }
  set(norm: readonly number[], mask: readonly number[], n: number, res: EvalResult): void {
    this.map.set(this.key(norm, mask, n), res)
    if (this.map.size > this.max) {
      const oldest = this.map.keys().next().value
      if (oldest !== undefined) this.map.delete(oldest)
    }
  }
}

export const boardMemo = new BoardMemo()

class NativeEngine implements EvalEngine {
  constructor() {
    load()
  }

  evalPosition(cells: readonly Cell[], player: Cell, n?: number): EvalResult {
    const size = n ?? Math.round(Math.cbrt(cells.length))
    const norm: number[] = new Array(cells.length)
    const mask: number[] = new Array(cells.length)
    for (let i = 0; i < cells.length; i++) {
      const c = cells[i]
      norm[i] = c === 0 ? 0 : c === player ? 1 : 2
      mask[i] = c === 0 ? 1 : 0
    }
    const cached = boardMemo.get(norm, mask, size)
    if (cached) return cached
    const res = evalPosition(norm, mask, size)
    boardMemo.set(norm, mask, size, res)
    return res
  }

  evalPositions(
    boards: readonly number[],
    masks: readonly number[],
    n: number,
  ): { values: number[]; policies: number[] } {
    return evalPositions(boards as number[], masks as number[], n)
  }

  searchScored(
    cells: readonly Cell[],
    ai: Cell,
    depth: number,
    topK: number,
    maxNodes: number,
    aggression: number,
    n: number,
  ): { moves: number[]; values: number[] } {
    return searchScored(cells as number[], ai, depth, topK, maxNodes, aggression, n)
  }

  async searchScoredAsync(
    cells: readonly Cell[],
    ai: Cell,
    depth: number,
    topK: number,
    maxNodes: number,
    aggression: number,
    n: number,
  ): Promise<{ moves: number[]; values: number[] }> {
    return searchScoredAsync(cells as number[], ai, depth, topK, maxNodes, aggression, n)
  }

  predictedLine(
    cells: readonly Cell[],
    ai: Cell,
    chosen: number,
    depth: number,
    n: number,
  ): { players: number[]; indices: number[] } {
    return predictedLine(cells as number[], ai, chosen, depth, n)
  }
}

export function createNativeEngine(): EvalEngine {
  if (!isAvailable()) {
    throw new Error(
      'Native TfmEngine unavailable — build the app via `npx expo run:android|ios`',
    )
  }
  return new NativeEngine()
}

export function createMockEngine(
  fn: (cells: readonly Cell[], player: Cell, n: number) => EvalResult,
): EvalEngine {
  return {
    evalPosition(cells, player, n) {
      return fn(cells, player, n ?? Math.round(Math.cbrt(cells.length)))
    },
  }
}