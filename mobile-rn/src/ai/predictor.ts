/**
 * Model-based opponent predictor. Byte-for-byte port of
 * backend/ml/predictor.py.
 *
 * Predicts where the human is likely to move next from the trained network:
 * candidate moves are scored by the net's value head (how good the position
 * becomes for the human if they play there), blended with the player's learned
 * affinity. Probabilities come from softmax. Pure TS — no RN/three imports.
 */

import { Board } from '../game/board'
import { EMPTY, P1, P2 } from '../game/types'
import type { Cell } from '../game/types'
import type { EvalEngine } from './types'
import { sigmoid, softmax } from './math'
import { requireEngine } from './mover'

export class OpponentPredictor {
  readonly board: Board
  readonly model: EvalEngine | null
  readonly error: string | null
  readonly decay: number
  affinity: Map<number, Map<number, number>>
  played: Record<number, number[]>

  constructor(board: Board, model: EvalEngine | null, decay = 0.9) {
    const { engine, error } = requireEngine(model)
    this.board = board
    this.model = engine
    this.error = error
    this.decay = decay
    this.affinity = new Map<number, Map<number, number>>()
    this.played = { [P1]: [], [P2]: [] }
  }

  record(player: Cell, cell: number): void {
    if (!this.played[player]) this.played[player] = []
    this.played[player].push(cell)
    let aff = this.affinity.get(player)
    if (!aff) {
      aff = new Map<number, number>()
      this.affinity.set(player, aff)
    }
    aff.set(cell, (aff.get(cell) ?? 0.0) + 1.0)
  }

  newGame(): void {
    for (const aff of this.affinity.values()) {
      for (const k of Array.from(aff.keys())) {
        aff.set(k, (aff.get(k) ?? 0.0) * this.decay)
      }
    }
    this.played = { [P1]: [], [P2]: [] }
  }

  /** Net's win-probability for `player` on this position. */
  private _valueOf(board: Board, player: Cell): number {
    if (this.model === null) {
      throw new Error(this.error ?? 'no model')
    }
    const { value } = this.model.evalPosition(board.cells, player)
    return sigmoid(value)
  }

  private _scoreCell(player: Cell, idx: number, board: Board): number {
    const aff = this.affinity.get(player)?.get(idx) ?? 0.0
    board.apply(idx, player)
    const strength = this._valueOf(board, player)
    board.cells[idx] = EMPTY
    return aff + strength
  }

  predictDistribution(
    player: Cell,
    temperature = 1.0,
    topK = 8,
    board?: Board,
  ): Array<[number, number]> {
    const b = board ?? this.board
    const scored: Array<[number, number]> = []
    for (const idx of b.moves()) {
      scored.push([idx, this._scoreCell(player, idx, b)])
    }
    if (scored.length === 0) return []
    let mx = -Infinity
    for (const [, s] of scored) {
      if (s > mx) mx = s
    }
    const exps: Array<[number, number]> = []
    for (const [idx, s] of scored) {
      exps.push([idx, Math.exp((s - mx) / temperature)])
    }
    let total = 0.0
    for (const [, e] of exps) total += e
    const ranked = exps
      .map(([idx, e]) => [idx, e / total] as [number, number])
      .sort((a, b) => b[1] - a[1])
    return ranked.slice(0, topK)
  }

  predictNext(player: Cell, board?: Board): number | null {
    const dist = this.predictDistribution(player, 1.0, 1, board)
    return dist.length > 0 ? dist[0][0] : null
  }

  /** Fast opponent-reply prediction straight from the model's policy head. */
  likelyMoves(player: Cell, topK = 5, board?: Board): Array<[number, number]> {
    if (this.model === null) {
      throw new Error(this.error ?? 'no model')
    }
    const b = board ?? this.board
    const { policy } = this.model.evalPosition(b.cells, player)
    const probs = softmax(policy, 1.0)
    const scored: Array<[number, number]> = []
    for (let i = 0; i < probs.length; i++) {
      if (b.cells[i] === EMPTY) scored.push([i, probs[i]])
    }
    scored.sort((a, b) => b[1] - a[1])
    return scored.slice(0, topK)
  }
}