/** Shared contracts for the AI + game state. */

import type { Cell, Coord } from '../game/types'
import { P1 } from '../game/types'

export type Difficulty = 'easy' | 'medium' | 'hard'

/** One forward pass of the transformer: value logit + policy logits. */
export interface EvalResult {
  value: number
  policy: number[]
}

/**
 * The inference seam. The real implementation calls the native C++ engine
 * (src/native/TfmEngine); tests inject a mock. Mirrors the Python
 * `eval_position(cells, player, n)` protocol exactly.
 */
export interface EvalEngine {
  evalPosition(cells: readonly Cell[], player: Cell, n?: number): EvalResult
}

export interface GameConfig {
  size: number
  difficulty: Difficulty
  /** 1 = play X (first), 2 = play O (second) */
  humanSide: Cell
}

export interface ScoredMove {
  index: number
  value: number
}

export interface LineStep {
  player: Cell
  index: number
  coord: Coord
}

/** The AI's chosen move + the search it ran (for UI telemetry). */
export interface AiDecision {
  player: Cell
  chosen: number
  coord: Coord
  kind: 'search' | 'blunder' | 'shift' | 'random'
  value: number | null
  depth: number
  scored: ScoredMove[]
  line: LineStep[]
}

/** Full game view — the on-device equivalent of the backend snapshot. */
export interface GameState {
  cells: Cell[]
  size: number
  currentPlayer: Cell
  winner: Cell
  winningLine: Coord[] | null
  over: boolean
  thinking: boolean
  movesPlayed: number[]
}

export function emptyState(size: number, _humanSide: Cell): GameState {
  return {
    cells: new Array<Cell>(size ** 3).fill(0),
    size,
    currentPlayer: P1,
    winner: 0,
    winningLine: null,
    over: false,
    thinking: false,
    movesPlayed: [],
  }
}