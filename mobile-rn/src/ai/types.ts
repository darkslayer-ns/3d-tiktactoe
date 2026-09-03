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
  /**
   * Optional batched forward: evaluate several boards (all the same `n`) in
   * one native call, computed in parallel. `boards`/`masks` are flat
   * concatenated n^3 slices. When absent, callers fall back to evalPosition.
   */
  evalPositions?(
    boards: readonly number[],
    masks: readonly number[],
    n: number,
  ): { values: number[]; policies: number[] }
  /**
   * Optional native expectimax: scores every legal move on `cells` in ONE
   * native call (the whole lookahead runs in C++, on a background thread so
   * the JS thread / UI never blocks). Returns parallel `moves`/`values`
   * arrays in ascending index order, resolved as a Promise. When absent, the
   * mover falls back to the TS recursion.
   */
  searchScored?(
    cells: readonly Cell[],
    ai: Cell,
    depth: number,
    topK: number,
    maxNodes: number,
    aggression: number,
    n: number,
  ): Promise<{ moves: number[]; values: number[] }>
  /**
   * Optional native predicted-line telemetry (greedy/likely continuation).
   * When absent, the mover falls back to the TS loop.
   */
  predictedLine?(
    cells: readonly Cell[],
    ai: Cell,
    chosen: number,
    depth: number,
    n: number,
  ): { players: number[]; indices: number[] }
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
  /** Index of the AI's most recent move, for a "just played" highlight. */
  lastAiMove: number | null
  /** Cell recommended by the Hint button (pulses cyan). */
  hintIndex: number | null
  /** True while an AI-vs-AI demo is running (no human input). */
  demo?: boolean
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
    lastAiMove: null,
    hintIndex: null,
    demo: false,
  }
}