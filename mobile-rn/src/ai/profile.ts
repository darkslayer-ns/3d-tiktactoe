/**
 * Player style profiling — pure, RN-free, unit-testable.
 *
 * Classifies each of the human's moves as one of:
 *   - "attack": the move wins, or creates an immediate win-threat (n-1 in a
 *     line with an open completing cell) for the player.
 *   - "defend": the move blocks an immediate win-threat the opponent had.
 *   - "neutral": neither (a setup/positioning move).
 *
 * A decaying running tally produces an `aggression` score in [-1, 1]:
 *   +1 = pure attacker, -1 = pure defender, 0 = unknown/mixed.
 */

import { Board } from '../game/board'
import { EMPTY, type Cell, type Coord } from '../game/types'

export type MoveStyle = 'attack' | 'defend' | 'neutral'

/** How "deep" a line runs through the cube (axis = easiest, space = hardest). */
export type LineAxis = 'axis' | 'face' | 'space' | 'none'

export function directionType(d: Coord): LineAxis {
  let nz = 0
  if (d[0] !== 0) nz++
  if (d[1] !== 0) nz++
  if (d[2] !== 0) nz++
  return nz === 1 ? 'axis' : nz === 2 ? 'face' : 'space'
}

/** True if `side` would win immediately by playing `cell` on `board`. */
export function wouldWin(board: Board, side: Cell, cell: number): boolean {
  if (board.cells[cell] !== EMPTY) return false
  board.cells[cell] = side
  const w = board.winner().player === side
  board.cells[cell] = EMPTY
  return w
}

/** True if `side` has at least one immediate winning move available. */
export function hasWinInOne(board: Board, side: Cell): boolean {
  for (const i of board.moves()) {
    if (wouldWin(board, side, i)) return true
  }
  return false
}

/**
 * Classify `move` (an empty cell) played by `player` on `board` — `board` is
 * the position BEFORE the move. Does not mutate `board`.
 */
export function classifyMove(board: Board, player: Cell, move: number): MoveStyle {
  return analyzeMove(board, player, move).style
}

/** Find the axis (axis/face/space) of the line `move` threatens/blocks. */
function threatAxis(board: Board, side: Cell, move: number): LineAxis {
  const c = board.coord(move)
  for (const line of board.lines) {
    let contains = false
    for (const p of line) {
      if (p[0] === c[0] && p[1] === c[1] && p[2] === c[2]) {
        contains = true
        break
      }
    }
    if (!contains) continue
    let cnt = 0
    for (const p of line) if (board.get(p[0], p[1], p[2]) === side) cnt++
    if (cnt >= board.n - 1) {
      const a = line[0]
      const b = line[1]
      return directionType([b[0] - a[0], b[1] - a[1], b[2] - a[2]])
    }
  }
  return 'none'
}

/**
 * Classify `move` and also report which line type it engages — used to profile
 * how well the player perceives the 3D (space-diagonal) lines.
 */
export function analyzeMove(
  board: Board,
  player: Cell,
  move: number,
): { style: MoveStyle; axis: LineAxis } {
  const opp: Cell = player === 1 ? 2 : 1
  if (wouldWin(board, opp, move)) {
    return { style: 'defend', axis: threatAxis(board, opp, move) }
  }
  board.cells[move] = player
  const win = board.winner().player === player
  const threat = !win && hasWinInOne(board, player)
  let axis: LineAxis = 'none'
  if (win || threat) axis = threatAxis(board, player, move)
  board.cells[move] = EMPTY
  return { style: win || threat ? 'attack' : 'neutral', axis }
}

/**
 * 3D-perception profile: weights the human's attacking/defending moves by the
 * depth of the line they play on (space diagonal = 1, face diagonal = 0.5,
 * axis = 0). `score()` ∈ [0,1]: 1 = sees deep lines, 0 = only surface lines.
 */
export class PerceptionProfile {
  axis = 0
  face = 0
  space = 0
  readonly decay: number

  constructor(decay = 0.95, init?: Partial<{ axis: number; face: number; space: number }>) {
    this.decay = decay
    this.axis = init?.axis ?? 0
    this.face = init?.face ?? 0
    this.space = init?.space ?? 0
  }

  record(axis: LineAxis): void {
    this.axis *= this.decay
    this.face *= this.decay
    this.space *= this.decay
    if (axis === 'space') this.space += 1
    else if (axis === 'face') this.face += 1
    else if (axis === 'axis') this.axis += 1
  }

  score(): number {
    const total = this.axis + this.face + this.space
    if (total < 1) return 0.5
    return (this.space + 0.5 * this.face) / total
  }

  toJSON(): { axis: number; face: number; space: number } {
    return { axis: this.axis, face: this.face, space: this.space }
  }
}

export interface ProfileCounts {
  attack: number
  defend: number
  neutral: number
}

export class PlayerProfile {
  attack: number
  defend: number
  neutral: number
  readonly decay: number

  constructor(decay = 0.95, init?: Partial<ProfileCounts>) {
    this.decay = decay
    this.attack = init?.attack ?? 0
    this.defend = init?.defend ?? 0
    this.neutral = init?.neutral ?? 0
  }

  /** Decay all counts, then credit the observed move style. */
  record(style: MoveStyle): void {
    this.attack *= this.decay
    this.defend *= this.decay
    this.neutral *= this.decay
    if (style === 'attack') this.attack += 1
    else if (style === 'defend') this.defend += 1
    else this.neutral += 1
  }

  /** [-1, 1]: +1 pure attacker, -1 pure defender, 0 unknown. */
  aggression(): number {
    const total = this.attack + this.defend
    if (total < 0.5) return 0
    return (this.attack - this.defend) / total
  }

  toJSON(): ProfileCounts {
    return { attack: this.attack, defend: this.defend, neutral: this.neutral }
  }
}