/**
 * n×n×n 3D tic-tac-toe board core. Byte-for-byte port of
 * backend/game/board.py.
 *
 * Board indexed as (x, y, z) with x,y,z in [0, n). Cells are stored in a
 * flat array at index x + n*(y + n*z). Values: 0 empty, 1 X, 2 O.
 */

import { EMPTY, SUPPORTED_SIZES, cellIndex, cellCoord } from './types'
import type { Cell, Coord, BoardSize } from './types'

export { EMPTY, P1, P2, SUPPORTED_SIZES, lineCount } from './types'
export type { Cell, Coord, BoardSize } from './types'

const _linesCache = new Map<number, Coord[][]>()

/** All primitive direction vectors (no opposites, zero excluded). */
function directions(_n: number): Coord[] {
  const dirs: Coord[] = []
  for (const a of [-1, 0, 1]) {
    for (const b of [-1, 0, 1]) {
      for (const c of [-1, 0, 1]) {
        if (a === 0 && b === 0 && c === 0) continue
        if (a > 0 || (a === 0 && b > 0) || (a === 0 && b === 0 && c > 0)) {
          dirs.push([a, b, c])
        }
      }
    }
  }
  return dirs
}

/** All winning lines: n consecutive cells along a direction vector.
 *
 * Every direction is a canonical primitive vector (no opposites, zero
 * excluded) and each line's starting cell is the one nearest the origin along
 * that direction, so no line is generated twice.
 *
 * Rather than scanning all n³ cells (as a naive triple loop would), we only
 * enumerate VALID starting cells per direction: a +1 component forces the
 * start coordinate to 0, -1 forces n-1, and 0 allows any value. Total work is
 * exactly the line count (3n²+6n+4) instead of 13·n³ candidate checks. Lines
 * are emitted per direction then re-sorted by (start cell, direction) so the
 * output order matches the original cell-major enumeration.
 *
 * Results are cached per board size.
 */
export function buildLines(n: number): Coord[][] {
  const cached = _linesCache.get(n)
  if (cached) return cached
  const dirs = directions(n)
  const entries: Array<{ key: number; line: Coord[] }> = []
  let dirIdx = 0
  for (const [dx, dy, dz] of dirs) {
    const xLo = dx === 1 ? 0 : dx === -1 ? n - 1 : 0
    const xHi = dx === 0 ? n - 1 : xLo
    const yLo = dy === 1 ? 0 : dy === -1 ? n - 1 : 0
    const yHi = dy === 0 ? n - 1 : yLo
    const zLo = dz === 1 ? 0 : dz === -1 ? n - 1 : 0
    const zHi = dz === 0 ? n - 1 : zLo
    for (let x = xLo; x <= xHi; x++) {
      for (let y = yLo; y <= yHi; y++) {
        for (let z = zLo; z <= zHi; z++) {
          const line: Coord[] = new Array(n)
          for (let k = 0; k < n; k++) {
            line[k] = [x + dx * k, y + dy * k, z + dz * k]
          }
          entries.push({ key: ((x * n + y) * n + z) * dirs.length + dirIdx, line })
        }
      }
    }
    dirIdx++
  }
  entries.sort((a, b) => a.key - b.key)
  const lines = new Array<Coord[]>(entries.length)
  for (let i = 0; i < entries.length; i++) lines[i] = entries[i].line
  _linesCache.set(n, lines)
  return lines
}

export interface WinResult {
  player: Cell
  line: Coord[] | null
}

export interface OutcomeResult {
  winner: Cell
  line: Coord[] | null
  over: boolean
}

export class Board {
  readonly n: number
  cells: Cell[]
  readonly lines: Coord[][]

  constructor(n: number, cells?: readonly Cell[]) {
    if (!SUPPORTED_SIZES.includes(n as BoardSize)) {
      throw new Error(`unsupported board size ${n} (supported: [3,4,5])`)
    }
    this.n = n
    this.cells = cells !== undefined ? [...cells] : new Array<Cell>(n ** 3).fill(EMPTY)
    this.lines = buildLines(n)
  }

  // -- indexing ----------------------------------------------------------

  idx(x: number, y: number, z: number): number {
    return cellIndex(x, y, z, this.n)
  }

  get(x: number, y: number, z: number): Cell {
    return this.cells[this.idx(x, y, z)]
  }

  set(x: number, y: number, z: number, value: Cell): void {
    this.cells[this.idx(x, y, z)] = value
  }

  coord(i: number): Coord {
    return cellCoord(i, this.n)
  }

  coordOfLine(line: readonly Coord[]): number[] {
    const out: number[] = []
    for (const c of line) {
      out.push(this.idx(c[0], c[1], c[2]))
    }
    return out
  }

  // -- state -------------------------------------------------------------

  isEmpty(x: number, y: number, z: number): boolean {
    return this.get(x, y, z) === EMPTY
  }

  emptyCount(): number {
    let count = 0
    for (const c of this.cells) {
      if (c === EMPTY) count++
    }
    return count
  }

  isFull(): boolean {
    for (const c of this.cells) {
      if (c === EMPTY) return false
    }
    return true
  }

  /** Indices of empty cells. */
  moves(): number[] {
    const out: number[] = []
    for (let i = 0; i < this.cells.length; i++) {
      if (this.cells[i] === EMPTY) out.push(i)
    }
    return out
  }

  moveCount(): number {
    return this.emptyCount()
  }

  copy(): Board {
    return new Board(this.n, this.cells)
  }

  apply(index: number, player: Cell): void {
    if (this.cells[index] !== EMPTY) {
      throw new Error(`cell ${index} already occupied`)
    }
    this.cells[index] = player
  }

  // -- outcome -----------------------------------------------------------

  /** Return {player, line} or {player: EMPTY, line: null}. */
  winner(): WinResult {
    for (const line of this.lines) {
      const first = this.get(line[0][0], line[0][1], line[0][2])
      if (first === EMPTY) continue
      let all = true
      for (let i = 1; i < line.length; i++) {
        if (this.get(line[i][0], line[i][1], line[i][2]) !== first) {
          all = false
          break
        }
      }
      if (all) {
        const out: Coord[] = []
        for (const c of line) out.push([c[0], c[1], c[2]])
        return { player: first, line: out }
      }
    }
    return { player: EMPTY, line: null }
  }

  /** Return {winner, line, over}. */
  outcome(): OutcomeResult {
    const { player, line } = this.winner()
    if (player !== EMPTY) {
      return { winner: player, line, over: true }
    }
    if (this.isFull()) {
      return { winner: EMPTY, line: null, over: true }
    }
    return { winner: EMPTY, line: null, over: false }
  }

  signature(): [number, Cell[]] {
    return [this.n, [...this.cells]]
  }
}