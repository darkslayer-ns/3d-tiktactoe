/** Core game constants/types. 1:1 mirror of backend/game/board.py. */

export type Cell = 0 | 1 | 2

export const EMPTY: Cell = 0
export const P1: Cell = 1
export const P2: Cell = 2

export type Coord = [number, number, number]

export const SUPPORTED_SIZES = [3, 4, 5] as const
export type BoardSize = (typeof SUPPORTED_SIZES)[number]

/** index = x + n*(y + n*z) */
export function cellIndex(x: number, y: number, z: number, n: number): number {
  return x + n * (y + n * z)
}

/** Inverse of cellIndex: flat index -> (x, y, z) coords. */
export function cellCoord(i: number, n: number): Coord {
  const z = Math.floor(i / (n * n))
  const r = i % (n * n)
  const y = Math.floor(r / n)
  const x = r % n
  return [x, y, z]
}

/** Number of winning lines for an n×n×n board: ((n+2)^3 - n^3)/2. */
export function lineCount(n: number): number {
  return ((n + 2) ** 3 - n ** 3) / 2
}