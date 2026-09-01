/**
 * Persistent opponent memory — pure, RN-free logic so it can be unit-tested.
 *
 * The predictor's `affinity` map (side -> cell -> weight) is what drives
 * "where is the human likely to play". This module owns the cross-game part:
 * a win REWARDS the winning side's cells (so the AI learns the moves that beat
 * it), a loss PENALIZES the losing side's cells, and everything is serialized
 * to AsyncStorage so the memory survives app restarts.
 */

import { EMPTY, type Cell } from '../game/types'

/** side (P1|P2) -> cell index -> weight. Same shape as OpponentPredictor.affinity. */
export type Affinity = Map<number, Map<number, number>>

/** Multiply the winner's played cells by this after a win. */
export const WIN_BOOST = 1.25
/** Multiply the loser's played cells by this after a loss. */
export const LOSS_DECAY = 0.5

export function emptyAffinity(): Affinity {
  return new Map()
}

/** Reward the winner's cells / penalize the loser's. No-op on a draw. */
export function applyResult(aff: Affinity, winner: Cell, loser: Cell): Affinity {
  if (winner === EMPTY) return aff
  const scale = (side: Cell, factor: number) => {
    const row = aff.get(side)
    if (!row) return
    for (const k of Array.from(row.keys())) {
      row.set(k, (row.get(k) ?? 0) * factor)
    }
  }
  scale(winner, WIN_BOOST)
  scale(loser, LOSS_DECAY)
  return aff
}

/** JSON round-trip so the Map survives AsyncStorage. */
export function serializeAffinity(aff: Affinity): string {
  const obj: Record<string, Record<string, number>> = {}
  for (const [side, row] of aff) {
    const out: Record<string, number> = {}
    for (const [cell, v] of row) out[String(cell)] = v
    obj[String(side)] = out
  }
  return JSON.stringify(obj)
}

export function parseAffinity(json: string | null | undefined): Affinity {
  const aff: Affinity = new Map()
  if (!json) return aff
  try {
    const obj = JSON.parse(json) as Record<string, Record<string, number>>
    for (const sideStr of Object.keys(obj)) {
      const row = new Map<number, number>()
      const src = obj[sideStr]
      for (const cellStr of Object.keys(src)) {
        row.set(Number(cellStr), Number(src[cellStr]) || 0)
      }
      aff.set(Number(sideStr), row)
    }
  } catch {
    return emptyAffinity()
  }
  return aff
}