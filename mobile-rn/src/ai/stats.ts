/**
 * Game-outcome stats for adaptive difficulty — pure, RN-free.
 *
 * Tracks the human's recent results with decay so the adaptive level responds
 * to current form, not the all-time record.
 */

import { EMPTY, type Cell } from '../game/types'

export interface GameStats {
  wins: number
  losses: number
  draws: number
}

export function emptyStats(): GameStats {
  return { wins: 0, losses: 0, draws: 0 }
}

/** Decay past results, then credit the latest outcome (human perspective). */
export function recordResult(s: GameStats, winner: Cell, humanSide: Cell, decay = 0.9): void {
  s.wins *= decay
  s.losses *= decay
  s.draws *= decay
  if (winner === EMPTY) s.draws += 1
  else if (winner === humanSide) s.wins += 1
  else s.losses += 1
}

/** Human win rate over wins+losses (draws excluded); 0.5 while too few games. */
export function winRate(s: GameStats): number {
  const total = s.wins + s.losses
  if (total < 2) return 0.5
  return s.wins / total
}

/**
 * Adaptive strength: +1 = ease the AI (human losing), -1 = harden it (human
 * winning), 0 at the ~55% target win rate.
 */
export function adaptiveLevel(s: GameStats): number {
  const r = winRate(s)
  return Math.max(-1, Math.min(1, (0.55 - r) * 2.5))
}