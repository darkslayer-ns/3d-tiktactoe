/**
 * Builds the "what the model knows about you" snapshot for the current board.
 *
 * Pure (no RN imports) so jest can test it. Reads the live AI seams — engine,
 * predictor, mover, profiles, stats — and flattens them into a displayable
 * snapshot. Nothing here mutates game state.
 */

import type { Board } from '../game/board'
import { P1, type Cell } from '../game/types'
import type { AiDecision, Difficulty, EvalEngine } from '../ai/types'
import type { OpponentPredictor } from '../ai/predictor'
import type { LookaheadMover } from '../ai/mover'
import type { PerceptionProfile, PlayerProfile } from '../ai/profile'
import type { GameStats } from '../ai/stats'
import { argmax, sigmoid } from '../ai/math'

export interface PredictionRow {
  index: number
  prob: number
}

export interface AffinityRow {
  index: number
  weight: number
}

export interface LineBar {
  label: string
  value: number
  weight: number
}

export interface ModelKnowledgeSnapshot {
  winProbHuman: number | null
  winProbAi: number | null
  bestMoveIndex: number | null
  /** Every empty cell, ranked by the model's predicted probability. */
  predictions: PredictionRow[]
  /** The human's remembered cells, ranked by affinity weight. */
  affinity: AffinityRow[]
  aggression: number
  perception: number
  perceptionBars: LineBar[]
  stats: GameStats
  winRate: number
  adaptive: number
  lastDecision: AiDecision | null
  difficulty: Difficulty
}

export interface KnowledgeInput {
  engine: EvalEngine | null
  board: Board | null
  humanSide: Cell
  difficulty: Difficulty
  predictor: OpponentPredictor | null
  mover: LookaheadMover | null
  profile: PlayerProfile | null
  perception: PerceptionProfile | null
  stats: GameStats
  adaptive: number
}

const PERCEPTION_BARS: Array<[string, keyof Pick<PerceptionProfile, 'axis' | 'face' | 'space'>]> = [
  ['AXIS', 'axis'],
  ['FACE', 'face'],
  ['SPACE', 'space'],
]

export function buildKnowledgeSnapshot(input: KnowledgeInput): ModelKnowledgeSnapshot {
  const { engine, board, humanSide, difficulty, predictor, mover, profile, perception, stats, adaptive } = input

  let winProbHuman: number | null = null
  let winProbAi: number | null = null
  let bestMoveIndex: number | null = null

  if (engine && board) {
    try {
      const humanEval = engine.evalPosition(board.cells, humanSide)
      winProbHuman = sigmoid(humanEval.value)
      if (board.emptyCount() > 0) {
        const aiSide: Cell = humanSide === P1 ? 2 : 1
        const aiEval = engine.evalPosition(board.cells, aiSide)
        winProbAi = sigmoid(aiEval.value)
        const policy = aiEval.policy
        let best = -1
        let bestV = -Infinity
        for (let i = 0; i < policy.length; i++) {
          if (board.cells[i] === 0 && policy[i] > bestV) {
            bestV = policy[i]
            best = i
          }
        }
        bestMoveIndex = best >= 0 ? best : argmax(humanEval.policy)
      }
    } catch {
      // engine hiccup — leave eval fields null
    }
  }

  let predictions: PredictionRow[] = []
  if (predictor && predictor.model && board) {
    try {
      const topK = Math.min(64, board.emptyCount())
      const dist = predictor.predictDistribution(humanSide, 1.0, topK, board)
      predictions = dist.map(([index, prob]) => ({ index, prob }))
    } catch {
      // predictor requires the engine — fall back to empty
    }
  }

  let affinity: AffinityRow[] = []
  if (predictor) {
    const row = predictor.affinity.get(humanSide)
    if (row) {
      affinity = Array.from(row.entries())
        .map(([index, weight]) => ({ index, weight }))
        .filter((r) => r.weight > 0)
        .sort((a, b) => b.weight - a.weight)
    }
  }

  const aggression = profile ? profile.aggression() : 0
  const perceptionBars: LineBar[] = PERCEPTION_BARS.map(([label, key]) => ({
    label,
    value: perception ? perception[key] : 0,
    weight: perception ? perception.score() : 0,
  }))

  return {
    winProbHuman,
    winProbAi,
    bestMoveIndex,
    predictions,
    affinity,
    aggression,
    perception: perception ? perception.score() : 0.5,
    perceptionBars,
    stats,
    winRate: stats.wins + stats.losses >= 2 ? stats.wins / (stats.wins + stats.losses) : 0.5,
    adaptive,
    lastDecision: mover ? mover.lastDecision : null,
    difficulty,
  }
}